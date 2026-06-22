# TOS-CMI — (aspirationally) Task-Orthogonal Selective Conditional MI for EEG

> Working title: *Selective Conditional Invariance in Task-Orthogonal EEG Subspaces.*
> **Current status: a synthetic-only proof-of-concept of the selection + projection
> scaffold.** It is not yet EEG-validated and not yet wired into the trainer/TSMNet — see
> "Honest status" below and [`THEORY.md`](THEORY.md) §8.

An **isolated** research package (peer to `h2cmi/`; does not import-with-side-effects or
mutate the AAAI `cmi/` package). The whole pipeline runs without real EEG on a controllable
feature simulator, so every component is exercised by a unit test.

## The idea in one line

Don't erase **all** conditional domain information `I(Z;D|Y)` (global LPC — which collapses
TSMNet, is λ-fragile on 2a, and ignores that leakage is uneven). Instead estimate a label
scatter `F_Y` and a class-conditional domain scatter `F_{D|Y}`, take the generalized spectrum
`F_{D|Y} v = ρ (F_Y + ηI) v`, and apply the leakage penalty **only on the domain-rich /
label-light subspace** it selects — **refusing** (identity) when no such subspace is
risk-feasible.

```
L = CE(Z) + λ · I( P_N Z ; D | Y )         # invariance only on the nuisance subspace P_N
```

## Two honesty caveats up front (the things the name overclaims)

* **First-moment only.** `F_Y`, `F_{D|Y}` are between-group **mean** scatters. They are blind
  to task/domain information in covariance/SPD geometry, higher moments, or nonlinear
  interactions, so the selected subspace is **label-mean-scatter-light**, not provably
  task-orthogonal. [`tests/test_limits.py`](tests/test_limits.py) is the explicit
  covariance-only counterexample where the selector (correctly, given what it measures)
  no-ops. The "task-orthogonal" name needs the **score-Fisher** version
  ([`score_fisher.py`](score_fisher.py), now implemented through Phase 1.2: model-expected
  score Fishers + coordinate-covariant metric + source-risk UCB rank gate). The
  parameter-level gradient-conflict (PCGrad) and the conditional-on-task training critic
  remain future work.
* **`F_{D|Y}` is a proxy, not CMI.** It is a first-moment linear surrogate for `I(Z;D|Y)`,
  not the CMI nor a bound on it. `domadv_*` in the eval is a *linear-probe advantage*, not
  mutual information.

## Relation to prior work (the novelty is conditional, see THEORY §8)

Class-conditional scatter + a generalized eigenproblem ≈ **Scatter Component Analysis**;
class-conditional first moments for invariant/spurious subspaces ≈ **ISR**; minimal-damage
linear concept erasure ≈ **LEACE**; task-covariance-preserving erasure ≈ **SPLINCE**. On its
own the mean-scatter selector here is *not* a new contribution. The defensible delta is the
**score-Fisher task/domain subspace + a source-risk upper-bound rank gate** (both implemented
in [`score_fisher.py`](score_fisher.py), Phase 1.1–1.2, synthetic-only), plus the still-to-build
**parameter-level conflict projection + one budget across layer/channel/node/edge to cure the
observed global-CMI collapse** (THEORY §8).

## Run it

```bash
# unit tests (standalone; pytest-compatible too) — env: conda `icml` (py3.9, torch 2.8, scipy)
for t in test_fisher test_subspace_identity_fallback test_projection_ablation \
         test_stability test_limits test_smoke; do
  conda run -n icml python -m tos_cmi.tests.$t
done

# end-to-end demo (writes a results artifact with env + seeds + numbers)
conda run -n icml python -m tos_cmi.run_synthetic --out results/tos_cmi_synthetic.json
```

## Component map

| Concept (THEORY §) | Module | Key surface |
|---|---|---|
| Mean Fishers `F_Y`, `F_{D\|Y}` (§1) + within-Y permutation-null floor (§4) | [`fisher.py`](fisher.py) | `label_fisher`, `conditional_domain_fisher`, `fisher_pair`, `null_domain_energy_floor` |
| Generalized eig + risk-feasible selection + identity fallback (§2,§4) | [`subspace.py`](subspace.py) | `solve_generalized`, `select_nuisance`, `SubspaceSelector` (`nn.Module`, buffered `P`) |
| Selective penalty `λ·I(P_N Z;D\|Y)` (§3) | [`selective_cmi.py`](selective_cmi.py) | `SelectivePenalty`, `ConditionalDomainCritic`, `label_prior` |
| Controllable worlds | [`data/synthetic.py`](data/synthetic.py) | `make` (overlap), `make_collinear` (no-safe-subspace), `make_covariance_only` (first-moment blind spot) |
| Leakage-free projection ablation (§5) | [`eval/projection_ablation.py`](eval/projection_ablation.py) | `linear_probe_projection_ablation` (3-way split) |
| Stability / recovery / termination gate (§6) | [`eval/stability.py`](eval/stability.py) | `projection_distance`, `precision_recall`, `selection_stability` |
| Config dataclasses | [`config.py`](config.py) | `FisherConfig`, `SubspaceConfig`, `PenaltyConfig`, `TOSConfig` |
| Run on TSMNet / 2a / GraphCMI (PLAN, not done) | [`INTEGRATION.md`](INTEGRATION.md) | wiring into `cmi/` + the hardest counterexamples |

## What `run_synthetic` shows (synthetic, aligned with the method's assumptions)

* **overlap sweep** — `overlap=0`: selection sits inside the planted nuisance span
  (precision≈1, recall partial) and removing it keeps linear label accuracy while removing
  the linear conditional-domain advantage; as task/domain entangle the deletable subspace
  shrinks toward identity. Estimated on selector-train, read on a disjoint probe-test.
* **stability gate** — across **sample draws of one fixed world**, selection is
  **core-stable** (nested spans, bounded k-flicker, consistent identity decisions); the
  **strict projection-distance bar is NOT yet met** (`proj_dist_strict_pass=false`, k
  flickers 2↔3 — eigengap/hysteresis is future work, surfaced not hidden); pure noise →
  identity on every draw.
* **train compare** — ERM vs global LPC vs selective, over **multiple seeds and λ**, reporting
  **both** test bAcc **and** post-training linear domain advantage: global LPC trades accuracy
  for leakage reduction once subspaces touch; selective aims to keep the label-bearing part.

## The bar this direction must clear (and when to stop)

A real-EEG win is: at matched source-only-selected λ, TOS-CMI **beats global LPC on the
TSMNet collapse case**, the selected subspace is **stable** (projection distance) across
seed/probe/fold, and it is **λ-robust** on 2a. If the ranking is unstable or it can't save
the clearest collapse case, this is "a more complicated regularizer" — abandon it
(THEORY §6/§8, [`INTEGRATION.md`](INTEGRATION.md)).

## Honest status

Synthetic-only research prototype. The selection/projection/null-floor/identity-fallback
machinery is implemented and unit-tested; the projection ablation is leakage-free and the
stability metrics are dimension-sensitive. **Not** done: a true source-risk gate, the
score-Fisher gradient-conflict core, the conditional-on-task objective, harder synthetic
stressors, and any EEG/trainer/TSMNet wiring. `repos/TSMNet` referenced in INTEGRATION is an
**untracked external checkout**, not part of this branch.
