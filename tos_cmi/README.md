# TOS-CMI — (aspirationally) Task-Orthogonal Selective Conditional MI for EEG

> Working title: *Selective Conditional Invariance in Task-Orthogonal EEG Subspaces.*
> **Current status (updated): synthetic scaffold + a completed frozen-feature EEG diagnostic study**
> on BCI-IV-2a (BNCI2014_001), LOSO, two backbones (TSMNet z=210, EEGNet z=16). The score-Fisher
> selector + projection + safety gate run on real frozen latents; flag-gated per-epoch instrumentation
> is wired into `cmi/train/trainer.py` (`log_curves`, default-off). **Still NOT done / honest negatives:**
> end-to-end TOS training (selective penalty in the loss), a *source-OOD benefit* gate, the architecture
> vs latent-dimension factorial, and *certified default-on deletion* (an honest negative — see
> [`notes/PHASE131_CERTIFICATION.md`](notes/PHASE131_CERTIFICATION.md)). Per-claim provenance:
> [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md). Paper draft + TMLR build: [`paper/`](paper/). `repos/TSMNet`
> is a SYMLINK and must be pinned before camera-ready ([`INTEGRATION.md`](INTEGRATION.md)).
>
> Headline finding (measurement-to-control gap): conditional domain leakage is measurable and sometimes
> (partially) removable, but neither measurement nor removal implies a cross-subject generalization
> benefit. See [`THEORY.md`](THEORY.md) §8 for the score-Fisher redesign rationale.

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

**Task-protected certification line: COMPLETED AS AN HONEST NEGATIVE. Default-on NOT certified at
delta_Y=0.10, n=6000** (tag `tos-task-cert-negative-v1`; full chain in
[`notes/PHASE131_CERTIFICATION.md`](notes/PHASE131_CERTIFICATION.md)). Package defaults keep
`task_protect=False`, `task_power_floor=False` — the gate is a **safety diagnostic + refuse-to-delete
module**, not a certified auto-deleter. The only deleting config is the
`certified_synthetic_experimental(table)` preset (deletes only on an EXACT fingerprint+scope
certificate hit; else identity).

What the certification framework established (synthetic, oracle-validated):
- **mean-scatter baseline** (`fisher.py`/`subspace.py`, tag `mean-scatter-v2`): first-moment,
  blind to covariance/synergy leakage — honest baseline.
- **score-Fisher selector + source-risk UCB gate** (`score_fisher.py`): model-expected score
  Fishers, covariant whitening metric, task-protected direct-sum projector, group-aware cross-fit,
  simultaneous cluster-bootstrap bands.
- **Bayes oracle** (`eval/bayes_oracle.py`): exact `I(Y;deleted|kept)` ground truth — proved the
  nested critic could UNSAFE-ACCEPT, and that the conservatism is an ESTIMATOR bottleneck (not
  n / delta_Y / intrinsic).
- **power floor / competence certificate** (`eval/power_certificate.py`): matched positive-control
  power, exact-cell + scope lookup, estimator fingerprint — abstains unless power-qualified.
- **plug-in → stacked log-ratio critic**: nested 0/30 → plug-in true-rate ~0.80 (independent-seed
  cert BORDERLINE) → stacking did not robustly improve. Verdict: safe-but-conservative.

Contribution framing: a **measurement-to-control certification framework for selective conditional
invariance** — it prevents unsafe deletion where geometry-only / weak learned gates would accept,
and is intentionally conservative at moderate n. (Answers LPC collapse / lambda-sensitivity by
proving WHEN not to delete, not by tuning lambda.)

**Frozen / not done:** conditional-on-task training critic; parameter-level PCGrad; sequential
subset search; any EEG/trainer/TSMNet wiring. Next line (separate): frozen-feature EEG pilot with
the gate as a conservative optional diagnostic. `repos/TSMNet` in INTEGRATION is an untracked
external checkout, not part of this branch.
