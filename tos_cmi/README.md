# TOS-CMI — Task-Orthogonal Selective Conditional MI for EEG

> Selective Conditional Invariance in Task-Orthogonal EEG Subspaces.

An **isolated** research package (peer to `h2cmi/`; it does not import-with-side-effects
or mutate the AAAI `cmi/` package). The whole method runs without real EEG on a
controllable feature simulator, so every component is exercised by a unit test.

## The idea in one line

Don't erase **all** conditional domain information `I(Z;D|Y)` (global LPC — which
collapses TSMNet, is λ-fragile on 2a, and ignores that leakage is uneven). Instead
estimate a **label Fisher** `F_Y` and a **class-conditional domain Fisher** `F_{D|Y}`,
take the generalized spectrum `F_{D|Y} v = ρ (F_Y + ηI) v`, and apply the leakage
penalty **only on the domain-rich / label-light subspace** it selects — **refusing**
(identity) when no such subspace is risk-feasible.

```
L = CE(Z) + λ · I( P_N Z ; D | Y )         # invariance only on the nuisance subspace P_N
```

## Why this is not "just disentanglement"

The novelty is concentrated in five places (THEORY.md):

1. the domain Fisher is **conditional on `Y`** (`F_{D|Y}`, not `F_D`);
2. the deletable subspace is decided by a **domain-vs-label generalized eigenproblem**;
3. CMI is estimated **only on the label-light subspace**, never globally;
4. a **risk-feasibility gate** + **within-`Y` permutation-null floor** make the method
   **degrade to identity** when no safe subspace exists (the falsifiable "refuse to
   delete" — what global LPC structurally cannot do);
5. channel / layer / graph-node use **one** CMI budget on the same selected subspace.

## Run it

```bash
# unit tests (standalone; pytest-compatible too)
for t in test_fisher test_subspace_identity_fallback test_proposition \
         test_stability test_smoke; do
  conda run -n icml python -m tos_cmi.tests.$t
done

# end-to-end demo: overlap sweep + stability gate + ERM/global-LPC/selective compare
conda run -n icml python -m tos_cmi.run_synthetic
```

## Component map

| Concept (THEORY §) | Module | Key surface |
|---|---|---|
| Two Fishers `F_Y`, `F_{D\|Y}` (§1) + permutation-null floor (§4) | [`fisher.py`](fisher.py) | `label_fisher`, `conditional_domain_fisher`, `fisher_pair`, `null_domain_energy_floor` |
| Generalized eig + risk-feasible selection + identity fallback (§2,§4) | [`subspace.py`](subspace.py) | `solve_generalized`, `select_nuisance`, `SubspaceSelector` |
| Selective penalty `λ·I(P_N Z;D\|Y)` (§3) | [`selective_cmi.py`](selective_cmi.py) | `SelectivePenalty`, `ConditionalDomainCritic`, `label_prior` |
| Controllable `(Z_Y,Z_N,overlap)` world | [`data/synthetic.py`](data/synthetic.py) | `SynthSpec`, `make`, `make_collinear` |
| Bayes-risk-preservation check (§5) | [`eval/proposition.py`](eval/proposition.py) | `bayes_risk_check` |
| Stability / termination gate (§6) | [`eval/stability.py`](eval/stability.py) | `principal_angles`, `subspace_overlap`, `selection_stability` |
| Config dataclasses | [`config.py`](config.py) | `FisherConfig`, `SubspaceConfig`, `PenaltyConfig`, `TOSConfig` |
| Run on TSMNet / 2a / GraphCMI | [`INTEGRATION.md`](INTEGRATION.md) | wiring into `cmi/` + the hardest counterexamples |

## What `run_synthetic` shows

* **overlap sweep** — at `overlap=0` the selector recovers the planted nuisance subspace
  (principal-angle overlap ≈ 1) and removing it preserves label accuracy while removing
  the leakage (the proposition); as task/domain entangle, the deletable subspace shrinks.
* **stability gate** — across seeds the selected subspace is the same object (overlap
  > 0.9) under a clear signal; pure noise yields identity on every seed (no false
  subspace). Instability here is the stated **stop** condition.
* **train compare** — ERM vs global LPC vs selective: global LPC over-erases once the
  subspaces touch; selective keeps the label-bearing directions.

## The bar this direction must clear (and when to stop)

A real-EEG win is: at matched source-only-selected λ, TOS-CMI **beats global LPC on the
TSMNet collapse case**, the selected subspace is **stable** across seed/probe/fold, and
it is **λ-robust** on 2a. If the domain-rich/label-light ranking is unstable, or it can't
save the clearest collapse case, this is "a more complicated regularizer" — abandon it.
See [`THEORY.md`](THEORY.md) §6 and [`INTEGRATION.md`](INTEGRATION.md).

## Status / honesty

Research implementation on a **simulator**: every piece is correct, differentiable,
null-calibrated, and composes end-to-end, and the proposition is what the code computes.
It is **not** a real-EEG SOTA claim. The confirmatory protocol is in `INTEGRATION.md`.
