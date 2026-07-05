# CIGL R1 — Evidence hardening + multi-probe stress audit (non-GPU scaffold)

Branch `project/cigl-r123-scaffold` off the CIGL audit line `fb5761d`. Implements the CIGL_45 R1 program:
make "graph/node leakage exists and CIGL reduces it" **statistically unimpeachable** and **probe-family
robust**. Code + tests only, **no GPU**. (Note: this doc is CIGL_57 on the audit line; a different CIGL_57
exists on the P10 sidecar branch — distinct content, different branch.)

## What R1 adds (on top of the existing audit)

The phase-3A audit today uses a single 1-layer MLP probe, `n_perm` 20–50, 3 seeds, **no FDR**. R1 hardens that:

### `cmi/eval/evidence_hardening.py`
- `exact_permutation_pvalue(observed, null, tail)` — `p = (1 + #{null ≥ obs}) / (B + 1)` (Phipson–Smyth;
  never 0).
- `benjamini_hochberg(pvalues, alpha)` — step-up FDR across the family of (fold/seed/probe) tests; returns
  rejected mask + BH-adjusted p + critical p.
- `hierarchical_bootstrap(records, levels=(dataset,fold,seed))` — **cluster** bootstrap that resamples
  dataset→fold→seed with replacement (a flat bootstrap understates the CI by ignoring within-fold/-dataset
  correlation).
- `harden_leakage_table(rows)` — end-to-end: exact p per row → BH-FDR → fraction cleared → hierarchical CI on
  the reduction.

### `cmi/eval/multiprobe_audit.py`
- Wires the EXISTING 7-probe suite (`leakage_audit.audit`: `linear/mlp_s/mlp_l/rf/hgbm` advantages + `hsic` +
  `knn_cmi`) over frozen features with a **per-probe within-label permutation null** (shuffle D within Y) +
  BH-FDR + the **≥5/7 agreement** criterion. `leakage_exists` iff ≥ `min_agree` probe families clear FDR.

### `cmi/eval/probe_calibration.py`
- `expected_calibration_error` (top-label ECE) + `fit_temperature` (1-parameter NLL min) + `calibration_report`
  (ECE before/after temperature scaling). Shows the KL leakage proxy is not a probe-miscalibration artifact.

## Tests (CPU, pass)
`test_evidence_hardening.py` (exact-p bounds/formula, BH known-case + adjusted-p monotone, hierarchical CI
brackets the mean & ≥ flat CI, end-to-end table); `test_multiprobe_audit.py` (≥5/7 detect a planted
within-label leak; a no-leak rep clears under FDR); `test_probe_calibration.py` (ECE=0 for perfect calibration,
temperature scaling reduces over-confidence, fitted T scales with logit inflation).

## Gated GPU runs (NOT launched — for the PI's run-spec review)
Per CIGL_45 R1: (2a) 2a fold-0 hardened pilot `n_perm=1000`, seeds 3, ERM + `graph_node_010`; (2b) graph-only /
node-only / graph+node ablation; (2c) λ-curve {0,.001,.003,.01,.03,.1}; (3a) multi-fold × 10-seed confirmation
`n_perm=1000` with FDR + hierarchical CI; (3b/4a) multi-probe + calibration post-hoc on frozen features (CPU).
**Success:** `graph_node_010` clears the **BH-FDR** null on ≥2/3 seeds (pilot) / primary folds (confirmation),
**≥5/7 probes** agree, ranges hold under 10 seeds. **Honest stop:** if the hardened null is not cleared, the
effect was under-powered — report and re-scope. No GPU until the run-spec is reviewed.
