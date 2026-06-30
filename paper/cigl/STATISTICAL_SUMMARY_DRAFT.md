# CIGL Statistical Summary (Phase 4C draft)

> Aggregates computed from the **existing** tracked summary JSON (no new runs):
> `results/cigl/phase3a_dgcnn_gn_multifold_confirmation/BNCI2014_001_dgcnn_gn_multifold_summary.json` and
> `results/cigl/phase3a_dgcnn_gn_second_dataset_confirmation/BNCI2015_001_dgcnn_gn_2nd_dataset_summary.json`.
> Ranges are per-fold min–max (mean); seeds 0,1,2; n_perm=50; gate α=0.05.

## BNCI2014_001 (CIGL_29) — 9 LOSO folds; **primary = folds 1–8** (fold-0 = dev, excluded)

| quantity | value |
|---|---|
| ERM source bAcc | mean 0.484, range [0.457, 0.508] (chance 0.25) |
| reg (`graph_node_010`) source bAcc | mean 0.486, range [0.455, 0.516] |
| source drop vs ERM | mean −0.002, max +0.008 → **retain (≤0.02): 9/9** (primary 8/8) |
| graph KL reduction | 35–58% (mean ≈ 44%) |
| node KL reduction | 31–45% (mean ≈ 37%) |
| ERM leakage clears / reg reduces ≥30% / source retained / target guardrail | **8 / 8 / 8 / 8** of the 8 primary folds |
| regularized leakage still clears null | **every fold** (partial reduction, not elimination) |
| decision | A (primary folds 1–8) |

## BNCI2015_001 (CIGL_31) — 12 LOSO folds (all confirmation; no dev fold)

| quantity | value |
|---|---|
| ERM source bAcc | mean 0.706, range [0.682, 0.734] (chance 0.50) |
| reg source bAcc | mean 0.700, range [0.676, 0.722] |
| source drop vs ERM | mean +0.007, max +0.024 → **retain (≤0.02): 11/12** (fold9 = +0.024 miss) |
| graph KL reduction | 43–77% (mean ≈ 66%) |
| node KL reduction | 37–61% (mean ≈ 52%) |
| ERM adequacy / ERM leakage / reg reduces / source retained / target guardrail | **12 / 12 / 12 / 11 / 12** of 12 folds |
| three-layer verdict | `source_only_confirmed`=T, `target_guardrail_pass`=T, `confirmed_with_target_guardrail`=T → **A** |
| regularized leakage still clears null | **every fold** (partial reduction, not elimination) |

## Reading

- Both datasets: ERM is adequate, leakage exists, the fixed `graph_node_010` reduces graph/node leakage in
  every (primary) fold, and source task is retained within the pre-registered gate (BNCI2014_001 9/9;
  BNCI2015_001 11/12). Target guardrail holds throughout.
- The reduction is **partial** — regularized leakage still clears the permutation null in every fold; we
  report reduction, not elimination.
- **TODO: finalize** mean ± 95% CI per quantity for the camera-ready (bootstrap over folds×seeds from the
  per-seed JSON); the ranges above are exact per-fold min–max/means from the summary JSON, no invented
  numbers. Regenerate with `python scripts/collect_cigl_evidence_tables.py --out_dir <dir>` (gitignored).
