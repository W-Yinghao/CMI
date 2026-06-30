# CIGL Statistical Summary (Phase 4D draft)

> Aggregates + **bootstrap 95% CIs** computed from the **existing** tracked summary JSON (no new runs):
> `results/cigl/phase3a_dgcnn_gn_multifold_confirmation/BNCI2014_001_dgcnn_gn_multifold_summary.json` and
> `results/cigl/phase3a_dgcnn_gn_second_dataset_confirmation/BNCI2015_001_dgcnn_gn_2nd_dataset_summary.json`.
> **Bootstrap method:** resample folds with replacement, 10 000 resamples, percentile 2.5/97.5; **seed 0**
> (`numpy.random.default_rng(0)`); statistic = mean over folds (each fold value is its 3-seed mean). Per-fold
> values are the summary JSON; CIs are fold-level. The per-fold **tables** regenerate via
> `scripts/collect_cigl_evidence_tables.py` (stdlib-only, no CI); the **CIs** here are from the documented
> numpy fold-bootstrap above (re-run that snippet to reproduce). No numbers are invented.

## BNCI2014_001 (CIGL_29) — **primary = folds 1–8** (fold-0 = dev, excluded); chance 0.25

| quantity | mean | 95% CI (bootstrap, folds 1–8) | per-fold range |
|---|---|---|---|
| ERM source bAcc | 0.484 | — | [0.457, 0.508] |
| reg (`graph_node_010`) source bAcc | 0.488 | [0.471, 0.505] | [0.455, 0.516] |
| source drop vs ERM | −0.002 | [−0.007, 0.003] | max +0.008 → **retain ≤0.02: 8/8** |
| graph KL reduction | 44.0% | [39.5%, 49.0%] | 35–58% |
| node KL reduction | 36.9% | [33.6%, 40.3%] | 31–45% |

Criteria (folds 1–8): ERM leakage clears **8/8**, reg reduces ≥30% **8/8**, source retained **8/8**, target
guardrail **8/8**; decision **A**.

## BNCI2015_001 (CIGL_31) — all 12 LOSO folds (no dev fold); chance 0.50

| quantity | mean | 95% CI (bootstrap, 12 folds) | per-fold range |
|---|---|---|---|
| ERM source bAcc | 0.706 | — | [0.682, 0.734] |
| reg source bAcc | 0.700 | [0.693, 0.707] | [0.676, 0.722] |
| source drop vs ERM | +0.007 | [+0.001, +0.012] | max +0.024 → **retain ≤0.02: 11/12** (fold9 miss) |
| graph KL reduction | 66.2% | [60.6%, 71.0%] | 43–77% |
| node KL reduction | 51.9% | [47.8%, 55.6%] | 37–61% |

Criteria (12 folds): ERM adequacy **12/12**, ERM leakage **12/12**, reg reduces ≥30% **12/12**, source
retained **11/12**, target guardrail **12/12**; `source_only_confirmed=T`, `target_guardrail_pass=T`,
`confirmed_with_target_guardrail=T` → decision **A**.

## Reading

- Both datasets: ERM adequate, leakage exists, the fixed `graph_node_010` reduces graph/node leakage in
  every (primary) fold, and source task **meets the pre-registered retention gate** (BNCI2014_001 8/8 primary;
  BNCI2015_001 11/12, with fold9 missing the per-fold threshold while the dataset-level gate passes). Target
  guardrail holds throughout.
- The reduction is **partial** — the *regularized* leakage **still clears the permutation null in every
  fold**; we report reduction, not elimination.
- CIs above are **fold-level bootstrap** (seed 0). A camera-ready may additionally bootstrap over
  folds×seeds from the per-seed JSON (`TODO: finalize per-seed CI`); no numbers here are invented.
