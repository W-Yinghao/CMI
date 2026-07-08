# Validation Status

All checks in this file are CPU/artifact checks. No GPU jobs were launched for this hygiene step.

| gate | status | evidence |
|---|---|---|
| Four-branch additive identity residual max | PASS | `four_branch_complete_ci.json`: max residual across MI/Sleep panels is `5.551115123125783e-17`. |
| No empty-result CSV is described as a result | PASS | `orthogonal_score_*_results.csv` and `spdim_official_baseline_results.csv` are documented as blocker/placeholders, not negative results. |
| V2P corrected unit key preserves BNCI2014-004 transitions | PASS | `v2p_corrected_unit_key_audit.md` maps key to `(dataset,pair,subject,tgt_sess,seed,estimator)`; `pair`/`tgt_sess` distinguish repeated BNCI2014-004 transitions. |
| Sleep replay terminal values match accepted report | PASS | `sleep_replay_hash_audit.md`: W0.1 deterministic replay accepted; `G -0.020007 -> -0.020125`, `P -0.143848 -> -0.143875`, residual `1.85e-17`. |
| Offdiag raw JSONL row counts match audit | PASS | `offdiag_completion_audit.md`: 1,080 raw rows, 0 bad JSON rows, 360 unit x perturbation cells, CSV 128 data rows. |
| SPDIM status is consistent across docs | PASS | `BLOCKERS.md`, `baseline_inventory_and_blockers.md`, `spdim_external_repo_assessment.md`, `spdim_official_baseline_blocker.md`, and `spdim_protocol_mapping.md` now agree: official code exists and imports, but no same-split H2CMI official SPDIM result exists. |
| Branch reconciliation complete | PASS | `BRANCH_RECONCILIATION.md`: current artifact branch is `exp/h2cmi-wave0-mechanism`; `29a2195` is not on `origin/exp/h2cmi-responsibility-qxu`; exact cherry-pick/merge commands recorded but not run. |
| Geometry status is consistent | PASS | `geometry_capacity_blockers.md`, `geometry_capacity_stress_methods.md`, `REVIEW_COMPLETION_SUMMARY.md`, and `MANUSCRIPT_NUMBERS_READY.md` agree: frozen stress covers null/reref/gain/dropout; offdiag stress covers rotation/mixing/strong_reref/block_mixing; montage remapping remains untested. |

Remaining blockers after this hygiene step:

- Official same-split SPDIM benchmark: not run; requires preregistered adapter/source-training probe first.
- Orthogonal-score diagnostic: not run; current code lacks exported score/Fisher APIs and no frozen run artifact exists.
