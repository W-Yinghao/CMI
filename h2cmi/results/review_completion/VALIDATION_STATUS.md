# Validation Status

All checks in this file are CPU/artifact checks. No GPU jobs were launched for this hygiene step.

| gate | status | evidence |
|---|---|---|
| Four-branch additive identity residual max | PASS | `four_branch_complete_ci.json`: max residual across MI/Sleep panels is `5.551115123125783e-17`. |
| No empty-result CSV is described as a current result | PASS | `orthogonal_score_*_results.csv` remain blocker placeholders; `spdim_official_baseline_results.csv` is a superseded historical placeholder. The canonical SPDIM result is `spdim_w1_repaired_three_seed_results.csv`. |
| V2P corrected unit key preserves BNCI2014-004 transitions | PASS | `v2p_corrected_unit_key_audit.md` maps key to `(dataset,pair,subject,tgt_sess,seed,estimator)`; `pair`/`tgt_sess` distinguish repeated BNCI2014-004 transitions. |
| Sleep replay terminal values match accepted report | PASS | `sleep_replay_hash_audit.md`: W0.1 deterministic replay accepted; `G -0.020007 -> -0.020125`, `P -0.143848 -> -0.143875`, residual `1.85e-17`. |
| Offdiag raw JSONL row counts match audit | PASS | `offdiag_completion_audit.md`: 1,080 raw rows, 0 bad JSON rows, 360 unit x perturbation cells, CSV 128 data rows. |
| SPDIM status is consistent across docs | PASS | `BLOCKERS.md`, `baseline_inventory_and_blockers.md`, `spdim_external_repo_assessment.md`, and `spdim_official_baseline_blocker.md` point to the complete P9 repaired-split three-seed result. |
| Branch reconciliation complete | PASS | `BRANCH_RECONCILIATION.md`: current artifact branch is `exp/h2cmi-wave0-mechanism`; `29a2195` and `483ff8c` are not on `origin/exp/h2cmi-responsibility-qxu`; exact cherry-pick/merge commands recorded but not run. |
| Geometry status is consistent | PASS | `geometry_capacity_blockers.md`, `geometry_capacity_stress_methods.md`, `REVIEW_COMPLETION_SUMMARY.md`, and `MANUSCRIPT_NUMBERS_READY.md` agree: frozen stress covers null/reref/gain/dropout; offdiag stress covers rotation/mixing/strong_reref/block_mixing; montage remapping remains untested. |
| P0.5 provenance head recorded | PASS | `RUN_PROVENANCE.md` records `analysis_base_commit=2838327`, `artifact_commit=29a2195`, `hygiene_digest_commit=483ff8c`, current remote head `483ff8c`, artifact branch, and responsibility branch head `09e9249`. |

Current unresolved blockers:

- Orthogonal-score diagnostic: not run; current code lacks exported score/Fisher APIs and no frozen run artifact exists.
- Montage-layout or cross-montage remapping stress: not run by the bounded
  geometry panels.

Official SPDIM is resolved by P9; no additional GPU work is required for the
canonical repaired-W1 result.
