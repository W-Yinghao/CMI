# Review Completion Summary

This package is an additive review-completion artifact audit. It should be accepted as a result/support package, not as proof that all reviewer concerns are solved.

## Completed Confirmatory Reanalysis

- Four-branch decomposition CIs are complete in `four_branch_complete_ci.csv/json`, including `I_int` for MI/W1 and Sleep/W2.
- MI dataset heterogeneity is complete in `mi_dataset_heterogeneity_complete_ci.csv`; the MI aggregate is positive but concentrated in Cho2017.
- Sleep deterministic branch replay is accepted in `sleep_replay_hash_audit.md`; confusion matrices and per-stage recalls are in `sleep_branch_confusion_matrices.json` and `sleep_per_stage_recall.csv`.
- V2P corrected unit-key reanalysis is complete in `v2p_corrected_*`; the corrected key preserves `(dataset,pair,subject,target_session,source_seed,method)` and avoids collapsing BNCI2014-004 repeated transitions.
- Existing frozen W1 geometry stress for null/reref/gain/dropout is summarized in `geometry_capacity_existing_ci.csv`.

## Completed Exploratory/Supplemental Run

- Off-diagonal geometry stress was newly executed by SLURM GPU jobs for rotation, cross-channel mixing, stronger reref, and block mixing.
- Raw rows are under `results/h2cmi/review_completion_offdiag/`.
- Aggregate table is `geometry_capacity_offdiagonal_results.csv`.
- Completion validation is in `offdiag_completion_audit.md`: 1,080 raw rows, 0 bad JSON rows, 360 unit x perturbation cells after seed averaging, and 128 CSV data rows.
- Because `sacct` was unavailable, validation uses queue exit, log/output existence, parse checks, row counts, and checksums rather than Slurm accounting DB.

## Manuscript-Ready Numbers

Use `MANUSCRIPT_NUMBERS_READY.md` as the single writer-facing number digest. Key headline values:

- MI/W1: `G=+0.0604 [+0.0411,+0.0811]`, `P=-0.0065 [-0.0147,+0.0021]`, `I_int=+0.0043 [-0.0012,+0.0099]`, n=115.
- Sleep/W2 primary: `G=-0.0201 [-0.0407,+0.0010]`, `P=-0.1439 [-0.1593,-0.1285]`, `I_int=+0.0588 [+0.0425,+0.0758]`, n=75.
- MI heterogeneity: Cho2017 drives the MI geometry aggregate (`G=+0.1227 [+0.0866,+0.1602]`), while BNCI2014-001 and Lee2019-MI are near zero.
- V2P q-grid is `{0.1,...,0.9}`; displacement should not be described as utility.
- Offdiag geometry contrasts are near zero with CIs spanning zero for rotation/block_mixing and non-positive for mixing/strong_reref; use bounded stress wording.

## Blockers

- Official SPDIM: external official code exists and imports/smoke-passes, but no same-split H2CMI official SPDIM result has been run. Provided official pretrained weights target BNCI2015_001 with 13 channels and are not compatible with H2CMI 22/62/3-channel MI tensors.
- Orthogonal-score diagnostic: no result exists. Current code lacks exported score/Fisher APIs and no frozen run artifact exists. Header-only CSVs are blocker placeholders, not negative results.
- Geometry montage stress: cross-montage/channel-layout remapping remains untested.

## Claims To Strengthen

- The main sleep result can be described as a metric-prior diagnostic under balanced accuracy, supported by deterministic replay and per-stage recall tables.
- The V2P corrected analysis preserves repeated transitions and supports the distinction between displacement and utility.
- MI can be described as aggregate-positive but heterogeneous, with Cho2017 carrying most of the geometry effect.

## Claims To Weaken

- Do not claim an official SPDIM comparison.
- Do not call internal `Latent-IM-Diag` SPDIM.
- Do not claim orthogonal-score evaluation.
- Do not claim universal diagonal-geometry adequacy; say bounded operator-family stress.
- Do not imply montage remapping was tested.

## Files To Cite In The Paper

- `MANUSCRIPT_NUMBERS_READY.md`
- `four_branch_complete_ci.csv`
- `mi_dataset_heterogeneity_complete_ci.csv`
- `sleep_replay_hash_audit.md`
- `sleep_branch_confusion_matrices.json`
- `sleep_per_stage_recall.csv`
- `v2p_corrected_unit_key_audit.md`
- `v2p_corrected_grid_summary.csv`
- `v2p_corrected_method_summary.csv`
- `geometry_capacity_existing_ci.csv`
- `geometry_capacity_offdiagonal_results.csv`
- `offdiag_completion_audit.md`
- `spdim_external_repo_assessment.md`
- `spdim_official_baseline_blocker.md`
- `orthogonal_score_blockers.md`
