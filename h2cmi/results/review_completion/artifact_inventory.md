# Artifact Inventory

| task | status | artifacts | note |
|---|---|---|---|
| MI/W1 repaired four-branch | complete, canonical | `w1_repaired_h2cmi_results.csv` + `w1_repaired_h2cmi_four_branch_ci.csv` | P7 repaired split; target-subject units; seeds averaged within unit |
| MI/W1 original contiguous split | superseded legacy | `p0_w1_all.jsonl` + `four_branch_complete_ci.csv/json` | diagnostic history only; not a current W1 result |
| Sleep/W2 deterministic confusion | complete | wave0_w2.report.json + results/h2cmi/wave0_w2det/*.jsonl | W0.1 deterministic eval-only reuse; confusion admissible |
| V2P corrected q-grid | complete | wave0_v2p.report.json + results/h2cmi/wave0_v2p/*.jsonl | 9-point q-grid, cluster=(dataset,subject) |
| Geometry capacity existing perturbations | complete | wave1_geom.report.json + w1g_*.jsonl | none/reref/gain/dropout |
| Off-diagonal geometry perturbations beyond frozen W1 | complete exploratory | `geometry_capacity_offdiagonal_results.csv` + `offdiag_completion_audit.md` | bounded rotation/mixing/strong-reref/block-mixing stress |
| Official SPDIM | complete, canonical | `spdim_w1_repaired_three_seed_results.csv` + summary/CI artifacts | P9 official repaired-split three-seed baseline; internal Latent-IM-Diag remains a separate comparator |
| Orthogonal score diagnostic | blocked | none | ell_theta/ell_eta/Fisher blocks not implemented/exported |
| Montage-layout remapping stress | blocked | none | bounded perturbations do not test cross-montage channel-layout remapping |
