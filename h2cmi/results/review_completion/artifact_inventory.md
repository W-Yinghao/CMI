# Artifact Inventory

| task | status | artifacts | note |
|---|---|---|---|
| MI/W1 four-branch | runnable from existing raw | p0_w1_all.jsonl + review_p0.report.json | target-subject units; seeds averaged within unit |
| Sleep/W2 deterministic confusion | complete | wave0_w2.report.json + results/h2cmi/wave0_w2det/*.jsonl | W0.1 deterministic eval-only reuse; confusion admissible |
| V2P corrected q-grid | complete | wave0_v2p.report.json + results/h2cmi/wave0_v2p/*.jsonl | 9-point q-grid, cluster=(dataset,subject) |
| Geometry capacity existing perturbations | complete | wave1_geom.report.json + w1g_*.jsonl | none/reref/gain/dropout |
| Off-diagonal geometry perturbations beyond frozen W1 | blocked | none | not pre-registered or run in existing artifacts; do not fabricate |
| Official SPDIM | blocked | none found | only `h2cmi/eval/spdim.py` Latent-IM-Diag exists; not official SPDIM |
| Orthogonal score diagnostic | blocked | none | ell_theta/ell_eta/Fisher blocks not implemented/exported |
