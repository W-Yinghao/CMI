# Off-Diagonal Geometry SLURM Submission

Submitted on 2026-07-08 from `/home/infres/yinwang/CMI_AAAI_qxu`.

| role | job id | dependency | script | expected output |
|---|---:|---|---|---|
| probe | 888741 | none | `h2cmi/results/review_completion/slurm/w1offdiag_probe.slurm` | `results/h2cmi/review_completion_offdiag/probe_bnci001_s1_seed0.jsonl` |
| array | 888742 | `afterok:888741` | `h2cmi/results/review_completion/slurm/w1offdiag_array.slurm` | `results/h2cmi/review_completion_offdiag/w1offdiag_*.jsonl` |
| post | 888743 | `afterok:888742` | `h2cmi/results/review_completion/slurm/w1offdiag_post.slurm` | `h2cmi/results/review_completion/geometry_capacity_offdiagonal_results.csv` |
| watcher | 888784 | none | `h2cmi/results/review_completion/slurm/w1offdiag_watch.slurm` | `h2cmi/results/review_completion/offdiag_watch_status.md` |

Probe completion check: job `888741` finished successfully and wrote 36 rows
(`BNCI2014_001` seed 0, 9 subjects x 4 perturbations) with SHA prefix
`ce1de9c68ea7`. The initial probe filename includes `s1`, but the first runner
revision did not yet enforce `--subjects`; the runner has since been patched so
future restricted probes honor subject filters.

Array status after probe: job array `888742_[0-4]` released and running on V100.
Post job `888743` remains pending on `afterok:888742`.

Watcher `888784` was submitted afterward to poll jobs `888742,888743` and record
queue state, artifact sizes, and final accounting.

Post-completion validation is recorded in `offdiag_completion_audit.md`.
