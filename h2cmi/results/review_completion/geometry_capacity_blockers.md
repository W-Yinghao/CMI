# Geometry-Capacity Blockers

Resolved: additional off-diagonal perturbations beyond frozen W1.geometry were executed by SLURM GPU jobs after the initial audit. The run covers sensor/channel rotation, cross-channel linear mixing, stronger reference mixing, and block-wise channel mixing. Raw rows are under `results/h2cmi/review_completion_offdiag/`; the post-analysis table is `geometry_capacity_offdiagonal_results.csv`; validation and checksums are in `offdiag_completion_audit.md`.

Remaining limitation: no montage/channel-layout stress with a different physical montage was run, because the existing W1 generator operates on fixed channel tensors and does not implement cross-dataset montage remapping for this stress test. Consequence: claims can cover the executed off-diagonal linear perturbations, but not arbitrary montage-layout changes.
