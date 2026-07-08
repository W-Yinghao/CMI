# Geometry-Capacity Stress Methods

Existing W1.geometry reuses frozen V2P bundles and applies channel-space perturbations `none`, `reref`, `gain`, and `dropout` to adaptation/evaluation signals. Unit is `(dataset, subject)` after averaging source seeds within unit; bootstrap cluster is `(dataset, subject)` with 10,000 percentile resamples. Primary falsification contrast is `max(CORAL-latent, EA-sensor) - max(FRSC, fixed_iterative, joint, latent_im_diag, pooled)`. Results are in `geometry_capacity_existing_ci.csv`.

Off-diagonal review-completion stress is launched additively with raw rows under `results/h2cmi/review_completion_offdiag/`. Perturbations are `rotation`, `mixing`, `strong_reref`, and `block_mixing`. Unit is `(dataset, pair, subject)` after averaging source seeds within unit; bootstrap cluster is `(dataset, pair, subject)` with 10,000 percentile resamples. The post job writes `geometry_capacity_offdiagonal_results.csv`.
