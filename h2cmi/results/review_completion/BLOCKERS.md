# Blockers

1. Official SPDIM: external official code was identified at `github.com/fightlesliefigt/SPDIM` and import/smoke checks passed, but no same-split H2CMI official-code run has been executed. Provided official pretrained weights target a different BNCI2015_001 demo protocol and are not compatible with H2CMI channel counts. Consequence: manuscript must not label Latent-IM-Diag as SPDIM and must not report official SPDIM numbers until an adapter/source-training run is completed.
2. Orthogonal-score diagnostic: missing score/Fisher APIs and no frozen run artifact. Consequence: no orthogonal estimator result is available.

Resolved after this audit:

- Off-diagonal geometry beyond frozen W1: executed SLURM GPU jobs for rotation, cross-channel mixing, stronger reference mixing, and block mixing. Raw rows are under `results/h2cmi/review_completion_offdiag/`; aggregate CSV is `geometry_capacity_offdiagonal_results.csv`; completion audit is `offdiag_completion_audit.md`.
