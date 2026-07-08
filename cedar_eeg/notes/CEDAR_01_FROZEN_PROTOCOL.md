# CEDAR P0 Frozen Protocol

P0 consumes saved features and labels:

1. load source `z`, `y`, and domain `d`;
2. optionally load `groups` for recording/session grouped cross-fit;
3. fit label-conditional domain probes `q(D | Z, Y)`;
4. estimate leakage advantage over the `q(D | Y)` prior baseline;
5. rank latent dimensions by domain-rich / task-light source evidence;
6. evaluate cumulative masks at fixed sparsity fractions;
7. apply P0 gates;
8. write JSON with candidate decisions and an atlas.

Target labels, if available, are evaluation-only and must not be used to select
drop fractions or thresholds.

Real EEG jobs must be submitted through Slurm from this login node.
