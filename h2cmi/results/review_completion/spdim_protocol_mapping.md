# SPDIM Protocol Mapping

Status: implemented and completed by P9 for repaired W1. Canonical result:
`spdim_w1_repaired_three_seed_results.csv`.

Available same-split comparator: `latent_im_diag_uniform` / `spdim` in internal
reports. It operates on frozen latent embeddings with a diagonal affine IM-style
recentering and is not an official SPD-manifold SPDIM implementation.

External official code checked: `https://github.com/fightlesliefigt/SPDIM`,
revision `1b0de0ccd4c48a4ff28f087b866a0b671b029c39`.

Fair H2CMI mapping for an official-code MI/W1 baseline:

1. Use H2CMI raw tensors and split IDs from the frozen MI/W1 or V2P unit.
2. Instantiate official `spdnets.models.TSMNet` with the dataset-specific
   `nchannels`/`nsamples`.
3. Train the source model on source-session data only; use source labels only.
4. Build target adaptation loader from H2CMI adaptation indices without target
   labels entering model fitting or target subsampling.
5. Run official source-free SPDIM routines:
   `get_information_maximization_geodesic` and/or
   `get_information_maximization_bias`.
6. Evaluate on H2CMI evaluation indices; use labels only for metrics.
7. Report bAcc, CI, negative-change rate, runtime, and failures with the same
   unit keys used by the review-completion MI tables.

Do not use the repository's provided pretrained demo weights for H2CMI
head-to-head: they are tied to `BNCI2015_001` and a 13-channel model. Do not use
the demo's target-label-based artificial subsampling in H2CMI confirmatory runs.

P9 followed this mapping for all 115 targets and source seeds 0/1/2. The
official result is a same-split full-pipeline baseline, not a controlled
adapter-only comparison against the H2CMI backbone.
