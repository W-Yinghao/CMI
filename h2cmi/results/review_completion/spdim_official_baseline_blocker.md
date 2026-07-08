# Official SPDIM Baseline Blocker

Updated after checking `https://github.com/fightlesliefigt/SPDIM` at revision
`1b0de0ccd4c48a4ff28f087b866a0b671b029c39`.

An official external SPDIM codebase exists and imports in the current `icml`
environment when added to `PYTHONPATH`; see `spdim_external_repo_assessment.md`.
However, no same-split official SPDIM H2CMI result has been run yet.

The official pretrained weights are not directly usable for H2CMI because they
belong to the repository demo protocol on `BNCI2015_001` with 13-channel input,
while the H2CMI MI/W1 binary tensors use `BNCI2014_001` (22 channels),
`Lee2019_MI` (62 channels), and `BNCI2014_004` (3 channels). A fair comparison
requires training official TSMNet source models on the exact H2CMI source splits
and then running official source-free SPDIM adaptation on the H2CMI adaptation
batch.

Consequence: the blocker is no longer "official code missing"; it is now
"official-code adapter and same-split source training not yet executed." The
manuscript still must not label internal `Latent-IM-Diag` as SPDIM.
