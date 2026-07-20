# C41 Red-Team Verification

Scope: C41 Global Leakage-Target Utility Objective Field Audit.

## Checks

- Artifact scope: C41 reads committed C30/C34S/C35/C36/C37/C38/C40 tables only. No training, GPU, or reinference path is used.
- Field availability: selection UCL is marked selected-only and is not proxied as a global candidate field; target gauge and target-unlabeled fields stay local/non-source-only.
- Sign orientation: lower-better source fields are sign-flipped before AUC/Spearman; target utility uses C34 `continuous_joint_min_margin` as the frozen higher-better scalar.
- Pooling: candidate-level alignment is computed within trajectory first, then summarized across the 162 trajectories.
- Enrichment baseline: low-leakage top-k/quantile audits use trajectory-conditioned random baselines and Bonferroni per-row p values.
- Taxonomy restraint: O4 is not activated because robust-local-alternative labels show sparse ratio enrichment despite joint-good being below baseline. O8 is not activated because 0.789 is below the pre-registered 0.800 gate.
- Claim boundary: C41 makes no atom mechanism claim and emits no selected-checkpoint method artifact.

## Verification

- C41 focused Slurm job `890084`: `8 passed in 0.24s`.
- C23-C41 regression Slurm job `890085`: `173 passed in 33.86s`.

## Verdict

C41 is internally consistent with its pre-registered gates. The accepted taxonomy is conservative: `O2 + O5 + O6`; `O4` and `O8` remain inactive.
