# C42 Red-Team Verification

Scope: C42 Source-Rank Actionability / Rank-to-Selector Gap Audit.

## Checks

- Artifact scope: C42 reads committed C30/C32/C35/C37/C38/C40/C41 artifacts only. No training, GPU, score tuning, feature selection, or reinference path is used.
- Score registry: C30 source-rank orientation is fixed as higher-better before analysis; C19 robust-core and C30 source-rank are explicitly marked as the same frozen score under different interpretations.
- AUC-to-top-k boundary: C30 AUC and C42 continuous-utility AUC are reported separately from top1/top-k localization. The report does not infer reliability from pairwise AUC alone.
- Base-rate check: every top-k result is paired with a trajectory-conditioned random baseline. The source-rank top1 gain over random is only 0.076.
- Top-region check: source-rank top regions are plateaued under the fixed eps=0.02 rule; this supports R7 and blocks a clean top1 claim.
- Gauge check: regime/target centering does not improve source-rank top1 localization; R5 remains inactive. The target-rank row is a diagnostic ceiling only.
- Leakage conflict check: leakage-vs-rank conflict rows omit candidate ids, checkpoint hashes, and model hashes. Target-gauge delta for rank-top vs OACI is marked unavailable, with no proxy.
- Claim boundary: C42 makes no deployment claim and emits no selected-checkpoint method artifact.

## Verification

- C42 focused Slurm job `890091`: `10 passed in 0.16s`.
- C23-C42 regression Slurm job `890090`: `183 passed in 32.24s`.

## Verdict

C42 is internally consistent with its pre-registered gates. The accepted candidate taxonomy for remote review is conservative: `R1 + R2 + R3 + R6 + R7 + R8 + R9`; `R4`, `R5`, and `R10` remain inactive.
