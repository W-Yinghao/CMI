# C33 - Red-Team Verification

Scope: adversarial review of the C33 Local Trajectory Boundary / Checkpoint Neighborhood Audit before commit.

## Blocking Finding And Fix

1. **B2 source-flat was initially invalid.**
   - First implementation counted selected-hit self-pairs as selected-vs-nearest-joint pairs. Those pairs have
     order delta 0 and source-score delta 0, inflating the source-flat fraction.
   - Red-team split showed most source-flat pairs were selected-hit self-pairs; among actual misses, source-flat was
     only 4/81 = 0.049.
   - Fix: selected-hit self-pairs are classified as `selected_already_joint_good` and excluded from miss-conditioned
     source-flat / source-wrong / gauge-jump fractions.
   - Result: B2 removed; B3 active local misranking enters primary (miss-conditioned source-wrong = 25/81 = 0.309).

2. **B4 wording and gate were tightened.**
   - Pair-level gauge-jump-unseen is 0.358 after miss conditioning, and transition gauge-jump fraction is 0.800.
   - C33 now reads B4 as common target-margin jumps with weak source transition alignment (source agreement 0.483),
     not as a clean pairwise unseen-gauge claim.

3. **Target-grouped local rung was downgraded.**
   - In same-target local neighborhoods, target-grouped centering is rank-invariant with source score.
   - Report now says target-grouped has zero local gain and is a non-deployable diagnostic, not a local ceiling.

## Checks That Passed

- B1 is not established: selected +/-1 neighborhood contains joint-good in 0.574 of units, below the 0.70 gate.
- B6 is not established: target-unlabeled pm1/pm2 gains are +0.025 / -0.049, below the local-help gate.
- B9 is not established: plateau size is nontrivial, but selected-bad plateau contains joint-good in only 0.256.
- No `model_hash` appears in C33 reports/tables.
- No selected-checkpoint artifact is emitted.
- Target-unlabeled R3 uses fixed label-free C24 features and LOTO scoring; target labels are diagnostic targets only.

## Surviving C33 Verdict

Primary cases: `B3 + B4 + B7 + B8`.

Interpretation: actual selected misses are not mostly source-score ties. They show active local misranking
(source prefers the selected non-joint checkpoint in 0.309 of miss pairs), plus common target-margin jumps with weak
source transition alignment. Target-unlabeled R3 does not rescue local neighborhood top-k localization. Robust margin
changes the local taxonomy by removing B8 from the robust-primary case list.
