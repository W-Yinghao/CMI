# V2P_MECHANISM_AUDIT — controlled real-signal prevalence intervention (frozen pre-registration)

INDEPENDENT of Stage V. Does NOT reopen or rewrite any Stage V verdict. Cannot change:
  STAGE_V_COMPLETE ; V2_A_OPERATOR_SUPPORT_SAFETY_HOLDS ; V2_B_SUPPORTED_UTILITY_NOT_ESTABLISHED.
Even if an operator looks good here, V2-B is NOT retroactively "passed". This is a CONTROLLED
intervention (3:1 / 1:3 pools are BUILDER-CONSTRUCTED from labels), NOT a natural label-shift
benchmark, and NOT part of metadata-routing pass/fail.

## Question
fixed prior removes adaptive prior FEEDBACK, but may not remove prevalence-MISSPECIFICATION bias:
fixed-reference soft responsibilities can still move the geometry with adaptation-pool prevalence
under class overlap. So: when the real EEG trials are held fixed and ONLY the unlabeled adaptation-
pool class composition changes, does the geometry estimator chase prevalence?

## Frozen design (no extra ratios / methods / tuning)
ratios:   1:1, 3:1, 1:3            (left:right; same TOTAL pool size M across ratios)
methods:  identity, always_pooled, always_canonical_CC, current_joint
data:     the V2-B supported cross-session targets (BNCI2014_001 0->1, Lee2019_MI 0->1,
          BNCI2014_004 0->1 / 2->3 / 2->4)
source:   REUSE the frozen V2-B source checkpoints exactly (no retraining)
eval:     ONE fixed evaluation set per target unit, identical across all three ratios
guarantees: equal total adaptation-pool N across ratios; identical eval trials; pool construction
  trial-IDs + seed written to a manifest BEFORE estimator runs; target labels used ONLY by the
  benchmark builder to construct pools, NEVER entering any estimator; no result-driven selection of
  subjects / ratios / datasets.

## Primary estimands (prevalence SENSITIVITY, not a bAcc leaderboard)
transform displacement vs the 1:1 pool:  D_m(r) = || theta_{m,r} - theta_{m,1:1} ||,  r in {3:1,1:3},
  theta = (diag log-scale a, bias b). (transform norm is the primary scalar; predicted occupancy and
  affine shift direction are secondary diagnostics -- no reliable signed projection on this artifact.)
signed movement: do the two imbalance directions produce OPPOSITE signed geometry movement (predicted
  occupancy at 3:1 vs 1:3)?
fixed-eval effect: DbAcc_{m,r} = bAcc_{m,r} - bAcc_{m,1:1}.
retained diagnostics: subject-level harm, predicted class occupancy, estimated target prior, transform
  scale/translation magnitude, dataset-stratified paired contrasts.
THE key statistical object: method x SIGNED adaptation-pool log-odds INTERACTION (not whether pooled/
  CC/joint are each individually significant).

## Interpretation grid (decided in advance)
pooled & joint move, CC stable        -> supports the mechanism split (fixed-prior CC prevalence-robust)
fixed-prior CC ALSO moves             -> supports the revised bias theorem; refutes "fixed-prior =
                                         prevalence-invariant"
none move (at 3:1 strength)           -> no detectable contamination in these MI data; SHRINK the claim

## After V2P: STOP experiment search
No 7:1/9:1; no target-only gate; no per-dataset operator; no orthogonal estimator in this cycle; no
retuning canonical CC. An orthogonal geometry estimator, if ever built, is a NEW method stage with a
NEW pre-registration -- not a patch to Stage V.
