# V2P_MECHANISM_AUDIT — results (frozen; 1080 rows, 72 units, 270 pool-manifest entries)
# raw sha256: v2p.jsonl=4184c142db866f21  v2p_pools.jsonl=55faa49fb8de5234 . Reused frozen V2-B checkpoints (code_sig 763bf49d).

Controlled real-signal prevalence intervention: real EEG trials FIXED; only the unlabeled adaptation-
pool class composition varies {1:1,3:1,1:3} at equal total size on one fixed eval set. Labels used
ONLY by the pool builder (never by an estimator). INDEPENDENT of Stage V; changes NO Stage V verdict.

## Key estimand: method x signed pool log-odds interaction (occupancy slope, CI95 over 72 units)
    identity              occ_slope -0.000 [-0.000,-0.000]   (zero-control: prevalence-invariant by construction)
    always_pooled         occ_slope -0.062 [-0.072,-0.051]*  disp(3:1)=0.735 disp(1:3)=0.707
    current_joint         occ_slope -0.042 [-0.059,-0.025]*  disp(3:1)=0.637 disp(1:3)=0.561
    always_canonical_CC   occ_slope -0.020 [-0.028,-0.013]*  disp(3:1)=0.309 disp(1:3)=0.297
(* = CI excludes 0.) signed_occ(1:3 - 3:1): pooled -0.135, joint -0.092, CC -0.045, identity 0.000.
fixed-eval DbAcc: all |Δ| <= 0.015 (balanced eval; geometry moves but accuracy barely changes).

## VERDICT (in-advance grid): FIXED_PRIOR_CC_ALSO_MOVES -> supports the revised bias theorem
Every adaptive operator shows statistically-clear prevalence-INDUCED geometry movement (CIs exclude
0); identity is exactly flat (clean measurement). fixed-prior CC MOVES (slope -0.020, CI excl 0): it
is ~3x more prevalence-ROBUST than classless pooled (-0.062) and the joint (-0.042) but NOT
prevalence-INVARIANT. This REFUTES "fixed-prior = prevalence-invariant" and supports the revised
theorem: fixed-reference soft responsibilities still shift the geometry with adaptation-pool
prevalence under class overlap. Magnitude order pooled > joint > CC > identity(0).

Direction: movement is COMPENSATORY (a more-imbalanced pool makes the diagonal moment-match /
fixed-reference responsibilities suppress the over-represented class on the fixed eval), not naive
prior-following. The effect is real in geometry but small on balanced-eval accuracy (|DbAcc|<=0.015),
consistent with V2-B's null utility.

## Scope (binding)
Controlled intervention (3:1/1:3 are builder-constructed from labels), NOT a natural label-shift
benchmark, NOT in metadata-routing pass/fail. Does NOT change: STAGE_V_COMPLETE,
V2_A_OPERATOR_SUPPORT_SAFETY_HOLDS, V2_B_SUPPORTED_UTILITY_NOT_ESTABLISHED. Experiment search STOPS
here (no 7:1/9:1, no target-only gate, no per-dataset operator, no orthogonal estimator, no CC retune).

## AUDIT CORRECTION (post-hoc; supersedes the numbers above)
The analyzer unit key was (dataset, subject), which COLLAPSED BNCI2014_004's 3 cross-session units per
subject (90 -> 72, with overwrite). FIXED: unit key = (pair, subject) [encodes session]; bootstrap is
now SUBJECT-CLUSTERED. Raw v2p.jsonl was complete (90 units).
CORRECTED (n_units=90, 72 subject clusters): occupancy slope vs signed pool log-odds --
    identity              -0.000 [-0.000,-0.000]   (zero-control)
    always_pooled         -0.063 [-0.073,-0.054]*  (CI excludes 0)
    always_canonical_CC   -0.016 [-0.023,-0.009]*  (CI excludes 0)   <-- the KEY claim survives clustering
    current_joint         -0.019 [-0.043,+0.005]   (CI now INCLUDES 0 -> joint sensitivity DESCRIPTIVE)
(* CI excludes 0.) The fixed-prior-CC-MOVES claim (revised bias theorem) remains CONFIRMATORY after
the unit-key fix + subject clustering: CC slope CI [-0.023,-0.009] excludes 0 and is ~1/4 of pooled
(-0.063). The current_joint slope CI now includes 0 (its per-session variance, hidden by the old
collapse, is large), so the joint's prevalence-sensitivity is DESCRIPTIVE, not confirmatory. Net:
pooled & fixed-prior-CC are CI-confirmed prevalence-sensitive; CC is relatively-more-robust but not
prevalence-invariant; the joint claim is descriptive.
