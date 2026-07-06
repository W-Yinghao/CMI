# B6-FM fixed-margin / class-preserving C-randomization canary (diagnostic-only)

Same idea as B6.0 (randomize C~P(C|Z,S), fix Z,Y, recompute byte-reused B3 T) BUT restrict the exact-CB Metropolis
odds-swap to SAME-CLASS within-subject pairs -> condition x class margins held EXACTLY fixed (Y only as swap
constraint; propensity still marginal P(C|Z,S), no Y). Run on the SAME B6.0 cohorts (base 200e6) for direct
OLD/B6.0/B6-FM comparison. NO tag, NO validity claim.

RESULT (red-team CLEAN, both lenses PASS, 0 serious): margin_fidelity=0 on all 400 (class-preserving swap exactly
correct); strong-cov fix RETAINED (0/50); POS retained (12/9). BUT FAILS hard screen -- NULL_label 22/50 (B6.0 25),
NULL_cov_plus_label 15/50. Verified driver: the pure prior IS held fixed exactly, but the prior-shift DGP ALSO induces
a WITHIN-CLASS session-covariate drift that the marginal Y-free propensity cannot reproduce -> reads as concept. So a
single observational C-randomization null cannot separate within-class covariate from within-class concept.

Row-level intersections (DECISION INPUT ONLY, not a B7 pass): old&plain cancels both witnesses' prior-shift
false-confirm (OLD=prior-witness, plain/FM=covariate-witnesses) -> all nulls<=3, POS 7/7; plain&fm (two same-type
C-nulls) does NOT cancel (NULL_label 22). Motivates a DUAL-WITNESS old_B3 AND B6_plain -- but that needs its own
pre-registered packaged canary + red-team (post-hoc here; residual soft-cov; AND costs power POS 7 vs OLD 12/13).

VERDICT: single observational C-randomization line reached its structural limit (Case B). See
notes/b6_fm_condition_randomization.md. Related: b6_0_condition_randomization.md.
