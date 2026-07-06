# Router R1 SCALE-UP — soft-covariate-regime tool with a hard covariate-strength boundary (diagnostic-only)

Two development-diagnostic experiments consolidating the R1 abstention router at the LOCKED tau_R1=0.004587196
(never re-derived). B3 certifier BYTE-UNCHANGED.
- multiblock/ : evaluate locked tau on 3 fresh RNG-disjoint blocks (100/110/120e6) + orig 90e6. NULL_cov allow
  [5,1,5,3] all <=7-cap; POS ~18%. Red-team: this is SUBSAMPLING stability on ONE dataset at ONE soft covariate
  (session_auc~0.52), NOT distributional/covariate drift or generalization.
- strongcov/ : a ground-truth NO_CONCEPT null with an AMPLIFIED session covariate (boundary-orthogonal shift; GT
  no-concept to machine precision). @ locked tau: delta1.5(auc0.81)->22/300=7.3% (breach), delta2.5(auc0.94)->78/300
  =26% (breach). Red-team PASS (sound): oracle-null test proves observed T is a typical draw from the TRUE null; the
  failure is the certifier's fixed-margin swap-null being mis-centered ~7-10x + under-dispersed under a strong
  covariate. The STATISTIC is fine; the NULL is the problem.
Envelope: robust to seed/subsample drift + weak covariates; NOT robust to covariate strength. NOT frozen-protocol
eligible as a general type-I controller. See notes/router_r1_scaleup.md. r1_scaleup_redteam_checks.json.
