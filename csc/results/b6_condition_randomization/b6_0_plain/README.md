# B6.0 plain condition-randomization canary (diagnostic-only)

Root-cause redesign: test H0 Y-perp-C|Z,S by resampling the CONDITION C ~ P(C|Z,S) (exact-CB Metropolis odds-swap,
count-preserving within subject), NOT the label Y. B3 contrast T byte-reused; propensity NEVER sees Y/synthetic/oracle.
8 conditions x n=50, base 200e6, SM16 cache.

RESULT (red-team-verified: accounting MINOR / decision_contract PASS / science MINOR+1-high): COVARIATE-ROOT FIX REAL
(strong-cov OLD 18/28 -> B6 0/0; soft 11 -> 2; random 0) + POS retained (clean concept = prior-blind OLD 12/13), BUT
PRIOR-SHIFT ESTIMAND GAP -- B6's Y-perp-C|Z also fires on pure prior shift (NULL_label 0 -> 25/50, NULL_cov_plus_label
4 -> 15/50), a CORRECT detection (random_label_control stays 0/50) but a hard CSC-taxonomy failure that BLOCKS a
confirmatory concept-certifier claim. Accounting: n_total=400/n_valid=399/n_invalid=1 (the nan is OLD-certifier-side
degeneracy, crt valid); UNIDENTIFIABLE_COVARIATE_LOCK state INERT (global-sum eff_randomization floor never reached;
p-value carried the strong-null abstention).

VERDICT: solves the strong-covariate false-confirm MECHANISM but fails the CSC null taxonomy on prior-shift controls.
NEXT = B6-FM class-preserving/fixed-margin C-randomization (same-class within-subject swaps). NO tag, NO validity claim.
See notes/b6_0_condition_randomization.md. Related: csc-b6-condition-randomization (memory), p3_oracle_diagnostic.md.
