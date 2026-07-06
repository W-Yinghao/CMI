# CSC B6-FM fixed-margin C-randomization — pure prior held fixed, but within-class covariate drift still fires

```
Scope: B6-FM class-preserving / fixed-margin C-randomization canary | development diagnostic | NOT confirmatory | NO tag | NO validity claim
  same-class within-subject Metropolis swaps -> condition x class margins held EXACTLY fixed | Y ONLY as swap constraint
  propensity marginal P(C|Z,S) (Y-free) | B3 contrast T byte-reused | SAME B6.0 cohorts (base 200e6) | NOT a param-tune
```

**Safe headline (red-team-verified, both lenses PASS, 0 serious):** *B6-FM restricts the exact-CB Metropolis odds-swap
to same-class within-subject pairs, so it holds the condition×class margins EXACTLY fixed (margin_fidelity=0 on all
400 cohorts) — the pure prior shift is now a held-fixed nuisance. It RETAINS B6.0's strong-covariate fix (0/50 at
auc0.81 and auc0.94) and POS power (12/50). BUT it FAILS the hard screen: NULL_label still false-confirms 22/50 (B6.0
was 25/50) and NULL_cov_plus_label 15/50. The precise driver, verified: the prior-shift DGP ALSO induces a WITHIN-CLASS
session-covariate drift (session-1 vs session-2 Z differ within each class) that the MARGINAL Y-free propensity cannot
reproduce → it reads as concept. So a single observational C-randomization null with a marginal propensity structurally
cannot separate within-class covariate from within-class concept — exactly the design-red-team prediction. This is NOT
a param-tune fix; it closes the single-C-null line.*

## Result (n=50/cond, SAME B6.0 cohorts; OLD certifier / B6.0-plain / B6-FM confirm counts)
| condition | GT | OLD | B6.0 | **B6-FM** | med p_C_FM | prop_auc | feasible_swaps |
|---|---|---|---|---|---|---|---|
| NULL_cov_soft | null | 11 | 2 | 2 | 0.498 | 0.61 | 2781 |
| NULL_cov_plus_label | null (cov+prior) | 4 | 15 | **15** | 0.047 | 0.62 | 2100 |
| NULL_cov_strong_auc0.81 | null | 18 | 0 | **0** | 0.731 | 0.98 | 2786 |
| NULL_cov_strong_auc0.94 | null | 28 | 0 | **0** | 0.818 | 1.00 | 2778 |
| **NULL_label** | null (prior) | 0 | 25 | **22** | 0.007 | 0.63 | 2100 |
| random_label_control | null | 0 | 0 | 0 | 0.540 | 0.61 | 2838 |
| POS_concept | concept | 12 | 15 | 12 | 0.107 | 0.61 | 1665 |
| POS_concept_plus_cov | concept | 13 | 12 | 9 | 0.182 | 0.61 | 1614 |

Accounting: n_total=400, n_valid_fidelity=399 (1 nan = OLD-side POS #45 degeneracy, crt valid, touches no confirm),
`max|T_old−T_crt|=0`, **margin_fidelity_max_err=0 and max_subject_count_err=0 on EVERY cohort**, sampler_invalid=0,
margin-lock 0/400.

## Reading (red-team-verified)
1. **The class-preserving swap is exactly correct** (margins invariant by detailed balance; verified on all 400).
2. **The pure prior is held fixed, yet NULL_label still fires (22/50)** — because the prior-shift DGP (`_prior_resample`,
   P(Y|session)=0.35/0.65) ALSO shifts P(Z|session) via the class mix, creating a WITHIN-CLASS session-covariate drift.
   The marginal Y-free propensity P(C|Z,S) (auc≈0.63) cannot reproduce that within-class structure, so the within-class
   null degenerates toward a uniform reshuffle and the drift reads as concept. `covariate_auc_gap`≈0 is marginal-only
   and BLIND to this within-class covariate — it must NOT be read as "covariate preserved".
3. **Strong-cov fix retained** (0/50, p-value-driven, lock=0) but does not rescue B6-FM (fails the null taxonomy).
4. **POS power:** POS_concept 12 (=OLD exactly, genuine); POS_concept_plus_cov 9 vs OLD 13 / B6.0 12 — a modest power
   cost of the tighter fixed-margin null on covariate-entangled concept (disclosed, not folded into "retained").
5. **Structural conclusion:** a single observational C-randomization null cannot simultaneously exclude prior/label
   shift AND covariate drift — not-see-Y fires on prior (B6.0), fix/see-Y-margins can't separate within-class covariate
   from concept (B6-FM). The fix is NOT another single C-null variant, NOT P(C|Z,S,Y) (conditions out concept), NOT
   within-class Z balancing (kills within-class concept too).

## Row-level intersections (DECISION INPUT ONLY — verified exact task_id join; NOT a B7 pass)
| condition | old | plain | FM | old∧plain | old∧FM | plain∧FM | triple |
|---|---|---|---|---|---|---|---|
| NULL_cov_soft | 11 | 2 | 2 | 1 | 1 | 2 | 1 |
| NULL_cov_plus_label | 4 | 15 | 15 | **3** | 3 | 13 | 2 |
| NULL_cov_strong_auc0.81 | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| NULL_cov_strong_auc0.94 | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| NULL_label | 0 | 25 | 22 | **0** | 0 | 22 | 0 |
| random_label_control | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| POS_concept | 12 | 15 | 12 | **7** | 7 | 9 | 6 |
| POS_concept_plus_cov | 13 | 12 | 9 | **7** | 6 | 7 | 5 |

The complementary blind spots are structural: **OLD is the prior-witness** (NULL_label 0, but strong-cov 18/28);
**plain/FM are covariate-witnesses** (strong-cov 0, but NULL_label 25/22). So `old∧(a covariate-witness)` cancels BOTH
failure modes (summed NULL false-confirm old∧plain=4 vs plain∧fm=37), while two same-type C-nulls (`plain∧FM`) cannot
(NULL_label 22). This MOTIVATES a dual-witness `old_B3 ∧ B6_plain` — but it is a **DECISION INPUT ONLY**: post-hoc on
the same 50 cohorts, still leaves residual soft-cov (old∧plain NULL_cov_plus_label=3, soft=1), and the AND costs
genuine power (POS 7 vs OLD 12/13). A validated result requires its own pre-registered packaged canary + red-team; the
AND-rule is set-conservative / nuisance-disjunctive, and its formal validity depends on each component-witness's
validity for its component null — NOT a universal type-I guarantee.

## Verdict + next (reviewer decision tree = Case B)
**B6-FM FAILS the hard screen (NULL_label 22/50) — the single observational C-randomization line has reached its
structural limit.** Next (reviewer-authorized IF this verdict holds): the **B7-primary dual-witness row-level canary**
(`old_B3_CONFIRMED ∧ B6_plain_CONFIRMED`, pinned; B6-FM secondary only) — a formal, pre-registered, red-teamed
canary, not this preview. If B7 fails (NULL_cov_plus_label high or POS=0) → B8 information-contract / randomized paired
audit design. NO B6-FM tuning, NO P(C|Z,S,Y), NO within-class Z balancing, NO B6.1, router/B5/B4 CLOSED. Builds on
[[b6_0_condition_randomization]], [[csc-b6-condition-randomization]]. Package: `csc/results/b6_condition_randomization/b6_fm/`.
