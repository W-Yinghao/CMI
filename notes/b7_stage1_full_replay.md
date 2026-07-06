# CSC B7.1 dual-witness full replay — SAFETY-FAIL on the mixed covariate+prior null (observational line exhausted)

```
Scope: B7.1 exposed FULL dev replay | dual-witness old_B3 AND B6_plain | fresh disjoint base 300e6, 8x300 = 2400
  development-only | NOT confirmatory | NOT deployable | NOT a universal type-I guarantee | NO tag | NO retuning on fail
  protocol pre-registered + committed BEFORE run (72085b7, sha ed986506) | witnesses FROZEN | B6-FM NOT used
```

**Safe headline (red-team-verified: provenance PASS + science MINOR_ISSUE, 0 serious):** *On a FRESH disjoint n=300
block the dual-witness rule (`old_B3_confirm AND B6_plain_confirm`) cancels PURE nuisances and retains STRONG concept
power — but SAFETY-FAILS on the MIXED nuisance. It controls pure covariate (strong-cov AND 2/2), pure prior
(NULL_label AND 0), soft/random (3/1), and keeps STRONG utility (POS_concept 44/300, POS+cov 53/300 ≥ 20/15) — yet
NULL_cov_plus_label (session-covariate AND class-prior shift, the literal superposition of the two) false-confirms at
24/300 = 8% (CP95u 0.11), far over the 7/300 cap. This is STRUCTURAL, not tunable: the two witnesses co-fire on the
mixed null (OLD covariate-blind fires on the covariate part, B6 prior-blind fires on the prior part), and their
co-occurrence is POSITIVELY CORRELATED (both=24 vs independence 9.24 → 2.6× enrichment), so no threshold/independence
tuning can rescue the AND. Per protocol SAFETY-FAIL → no retuning. The observational-null-repair line (B3→B4→B5→B6→B7)
has exhausted its structural options on real EEG; the next line is B8 information-contract / randomized paired audit.*

## Result (fresh base 300e6, n=300/cond; B7-primary = both witnesses confirm)
| condition | GT | old | B6_plain | **B7** | CP95u | both | old_only | B6_only | neither_v | inval | Σ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NULL_cov_soft | null | 68 | 6 | **3** | 0.026 | 3 | 65 | 3 | 229 | 0 | 300 |
| **NULL_cov_plus_label** | null (cov+prior) | 33 | 84 | **24** | **0.111** | 24 | 9 | 60 | 207 | 0 | 300 |
| NULL_cov_strong_auc0.81 | null | 113 | 4 | **2** | 0.021 | 2 | 111 | 2 | 185 | 0 | 300 |
| NULL_cov_strong_auc0.94 | null | 172 | 3 | **2** | 0.021 | 2 | 170 | 1 | 127 | 0 | 300 |
| NULL_label | null (prior) | 0 | 165 | **0** | 0.010 | 0 | 0 | 165 | 135 | 0 | 300 |
| random_label_control | null | 1 | 5 | **1** | 0.016 | 1 | 0 | 4 | 295 | 0 | 300 |
| POS_concept | concept | 71 | 90 | **44** | — | 44 | 27 | 46 | 169 | 14 | 300 |
| POS_concept_plus_cov | concept | 88 | 91 | **53** | — | 53 | 35 | 38 | 167 | 7 | 300 |

Provenance/accounting: protocol committed BEFORE the run; 2400 unique seeds (301e6–309e6) disjoint from every prior
block (0 collisions); disjoint partition sums to 300/condition (`both==DUAL`, `invalid==UNIDENTIFIABLE`);
`fidelity_dT=0` on all VALID cohorts (NaN/excluded on the 14+7 old-side NEED_MORE_LABELS invalid rows); contamination
clean (only `old_B3 ∧ B6_plain` in the primary — no B6-FM/oracle/router/observed_T/p-recalibration); NULL kinds never
pooled.

## Reading (red-team-verified)
1. **Pure nuisances cancel; power retained.** Pure covariate → B6 silent → AND 2/2; pure prior (NULL_label) → OLD
   silent → AND 0. POS: both witnesses fire on genuine concept → STRONG (44/300, 53/300), true dual-confirms.
2. **The mixed cov+prior null FAILS — and it's the important cell.** NULL_cov_plus_label = `_draw(pooled)` (soft
   session covariate, auc≈0.62) + `_prior_resample` (0.35/0.65 prior) — the superposition of the two pure nulls. Both
   witnesses false-confirm (OLD 33 on the covariate part, like its 68/113/172 on pure covariate; B6 84 on the prior
   part, like its 165 on NULL_label). Neither is silent → the AND (24) survives. This is the CSC-taxonomy cell that
   MUST be safe and the MOST operationally realistic real-EEG nuisance — the fail lands in the important cell, so it is
   more serious for being here, not less.
3. **Structural, not tunable (2.6× enrichment).** The dual false-confirm is positively correlated within cohort:
   both=24 vs the independence expectation `33×84/300 = 9.24` → 2.6× enrichment. The AND cancels LESS than independent
   witnesses would, so no witness-independence or threshold tuning can rescue it. This is a structural limit of the
   nuisance-disjunction design.
4. **Selection vs confirmation (honest).** The exposed n=50 was already 3/50 = 6% (CP95u≈0.148), ABOVE the pre-reg
   7/300=2.33% bar; the Stage-0 screen (≤5/50) simply TOLERATED it (more permissive than the confirmation bar). The
   fresh n=300 (24/300 = 8%, CP95u 0.111) makes the screen-tolerated breach DECISIVE — healthy selection-vs-confirmation,
   NOT a failure that was invisible at n=50.

## Verdict + next (per pre-registered protocol)
**SAFETY-FAIL (kind-specific: NULL_cov_plus_label 24/300 ≫ 7/300). Per protocol: NO retuning.** The observational
null-repair line has now been exhausted on real EEG:
- single Y-null (B3): covariate-blind → fails strong covariate.
- estimable-null repair (B4): CLOSED.
- richer features (B5.0 random, B5.1 SSL): fitted-null FAIL intact.
- single C-null (B6.0 plain / B6-FM fixed-margin): fails prior / within-class covariate.
- dual-witness AND (B7): cancels pure nuisances + retains power, but fails the mixed cov+prior cell (structural, 2.6×).

**Next line = B8 information-contract / randomized paired audit design** (reviewer decision; NOT another witness/variant):
counterbalanced condition order, within-session matched blocks, randomized task/condition assignment, pre-specified
covariate-balance audit, minimal target-label audit, condition-randomization-support diagnostics — i.e. change the
data-generating process so concept shift is falsifiable, rather than repairing the null on observational data. Builds on
[[csc-b6-condition-randomization]], [[b7_stage0_dual_witness]]. Package: `csc/results/b7_stage1_full_replay/`.
