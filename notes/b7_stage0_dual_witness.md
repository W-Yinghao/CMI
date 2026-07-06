# CSC B7-primary Stage-0 dual-witness canary — nuisance-disjunction of OLD B3 ∧ B6_plain (exposed re-analysis)

```
Scope: B7-primary Stage-0 dual-witness | EXPOSED row-level re-analysis of COMMITTED B6.0 cohorts | NO new compute
  development-only | NOT confirmatory | NOT deployable | NOT a universal type-I guarantee | NO tag
  primary PINNED = old_B3_confirm AND B6_plain_confirm (gated states) | B6-FM = secondary diagnostic only
```

**Safe headline (red-team-verified, both lenses PASS, 0 serious):** *B6 as a standalone certifier is CLOSED (B6.0
covariate-correct/prior-wrong; B6-FM within-class-covariate-wrong). The reconstruction is a DUAL-WITNESS rule: the OLD
B3 fixed-margin Y-null is the PRIOR-shift witness, B6.0 plain C-null is the COVARIATE-process witness, and a concept
alert (`DUAL_CONCEPT_ALERT`) is allowed ONLY when BOTH nuisance explanations are rejected. On the exposed B6.0 cohorts
(row-level, gated states) the AND cancels each witness's blind spot exactly — NULL_label 0/50, strong-cov 0/50 both
levels — and controls the hardest cell NULL_cov_plus_label at 3/50, while retaining POS_concept 7/50 (strong canary
band). It PASSES the development screen. This is set-conservative, exposed-data, NOT a confirmatory result and NOT a
universal type-I guarantee.*

## Method
B6 fails as a single observational C-null (no single null tested separates covariate nuisance, prior nuisance, AND
within-class concept — [[b6_0_condition_randomization]], [[b6_fm_condition_randomization]]). B7-primary combines two
complementary witnesses:
- **OLD B3 Y-null** (fix Z,C; resample Y w/ fixed margins + studentized + LCB gates) = PRIOR-shift witness
  (NULL_label 0, but covariate-blind: strong-cov 18/28).
- **B6.0 plain C-null** (fix Z,Y; resample C~P(C|Z,S)) = COVARIATE-process witness (strong-cov 0, but prior-sensitive:
  NULL_label 25).
`B7_primary_confirm = (old_b3_state==CONCEPT_CONFIRMED) AND (b6_state==CONCEPT_CONFIRMED_B6)` — STATE-based (each
witness's own gates). `p_dual = max(p_old, p_C)` is a DIAGNOSTIC only. Three states: `DUAL_CONCEPT_ALERT` /
`NO_ACTIONABLE_CONCEPT_EVIDENCE` / `UNIDENTIFIABLE_OR_INVALID` (never `NO_CONCEPT` — this is action evidence, not
semantic truth).

## Result (exposed B6.0 cohorts, n=50/cond; DISJOINT partition summing to 50 + B7-primary)
| condition | GT | old | B6_plain | **B7-primary** | both | old_only | B6_only | neither_valid | invalid | Σ |
|---|---|---|---|---|---|---|---|---|---|---|
| NULL_cov_soft | null | 11 | 2 | **1** | 1 | 10 | 1 | 38 | 0 | 50 |
| NULL_cov_plus_label | null (cov+prior) | 4 | 15 | **3** | 3 | 1 | 12 | 34 | 0 | 50 |
| NULL_cov_strong_auc0.81 | null | 18 | 0 | **0** | 0 | 18 | 0 | 32 | 0 | 50 |
| NULL_cov_strong_auc0.94 | null | 28 | 0 | **0** | 0 | 28 | 0 | 22 | 0 | 50 |
| NULL_label | null (prior) | 0 | 25 | **0** | 0 | 0 | 25 | 25 | 0 | 50 |
| random_label_control | null | 0 | 0 | **0** | 0 | 0 | 0 | 50 | 0 | 50 |
| POS_concept | concept | 12 | 15 | **7** | 7 | 5 | 8 | 29 | 1 | 50 |
| POS_concept_plus_cov | concept | 13 | 12 | **7** | 7 | 6 | 5 | 32 | 0 | 50 |

DISJOINT partition (Option A): `both + old_only + B6_only + neither_valid + invalid = 50` (fail-closed asserted;
`both == DUAL_CONCEPT_ALERT`, `invalid == UNIDENTIFIABLE_OR_INVALID`). The earlier display mixed the four-cell with the
separate invalid count (POS summed to 51 — POS #45 counted in both `neither` and `INVALID`); corrected here to a single
disjoint partition regenerated from raw rows (B7-primary counts unchanged).

**Set-conservative (verified):** `b7 == both ≤ min(old, B6_plain)` for every condition; every `DUAL_CONCEPT_ALERT`
cohort has both witnesses confirming (0 gated-confirms invalid). The complementary cancellation is exact: NULL_label
(plain fires 25, OLD silent) → 0; strong-cov (OLD fires 18/28, plain silent) → 0/0; POS retained where both fire.

## Development screen — PASSED (NOT a CP safety claim)
NULL_cov_soft 1 (≤5), NULL_cov_plus_label 3 (≤5, at preferred cap), strong_0.81 0 (≤3), strong_0.94 0 (≤3),
NULL_label 0 (≤3), random 0 (≤1), POS_concept 7 (>0, ≥5 strong). Honest residual: 4 dual FALSE-confirms on nulls
(NULL_cov_soft #27 ×1; NULL_cov_plus_label 0020/0040/0046 ×3) — genuine type-I events of the AND rule on null data,
disclosed; **NCPL=3 (at the preferred cap) is the flagged residual safety risk**. The AND-rule's formal validity
depends on each component witness being valid for its component null; this is NOT a universal type-I guarantee.

## Scope / disclosures (verified in the artifact)
Exposed row-level re-analysis of already-committed B6.0 cohorts (post-hoc, same 50/cond); development-only, not
confirmatory, not deployable. `p_dual` is diagnostic; B6-FM (`old∧FM`, `plain∧FM`, `triple`) is SECONDARY only, never
primary. No oracle/router/observed_T threshold/p-recalibration/condition-label rule enters the decision. Low notes:
the B6-invalid routing branch is structurally correct but vacuously exercised here (0 invalid/lock); the gated-vs-p-only
B6 confirm coincides on this data (the stricter gate removes nothing yet).

## Verdict + next
**B7.0 dual-witness canary PASSES the development screen on exposed data.** If the reviewer authorizes, next is
**B7.1 exposed full dev replay** (8 conditions × 300 cohorts, SAME old_B3 + B6_plain paths, SAME AND rule, NO tuning /
NO threshold / NO B6-FM substitution; primary = each null kind separately CP95 upper < 0.05, NO pooling; utility
POS_concept > 0). If B7.0/B7.1 fails (NULL_cov_plus_label over screen or POS intersection collapses) → B8
information-contract / randomized paired audit design. NO B6.1, NO B6-FM rescue, NO more C-randomization variants,
router/B5/B4 CLOSED. Builds on [[csc-b6-condition-randomization]]. Package: `csc/results/b7_stage0_dual_witness/`.
