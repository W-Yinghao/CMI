# CSC-realEEG Router R1 — clean single-block SAFETY-PASS, but thin/drift-exposed margin against a SOFT covariate

```
Scope: Router R1 fresh-calibration + held-out validation | development diagnostic | NOT confirmatory | NO tag
  B3 certifier path BYTE-UNCHANGED | router = pure post-hoc MONOTONE threshold on the deployable score observed_T
  tau LOCKED from calibration (null-only) BEFORE validation | no retune on validation | no oracle / no future outcomes
  pre-registered endpoints + seed disjointness (calib 80e6 / valid 90e6, disjoint from P3 20/50/60/70e6)
```

**Safe headline (red-team-corrected — down-scoped):** *A monotone abstention router — ALLOW iff B3 says
CONCEPT_CONFIRMED AND observed_T ≥ τ, with τ LOCKED on a fresh null-only calibration block and evaluated ONCE on a
disjoint held-out block — passes the pre-registered held-out safety endpoints (NULL_cov 5/300, NULL_cov+label 4/300,
both ≤7 cap, CP95u < 5%) with genuine, concept-driven utility (POS 52–53/300). The result is COMMIT-CLEAN on
integrity (no peeking; τ re-derives byte-exact; monotonicity holds; no_retune honored) and the power is genuinely
concept-driven (observed_T separates POS from covariate-null at AUC≈0.90). BUT it is a SINGLE-BLOCK, THIN-MARGIN pass
against a SOFT covariate (session_auc≈0.52): a verified 2× calibration→validation drift (allow 3→5) already consumed
most of the buffer, τ×0.90 breaches the cap, and utility is ~17% recall on an EASY linear concept. This is a genuine
but NARROW positive — a candidate for frozen-protocol DESIGN DISCUSSION, NOT a confirmatory result.*

## Protocol (pre-registered, commit-sequenced)
1. Commit protocol before running. 2. CALIBRATION (base 80e6; **NULL_cov + NULL_cov_plus_label ONLY, null-only**; POS
never run/inspected pre-lock). 3. Lock `τ_R1 = 0.004587196` as a deterministic grid point (sorted-unique observed_T
among calibration B3-method-confirms + sentinels {0,+∞}; conservative ≤3/300 rule; grid rank 81/88; hexfloat
`3f72ca060cce336e`) → hashed before validation. 4. HELD-OUT VALIDATION (base 90e6; all 6 conditions, n=300 = 1800
cohorts; merged sha256 `065c681b…`). 5. Evaluate LOCKED τ (never re-derived). 6. Red-team. 7. Commit. `no_retune`.

## Held-out result @ locked τ_R1 (comparison ≥) — every number independently reproduced from the 1800 records
| condition | GT | n | n_inv | method_confirm | **router_allow** | rate | CP95u | medT_allow |
|---|---|---|---|---|---|---|---|---|
| **NULL_cov** | null | 300 | 0 | 72 | **5** | 0.017 | 0.0347 | 0.00486 |
| **NULL_cov_plus_label** | null | 300 | 0 | 27 | **4** | 0.013 | 0.0303 | 0.00541 |
| NULL_label | null | 300 | 0 | 2 | 1 | 0.003 | 0.0157 | 0.00498 |
| random_label_control | null | 300 | 0 | 0 | 0 | 0.000 | 0.0099 | — |
| POS_concept | concept | 300 | 16 | 69 | **52** | 0.173 | — | 0.01282 |
| POS_concept_plus_cov | concept | 300 | 10 | 70 | **53** | 0.177 | — | 0.01607 |

- **PRIMARY safety PASS:** NULL_cov 5/300, NULL_cov+label 4/300 — both ≤7 cap; CP95u 3.5%/3.0% < 5%. Router cuts the
  raw B3 method-confirm on NULL_cov from 72/300 (24%) to 5/300 (1.7%). **SECONDARY ok:** NULL_label 1, random 0.
- **UTILITY genuinely concept-driven** (not covariate artifact): NULL_cov and POS_concept share identical Z + session
  covariate, differ ONLY by the injected 25° session-2 rotation, so the T-gap is the concept; observed_T separates POS
  from NULL_cov confirms at **AUC≈0.90**; within POS, confirm-T tracks subject-consistency (+0.86), not covariate (+0.16);
  NULL_cov "confirmations" are near-null noise (confirm-T 0.0027 ≈ non-confirm-T 0.0024) the router correctly discards.

## Verified CLEAN (independent recompute + re-derivation)
- **No peeking / tau-lock (PASS):** τ re-derives byte-exact from the sealed calibration; calibration was genuinely
  null-only (zero POS records → τ cannot be POS-tuned); eval loads `LOCK["tau_R1"]`, never re-derives (a
  validation-optimal τ would be 0.005128 ≠ locked). Router score uses only `b3_state` + `observed_T`; oracle/condition/
  ground-truth are report-only. Seed formula 0 violations; calib/valid/P3 blocks pairwise disjoint. All sha256 pass.
- **Monotonicity:** 0 allowed-but-unconfirmed records anywhere; allow ⊊ method-confirm in every condition.
- **Invalid accounting conservative:** nulls have 0 invalid; the 16+10 POS invalids are NEED_MORE_LABELS, non-confirmed,
  never allowed → they DEPRESS utility (POS valid rate 52/284=18.3%), never flatter safety.

## Honest LIMITATIONS (red-team; must precede any "frozen-protocol" language)
1. **SOFT covariate only (the #1 caveat).** session_auc ≈ 0.52, overlap ≈ 0.99 in EVERY condition — the real Lee2019
   session covariate is nearly negligible in SM16 log-bandpower. Type-I control is demonstrated only against a weak
   covariate. The raw certifier still false-confirms 72/300 here (small artifact-T); the router works ONLY because
   those artifact observed_T's stay below τ. Against a genuinely strong covariate, null-T could exceed τ and the router
   could FAIL — **untested**. The router masks, does not fix, the under-dispersion.
2. **Thin, τ-fragile, drift-exposed margin.** NULL_cov 5 vs cap 7 (headroom 2). Verified calibration→validation drift
   ≈2× at fixed τ (NULL_cov allow 3→5; method-confirm 62→72) — one more such drift breaches the cap, and with n=1
   validation block there is NO variance estimate on it. τ×0.90 → NULL_cov 11, NULL_cov+label 8 (both breach); the
   sibling pre-registered `tau_boundary` (0.004049) would also FAIL. Cohort knife-edge: min allowed T is +3.4e-5 above τ.
3. **Utility is high-precision / LOW-recall on an EASY concept.** 52/300 = 17.3% allowed → 83% of true-concept cohorts
   abstained; "STRONG" is only relative to the 6.7% pre-reg bar. The concept is a LINEAR 25° rotation matched to a
   linear rank-3 certifier — a nonlinear real concept shift would likely give lower power. 17% is an optimistic ceiling.
4. **The two utility endpoints are NOT independent** — POS_concept and POS_concept_plus_cov are the same build_cohort
   branch differing only by seed (52 vs 53 allow), not a distinct covariate-augmented corroboration.
5. **Single dataset, single held-out block, semi-synthetic, post-hoc router.** Lee2019 SM16 only; injected concept,
   not a natural shift.

## Path to a confirmatory tag (red-team-specified)
Multiple independent validation blocks (bound the drift) + a STRONG-covariate null (test the router's actual failure
mode) + a second dataset / nonlinear concept. None done here; this is development-diagnostic.

## Relation to the rest of the line
R1's type-I lever is the calibrated abstention layer, working because on SM16 log-bandpower observed_T separates
concept from covariate-null. [[b5_0_random_encoder]] (B5.0) is a SEPARATE, weaker probe — a frozen RANDOM deep encoder
is a near-isometry no-op that reproduces SM16 exactly with no usable concept power, so it does NOT establish
feature-invariance (that is B5.1's job with a learned encoder) and does not bear on this router result. Builds on
[[router_stage0_frontier]] (R0 frontier) and [[p3_oracle_diagnostic]] (the under-dispersion the router routes around).

## Package (diagnostic-only, no tag)
`csc/results/router_stage1_validation/`: protocol, tau_lock, calibration rows, `r1_validation_merged.json` (+sha256),
`router_stage1_validation_tables.json` (+ red-team down-scope annotation), `r1_redteam_checks.json`, scripts
(`realeeg_r1_forensic.py`, `realeeg_r1_merge.py`, `r1_tau_lock.py`, `r1_validation_eval.py`, sbatch). Red-team:
nopeek/tau-lock PASS, safety_real MINOR_ISSUE, utility_real MINOR_ISSUE — verdict "commit-clean; single-block SAFETY-PASS
with a thin drift-exposed margin and ~17% power against a soft covariate; candidate for frozen-protocol DISCUSSION, not
confirmatory."
