# C20 — Frozen-probe new-regime validation (locked to C19 `664007686afb520f`)

> Validates the FROZEN C19 robust-core diagnostic probe by cross-regime LOTO: train on development regimes S0/S2/S3 (non-held-out targets), evaluate on the held-out target in DELETION regimes S4/S5/S6/S7. Nothing about the probe changed. DIAGNOSTIC-ONLY; no selector.

## Verdict (layered — headline is deliberately conservative)

- **primary: `external_new_regime_generalization_not_established`**  ·  case `largely_regime_local_with_marginal_exceptions`
- secondary observation: `marginal_above_chance_transfer_in_S6_boundary_aligned_mask_S7_random_matched_mask` (MARGINAL exceptions, not a generalization claim)
- failure mode: `relationship_level_regime_shift_not_feature_availability`  ·  availability: `robust_core_available_all_heldout_regimes`  ·  strength: `weak_diagnostic_only`
- Broad external new-regime generalization is NOT established. Robust-core features are available in all held-out regimes, so the failures are RELATIONSHIP-level regime shift, not feature availability. 2/4 held-out regimes (['S6_boundary_aligned_mask', 'S7_random_matched_mask']) only MARGINALLY clear the strict chance bar and are reported as secondary exceptions, not a generalization claim. Remain diagnostic-only; do NOT proceed to C20-B external execution on this evidence.

## 1. Gate correction (superseded first run)

> The first C20-A run reported `validity_limited_by_feature_availability` — a metric BUG: `abstention.availability` fed the fragile accuracy ENDPOINTS to the estimability gate, so the reported 'robust_core_scored_rate' was really the robust+endpoint rate → 0 under deletion (endpoints NaN) while the robust-core features are fully finite. That first taxonomy is DISCARDED. Fixed: robust-only availability reported separately; failures are now correctly attributed to the source→target relationship, not feature availability.

## 2. C19 lock audit

- locked C19 config hash `664007686afb520f` · feature-lock True · dev/held-out disjoint True · no-selector True (no feature / threshold / regularization tuning)

## 3. Corrected availability

| held-out regime | robust-core scored_rate | endpoint_available_rate | endpoint_nonestimable_rate |
|---|---:|---:|---:|
| S4_missing_cells | +1.000 | +0.000 | +1.000 |
| S5_block_class_by_domain | +1.000 | +0.490 | +0.510 |
| S6_boundary_aligned_mask | +1.000 | +0.000 | +1.000 |
| S7_random_matched_mask | +1.000 | +0.000 | +1.000 |

> Robust-core available in ALL held-out regimes; endpoint availability stays fragile (as C18 predicted). So failures are relationship-level, not availability-level.

## 4. Held-out regime results (severity in parens)

| held-out regime | sev | loto_auc | margin vs 0.5 | clears 0.03 bar by | passes |
|---|:--:|---:|---:|---:|:--:|
| S4_missing_cells | 4 | +0.500 | -0.000 | n/a | False |
| S5_block_class_by_domain | 4 | +0.511 | +0.011 | n/a | False |
| S6_boundary_aligned_mask | 3 | +0.532 | +0.032 | +0.002 | True |
| S7_random_matched_mask | 3 | +0.536 | +0.036 | +0.006 | True |

> **Permutation p is NOT the discriminator**: the LOTO shuffle null centers below 0.5, so perm p hits the floor (~0.005) for ALL four regimes including at-chance S4 — do NOT report 'all beat permutation'. The BINDING criterion is the strict chance-margin (>=0.03). Under it, S4 (0.500, at chance) and S5 (+0.011) FAIL; S6 (+0.032) and S7 (+0.036) pass but clear the bar by only ~0.002 and ~0.006 — threshold-level, fragile.
> **Not boundary-specific / severity-local**: S7 random (+0.536) >= S6 boundary-aligned (+0.532), so the marginal transfer is NOT a boundary mechanism. The only above-chance regimes (S6/S7) are the LOWER-severity ones (sev 3) vs the failing S4/S5 (sev 4): severity_local_marginal_transfer = **True** → a severity gradient, not novel-regime generalization. (S7 is C18's RANDOM NEGATIVE CONTROL, yet 'passes' above the S6 treatment — the probe cannot separate structured from random deletion.)

## 4b. Robustness disclosure (post-hoc; pre-registered pass/fail unchanged)

- Multiplicity: 2/4 pre-registered passes collapse to **0/4** surviving Holm correction (one-sided z vs 0.5, Hanley SE~0.016). The permutation gate is non-discriminating (sub-0.5 null → p at the ~0.005 floor for ALL regimes incl at-chance S4), so only the single margin-vs-0.5 gate does any work and it does not survive multiplicity.
- **Simpson / pooling confound**: per-target mean AUC is essentially constant across ALL four regimes (range [0.632, 0.652]) while the pooled pass/fail AUC ranges [0.500, 0.536] — so the pooled statistic that drives pass/fail is a between-target offset artifact DECOUPLED from within-target signal. simpson_decoupled = **True**.
- **Under-claim (disclosed, honest)**: a consistent WITHIN-target AUC ≈ +0.640 exists in EVERY held-out regime — a genuine competence signal — BUT it is regime-nonspecific and is NOT the pre-registered pooled estimand, so it cannot rescue the generalization claim.

## 5. Endpoint-augmented (SECONDARY — cannot rescue primary)

| held-out regime | n_endpoint_estimable | loto_auc | passes |
|---|---:|---:|:--:|
| S4_missing_cells | 0 | n/a | False |
| S5_block_class_by_domain | 621 | +0.519 | False |
| S6_boundary_aligned_mask | 0 | n/a | False |
| S7_random_matched_mask | 0 | n/a | False |

## 6. Claim boundary

> DIAGNOSTIC-ONLY. Broad external new-regime generalization is NOT established. The C19 robust-core competence signal is real in its pre-registered development regimes but its held-out support-regime transfer is weak and largely regime-local; S6/S7 are marginal above-chance exceptions on the mildest deletion regimes, not a generalization result. It remains a weak diagnostic signal — not a detector, not a chooser, no OACI rescue. C20-B external-dataset work stays a PROTOCOL only; do NOT proceed to execution on this evidence.