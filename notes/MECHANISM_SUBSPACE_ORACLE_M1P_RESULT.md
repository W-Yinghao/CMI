# Mechanism-Subspace Oracle — M1-P confirmatory PRIMARY tranche RESULT

Real EEG, full LOSO. EEGNet × contrast_disagreement × (9+12) subjects × 3 seeds = **63/63 cells** (single
git_sha `a05ee3a5`, config_hash `753ea83de4f54fe1`; all `SHARED_NULL_HAAR` primary control, no LOW_DOF fallback).
Amendment 03 shared-null conditional estimand, config v4 FROZEN. NOT a scientific stop — a graded outcome that
terminates the current post-hoc hypothesis and hands to the next falsifiable one. Manuscript FROZEN.

## Question
Within the space that does NOT delete the shared class-contrast mechanism (`B = N·TopEig(NᵀG_dis N)`,
`N = null(C̄)`), are the directions most aligned with cross-subject mechanism DISAGREEMENT more enriched for
future-session-harmful deletable mechanism than RANDOM directions in the SAME shared-null space?

## Result = C (post-hoc disagreement-dictionary enrichment NOT supported)
| dataset | n | ΔU_safe_specific [95% cluster CI] | Holm sign-flip p | ΔU_unc_specific | q95-exceedance (null .05) | capDict inf/rnd | capSel inf/rnd |
|---|---|---|---|---|---|---|---|
| BNCI2014_001 | 9 | **+0.0015 [−0.0012, +0.0047]** | 0.414 | +0.0015 | 0.037 | 0.954 / 0.613 | 0.256 / 0.181 |
| BNCI2015_001 | 12 | **+0.0000 [−0.0019, +0.0019]** | 0.491 | +0.0000 | 0.028 | 0.990 / 0.533 | 0.213 / 0.123 |

**The construction WORKS** — the informed dictionary captures 95–99% of the shared-null `G_dis` energy vs 53–61%
for matched random, so the mechanism-disagreement subspace is real and well-identified (the null is NOT an
estimator/basis defect). **But deleting it confers NO future-session-harm specificity over random directions in the
same shared-null space:**
- ΔU_safe_specific and ΔU_unc_specific both STRADDLE / sit AT zero; practical_state = NO_PRACTICAL_ENRICHMENT
  (even the CI upper bound ≤ +0.005 bAcc is negligible).
- Holm-adjusted EXACT sign-flip p = 0.41 / 0.49; per-subject sign counts ≈ coin flip (6/9, 7/12).
- q95-exceedance = 0.037 / 0.028 — BELOW the 0.05 null (informed beats the random q95 LESS than chance).
- Absolute effects: informed_safe dU ≈ +0.003 / +0.001 ≈ shared-null-Haar random (+0.002 / +0.001) — deletions DO
  remove some future-session error, but the disagreement dictionary is not more useful than matched random; for
  BNCI2015, ambient random (+0.0075) even helps MORE.

## Why C and not B or D (decisive discriminators)
- **Not B** (beneficial-but-entangled): the UNCONSTRAINED oracle — target labels, exhaustive rank≤3, NO safety
  filter — is ESSENTIALLY IDENTICAL to the safe oracle (+0.0015/+0.0000, same 6/9, 7/12 sign counts). There is no
  hidden beneficial direction that the source-safety filter is blocking; the target-label oracle also fails to beat
  matched random. Safe selection is non-trivial (0% / 11% identity, mean selected rank 2.04 / 1.75), so this is not
  a "safety collapsed to identity" artifact.
- **Not D** (uncertain-but-consistently-positive): BNCI2015 is a DEAD null (+0.0000, 7/12 ≈ chance) and BOTH
  datasets have q95-exceedance BELOW the null rate — this is consistent with the NULL, not a masked positive.
  Adding seeds would not rescue it (uncertainty is not a consistent-positive-point-estimate-with-seed-noise case);
  between-subject vs between-seed sd is mixed (0.0045/0.0024, 0.0034/0.0013), not seed-dominated.
- **Not a query-regime-flat artifact**: deletions produce non-zero query dU (+0.003 / +0.0075 for ambient), so
  there IS removable future-session error — it is simply not disagreement-specific.

## Disposition (per PM A/B/C/D map)
Result C terminates the **post-hoc disagreement-dictionary enrichment hypothesis** and the **post-hoc erasure
objective** — NOT the subspace science line. Interpretation: the ERM representation does not contain a redundant,
stable, post-hoc-DELETABLE disagreement "ticket" whose removal specifically helps future sessions; the disagreement
directions exist and are captured, but they are not future-harm-enriched relative to matched shared-null random.

**Next falsifiable hypothesis (PM to authorize):** mechanism-consistency TRAINING — reshape the encoder DURING
training to CREATE a shared, deletable/shapeable mechanism, e.g.
`L = L_task + λ Σ_d ‖C_d − C̄‖²_F` (or a direction-normalized variant), then re-run the SAME M1-P oracle to test
whether training changed (a) the shared contrast, (b) the disagreement rank, (c) the random-specific utility. This
is an encoder-RETRAINING workload → GPU. Secondary post-hoc runs (DGCNN contrast, EEGNet rule/grad) are NOT
launched: with a clear no-enrichment primary, the PM map routes to training-time, not to spending compute on all
post-hoc secondaries.

HELD: M2 source selector, learned oblique projector, TTE (unless a future run shows unc-positive/safe-negative),
CMI-GPU certification, full all-family/all-backbone M1, new amendments. Manuscript FROZEN. Scientific line ACTIVE.
