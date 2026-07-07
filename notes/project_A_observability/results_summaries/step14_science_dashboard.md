# Step 14 — Science Dashboard (rich R1 diagnostics + real minimal-label curves)

Scope: rich R1 diagnostics + real minimal-label curves (coverage-decomposed) + power; not SOTA.

## Key metrics

- real runs: **54** · harm-rate **0.8519** · R1 diagnostics available **1.0**
- harm-predictor bAcc — R0 **0.3342** · R1 **0.6522** · perm-null p95 **0.6821** · margin **-0.0299** · robust **False** · verdict **above_baseline_but_within_permutation_null_overfitting_artifact**
- power: underpowered **True** · min detectable bAcc ≈ **0.7121**
- real minimal-label: k256 coverage **0.318** · accuracy when decisive **0.9985** · best k uncond≥0.8 **None** · cond≥0.8 **8**
- claim boundary ok **True** · target labels used in R1 diagnostics **False**

## What changed from Step 12

1. Richer R1 diagnostics beat 0.5 but do NOT clear their permutation null (p95 0.6821) -> OVERFITTING ARTIFACT (19 features / 54 rows / 8 minority), not signal.
2. Real minimal-label curves are COVERAGE-limited, not inaccurate: at k=256 coverage (decisive rate) is 0.318 while accuracy WHEN decisive is 0.9985; best k for unconditional≥0.8 = None, conditional≥0.8 = 8. The burden is coverage, not accuracy (labeled slice under an iid sampling contract).
3. Power: minority n small, underpowered=True; a bAcc below ~0.7121 is indistinguishable from the overfitting null.
4. R1 diagnostics remain label-free; oracle per-trial labels used only for R2 curves / evaluation.

> R1 diagnostics are label-free; R0/R1 harm prediction is retrospective, not identifiability; real k>0 curves are labeled slices under an iid sampling contract (coverage-limited). No SOTA claim.
