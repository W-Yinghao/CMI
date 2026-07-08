# C28 — Source-Target Logit-Factor Homology Audit (frozen C19 `664007686afb520f`)

> C27 localized the target score-offset carrier to CLASS-CONDITIONED CONFIDENCE. C28 asks whether an analogous SOURCE-side factor exists and can predict the target factor / offset (pushing C23's source-unobservable result to the logit-factor level). Read-only; identical source/target factor definition; target labels post-hoc only. DIAGNOSTIC-ONLY.

- **PRIMARY: `H7_logit_factor_confirms_source_unobservability`** — a source analog of the carrier exists but CANNOT identify the target offset -> source-unobservability is confirmed at the logit-factor level (C23 explained, not overturned).
- established: **H7_logit_factor_confirms_source_unobservability, H2_source_factor_misaligned, H4_source_factor_tracks_source_error_only, H5_target_residual_carries_offset**

## Q3 — does the SOURCE factor predict the TARGET offset? (decisive)

- target-carrier reference (C27) gap **+0.524**; best source-carrier gauge **source_carrier__source_guard** gap **-0.378** → source predicts offset: **False**. the source-only class-conditioned confidence gauge does NOT recover the target offset -> source-unobservability confirmed at the logit-factor level

| source gauge | gap closed | survives |
|---|---:|:--:|
| target_carrier_reference | +0.524 | True |
| source_carrier__source_guard | -0.378 | False |
| source_occupancy__source_guard | +0.384 | False |
| source_global_confidence__source_guard | -0.410 | False |
| source_carrier__source_audit | -0.797 | False |
| source_occupancy__source_audit | -0.499 | False |
| source_global_confidence__source_audit | -0.626 | False |

## Q2 — source↔target factor homology (informative = CENTERED, not raw cosine)

| role | raw cosine | class-wise corr | CENTERED (informative) | aligned | mean-dominated |
|---|---:|---:|---:|:--:|:--:|
| source_guard | +0.986 | +0.381 | +0.333 | False | True |
| source_audit | +0.983 | +0.257 | +0.180 | False | True |

- **raw cosine +0.986 is a MEAN-STRUCTURE artifact** (two positive confidence 4-vectors); the offset-relevant CENTERED alignment is only +0.333 (guard) / +0.180 (audit) → informatively aligned: **False**. raw cosine 0.986 is high but CENTERED alignment is only 0.333 (mean-structure artifact) -> the source and target factors share the confidence baseline but their OFFSET-RELEVANT variation is weakly aligned; the carrier lives in target decision occupancy, not source logit geometry

## Q4 — source vs target error geometry

| role | src factor↔src recall | src factor↔tgt factor | tracks source error only |
|---|---:|---:|:--:|
| source_guard | +0.662 | +0.381 | False |
| source_audit | +0.393 | +0.257 | True |

## Q5 — target-factor residual decomposition (source_guard)

- full target carrier gap **+0.524**; source-explained **-0.378**; TARGET RESIDUAL **+0.260** (survives False) → residual carries offset: **False**. neither the source-explained component nor the residual cleanly dominates the offset recovery

## Boundary of the claim

> DIAGNOSTIC-ONLY. Source and target factor definitions IDENTICAL; no feature selection. A source factor that carried diagnostic offset information would be a DIAGNOSTIC, NOT a selector or deployable gauge. Target labels entered ONLY the post-hoc error geometry, never the source-factor construction.