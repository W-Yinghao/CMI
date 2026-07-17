# MCC training viability — RESULT = C-provisional (controllable-but-inert mechanism); stronger-lever test pre-registered

Real EEG, full LOSO. 63 bundles × 3 arms = 189/189 cells complete (training code identical across git c0b2f8ee/
39293f9e; config cmi_trace_mechanism_consistency.yaml frozen; 3 arms/bundle share a warm-up hash). EEGNet 16-d,
warm-up = existing ERM, +20 continuation epochs, λ=0.25 ramp/5ep, LR 0.1×, source-only checkpoint selection.
**This note was CORRECTED after a 4-agent adversarial panel (wf_93d5d436) — the routing was right but the first
draft justified it with two invalid arguments and omitted the decisive C-vs-E number.** Manuscript FROZEN.

## Result (per-bundle; inference unit = target subject, 3 seeds averaged per subject)
| dataset | n | ΔU_MCC−ERM [95% CI] | signflip p | ΔU_MCC−shuffle | signflip p | Δdir_consistency **B−A** (vs ERM) | **Δdir_consistency B−C (vs shuffle)** | Δ tr(G_dis)/tr(G_shared) B−C |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | 9 | +0.0002 [−0.0021,+0.0023] | 0.469 | −0.0005 | ~0.68 | +0.0006 (≈0) | **+0.00132 (9/9 subj, p≈0.002)** | −0.003 |
| BNCI2015_001 | 12 | −0.0004 [−0.0029,+0.0018] | 0.638 | −0.0001 | ~0.55 | −0.0001 (≈0) | **+0.00132 (11/12 subj, p≈0.002)** | −0.010 |

Absolute source dir-consistency (whitened, fleet): ERM 0.635 / 0.616 → MCC 0.637 / 0.616 (baseline NOT saturated:
bundle-to-bundle sd 0.030, range 0.16, so the metric is sensitive and the ~0 B−A is a real null). tr(G_dis)/
tr(G_shared) and A_t unchanged vs ERM. Fleet collapse guards: effective rank ≈ 11.8/10.9, contrast norm ≈
0.37/0.51 — no collapse.

## Reading = C-provisional (NOT clean C, NOT clean E)
Two facts pin it between the PM's C and E:
- **Strict E ("MCC == shuffle") is FALSIFIED.** MCC-true beats the matched shuffled-subject control in the
  mechanism's PREDICTED direction on two geometry metrics: Δdir_consistency **B−C = +0.00132** (correctly signed,
  9/9 & 11/12 subjects, sign-flip p≈0.002) and Δ tr(G_dis)/tr(G_shared) **B−C = −0.003 / −0.010** (MCC reduces
  cross-subject disagreement more than shuffle). So the true subject grouping does something specific — the
  mechanism is CONTROLLABLE.
- **Clean C ("geometry did not move") is OVERSTATED.** The controllable signal is 5–15× below any DG-relevant
  scale and INERT: vs ERM, Δdir_consistency B−A ≈ 0 (0/63 bundles exceed |0.01|), DG utility is a clean null
  (ΔU_MCC−ERM straddles 0, sign 15/27 & 15/36 ≈ coin-flip), and ΔU_MCC−shuffle ≈ 0 (p 0.68/0.55) — the mechanism
  buys nothing on target accuracy or vs ERM.
- **Not D (damage/collapse)**: source-val bAcc drop +0.0002/+0.0006 and an INDEPENDENT full-source train-bAcc check
  (panel V3) both < 0.02; no eff-rank/contrast-norm collapse.

**Corrections to the first draft (panel-caught, do not reuse):**
- The "~6% feature bite" (‖Z_B−Z_A‖/‖Z_A‖ ≈ 0.067/0.060) is NOT mechanism-specific — the shuffle control moves the
  features as much or MORE (0.067/0.062). It was wrong to cite it as evidence the MCC objective "bit."
- The "batch alignment 0.51 vs full-source 0.635" batch-noise mechanism is METRIC-MISMATCHED (raw training-loss vs
  whitened geometry) and REVERSES on BNCI2015 (whitened batch 0.667 > whitened full 0.610). Dropped.
- The real C-vs-E discriminator is the **B−C (vs shuffle)** separation above, which the draft table omitted.

## Disposition + PRE-REGISTERED flip criterion (per PM A–E, no unfalsifiable escape hatch)
The mechanism is controllable but DG-inert at this update strength → run ONE bounded stronger-update lever (PM's
choice, not both): **(i) λ_MCC = 1.0, or (ii) unfreeze full-encoder vs top-block**; re-run the SAME 3-arm protocol
+ M1-P oracle re-audit. **Pre-registered flip rule (decided BEFORE the rerun so "too-weak-update" cannot become a
permanent PENDING):**
- **→ Result A / continue** only if the stronger lever (a) AMPLIFIES the true-vs-shuffle geometry separation (B−C),
  **AND** (b) lifts Δdir_consistency B−A above ERM (geometry actually moves vs ERM, not just vs shuffle), **AND**
  (c) yields ΔU_MCC−shuffle > 0 (DG benefit that is mechanism-specific).
- **→ close as E** (generic-regularization / controllable-but-deployment-useless) if the stronger lever amplifies
  the geometry separation but STILL leaves ΔU_MCC−shuffle ≈ 0. Do NOT persist as C.

HELD: M2 selector, learned projector, TTE, CMI, λ sweep beyond the single lever, secondaries, new amendment,
manuscript. Scientific line ACTIVE. AWAITING PM choice of the single bounded lever.
