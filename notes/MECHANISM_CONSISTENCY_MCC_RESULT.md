# MCC training viability — RESULT = C (geometry did not move; update too weak, NOT falsification)

Real EEG, full LOSO. 63 bundles × 3 arms = 189/189 continuation cells complete (training code identical across
git c0b2f8ee/39293f9e; config cmi_trace_mechanism_consistency.yaml frozen; all 3 arms per bundle share a warm-up
hash). EEGNet, 16-d bottleneck, warm-up = existing ERM, +20 continuation epochs, λ=0.25 ramp/5ep, LR 0.1×,
source-only checkpoint selection. Manuscript FROZEN; graded verdict, not a stop.

## Result
| dataset | n | ΔU_MCC−ERM [95% cluster CI] | sign-flip p | ΔU_MCC−shuffle | Δdir_consistency | ΔA_t | Δ tr(G_dis)/tr(G_shared) | source drop |
|---|---|---|---|---|---|---|---|---|
| BNCI2014_001 | 9 | +0.0002 [−0.0021, +0.0023] | 0.469 | −0.0005 | +0.0006 | −0.0006 | −0.0021 | +0.0002 |
| BNCI2015_001 | 12 | −0.0004 [−0.0029, +0.0018] | 0.638 | −0.0001 | −0.0001 | +0.0002 | −0.0019 | +0.0006 |

**Absolute geometry (seed 0):** source dir-consistency ERM 0.635 / 0.616 → MCC 0.637 / 0.616 (room to 1.0 =
0.37/0.38 — the baseline is NOT saturated); tr(G_dis)/tr(G_shared) 1.16/1.46 unchanged; target→source alignment
A_t 0.48/0.40 unchanged.

## Reading = C (not A/B/D/E)
- **DG utility ≈ 0**: ΔU_MCC−ERM straddles zero, sign counts 15/27 and 15/36 (≈ coin flip), |B−A| target bAcc
  ~0.005; sign-flip p 0.47/0.64. MCC does not beat ERM-continuation.
- **MCC ≈ shuffle**: ΔU_MCC−shuffle ≈ 0.
- **Geometry did NOT move**: Δdir_consistency ≈ 0 despite ample room; tr(G_dis)/tr(G_shared) and A_t unchanged.
- **Not D (damage)**: source-val bAcc drop +0.0002/+0.0006 (≪ 0.02); effective rank ~12.3, contrast norm ~0.27 —
  no collapse.
- **The update was not literally zero** — MCC continuation changed the source features by ‖Z_B−Z_A‖/‖Z_A‖ ≈
  0.068/0.056 (~6%) relative to ERM-continuation — but that movement did NOT improve cross-subject contrast
  consistency, DG, or the disagreement ratio. So the fixed gentle continuation (λ=0.25, LR 1e-4, 20 epochs) is too
  weak / mis-placed to move the geometry, rather than the hypothesis being falsified.

Candidate mechanism (observation, not a spec change): the MCC gradient is computed on K=4-trial batch class-mean
contrasts whose batch alignment (~0.51, = 1−mean batch MCC loss 0.49) is well BELOW the full-source dir-consistency
(0.635) — i.e. the per-batch contrasts are dominated by sampling noise, so reducing batch-level inconsistency need
not move the true (full-source) consensus. This is a plausible reason the update "bit" the features (6%) without
moving the target geometry.

## Disposition (per PM A–E map)
Result C: the current loss/update position is too weak; this does NOT falsify mechanism-consistency training. PM's
prescribed response = ONE bounded fix, PM to choose (not both): (i) λ_MCC = 1.0, or (ii) unfreeze the full encoder
vs the top block. Re-run the SAME 3-arm protocol + oracle re-audit. If a stronger update STILL does not move the
geometry, that would begin to separate C-persists from E (generic-regularization) and would warrant the true/shuffle
gradient-cosine + margin + effective-rank analysis. M1-P re-audit is consistent with C: tr(G_dis)/tr(G_shared) is
unchanged, i.e. training did not alter the disagreement structure.

HELD: M2 selector, learned projector, TTE, CMI, λ sweep, secondaries, new amendment, manuscript. Scientific line
ACTIVE. AWAITING PM choice of the single bounded lever.
