# Phase 2.2 — LPC objective-scaling sanity ablation (reviewer-proofing)

**Question.** Phase 2.1 showed the high-λ global-LPC collapse is a sharp objective-scaling bifurcation
to feature-norm collapse (Z→0). The reviewer's inevitable follow-up: *if that's a Z→0 loophole, does a
simple scale-invariant penalty (or a warm-up schedule) fix it — and if so, does corrected LPC actually
remove the leakage?* This is a **closed, pre-registered ablation**, not a new method or a baseline rescue.

**Variants** (flag-gated in [cmi/train/trainer.py](../../cmi/train/trainer.py), default-off = verified
no-op; task head always sees raw Z):
- `raw_lpc` — original (= Phase 2.1). ERM (λ=0) reused as the uncollapsed leakage reference.
- `lpc_scale_invariant` — LPC penalty **and its Step-A posterior** see `Z_pen=(Z−sg μ_B)/(sg σ_B+ε)`
  (`lpc_pen_normalize=True`). Stop-grad on batch mean/std ⇒ Z→0 can no longer lower I(Z;D|Y) via scale.
- `lpc_warm_ramp` — ERM until ep100, λ ramps over ep100–150, fixed after (`lam_warm_ramp=True`). Tests
  whether the collapse is a high-λ cold-start basin.

**Matrix.** folds {1,5,9} × seeds {0,1,2} × λ {1,3} for the two variants, + raw_lpc λ=0 ERM baseline =
45 runs, 300ep, full Phase 2.1 per-epoch instrumentation + final-feature task/subject decode probes
(`_decode`, 50/50 MLP). Analyzer [eeg/variant_compare.py](../../tos_cmi/eeg/variant_compare.py).

## Results (medians over 9 runs/cell; per-run spread tight, deterministic — no median-masking)
| variant | λ | collapse | feat_norm | src bAcc | task_dec | **subj_dec** |
|---|---|---|---|---|---|---|
| ERM (raw_lpc λ=0) | 0 | 0/9 | 5.85 | 0.83 | 0.76 | **1.00** [.995,.999] |
| raw_lpc | 1 | 9/9 | 0.00 | 0.25 | — | — |
| raw_lpc | 3 | 9/9 | 0.00 | 0.25 | — | — |
| **lpc_scale_invariant** | 1 | **0/9** | 5.69 | 0.82 | 0.75 | **0.997** [.995,.999] |
| lpc_scale_invariant | 3 | 9/9 | 0.00 | 0.25 | 0.24 | 0.117 (chance) |
| **lpc_warm_ramp** | 1 | **0/9** | 5.85 | 0.83 | 0.75 | **0.997** [.995,.999] |
| **lpc_warm_ramp** | 3 | **0/9** | 5.85 | 0.83 | 0.75 | **0.997** [.995,.999] |

## Two robust conclusions

### 1. The collapse is an OPTIMIZATION pathology (fixable), not a data-geometry necessity
- **warm_ramp avoids the collapse at BOTH λ=1 and λ=3** (0/9, 0/9; task preserved) — starting from an
  ERM solution and ramping λ never enters the Z→0 basin. So the collapse is substantially a
  **cold-start basin** artifact.
- **scale_invariant avoids it at λ=1** (0/9) but **NOT at λ=3** (9/9) — blocking the feature-scale
  loophole helps at moderate λ; at extreme λ the objective still finds a degenerate solution. So the
  collapse is *both* a scale loophole *and* a basin issue (decision-rule label "C").
- Either way this **confirms Phase 2.1**: the λ-fragility is optimization/objective-scaling, definitively
  NOT a geometric necessity that leakage removal must destroy the representation.

### 2. KEYSTONE — no collapse-free LPC removes ANY leakage (the reviewer-proof result)
In **every** collapse-free, task-preserving cell (scale_invariant λ=1; warm_ramp λ=1 and λ=3) the
subject decode is **0.997 ∈ [0.995, 0.999] = ERM (1.00)** while task is preserved (task_dec ~0.75).
**Once the collapse is prevented — by either fix — the global CMI penalty reduces subject leakage by
ZERO.** Raw global-LPC's apparent "de-domaining" (Phase 2.0/2.1: dom_adv→−0.01 at λ≥1) was therefore
**entirely an artifact of the representation collapse**, not real invariance.

## Consequence for Phase 2
Combined with Phase 2.0 (low-rank selective deletion only DENTS the high-dim/redundant subject leakage,
0.997→0.955) and Phase 2.1 (global LPC "removes" leakage only by collapsing the representation), Phase
2.2 closes the loop: **on TSMNet/2a there is no task-preserving CMI mechanism — global penalty (any
collapse-free variant) OR low-rank deletion — that removes the subject leakage.** The leakage is
genuinely high-dimensional/redundant/task-entangled. The TOS framework's role is correctly **diagnostic**
(localize the leakage, demonstrate it is not removable by available controls, and abstain).

## Strict boundaries honored
No β tuning, no backbone/dataset change, no PCGrad, no TOS selective-penalty training, no
task_protect/power_floor change, no λ-schedule search beyond the single fixed warm-ramp. All variants
flag-gated, default-off. One-shot ablation; **Phase 2 (2.0+2.1+2.2) COMPLETE** → write-up.

Artifacts: `results/.../lpc_collapse_curves/{lpc_scale_invariant,lpc_warm_ramp,raw_lpc}_sub*_seed*.{json,npz}`,
`variant_compare.json`.
