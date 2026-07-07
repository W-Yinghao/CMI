# Phase 2 — Frozen-feature EEG pilot: consolidated report (2.0 + 2.1 + 2.2)

**Scope.** BNCI2014_001 / BCI-IV-2a (9 subj, 4-class, 2 sessions, 22ch, 250→128 Hz), TSMNet backbone,
frozen LogEig tangent latent Z (z_dim=210), LOSO protocol. Measurement/diagnostic only — no TOS-CMI
end-to-end training, no deletion deployed; the certified gate is a refuse-to-delete safety diagnostic.
Target subject is used for REPORT only (never in selector/gate/probe). Env `icml`, GPU via SLURM.

## Contribution (one paragraph)
On a real SPD/Riemannian EEG-DG backbone we show that conditional domain leakage I(Z;D|Y) is **readily
measurable but not controllable** by the two CMI mechanisms available: (i) a global LPC penalty removes
the leakage only by a sharp λ-tied **optimization collapse** of the representation (and once that
collapse is prevented, it removes nothing), and (ii) **low-rank selective deletion** of the
domain-rich/label-light subspace only dents the leakage because it is high-dimensional and redundant.
The TOS score-Fisher framework correctly **localizes** the leakage subspace, **demonstrates** that
low-rank deletion is insufficient, and **abstains** (the certified gate refuses to delete). This is a
measurement→control **gap**: a positive diagnostic, an honest negative for deployable removal here.

## 2.0 — Measurement / removability (162 dumps: 3 seeds × 9 folds × {ERM + LPC λ∈.03/.1/.3/1/3})
- **Leakage is real and large:** subject-identity decode from ERM Z ≈ 1.00 (chance 0.125), ≫ session
  (~0.90). Domain-probe advantage +0.87.
- **Not low-rank removable:** the score-Fisher selector finds a genuine domain-preferential,
  ~label-light low-rank subspace V_D (nDcand 2–5), but deleting it drops subject decode only
  0.997→0.955 (vs random-k →0.997) while task is fully preserved (Δ≈0). The leakage is high-dim /
  redundant — V_D removal *dents* but does not *remove* it. [3-seed stable; 4-agent adversarial
  workflow verified, see PHASE2_EEG_FROZEN_PILOT.md.]
- **Gate abstains:** ERM score-Fisher is borderline (task_ucb≈δ_Y, 5–7/9 "accept") and the accepts are
  *vacuous* (ablation shows they don't remove domain); at λ≥1 the gate correctly returns
  DOMAIN_GATE_CLOSED. No EEG exact-scope power certificate ⇒ any accept is `diagnostic_accept`, never
  `certified_accept`. No target leakage (audited).

## 2.1 — Why global LPC "works": the collapse mechanism (36 runs, per-epoch curves)
- Global LPC is λ-fragile: λ≤0.3 → 0/9 collapse, λ≥1 → 9/9 collapse (sharp cliff, deterministic).
- **Mechanism (adversarially verified, wf_c2880caf):** a sharp λ-tied **optimization objective-scaling
  bifurcation to feature-norm collapse at the ORIGIN** (Z→0): feat_norm 1.09→0.0000, top-1 singular
  value→0.001, penalty→~0, source CE→ln4=chance. **NOT a gradient explosion** (abs peak grad at collapse
  is ~10× *smaller* than healthy training; 0/36 non-finite) and **NOT geometric over-compression**
  ("eff_rank stays high" is a scale-invariant-metric artifact; only feat_norm/raw-SV reveal the
  collapse). Directly-opposed task/leakage gradients (cos=−0.99 at λ=3). [PHASE21_LPC_COLLAPSE_MECHANISM.md]

## 2.2 — Is the collapse a fixable loophole, and does fixing it recover invariance? (45 runs)
- **Collapse is an optimization pathology, fixable:** a warm-up schedule (ERM→ramp λ) avoids the
  collapse at BOTH λ=1 and λ=3 (0/9, 0/9); a scale-invariant penalty avoids it at λ=1 (0/9) but not
  λ=3. So the λ-fragility is optimization (basin + scale), definitively not a geometric necessity.
- **KEYSTONE:** in **every** collapse-free, task-preserving cell the subject decode stays at ERM
  (0.997 ∈ [.995,.999] vs ERM 1.00). **No collapse-free global LPC removes any leakage** — raw LPC's
  apparent de-domaining was *entirely* an artifact of the collapse. [PHASE22_LPC_OBJECTIVE_SCALING_ABLATION.md]

## Unified conclusion
On TSMNet/2a there is **no task-preserving CMI mechanism that removes the subject leakage**: the global
penalty does it only by collapsing the representation (and removes nothing once that is prevented), and
low-rank selective deletion is insufficient against high-dim/redundant leakage. Measurement→control is
**POSITIVE for diagnosis, NEGATIVE for both deployable knobs.** `task_protect`/power-floor stay OFF; the
gate is a diagnostic / refuse-to-delete module.

## Paper-ready claims
1. *Conditional domain leakage on frozen TSMNet/2a features is high-dimensional and redundant: a
   localized low-rank domain-preferential subspace exists, but deleting it dents subject decode only
   0.997→0.955 while a global penalty cannot remove it without collapse.*
2. *The global-LPC λ-collapse is an optimization objective-scaling bifurcation to feature-norm collapse
   at the origin (Z→0), not a gradient explosion and not geometric over-compression; it is removable by
   warm-up scheduling.*
3. *Every collapse-free global LPC leaves subject leakage at ERM levels — its apparent invariance was an
   artifact of representation collapse.*
4. *The TOS score-Fisher framework correctly localizes the leakage, certifies that low-rank deletion is
   insufficient, and abstains — a measurement→control certification gap, not a removal method.*

## Figures / artifacts
- Ablation removability (V_D vs random-k, linear+MLP, subject vs session) — `eeg/ablation.py`,
  `eeg/adversarial.py`, `aggregate3` outputs.
- Collapse-mechanism 5-row curves (task CE, λ·penalty, grad-norm, eff_rank [scale-invariant],
  feat_norm) — `results/.../lpc_collapse_curves/collapse_curves.png`.
- Variant comparison table + keystone — `results/.../lpc_collapse_curves/variant_compare.json`.

## Caveats / scope (must accompany any claim)
- Single backbone (TSMNet) + dataset (2a); folds {1,5,9} for the curve sweeps (3 folds × 3 seeds × 4 λ);
  300ep, curve_every=10 (collapse-epoch ±10ep). Do not generalize to other backbones/datasets/schedules
  without rerun.
- The per-epoch encoder grad is a between-epoch read-only diagnostic proxy (eval-mode, fixed batch,
  separate RNG), not the training-step gradient. eff_rank/stable_rank are scale-invariant (non-probative
  for compression). Domain==subject under LOSO ⇒ group-aware certification folds abstain
  (FOLD_COVERAGE_FAILURE); diagnostics use trial-level folds.
- No certified (exact-scope power) deletion is claimed on EEG; the synthetic certification line closed as
  an honest negative (PHASE131_CERTIFICATION.md).
