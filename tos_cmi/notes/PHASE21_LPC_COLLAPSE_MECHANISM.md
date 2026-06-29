# Phase 2.1 — LPC collapse-mechanism audit (BNCI2014_001 / TSMNet / LOSO frozen pilot)

**Question (from Phase 2.0):** is the global-LPC (`lpc_prior` = CE + λ·I(Z;D|Y)) representation collapse
at high λ an **optimization** failure or a **smooth geometric over-compression** of the data? Phase 2.0
could only infer "likely optimization" indirectly (bimodality 26/27 at λ=1). Phase 2.1 settles it with
**direct per-epoch training curves**.

**Method (instrumentation only — no dynamics change).** Flag-gated `log_curves` in
[cmi/train/trainer.py](../../cmi/train/trainer.py) (default-off = verified no-op): per epoch records
task-CE, λ·LPC penalty, encoder grad-norm + max + NaN flag, source-eval CE/bAcc/entropy, and feature
geometry (eff_rank, stable_rank, **feat_norm**, top-5 singular values) on a FIXED source subset; every
`curve_every` epochs a grad-decomposition (‖grad_task‖, ‖grad_LPC‖, cos) on the encoder (=backbone
minus task head) via `torch.autograd.grad` (never writes `.grad`, never `opt.step` → RiemannianAdam
state untouched; separate RNG `seed+12345`). Runner [run_lpc_curves.py](../../tos_cmi/run_lpc_curves.py),
analyzer [eeg/collapse_analysis.py](../../tos_cmi/eeg/collapse_analysis.py). Sweep = pre-registered
folds {1,5,9} × seeds {0,1,2} × λ {0, 0.3, 1, 3}, 300 epochs, curve_every=10.

## VERDICT (adversarially verified — wf_c2880caf-68a, 4 independent skeptics + synthesis)

**The high-λ collapse is a sharp, λ-tied OPTIMIZATION BIFURCATION to a degenerate trivial minimizer —
feature-norm collapse to the ORIGIN (Z→0) — which zeroes the penalty and the task simultaneously. It is
NOT a gradient explosion, and NOT a smooth geometric over-compression of the data.**

This is the "objective-scaling failure" branch (the global penalized objective is, above a critical λ,
minimized by the trivial Z→0 solution that satisfies I(Z;D|Y)=0 at the cost of all task signal). It is
NOT evidence that the data lacks a removable leakage subspace.

### Evidence (all re-derived from the 36 raw curve JSONs)
| λ | collapse | feat_norm init→final | top-1 SV final | penalty peak→final | CE final | abs peak grad | type |
|---|---|---|---|---|---|---|---|
| 0   | 0/9 | 1.09 → **5.85** | ~107 | 0 → 0 | 0.49 | 150 | no_collapse |
| 0.3 | 0/9 | 1.09 → **5.69** | ~107 | 0.62 → 0.62 | 0.50 | 181 | no_collapse |
| 1   | **9/9** | 1.09 → **0.0000** | **0.001** | 0.79 → **0.001** | **1.386** | **20.8** | feature_norm_collapse |
| 3   | **9/9** | 1.09 → **0.0001** | **0.001** | 0.36 → **0.002** | **1.386** | **7.9** | feature_norm_collapse |

- **Feature-norm collapse to a point (PRIMARY signal):** feat_norm → 0.0000 (controls grow to ~5.7);
  raw top-1 singular value → 0.001 (controls ~107); eval CE → ln(4)=1.386 (uniform 4-class output);
  source & target bAcc → chance 0.25. The encoder output shrinks to the origin.
- **Sharp λ-cliff = bifurcation:** λ≤0.3 NEVER collapse, λ≥1 ALWAYS collapse; final bAcc is only ever
  ~0.25 or ~0.8 (zero intermediate values); **deterministic in λ** (no within-λ seed bimodality).
- **λ-tied timing:** λ=3 collapses @ep1–9 (immediately as λ ramps); λ=1 @ep15–38 (as the linear warmup
  pushes effective λ past the critical point ≈0.4–0.9). The penalty ramps up, then crashes to ~0 once
  Z→0 trivially satisfies it.
- **Directly-opposed objectives:** median cos(grad_task, grad_LPC) = **−0.99** at λ=3 (the LPC gradient
  points opposite the task gradient — at high weight the penalty wins and kills the representation).
- **Instrumentation-neutral:** the uninstrumented Phase 2.0 dumps (`log_curves=False`) reproduce the
  λ=1 collapse in **26/27** runs (→ chance, dom_adv −0.01); collapse predates the instrumentation.

### Corrections forced by the adversarial verification (vs my first pass)
1. **"gradient explosion 50–300×" → REFUTED.** That ratio was peak / (post-collapse near-zero median):
   after collapse the dead encoder sits at a ~0-gradient fixed point, dragging the median to ~0.1.
   The honest peak / **pre-collapse** ratio is only **~2.4×**, and the **absolute** peak grad at
   collapse (8–21) is **~10× SMALLER** than in healthy low-λ training (150–181). 0/36 runs ever
   produced a non-finite grad; the peak grad epoch *precedes* the cliff, then the grad dies. It is a
   **quench into a degenerate fixed point, not a divergence.**
2. **"eff_rank stays high ⇒ not geometric compression" → RETRACTED as evidence.** eff_rank,
   stable_rank, top-1 share are all **scale-invariant** (eff_rank = exp(entropy(s/Σs))), structurally
   blind to magnitude. They stay ~ERM (153–167) at collapse but that is **non-probative**. Only
   feat_norm and raw singular values (magnitude-sensitive) reveal the collapse — and they go to ~0.
3. The mechanism is **coupled optimization+geometry**: a sharp optimization bifurcation that *produces*
   total geometric compression **to the origin** (not a rank-1 line, not divergence). Do not sell it as
   "optimization instead of geometry."

### Residual caveats (write into any paper text)
- `grad_total_encoder_norm` is a **between-epoch read-only diagnostic** on a fixed eval-mode batch
  (separate RNG), not the actual training-step gradient; it also already includes the λ-warmup weight.
  Claims about "explosion / no-explosion" are bounded to this proxy.
- eff_rank/stable_rank/top1-share are scale-invariant → never cite for/against compression; only
  feat_norm and raw singular values are magnitude-sensitive. "eff_rank preserved" = feature-NORM
  collapse (scale→0), NOT spectral-rank collapse.
- **Scope:** BNCI2014_001/2a, TSMNet, LOSO, folds {1,5,9} only, 3 seeds, 4 λ points, 300 epochs,
  curve_every=10 (collapse-epoch timing ±10ep). Do not generalize to other backbones/datasets/schedules
  without a rerun.

### One line for the paper
> At λ≥1 the global-LPC objective on TSMNet/2a undergoes a sharp, λ-tied bifurcation that collapses the
> encoder output to the origin (feature-norm and top-singular-value → 0, source CE → ln K, source bAcc
> → chance in 18/18 runs, deterministically and independent of instrumentation), driven by
> directly-opposed task/leakage gradients rather than by any gradient explosion (the diagnostic encoder
> gradient is ~10× smaller than in healthy low-λ training and never non-finite); the apparent
> "effective-rank is preserved" is a scale-invariant-metric artifact and does not contradict this norm
> collapse.

## Consequence for Phase 2 / the contribution
This **confirms and sharpens** the Phase 2.0 conclusion: global LPC's λ-fragility is an
**objective-scaling pathology** (degenerate Z→0 minimizer above a critical λ), *not* proof that
leakage is irremovable. Combined with Phase 2.0 (selective low-rank deletion only DENTS the high-dim
redundant subject leakage), the honest Phase 2 result stands: **measurement-to-control POSITIVE for
diagnosis (the framework localizes the leakage subspace, shows low-rank deletion is insufficient, and
the gate abstains), NEGATIVE for both deployable knobs available here** — global LPC collapses via
objective-scaling, and low-rank selective deletion is insufficient. `task_protect`/power-floor stay OFF.

Artifacts: `results/tos_cmi_eeg_frozen/lpc_collapse_curves/{summary.json, collapse_curves.png, sub*_lam*_seed*.json}`.
