# DualPC — consolidated summary (design · experiments · conclusions)

*Self-contained summary of the current `dualpc` algorithm for the AAAI EEG-DG paper. Sources:
`notes/AAAI_DUALPC_DESIGN.md`, `notes/dual_cmi_project_report_2026-06-16.md`,
`results/dualpc_protocol_paper/`. Companion: `notes/CONCEPT_SHIFT_SECTION.md` (diagnostics),
`notes/DUAL_CMI_THEORY.md` (tension theorem). Last updated 2026-06-14.*

---

## 1. Motivation
Zero-calibration cross-subject / cross-site EEG domain generalization: learn `Z=f(X)` that keeps the task
label `Y` while removing post-label domain information. The deployment story has **two** sides that should be
controlled together:
- **`P(z)` (representation):** feature space domain-stable;
- **`P(y|z)` (predictor):** decision boundary domain-stable;
- **label-prior first:** otherwise `P(z)` alignment erases labels and raw `I(Y;D|Z)` reads label-prior/calibration artifacts.

Naive `dual` co-minimizes `I(Z;D|Y)+I(Y;D|Z)`, but under label shift these **fight** (tension theorem,
verified 2.4e-15): `I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)`, so forcing the encoder term to 0 forces the
decoder term up. `dualpc` is the GLS-factorized objective that dissolves this.

## 2. The `dualpc` objective
```
L = CE(Y|Z)  +  λ · I_w(Z;D|Y)  +  γ · [ JS_w( h_full(Y|Z,D), h0(Y|Z,D) ) − τ ]_+
```
- `w_i = π*(y_i)/π_{d_i}(y_i)` = GLS label-shift weight, `π*` uniform (importance-weighted continuous form of "biased batch sampling").
- **`I_w(Z;D|Y)` = factorized `P(z)` control.** Under the GLS reference `Y⊥D`, forcing it →0 aligns the class-balanced mixture `P_w(Z|D)=Σ_y π*(y)P(Z|Y=y,D)` — i.e. aligns the per-class `P(z|y)` across domains while *preserving* `Y` (avoids the label-erasure of direct marginal `I_w(Z;D)`).
- **`JS_w(h_full, h0)` = `P(y|z)` control.** JS-consistency between the full domain decoder `h_full(Y|Z,D)` and the intercept-only decoder `h0(Y|Z,D)=u(Z)+b_D` (D may only shift a per-domain bias). Replaces `dualc`'s CE-residual *as the training loss* (CE-residual raised held-out decoder leakage while improving acc → kept only as a diagnostic).
- Gate `τ`: `dualpc` defaults `τ=0` (JS active); `dualc` keeps `τ=0.02`. `dualpc` forces the RAW sampler (the CMI terms already carry explicit GLS weights).

### Variant roles
| method | `P(z)` side | `P(y\|z)` side | role |
|---|---|---|---|
| `lpc_prior` | `I(Z;D\|Y)` | — | stable main baseline |
| `dual` | `I(Z;D\|Y)` | raw `I(Y;D\|Z)` | tension ablation (fights under label shift) |
| `dualc` | `I(Z;D\|Y)` | gated CE-residual `R_res` | concept diagnostic / CE-residual ablation |
| **`dualpc`** | **factorized `I_w(Z;D\|Y)`** | **gated JS to `h0`** | **candidate main algorithm** |
| `dualpc_marginal` | direct `I_w(Z;D)` | gated JS to `h0` | negative-control ablation (unstable) |

## 3. Experiments

### 3.1 Synthetic gates (CPU `synthetic/dualpc_validation.py`)
- **null-prior:** DualPC-JS matches ERM target bAcc, slightly lowers `P(z)` KL and JS — **null-safe**.
- **concept:** DualPC-JS improves both probes without hurting target (bAcc 53.5, `P(z)`KL 0.122, res 0.080) whereas `dualc` CE-residual raises target (63.1) but *worsens both probes* (0.156 / 0.172) → CE-residual is not a clean simultaneous optimizer; JS is.
- **all-three / covariate+label:** DualPC ≈ LPC ≈ DualC; `dualpc_marginal` worse → direct marginal demoted.

### 3.2 Paper protocol — the 5 SLURM rounds (2026-06-17/18)
Goal across rounds: pick `λ/γ/τ` **without target labels** (source-only selection), then prove null-safety vs `lpc_prior`. Iterative **selector hardening**:
- **R1** (paper-g1/g2): full protocol — regression checks + synthetic gate (null/concept/all_three) + LOSO (BNCI2014_001, MUMTAZ) + guarded selector + SCPS PD/SCZ (D=cohort), seeds 0/1/2.
- **R2** (paper-r2, r2b): fixed-grid vs strict selector; narrowed SCPS grids.
- **R3** (paper-p3, p3b): hinge/gate grid + selector guard.
- **R4** (r4): repeated-CV mean selector.
- **R5** (r5): leave-one-DATASET-out selector (most stringent, no target leakage).

### 3.3 Headline multi-seed results (3 seeds, `results/dualpc_protocol_paper/`)
| task | erm | lpc_prior | dualc | **dualpc** | leakKL_rw (erm→cmi) | JS `P(y\|z)` |
|---|---|---|---|---|---|---|
| LOSO BNCI2014_001 (4-cls MI) | 43.2 | 41.4 | 41.5 | **41.5** | 1.18 → 0.31 | ~0.0015 |
| SCPS PD (D=cohort) | 59.0 | 60.5 | 60.3 | **59.5** | 0.20 → 0.035 | ~0.0001 |
| SCPS SCZ (D=cohort) | 53.2 | 52.3 | 52.1 | **53.3** | 0.48 → 0.13 | ~0.0001 |

## 4. Conclusions (honest)
1. **`dualpc` is null-safe:** matches `lpc_prior`/`dualc` accuracy (within noise), cuts leakage 3–6×, JS `P(y|z)`≈0 (concept null re-confirmed on real data). It PASSES design gate-2 (null safety).
2. **`dualpc` does NOT beat vanilla ERM on accuracy** — because concept shift is ≈0 on these datasets, the JS decoder term has nothing to fix, and leakage removal alone is accuracy-neutral here.
3. **Decision gate = NEEDS_REVIEW** (`dualpc_decision.json`): protocol complete (4 comparison tasks w/ erm+lpc baselines, 2 selectors, synthetic gate PASS) but with baseline/selector warnings.
4. **Relation to Zhao-2020 entropy-DG (verified):** `P(z|y)` is the same nominal object, but the *mechanisms differ* — Zhao's entropy/JSD term is within-domain across-class (predictor route); its `P(z)` is the raw marginal aligned adversarially. `dualpc`'s `I_w(Z;D|Y)` directly aligns `P(z|y)` cross-domain and *rejects* raw-marginal alignment (= demoted `dualpc_marginal`). `dualpc` is closer to a **GLS-reweighted conditional-MI reformulation** of Zhao's intent. (Coincide only when class-balanced.)
5. **Honest positioning:** the paper's defensible core = conditional leakage removal (`lpc_prior`) + tension theorem + concept-shift diagnostic; `dualpc` = principled joint-`P(z)`-`P(y|z)` extension whose current value is **null-safety + source-only selection**, NOT accuracy gain.

## 5. The open gap → the new algorithm (Task 2)
**Every method here is accuracy-parity with ERM.** The remaining, hard objective: an algorithm that **jointly
optimizes `P(z)` and `P(y|z)` AND beats vanilla ERM on accuracy** (not just leakage), with clear theory and
ablations. Pure invariance (remove `I(Z;D|Y)`) is accuracy-neutral when leakage is "harmless"; the new method
must ADD a discriminative / target-risk-reducing mechanism. This is developed iteratively in
`notes/DUALPC2_DESIGN.md` (next).
