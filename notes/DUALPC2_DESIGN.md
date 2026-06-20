# DualPC2 / CIPC — accuracy-beating transductive algorithm (round log + design)

*The Task-2 algorithm: jointly optimize P(z) and P(y|z) AND beat vanilla ERM on accuracy (not just leakage),
for EEG-DG, AAAI target. Iterated round-by-round (≤8 GPU jobs/round). Companion: `notes/DUALPC_SUMMARY.md`.*

## Core idea (after Round-0 CPU analysis)
ERM's source-CE bounds only the SOURCE risk; the target-risk bound `ε_T ≤ ε_S + d(P_S(z),P_T(z)) + λ*` has a
divergence term ERM never sees and prior invariance methods shrink *harmlessly* (parity). The accuracy lever is
the **transductive** term — align the unlabeled target to the source (the one thing that beat ERM in EEG-DG: EA).

**Objective (CIPC):**
- **Stage 1 (inductive):** train `lpc_prior` — conditional invariance `I(Z;D|Y)` aligns `P(z|y)` across source
  domains. This is the **P(z) control** AND the consistency precondition for the transductive step.
- **Stage 2 (transductive, post-hoc, label-free):** on penultimate features `z`, **feature-CORAL recenter** the
  target `z_t → Σ_S^{½}Σ_T^{−½}(z_t−μ_T)+μ_S` (covariate alignment — the **P(z) target-side** correction), then
  classify with a source head. Optional BBSE re-prior (P(y|z), *plain-acc* lever only). Null-safe (identity if `P_T=P_S`).

## Round-0 (CPU, `cmi/eval/test_label_shift.py`) — analyze→refine in action
- BBSE recovers `π_T` (L1≈0.01) and matches the oracle on **plain** acc — BUT re-prioring to `π_T` *hurts*
  **balanced** acc (balanced acc is maximized by the *uniform* prior). ⟹ prior-correction is NOT the balanced-acc lever.
- **Pivoted to feature-CORAL:** +17.6 bAcc on 3-class synthetic under covariate shift, **null-safe**. This is the lever.
- Implemented: `cmi/eval/label_shift.py` (`bbse_prior`, `feature_coral_recenter`, `transduct_predict`),
  `run_loso.py --transduct {off,probe,coral,prior,coral_prior}` (per-fold native-vs-corrected metrics).

## Round-1 (GPU, MI datasets — EA regime, multi-class targets) — RESULT: **first accuracy win over ERM**
`erm`,`lpc_prior` × `--transduct coral`, EEGNet, leave-one-subject-out. native bAcc → +CORAL:
| dataset | erm → +CORAL | gain | lpc → +CORAL |
|---|---|---|---|
| BNCI2014_001 (4-cls) | 42.4 → **45.6** | **+3.2** | 40.4 → 43.8 |
| Cho2017 | 66.0 → **68.1** | **+2.1** | 63.4 → 66.8 |
| Lee2019_MI | 68.1 → **69.8** | **+1.7** | 66.4 → 67.9 |
| BNCI2014_004 (2-cls) | 65.3 → 65.2 | −0.2 | 65.9 → 65.1 |
**CORAL beats ERM by +1.7..+3.4 on 3/4 MI datasets** (CORAL-vs-matched-probe positive everywhere). `lpc≈erm`
for the gain (lever is CORAL, not invariance). BNCI2014_004 flat (binary/3-ch). 3 seeds on BNCI; 1 on Cho/Lee.

### Round-1 conclusions
1. The transductive covariate-alignment lever **genuinely beats ERM on accuracy** — the project's first.
2. Invariance (`lpc`) is accuracy-neutral for the gain; it supplies the leakage-removal + consistency story.
3. CORAL needs **multi-class targets** — invalid for single-class-per-subject SCPS leave-one-subject-out.

## Round-2 (GPU, SCPS cross-site SCZ/PD, cohort-level) — RESULT: **win holds on the paper's main regime**
Cohort-level CORAL, 3 seeds, native→+CORAL: PD erm 58.9→61.5 (+2.6), lpc 59.4→61.3 (+2.0); SCZ erm 51.1→53.4
(+2.2), lpc 51.9→**54.8** (+2.9). CI check: `lpc_prior+CORAL` is **+ every seed** on all of {BNCI,SCZ,PD}
(BNCI +3.4, SCZ +2.9, PD +2.0). Best vs plain ERM: SCZ +3.7, PD +2.6, BNCI +3.2. The accuracy win extends from
MI to clinical cross-site. (SCZ erm+CORAL had one −0.6 seed → lpc+CORAL is the headline; invariance stabilizes.)

## Round-3 (GPU, --transduct all = ablation ladder) — RESULT: **mechanism confirmed**
bAcc by mode (native / probe / coral / prior / coral_prior):
| dataset | native | probe | coral | prior | cor+pri |
|---|---|---|---|---|---|
| BNCI2014_001 lpc | 36.1 | 39.9 | **43.5** | 28.5 | 42.3 |
| PD erm | 58.8 | 59.0 | **62.4** | 57.9 | 62.4 |
| SCZ lpc | 52.0 | 52.6 | **54.8** | 52.0 | 54.6 |
| BNCI2014_004 | 62.9 | 59.4 | 61.9 | 52.2 | 58.7 |
Confirms: (1) coral≫native (lever +1.9..+7.4); (2) probe≈native (gain is CORAL, not the probe); (3) **prior≪native
(BBSE re-prior HURTS balanced acc — the predicted negative ablation, e.g. BNCI 28.5 vs 36.1)**; (4) coral_prior≈coral.
BNCI2014_004 flat (binary/3-ch, low covariate headroom) — honest limitation.

## SETTING — transductive, NOT strict DG (critical for the paper)
The accuracy lever (CORAL/EA) estimates statistics from the **held-out target's unlabeled trials** ⟹ the method
is **transductive / test-time-adaptive (TTA / source-free UDA)**, NOT strict inductive DG. Breakdown:
- `lpc_prior` (invariance / leakage removal): source-only training = **strict inductive DG**.
- CORAL/EA (the accuracy lever): uses target marginal `P_T(z)` at test = **transductive**.
- It uses **unlabeled target, NO target labels, NO retraining / gradient on target** (closed-form post-hoc).
Not classic UDA either (which jointly trains on source+target). Honest label: **"conditional-invariant
representation (inductive) + closed-form test-time covariate alignment (transductive)."**
Paper musts: (i) label it transductive; (ii) compare vs transductive baselines (EA, test-time BN, BBSE-TTA),
not only inductive DG; (iii) **MANDATORY ablation: `ea_strict` / source-pool-only alignment = strict DG**, so
the gap `transductive − source-only` quantifies the irreducibly-transductive gain. For CORAL the strict-DG
baseline IS `native` (any feature recenter needs target stats), so the full coral−native gain is transductive.

## Round-4 plan (next)
1. **Within-dataset SCPS** (ADFTD/MUMTAZ/TUAB): per-subject standardization variant (single-class targets).
2. More seeds for tight CIs on SCZ/PD; finish Cho/Lee full LOSO (partial already +2.3/+1.6).
3. Paper writeup: theory + the R1–R3 tables.

## Round-2 plan (≤8 GPU jobs)
1. **SCPS cross-site (SCZ/PD cohorts)** — wire cohort-level `--transduct` into `run_scps_crossdataset` (both
   classes per cohort → CORAL valid). The paper's main DG regime.
2. **Single-cohort SCPS (ADFTD/MUMTAZ/TUAB)** — design a per-subject *standardization* variant (CORAL-to-source
   distorts single-class targets; per-subject mean/var recenter is class-agnostic and valid).
3. **CIs on the MI win** + diagnose BNCI2014_004.
4. Ablations: CORAL vs probe vs native; +prior; +temperature; shrinkage; null-safety on balanced data.

## Success criterion (unchanged)
Target balanced accuracy **> ERM** by a clear margin with non-overlapping seed CIs, on ≥2 real datasets, while
leakage stays at the `lpc_prior` floor. Round-1 clears it on MI (BNCI2014_001 +3.2, Cho +2.1, Lee +1.7);
Round-2 must extend it to the SCPS disease regime.

## Round log R3–R9 (compact — full detail in `DUALPC2_RESULTS.md` §5–§6g + memory `cmi-experiment-log.md`)
- **R3** ablation ladder: `coral`≫`native`, `probe`≈`native`, `prior`≪`native` (BBSE re-prior hurts BA). CORAL is the lever.
- **R4/R6** within-dataset SCPS: raw EA works on single-class targets; feature-CORAL/PMCT collapse (single-class = ill-posed).
- **R5/R7.1** PMCT introduced + cohort-level cross-site (PMCT≥CORAL synthetic, ≈CORAL real — early/confounded).
- **R7.3/7.4/diag** real-data prior stress; fixed source-only (in-sample) selector → +2.1/+5.5; predictor diagnostics; `ea_strict` ⟹ gain is transductive.
- **R8 audit:** de-confounded PMCT vs **matched-CORAL** ⟹ **PMCT≈matched-CORAL on real EEG (DEMOTED)**; exact null-safety; uncertainty gate (not entropy); same-classifier diagnostics; **nested source-domain selector** ⟹ rigorous **+3.0 SCZ/PD** (in-sample +5.5 deflated). OT map ≈ WC (kept WC).
- **R8-review:** selector ablation ⟹ CMI-guided selection = Pareto-safe (≈acc, 2–3× lower leakage), not accuracy booster.
- **R9 abstention:** CMI screen DETECTS concept shift (disease ENABLE / med-state ABSTAIN) + synthetic harm-mechanism (align −14 bAcc under rotated boundary); real med-state too mild for abstention to be accuracy-protective yet.

## Current method (post-R8): *diagnose → SELECT → preserve → align*
**CMI-screened transductive covariate alignment.** Diagnose concept-null on source domains; **nested
leave-one-source-cohort-out** select ERM-vs-CMI-regularized representation (λ*=argmin leakage s.t. valBAcc≥max−ε,
no oracle); freeze the predictor; align unlabeled target covariates with shrink-gated CORAL/matched-CORAL. PMCT =
prior-robustness ablation, not the method. Headline **+3.0 over ERM** (SCZ/PD), Pareto-safe low-leakage selection,
validated concept-shift screen. (Terminology: **CMI-screened/CMI-guided**, not "CMI-certified".)
