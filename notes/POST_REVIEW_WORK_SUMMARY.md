> ⚠️ **SUPERSEDED (2026-06-21) by `notes/EVIDENCE_LEDGER.md`.** The "CMI-screened ... beats ERM by +3.0" headline is RETRACTED (deflated to ~+1.8/+2.3 2-seed; the lever is matched-CORAL, not CMI/LPC; CITA ties SPDIM). LPC is DROPPED as a deployment mechanism (P1.5), the CMI/residual-CMI gate is report-only DIAGNOSTIC, and the harm-gate/rollback/abstention line closed to `DIAGNOSTIC_ONLY` (A0-PILOT). Read the ledger for current status.

# Post-review work summary — from "dualpc is null-safe but accuracy-parity" to a transductive method that beats ERM

*Consolidated record of all code changes + experiments done after the review (the dualpc/Zhao positioning +
the goal "develop a new algorithm that beats vanilla ERM on accuracy, not just leakage"). Companions:
`notes/DUALPC2_RESULTS.md` (clean results — AUTHORITATIVE), `notes/DUALPC2_DESIGN.md` (round log),
`notes/DUALPC_SUMMARY.md` (old dualpc). Date: 2026-06-20.*

---

## LATEST STATE (R7–R9, authoritative — supersedes §1/§5/§6 below)
Three review rounds later, the honest, audit-survived position is:

**Headline (5-seed confirmatory running):** `CITA-nested + transductive alignment` (CMI-screened, nested
leave-one-source-cohort-out selection — no oracle, no target labels, no in-sample leakage) beats ERM by **+3.0
balanced accuracy on SCZ and PD** cross-site. The accuracy lever is the **transductive covariate alignment**
(CORAL/matched-CORAL); the CMI/LPC component is the leakage controller + the applicability screen.

**What each review round changed:**
- **R1 deciding experiments (§6c–6e in RESULTS):** LPC×alignment interaction ≈0 (LPC not necessary for the gain);
  fixed source-only selector; covariate-repair not mass-flip; gain is genuinely transductive (`ea_strict`).
- **R2 implementation audit (§6f):** de-confounded PMCT vs **matched-CORAL** ⟹ **PMCT ≈ matched-CORAL on real EEG**
  (the prior-matching wasn't the lever; the earlier edge was shrink/gate machinery) ⟹ **PMCT DEMOTED to a
  prior-robustness ablation**. Plus exact null-safety, uncertainty (not entropy) gate, same-classifier diagnostic,
  nested selector. The in-sample selector's +5.5 SCZ deflated to a rigorous +3.0.
- **R3 "why not ERM+CORAL?" (§6g):** (a) selector ablation — CMI-guided selection ≈ accuracy-only on accuracy but
  **2–3× lower leakage** (Pareto-safe, not an accuracy booster); (b) abstention — the residual-decoder-CMI screen
  **detects** concept shift (disease ENABLE vs med-state ABSTAIN) and the **harm mechanism is real** (synthetic:
  unconditional alignment costs −14 bAcc under a rotated boundary), but real med-state's shift is too mild for
  abstention to be accuracy-protective yet.
- **Gaussian-OT map (§5 of R2 review):** ≈ whiten-color (displacement is mean-dominated) ⟹ keep WC.

**Honest framing:** a **CMI-screened transductive covariate-alignment framework** — modest accuracy gain over
ERM+CORAL (+0.2…+1.1) but **2–3× lower deployment leakage** and a **validated concept-shift screen** (detector +
mechanism). PMCT, prior-correction, single-class feature transport, and the OT map are documented negative/ablation
controls. **Terminology: CMI-screened / CMI-guided, NOT "CMI-certified".** Remaining must-do: source-state
serialization (source-free claim), 5-seed CIs (running), SPDIM/T3A baselines, batch-size curve, TUAB lockbox, and a
real large-concept-shift task to make the abstention protective value load-bearing.

*(The sections below are the original R0–R7 record, kept for history; where they conflict with the above, the
above wins.)*

---

## 0. Why this work happened (the review verdict)
The review of the existing line (`lpc_prior`/`dual`/`dualc`/`dualpc`) concluded: **the methods remove domain
leakage 10–100× but are accuracy-PARITY with ERM** — because pure invariance is accuracy-neutral when the
leakage is "harmless", and on real cross-site EEG concept shift `I(Y;D|Z)≈0` (so the decoder term has nothing
to fix). Also clarified vs Zhao-2020 entropy-DG: same `P(z|y)` object, but different mechanisms; dualpc is a
GLS-reweighted conditional-MI reformulation, not term-for-term. **Goal set:** a new algorithm that jointly
controls `P(z)`/`P(y|z)` AND beats ERM on accuracy.

## 1. The method (final form): CITA / PMCT — *diagnose → preserve → align*
Not "jointly optimize `P(z)` and `P(y|z)`" (Stage 2 never updates the predictor). It is:
**diagnose** (CMI shows concept shift ≈0 on effective domains → the source predictor is domain-stable);
**preserve** it (do NOT re-fit / re-prior the classifier — re-prioring provably hurts balanced acc);
**align** only the unlabeled target *covariates* to the source's prior-matched class-conditional geometry.
- **Stage 1 (inductive, source-only):** `lpc_prior` — `L = CE + λ·I(Z;D|Y)` makes `P(z|y)` domain-invariant
  and supplies the source class-conditional moments `{μ_y^S, Σ_y^S}`.
- **Stage 2 (transductive, test-time, closed-form, no retraining):** **PMCT (Prior-Matched Conditional
  Transport)** — estimate target support `π̂_T`, build the prior-matched source reference `(μ_R, Σ_R)`,
  transport `T(z)=μ_R+Σ_R^{½}(Σ̂_T+εI)^{−½}(z−μ̂_T)` with identity interpolation. Reduces to mixture-CORAL when
  `π̂_T` is mixed; to a single-class reference when one-hot. The prior is used for the geometric reference
  ONLY, never to re-weight logits.
- **Setting:** **transductive / test-time-adaptive (TTA, source-free)** — uses the unlabeled target marginal,
  NOT strict DG. (Stage 1 alone is inductive DG.)
**Why it beats ERM:** the target-risk bound `ε_T ≤ ε_S + d(P_S(z),P_T(z)) + λ*` has a divergence term ERM never
sees; the transductive alignment lowers the achieved target divergence (moves the boundary onto the target's
class-conditionals). PMCT > global-CORAL because CORAL conflates label-prior shift with covariate shift.

## 2. Code changes (all new/modified files)
**New — `cmi/eval/label_shift.py` (219 lines):** the transductive correction library.
- `feature_coral_recenter` — CORAL/EA whitening in feature space (the balanced-acc lever).
- `pmct_transport` / `transduct_predict(mode=...)` — PMCT + the mode dispatch (`probe/coral/prior/coral_prior/pmct`).
- `bbse_prior` / `_mlls_em` / `apply_correction` — BBSE/MLLS label-shift prior estimate + re-prior (kept as
  the *negative* plain-acc ablation).
- `transduct_all` — compute every mode in one pass (the ablation ladder).
**New — `cmi/eval/test_label_shift.py` (162 lines):** CPU validation. Prior-estimation recovers `π_T`;
feature-CORAL gives +17.6 bAcc on 3-cls covariate shift; null-safe; **`run_stress`** = class-support sweep
proving PMCT ≥ CORAL under prior shift.
**Modified — `cmi/run_loso.py`:** `--transduct {off,probe,coral,prior,coral_prior,all}` + per-fold
native-vs-corrected metrics + summary aggregation (+ earlier per-fold checkpointing, no-walltime convention).
**Modified — `cmi/run_scps_crossdataset.py`:** cohort-level `--transduct` (target = held-out cohort) +
`--dec_domain` (hierarchical D) + per-cohort checkpointing.
**Docs:** `DUALPC_SUMMARY.md` (old dualpc), `DUALPC2_DESIGN.md` (round log), `DUALPC2_RESULTS.md` (clean
results), this file.

## 3. Experiments (Round-by-round) and results
**R0 — CPU design + validation.** Judge-panel → CIPC (prior-correction). **CPU caught the flaw**: re-prioring
to `π_T` HURTS balanced acc (BA is maximized by the uniform prior) → **pivoted to feature-CORAL**
(covariate alignment). Validated: +17.6 bAcc 3-cls, null-safe. (Later: PMCT added + `run_stress` validated.)

**R1 — MI (feature-CORAL), leave-one-subject-out (multi-class targets).** FIRST accuracy win:
| dataset | ERM → +CORAL | gain |
|---|---|---|
| BNCI2014_001 (4-cls) | 42.4 → **45.6** | **+3.2** (all seeds +) |
| Cho2017 | 66.0 → **68.1** | +2.1 |
| Lee2019_MI | 68.1 → **69.8** | +1.7 |
| BNCI2014_004 | 65.3 → 65.2 | −0.2 (flat) |

**R2 — disease cross-site (cohort-level CORAL), 3 seeds.** Win holds on the paper's main regime:
| cross-site | ERM | erm+CORAL | **lpc+CORAL** | best vs ERM |
|---|---|---|---|---|
| SCZ | 51.1 | 53.4 | **54.8** | **+3.7** |
| PD | 58.9 | 61.5 | 61.3 | **+2.6** |
`lpc+CORAL` positive **every seed** on BNCI/SCZ/PD.

**R3 — ablation ladder (`--transduct all`).** Mechanism confirmed: `coral`≫`native` (+1.9…+4.3); `probe`≈`native`
(gain is CORAL, not the probe); **`prior`≪`native`** (BNCI 29.9 vs 41.1 — BBSE re-prior HURTS BA, the predicted
negative ablation); `coral_prior`≈`coral`.

**R4 — within-dataset SCPS (raw EA), leave-one-subject-out (single-class targets).** EA works, feature-CORAL
distorts:
| dataset | ERM | erm+EA | lpc+EA | feature-CORAL |
|---|---|---|---|---|
| ADFTD | 45.4 | **48.0** | 43.5 | −8.1 (distorts) |
| MUMTAZ | 80.1 | 80.0 | **83.7** | −27.6/−31.9 (catastrophic) |

**R6 — PMCT on real SCPS (`--transduct all` incl pmct), MUMTAZ.** KEY NEGATIVE FINDING: feature-space transport
is ill-posed for single-class LOSO targets. `probe`=native 81.7 but **`coral`/`pmct` collapse to ≈50–53** —
PMCT does NOT rescue the single-class case (crashes like CORAL); the `prior` "gain" (83.3) is a degenerate
single-class-metric artifact. ⟹ feature transport requires multi-class targets; single-class SCPS uses raw EA
or pooled/cohort eval.

## 4. Key findings / decisions
1. **Transductive covariate alignment beats ERM by +2…+4 bAcc** across MI (R1) and disease cross-site (R2) —
   the project's first real accuracy win, leakage stays at the `lpc_prior` floor.
2. **Prior re-correction (BBSE) is a documented NEGATIVE ablation for balanced accuracy** (R0 predicted, R3
   confirmed). It helps *plain* accuracy only.
3. **PMCT > global-CORAL under prior shift** (CPU `run_stress`: 0.922 vs 0.865 at (0.8,0.2)), reduces to CORAL
   when balanced, graceful single-class limit — but on real single-class LOSO it still crashes (R6): feature
   transport needs multi-class targets.
4. **The method is transductive/TTA, not strict DG** — must be labeled honestly; the `ea_strict` ablation
   (pending) quantifies the transductive gain.
5. **Regime map:** multi-class targets (MI LOSO, cross-site cohorts) → feature-CORAL/PMCT; single-class targets
   (within-dataset SCPS LOSO) → raw EA or pooled/cohort-level evaluation.

## 5. Current state + next (Round-7)
- **Done:** R1, R2, R3, R4 (8 each), R6 MUMTAZ; CPU validation + PMCT stress test.
- **Running:** R6-ADFTD (single-class, expect same crash), R1-Cho/Lee (large MI, finishing), R4-ADFTD seeds.
- **Round-7 (planned, see `DUALPC2_RESULTS.md` §9):** (1) **PMCT in the correct protocol** — SCZ/PD cohort-level
  `--transduct all` (the PMCT-wins regime); (2) **pooled-target SCPS evaluation** (`--transduct_pool`) to remove
  the single-class degeneracy; (3) **`ea_strict` strict-DG ablation**; (4) hardening (shrinkage/`α` sweep,
  `π̂_T` confidence gate); (5) more seeds/CIs + TUAB.

## 6. Bottom line
A principled, honest **transductive EEG-DG** method (conditional-invariant representation + closed-form
test-time covariate alignment) that **beats vanilla ERM by +2…+4 balanced accuracy** on MI and clinical
cross-site, with a clean ablation story (alignment is the lever; prior-correction and single-class feature
transport are documented negative controls). The PMCT upgrade makes the covariate alignment prior-aware
(beats CORAL under prior shift); its boundary (multi-class targets only) is now understood and will be
evaluated in the correct protocol in Round-7.
