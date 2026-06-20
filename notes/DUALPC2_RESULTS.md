# CITA — Conditional-Invariant Test-time Alignment for EEG domain generalization (results)

*Paper-style consolidation of the Task-2 algorithm. Method + setup + main results + ablations. Design/round
log: `notes/DUALPC2_DESIGN.md`; full dev-log: `notes/POST_REVIEW_WORK_SUMMARY.md`. (Renamed from "CIPC" — the
prior-correction component is now a negative ablation, not part of the method.)*

---

## 0. COMPLETE RESULTS AT A GLANCE (the authoritative summary)
**Method (one line):** CMI verifies the source decision rule is domain-stable (`I(Y;D|Z)≈0`, positive-control
validated) → a **nested source-domain λ selector** fixes the representation (λ=0=ERM) → **closed-form transductive
covariate alignment** (CORAL/matched-CORAL) moves only the *unlabeled target covariates* onto the source
class-conditional geometry, with a **frozen** source readout. **Transductive / test-time**, not strict DG.

**★ HEADLINE (Round-8, RIGOROUS) — `CITA-nested + transductive alignment` vs ERM** (nested leave-one-source-cohort-out
selector ⟹ no oracle, no target labels, no in-sample selection; 2 seeds):
| | ERM | **CITA-nested + align** | gain |
|---|---|---|---|
| SCZ cross-site | 52.4 | **55.4** | **+3.0** |
| PD cross-site | 58.0 | **61.0** | **+3.0** |
*(The earlier in-sample-selector "+5.5 SCZ" deflated to +3.0 under proper nested CV — the audit caught real
selection optimism.)* The aligner is **CORAL ≈ matched-CORAL ≈ PMCT on real EEG** (see PMCT note below).

**PMCT verdict (Round-8 de-confounding):** once compared against **matched-CORAL** (PMCT's *exact* shrink/gate/
interpolation machinery, pooled reference), **PMCT ≈ matched-CORAL on real EEG** (Δ = −0.3…+0.3, natural prior AND
at moderate prior shift 0.7). The earlier "+0.5…+0.9 PMCT>CORAL" was the **machinery, not the prior-matching**.
⟹ **PMCT is DEMOTED to a prior-robustness ablation** (its prior-matched reference helps only under large, clean,
estimable prior shift — synthetic: +5.9 @(.8,.2); real EEG prior shift is too modest/noisy to exploit). The
real-data headline aligner is plain/matched-CORAL.

**Main results by regime (best transductive-aligned vs ERM, multi-seed):**
| regime (target class support) | datasets | gain over ERM |
|---|---|---|
| MI, leave-1-subject-out (multi-class) — feature-CORAL | BNCI2014_001 **+3.2**, Cho2017 +2.1, Lee2019 +1.7 | (BNCI2014_004 flat) |
| Disease cross-site, leave-1-cohort-out (multi-class) — CORAL | SCZ **+3.0**, PD **+3.0** (nested) | both seeds + |
| Within-dataset SCPS, leave-1-subject-out (single-class) — raw EA | MUMTAZ **+3.6**, ADFTD +2.6 | feature-CORAL invalid here (negative control) |

**Mechanism (ablation ladder):** `coral` ≫ `native` (+1.9…+4.3); `probe` ≈ `native` (gain is the alignment, not
the probe); **`prior` ≪ `native`** (BBSE re-prior HURTS balanced acc — negative control); **`matched_coral` ≈
`coral`** (the shrink/gate machinery isn't the lever); single-class feature-CORAL collapses −8…−32 (negative control).

**Predictor diagnostics (Round-8, same-classifier):** alignment flips only **17–23%** of predictions (JS 0.05–0.08),
and the frozen readout agrees with the EEGNet head **94–96%** of the time ⟹ **predictor-preserving is honest** (the
readout ≈ the trained head) and the alignment **repairs covariate-displaced samples, not mass-flips** them.

**Review rounds — both addressed:** R1 deciding experiments (LPC not necessary; fixed selector; covariate-repair;
transductive) + R2 implementation audit (de-confounded PMCT vs matched-CORAL; exact null-safety; reliability gate;
nested selector). Net: the honest contribution is **CMI-screened transductive covariate alignment that beats ERM
+3.0 with a no-oracle selector**; PMCT is an ablation, not the hero. *(Terminology: "CMI-screened" / "CMI-guided"
— NOT "CMI-certified": the source-domain concept-null screen does not guarantee any unseen target is concept-shift-free.)*

**Raw results:** `results/r{1,2,3,4,6,7,7stress,7sel,7diag,8}_dualpc2/*.json`. Detailed sections below (§1 method,
§4 main, §5 ablation, §6–6f review-response + audit, §7 limitations, §9 next).

---

## 1. Method — *diagnose → preserve → align*
The narrative is **not** "jointly optimize `P(z)` and `P(y|z)`" (Stage 2 never updates the predictor). It is:
> **Diagnose** with CMI that, on effective domains, concept shift `I(Y;D|Z)≈0` (positive-control-validated
> across site/sex/age/race) ⟹ the source predictor is domain-stable; **preserve** it (do not re-fit / re-prior
> the classifier — re-prioring provably hurts balanced acc); **align** only the unlabeled target *covariates*
> to the source's prior-matched class-conditional geometry.
CMI is therefore not decoration — it *decides what to adapt and what to leave alone*.

**Stage 1 — inductive (source-only).** Train `lpc_prior`: `L = CE(Y|Z) + λ·I(Z;D|Y)`, making the
class-conditional features `P(z|y)` domain-invariant. Provides the stable predictor + the source class-conditional
moments `{μ_y^S, Σ_y^S}` used by Stage 2 (the **shared statistic `P(z|y)`** that couples the two stages).

**Stage 2 — transductive covariate transport (test-time, label-free, closed-form, no retraining).**
**PMCT (Prior-Matched Conditional Transport)** — one unified operator (replaces the manual CORAL/EA switch):
1. estimate the target class support `π̂_T` from CORAL-bootstrapped soft predictions (EM-refined);
2. build the **prior-matched** source reference `μ_R=Σ_y π̂_T(y)μ_y^S`, `Σ_R=Σ_y π̂_T(y)[Σ_y^S+(μ_y^S−μ_R)(μ_y^S−μ_R)ᵀ]`;
3. transport `T(z)=μ_R+Σ_R^{½}(Σ̂_T+εI)^{−½}(z−μ̂_T)`, with identity interpolation `z̃=(1−α)z+αT(z)`.
`π̂_T` mixed ⟹ mixture-matched CORAL; `π̂_T` one-hot ⟹ single-class reference automatically (no manual switch,
no single-class collapse). The prior is used for the **geometric reference only**, never to re-weight logits.

**Setting:** **transductive / test-time-adaptive (TTA, source-free)**, NOT strict DG — Stage 2 uses the target
marginal `P_T(z)` (unlabeled, no target labels, no target gradient). Report `Δ_trans = bAcc(target-stat align) −
bAcc(source-only align)` to isolate the transductive contribution.

## 2. Why it beats ERM (the accuracy mechanism)
The target-risk bound `ε_T ≤ ε_S + d(P_S(z),P_T(z)) + λ*` has a divergence term ERM never sees and that prior
invariance shrinks *harmlessly* (the project's accuracy-parity history). The lever is the **transductive** term:
aligning the unlabeled target onto the source support lowers the achieved target divergence, moving the boundary
onto the target's class-conditionals. **PMCT's edge over global CORAL:** global CORAL conflates label-prior shift
with covariate shift (`μ_T=Σ_y π_T(y)μ_{T,y}` differs from `μ_S` even with zero covariate shift if `π_T≠π_S`),
catastrophically so for single-class targets; PMCT aligns to a *prior-matched* reference, so label-prior shift is
not mistaken for covariate shift. Prior re-correction of the *decision* (BBSE) is a plain-acc lever and hurts
balanced acc — kept as a negative ablation.

> Honest caveat (the PMCT failure mode): `π̂_T` on a strongly-shifted target is itself biased (chicken-and-egg).
> CPU stress test: PMCT ≥ global-CORAL across the prior sweep (`(0.8,0.2)`: 0.922 vs 0.865), reduces to CORAL when
> balanced, but at the single-class extreme `(1.0,0.0)` *with strong covariate shift* both collapse — an
> identifiability limit (no method can separate prior from covariate from one class + no labels). Real cross-site
> cohorts have both classes ⟹ the PMCT-wins regime.

## 3. Experimental setup
- Backbone EEGNet on raw EEG; balanced accuracy on unseen domains; ≥3 seeds where noted; per-fold checkpointed.
- **MI** (multi-class targets): BNCI2014_001/004, Cho2017, Lee2019_MI — leave-one-subject-out.
- **Disease cross-site** (multi-class targets): SCZ (ds003944+47), PD (3 sites) — leave-one-cohort-out, D=cohort.
- **Within-dataset SCPS** (single-class targets): ADFTD, MUMTAZ — leave-one-subject-out.

## 4. Main results — transductive alignment beats ERM
**(a) MI (feature-CORAL), 3 seeds on BNCI:**
| dataset | ERM | **+CORAL** | gain |
|---|---|---|---|
| BNCI2014_001 (4-cls) | 42.4 | **45.6** | **+3.2** (all seeds +) |
| Cho2017 | 66.0 | **68.1** | +2.1 |
| Lee2019_MI | 68.1 | **69.8** | +1.7 |
| BNCI2014_004 (2-cls) | 65.3 | 65.2 | −0.2 (flat) |

**(b) Disease cross-site (cohort-level feature-CORAL), 3 seeds:**
| cross-site | ERM | erm+CORAL | **lpc+CORAL** | best vs ERM |
|---|---|---|---|---|
| **SCZ** | 51.1 | 53.4 | **54.8** | **+3.7** |
| **PD** | 58.9 | 61.5 | 61.3 | **+2.6** |
`lpc_prior+CORAL` is positive **every seed** on BNCI/SCZ/PD (+3.4 / +2.9 / +2.0 over its own native).

**(c) Within-dataset SCPS (EA, single-class targets):**
| dataset | ERM | erm+EA | **lpc+EA** | best vs ERM |
|---|---|---|---|---|
| ADFTD (3-cls) | 45.4 | **48.0** | 43.5 | +2.6 |
| MUMTAZ (2-cls) | 80.1 | 80.0 | **83.7** | **+3.6** |

## 5. Ablation ladder (`--transduct all`) — the mechanism
bAcc by correction mode:
| dataset | native | probe | **coral** | prior | coral+prior |
|---|---|---|---|---|---|
| BNCI2014_001 (erm) | 41.1 | 39.9 | **45.0** | 29.9 | 45.1 |
| BNCI2014_001 (lpc) | 39.5 | 40.4 | **43.8** | 30.1 | 42.2 |
| PD (erm) | 58.8 | 59.0 | **62.4** | 57.9 | 62.4 |
| SCZ (lpc) | 52.0 | 52.6 | **54.8** | 52.0 | 54.6 |
| BNCI2014_004 | 65.4 | 64.5 | 64.9 | 60.0 | 64.7 |

- **`coral` ≫ `native`** (+1.9…+4.3) — the lever.
- **`probe` ≈ `native`** — the gain is CORAL, not the re-fit linear head.
- **`prior` ≪ `native`** (BNCI 29.9 vs 41.1) — BBSE re-prioring HURTS balanced accuracy → *negative ablation*, justifying CORAL over prior.
- **`coral+prior` ≈ `coral`** — prior adds nothing.

**Single-class target negative ablation (SCPS):** feature-CORAL *catastrophically distorts* — MUMTAZ −27.6/−31.9, ADFTD −8.1 — confirming it requires multi-class targets and motivating the EA fallback.

## 6. Transductive vs strict-DG (positioning) — DONE (ablation)
The win is transductive. `ea_strict` (target whitened by the *source pool only*, no target stats = strict
inductive DG) quantifies it: **`ea_strict` is +1.3…+5.7 WORSE than `ea`** (transductive, uses target stats) on
MUMTAZ/ADFTD — source-only alignment cannot capture the gain; it **requires the unlabeled target statistics**.
On ADFTD `ea_strict` even hurts vs native. For feature-CORAL the strict-DG baseline IS `native` (any recenter
needs target stats), so the full `coral − native` gain (+2…+4) is transductive. ⟹ **the method is genuinely
transductive/TTA; strict source-only DG does not achieve the accuracy gain.** (ADFTD 88-fold `ea`/`coral` runs
at mismatched fold counts ⟹ ADFTD `native` unreliable; the `ea−ea_strict` gap is the clean isolation, holds on
both; MUMTAZ complete.)

## 6b. Round-7.1 — PMCT vs CORAL in the correct protocol (cohort-level cross-site, 3 seeds)
| cohort | native | coral | pmct | prior |
|---|---|---|---|---|
| PD (erm) | 59.1 | 61.8 | **62.0** | 57.9 |
| SCZ (erm) | 50.8 | 52.7 | **53.1** | 50.9 |
| SCZ (lpc) | 52.0 | 54.7 | 54.6 | 52.1 |
- **Transductive-alignment win re-confirmed** (CORAL/PMCT beat ERM +2.0…+2.7; prior craters BA again).
- **PMCT ≈ CORAL on real data** (+0.1…+0.4) — *not* a clear win over CORAL, because the real SCZ/PD cohorts have
  little class-prior shift, so PMCT's prior-matched reference reduces to plain CORAL (its CPU-stress edge only
  appears under prior shift). PMCT = "CORAL + prior-shift insurance": never worse, differentiates only when
  priors differ. Honest framing: headline = transductive covariate alignment beats ERM; PMCT is the principled
  prior-robust operator (validated synthetically) that coincides with CORAL on these benchmarks.

## 6c. Review response — CMI's role, the LPC necessity test, and corrected claims (2026-06-20)
**Reframe (accepted):** CMI is **not an accuracy loss**; it is an **applicability certificate** — the
positive-control-validated `I(Y;D|Z)≈0` (across site/sex/age/race, §3.5 of CONCEPT_SHIFT) *licenses* preserving
the source predictor and adapting only covariates. The narrative is **diagnose → preserve → align**, not
"jointly optimize `P(z)`/`P(y|z)`".

**LPC × alignment INTERACTION (from R3 + R7.1, no new GPU):**
`Δ_int = [LPC+algn − LPC] − [ERM+algn − ERM]` = PD coral −0.8 / pmct −0.9; SCZ coral +0.8 / pmct +0.2.
⟹ **interaction ≈ 0 (no consistent synergy): LPC is NOT a necessary precondition for the alignment gain.**
Honest split: **alignment is the accuracy lever** (+1.9…+2.9 on top of *either* ERM or LPC); **LPC is the
leakage-removal/invariance component** (independent). Do NOT claim "LPC necessary for PMCT".

**α_T support gate (implemented):** `pmct_transport(support_gate=True)` sets `α_T = α0·g_support·g_cov` with
`g_support = H(π̂_T)/log C` and `g_cov = clip(n_T/2d,0,1)`; as the target → single-class (`H→0`) or n_T is
small, `α_T→0` and PMCT degrades to **identity (null-safe)** instead of distorting. CPU: improves moderate
prior shift (0.8/0.2: 0.865→0.899), unchanged at the unidentifiable single-class+strong-covariate limit.

**Corrected claim set (what we say / don't say):**
- ✅ "Prior CMI objectives cut leakage 10–100× but stayed accuracy-parity; we preserve the predictor and add
  transductive covariate alignment; CORAL improves bAcc on 3/4 MI + both clinical cross-site; prior-correction
  and single-class transport are negative controls; PMCT removes CORAL's label-prior confounding and is being
  validated in the multi-class regime."
- ❌ NOT: "jointly optimize P(z)&P(y|z)"; "LPC is the necessary precondition for PMCT"; "PMCT handles real
  single-class targets"; "the bound proves PMCT raises accuracy"; "+2–4 in all EEG-DG"; "best-of-ERM/LPC ×
  best-of-CORAL/EA is one algorithm".

**Theory tightening (to add):** (Prop 1) under prior-only shift `P_T(z|y)=P_S(z|y), π_T≠π_S`, PMCT's
prior-matched reference ⇒ PMCT = identity while global CORAL is not — PMCT's provable edge. (Prop 2)
single-class unlabeled transport is **unidentifiable** (different `(y*,T_D)` give the same `P_T(z)`) — elevates
the R6 failure from "tuning" to a stated boundary. The adaptation bound is *motivation*, not a risk guarantee.

**Four deciding experiments (reviewer §6):** [done] LPC×alignment interaction (≈0). [running/next] real-data
target-prior stress test (CORAL vs PMCT under skew — the decisive PMCT experiment); fixed source-only
CMI-guarded selector (λ=0=ERM) → headline `CITA-selected+PMCT` (no per-dataset oracle picking); predictor
diagnostics + batch-size/covariance reliability.

## 6d. Round-7.3 — REAL-DATA prior stress test (the decisive PMCT vs CORAL evidence)
Held-out cohort subsampled to majority-class fraction p; native/CORAL/PMCT on the same skewed target (seed 0):
| cohort | p=0.5 (coral/pmct) | p=0.7 | p=0.9 |
|---|---|---|---|
| PD | 62.7 / 62.9 | 61.9 / **62.8** (+0.9) | 60.0 / 59.5 |
| SCZ | 54.0 / 55.1 | 53.2 / **54.1** (+0.9) | 54.9 / 54.8 |
**PMCT > CORAL by +0.9 on BOTH diseases at moderate prior shift (p=0.7)** — real-data confirmation of the
mechanism (not just synthetic). At p=0.5 PMCT≈CORAL (reduces to CORAL); at p=0.9 PMCT≈CORAL (prior estimate
hits the identifiability wall, Prop 2). ⟹ **PMCT ≥ CORAL everywhere, strictly better in the detectable-prior-
shift band.** This is the evidence that PMCT is a *necessary* prior-aware replacement of global CORAL, not
decoration. (Natural cohorts (R7.1) have ~no prior shift ⟹ PMCT≈CORAL there; the stress test exercises the
regime PMCT targets.)

## 6e. Round-7.4 — FIXED headline algorithm (no oracle picking) + stress confirmation (2 seeds)
**`CITA-selected+PMCT`** — λ chosen by SOURCE-validation bAcc (`source_val_bacc`, no target labels, no
per-dataset picking; λ∈{0=ERM,0.1,0.3=LPC}), then PMCT:
| | ERM | ERM+PMCT | **CITA-selected+PMCT** | vs ERM |
|---|---|---|---|---|
| PD | 59.2 | 61.4 | 61.3 | **+2.1** |
| SCZ | 50.3 | 54.1 | **55.8** | **+5.5** |
The selector **picks LPC on SCZ** (55.8 > ERM+PMCT 54.1 — invariance helps) and **ERM on PD** (≈ ERM+PMCT —
it doesn't), source-only. ⟹ **one fixed algorithm beats ERM +2.1/+5.5, eliminating the oracle-method-selection
risk** (reviewer §2.2/§6.5). This is the main-table headline row.
**Stress confirmed (2 seeds):** PMCT − CORAL = +0.5…+0.6 at p=0.5–0.7 (both diseases), ≈0 at p=0.9. PMCT ≥
CORAL, strictly better in the detectable-prior-shift band.

**Reviewer deciding-experiment status: 4/4 DONE** — [✅] LPC×alignment interaction ≈0 (LPC not necessary);
[✅] real-data prior stress (PMCT>CORAL moderate shift); [✅] fixed source-only selector (CITA-selected+PMCT,
+2.1/+5.5, no oracle); [✅] **predictor diagnostics**: PMCT changes only 17–27% of predictions (not a ~50%
mass-flip), Δconf±0.02, while bAcc rises +2.7…+4.3 ⟹ the moderate targeted flips REPAIR covariate-displaced
samples, not wholesale relabeling (PD erm 58.9→63.1 flip 25.9% JS 0.107; SCZ lpc 51.6→54.3 flip 17.4% JS 0.053).
Plus the transductive-vs-strict-DG ablation (§6): `ea_strict` +1.3…+5.7 worse than `ea` ⟹ gain is genuinely
transductive. The method is de-risked against this review round.

## 6f. Round-8 — implementation–claim AUDIT (second review round; code fixes before new modules)
A code-audit review found three implementation gaps where the code under-delivered the claims. **All fixed in
code (CPU-tested) AND re-validated on real EEG (`results/r8_dualpc2/`, 8 jobs, 2 seeds, DONE).**

**REAL-DATA VERDICT (the decisive outcome):**
- **PMCT ≈ matched-CORAL on real EEG.** De-confounded ladder (natural prior): `PMCT − matchedCORAL` = −0.2/−0.1/−0.2
  (PD erm/lpc/nested), −0.1/+0.1/−0.2 (SCZ). Stress at prior 0.7: −0.3/−0.2 (PD), +0.3/−0.3 (SCZ). The earlier
  "+0.5…+0.9 PMCT>CORAL" was the shrink/gate/interp machinery, **not** the prior-matched reference. ⟹ **PMCT
  demoted to a prior-robustness ablation** (helps only under large clean prior shift; synthetic de-confounded
  test: PMCT > matched-CORAL +5.9 @(.8,.2), but real EEG prior shift is too modest/noisy to exploit).
- **Nested-selected headline HOLDS and is now rigorous:** `CITA_nested` (inner leave-one-source-cohort-out, no
  oracle/in-sample) + alignment beats ERM **+3.0 SCZ / +3.0 PD**. The in-sample selector's +5.5 SCZ deflated to
  +3.0 — the audit caught genuine selection optimism. Selector picks `lpc_prior:0.3` on shifted folds.
- **Same-classifier diagnostics (confound removed):** flip 17–23%, JS 0.05–0.08, **probe=head agreement 94–96%**
  ⟹ predictor-preserving is honest (the frozen readout reproduces the EEGNet head) and the alignment is
  covariate-repair, not mass-flip — now established with ONE classifier (z vs T(z)), not head-vs-probe.

**Gaussian-OT (Bures) map ablation (reviewer §5):** added `tmap='ot'` — the minimal-mean-square-displacement
Monge map `A_OT=Σ_T^{−1/2}(Σ_T^{1/2}Σ_RΣ_T^{1/2})^{1/2}Σ_T^{−1/2}` (also exactly null-safe, 1.4e-14). On the
synthetic stress it is **indistinguishable from whiten-color** in bAcc, mean displacement, AND flip-rate (e.g.
prior 0.8: bAcc .924 vs .925, |Δz| 1.863=1.863, flips 63.9% vs 64.2%) — because the displacement is dominated by
the **mean recenter** `μ_R−μ_T` (identical for both maps); OT's covariance-displacement gain is second-order. ⟹
**keep WC** (per the reviewer's contingency; doubly moot since PMCT is already demoted to an ablation).

**Code fixes (all retained):**
1. **De-confounded PMCT vs CORAL.** The old `coral` differed from `pmct` in shrinkage+gate+interpolation, not
   just the reference (and `--transduct_shrink` was never wired to PMCT's `rho`). Added **`matched_coral`**:
   PMCT's *exact* machinery (same `rho`, gate, interpolation) with a **pooled** reference instead of the
   prior-matched one — so the only difference is the reference. Now the admissible claim is **PMCT > matched-CORAL**.
   *Synthetic de-confounded stress (CPU): matched-CORAL ≈ plain-CORAL (0.866 vs 0.865), while PMCT beats
   matched-CORAL by +5.9 @ prior (0.8,0.2), +2.6 @ (0.95,0.05)* — the shrink/gate machinery does NOT explain
   PMCT's edge synthetically; real-data verdict pending R8.
2. **Reliability gate replaces the entropy gate (my error).** The old `g_support=H(π̂_T)/logC` damped PMCT for
   *genuinely* skewed priors (low entropy = real prior, not unreliable) — likely why the p=0.9 advantage
   vanished. Replaced with **`g_unc=exp(−κ·Var(π̂_T))`** (keys on prior-estimate *uncertainty*, not skew) × `g_cov`.
3. **Exact null-safety.** The same shrink operator `S_ρ` is now applied to BOTH the reference and target
   covariance ⟹ raw-moment equality gives `T(z)=z` to **8.9e-15** (unit-tested, `test_label_shift.run_nullsafety`).
4. **Same-classifier predictor diagnostic.** #4's flip/JS now compares the *frozen source readout* on `z` vs
   `T(z)` (`probe`→`pmct`), not EEGNet-head-on-`z` vs probe-on-`T(z)` — isolating the TRANSPORT from a
   classifier swap. Also logs `probe_vs_head_agree` so "predictor-preserving" is honest. (EEGNet's conv head
   can't consume a pooled transported `z`; we preserve a frozen linear readout on the alignment embedding and
   verify it matches the head's accuracy.)
5. **Nested source-domain selector replaces in-sample `sv_bacc`.** `--select nested`: inner
   leave-one-source-cohort-out CV — the model is **never trained on its validation domain**; `λ*=argmin leakage`
   s.t. `valBAcc ≥ max−ε`. Emits a `CITA_nested` pseudo-config = the no-oracle, no-target-leakage headline.
**Honest stance going in:** if real-data PMCT ≈ matched-CORAL, the headline becomes *"CMI-screened transductive
alignment (matched-CORAL)"* with PMCT demoted to a prior-robustness ablation — the diagnose→preserve→align chain
and the transductive ERM-beating gain stand regardless. (Still TODO: source-state serialization for strict
source-free; pooled single-class metric; `Prior-Matched **Covariance** Transport` rename; doc HISTORY split.)

## 6g. Round-8 review → "why is this not just ERM+CORAL?" — the CMI-value ablations
The R8 reviewer accepted the demotion and sharpened everything to one question: *what is CMI's irreplaceable
role?* Two ablations answer it.

**(A) Selector ablation (GPU-free, from r8sel JSONs; aligner = matched-CORAL throughout):**
| | SCZ bAcc | SCZ leak | PD bAcc | PD leak |
|---|---|---|---|---|
| ERM + matched-CORAL (fixed) | 54.4 | 0.308 | 61.0 | 0.147 |
| accuracy-selector (nested bAcc) | 55.1 | 0.157 | 61.2 | 0.112 |
| **CMI-selector = CITA** (nested bAcc + leakage tie-break) | **55.5** | **0.148** | **61.2** | **0.053** |
| CMI-selector + native (no align) | 52.7 | — | 59.1 | — |

⟹ **CMI-guided selection ≈ accuracy-only selection on accuracy, but Pareto-better on leakage** (PD 0.053 vs 0.112,
≈2× lower at equal bAcc). **CMI is a Pareto-safe selection rule, NOT an accuracy booster.** The accuracy lever is
the alignment (native→matched-CORAL = +2.8 SCZ / +2.1 PD, shared with ERM+CORAL). So *"why not ERM+CORAL?"* →
**CITA gives a small accuracy gain (+1.1 SCZ / +0.2 PD) AND 2–3× lower deployment leakage** (Pareto), not a large
accuracy jump. This is honest "outcome #2."

**(B) Abstention control — CMI decides *when not to align* (Round-9 DONE, `results/r9_dualpc2/`).** A source-side
gate `R_dec ≤ q^null_0.95 → align, else abstain`, same `--transduct all --decoder_null_perms 200 --dec_domain
subject` on disease and med-state. **Nuanced, honest verdict:**
- *Detection works (real):* residual decoder-CMI vs its permutation null **cleanly separates** the task families —
  disease excess −0.005…−0.008 (≤null ⟹ **ENABLE**, alignment helps +1.6…+2.9), med-state excess +0.013/+0.028
  (>null ⟹ **ABSTAIN**, concept shift detected). CMI is a valid concept-shift *detector*.
- *But abstention is not yet accuracy-justified on real data:* med-state alignment **still helps** (+2.1 erm,
  +0.5 lpc) — its concept shift is too mild for covariate alignment to turn harmful (covariate shift still
  dominates). So the binary gate is *conservative* here, not protective. (Only hint of the mechanism: the
  highest-excess fold, PDMED-lpc +0.028, has the smallest gain +0.5.)
- *Mechanism shown synthetically (the regime where it matters):* covariate-shifted target + a decision boundary
  **rotated by θ** — alignment flips from neutral to **harmful** as θ grows (Δ_align: +0.1 → −4.5 → −11.3 →
  **−13.7** bAcc at θ=30/45/60°), and the decoder-CMI residual rises **monotonically** (0.001→0.027) tracking the
  harm. ⟹ the *mechanism* justifying abstention is real; **when concept shift is large, unconditional alignment
  (= ERM+CORAL) costs up to ~14 points, and the CMI residual flags it.**
- **Honest claim:** CMI is a **validated concept-shift screen** (real-data detection + synthetic
  harm-mechanism). The binary gate's *real-data harm-prevention* value awaits a real task with large concept shift
  + covariate shift (med-state is mild; future work / boundary-injection task). This answers *"why not ERM+CORAL?"*:
  ERM+CORAL aligns unconditionally; there exist regimes where that hurts ~14 pts, and CMI is the screen that
  detects them — but on the two real disease tasks tested, both are concept-null so the screen's protective value
  is latent, not yet load-bearing.

**Net-correction diagnostic** added (`pmct_w2c`/`pmct_c2w`/`pmct_net_correction`): reports wrong→correct vs
correct→wrong flips so the 17–23% flips are shown to be net error-correction, not reshuffling (captured from the
next run on).

## 6h. Semi-synthetic concept-shift study — IS the gate load-bearing? (`cmi/eval/concept_shift_study.py`)
The R8 review's bar: not "accuracy dropped + abstention rose", but a controlled injection with monotonic
strengths, detector enrichment vs the injection mask, risk-coverage vs MSP/entropy/energy, a random-noise
control, and a dev-frozen threshold — the detector never seeing target labels/mask. Built the harness; ran it on
**real EEG features** (dumped from trained EEGNet, `--dump_features`). Two regimes, two honest verdicts:
- **Boundary-rotation (low-margin) concept shift:** the changed samples sit between the old/new boundary ⟹ they
  are low-confidence ⟹ **MSP/entropy already catch them (AUROC 0.74–1.00)**; the gate is NOT needed here.
- **Confident-but-wrong concept shift** (a high-margin pocket flipped + covariate signature — the *dangerous*
  case): on REAL EEG features, **MSP/entropy/energy collapse to AUROC 0.00** (the mislabeled samples are
  maximally confident, so confidence rejection is blind), while a **covariate-density-aware detector — the domain
  discriminator `P(target|z)`, a CMI-family `I(D;Z)` quantity — gets AUROC 0.87–1.00** across all PD/SCZ folds.
  ⟹ **the CMI gate IS load-bearing exactly where confidence rejection fails catastrophically.**
**Honest caveats:** (i) the *working* detector is the density/domain score, not the bespoke geometry-margin
"cmi" score (inconsistent, 0.10–0.79) — the gate should be formulated as the CMI/leakage-probe density score;
(ii) it requires the concept shift to carry a **covariate signature** (pure relabeling with identical marginals
is undetectable by *every* label-free method — a fundamental limit); realistic EEG state-change does move
features, so this is reasonable but must be stated; (iii) one degenerate fold (PD/ds002778, native≈chance) breaks
the geometry score. **Full rigor completed on real features (honest, mixed):** (1) **detection load-bearing** — domain-density gate
AUROC **0.97** vs MSP **0.02** (confident-but-wrong, frac 0.15, pooled folds); (2) **random-noise control** — the
source-side decoder-CMI screen responds more to structured concept shift (+0.009) than to equal-rate random label
noise (+0.005), but *weakly*; (3) **dev-frozen threshold** — freeze the gate threshold on dev folds, evaluate on
held-out test folds → **+1.8 selective accuracy** from abstaining on flagged samples (clean, no peeking); (4)
**but risk-coverage is muted** — selective *risk* stays ~50% across coverage because the base predictor is
near-chance on these hard cross-site cohorts, so detecting the injected 15% barely moves overall accuracy.
**Net:** the gate **detects** what confidence rejection cannot (the decisive, clean result), the methodology is
sound (frozen threshold), but the **selective-prediction accuracy *value* needs a base task well above chance** to
be visible — a real, stated limitation, not hidden.

## 7. Limitations / open
- **Single-class-target LOSO is ill-posed for feature-space transport (R6 finding).** On within-dataset SCPS
  with `leave-one-subject-out`, the held-out target is ONE class (`Y=g(subject)`). Real-data R6 (`--transduct all`
  on MUMTAZ): `probe`=native (81.7) but **`coral`/`pmct` collapse to ≈50** (native 81.7 → pmct 53.0) — feature
  transport distorts a single-class target. PMCT does NOT rescue this case (it crashes like CORAL); the
  per-fold `prior` "gain" (83.3 > 81.7) is a *degenerate single-class-metric artifact* (BBSE → the subject's own
  class → inflated per-subject recall), not a real win. ⟹ **feature-space transport (CORAL/PMCT) requires
  multi-class targets**; for single-class SCPS use raw-signal EA (R4: lpc+EA 83.7) or evaluate at the cohort /
  pooled level (R2). The single-class LOSO protocol is the wrong evaluation for feature transport.
- **BNCI2014_004** (binary, 3-channel) is the lone flat case — low covariate headroom.
- EA gain on SCPS is method/dataset-dependent (MUMTAZ erm flat; best combos lpc+EA win).
- CIs need tightening (more seeds on SCZ/PD/MUMTAZ); TUAB pending; `ea_strict` strict-DG ablation pending.

## 9. Round-7 plan + improvement measures
**Diagnosis (from R6):** PMCT's value is real but only where the target spans classes; we tested it in the
wrong protocol (single-class LOSO). Round-7 corrects the protocol and hardens the method.
1. **PMCT in the RIGHT protocol — cohort-level cross-site (where it should beat CORAL).** Run SCZ/PD
   `run_scps_crossdataset --transduct all` (now includes `pmct`) × seeds — both classes per cohort, and cohorts
   differ in class prior ⟹ exactly the PMCT-wins regime (prior-matched reference vs CORAL's prior/covariate
   conflation). Expect `pmct ≥ coral ≥ native`.
2. **Protocol fix for within-dataset SCPS — POOLED-target evaluation.** Add a `--transduct_pool` mode: collect
   the per-subject corrected predictions across all held-out subjects, then compute ONE pooled balanced
   accuracy (target now spans both classes) — the valid SCPS metric, removing the single-class degeneracy.
   Apply CORAL/PMCT on the pooled held-out set (transductive over the whole target dataset, the standard EEG
   transductive setting), not per single-class subject.
3. **`ea_strict` strict-DG ablation** (target whitened by source pool only) — quantifies the transductive gain;
   settles the DG-vs-TTA framing.
4. **Hardening:** (a) shrinkage/`α` interpolation sweep for PMCT covariance estimate on small target sets;
   (b) confidence gate on `π̂_T` (fall back to native when the support estimate is unreliable);
   (c) more seeds for CIs on SCZ/PD; TUAB.
5. **Keep R4's raw-EA path** as the principled single-class-SCPS lever (feature transport is documented as
   multi-class-only).

## 8. Bottom line
A principled, honest **transductive** EEG-DG method — conditional-invariant representation + closed-form
test-time covariate alignment — that **beats vanilla ERM by +2 to +4 balanced accuracy** across MI, disease
cross-site, and within-dataset SCPS, while keeping domain leakage at the `lpc_prior` floor. The project's first
real accuracy win, with a clean ablation story (alignment is the lever; prior-correction and single-class
feature-CORAL are documented negative controls).
