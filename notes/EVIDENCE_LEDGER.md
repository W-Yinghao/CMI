# EVIDENCE LEDGER — authoritative claim status (2026-06-21, supersedes all earlier summaries)

This is the single source of truth for what the LPC-CMI / CITA project has actually established. It supersedes the
"LATEST STATE" sections of `notes/POST_REVIEW_WORK_SUMMARY.md`, the safety-gate / protective-abstention language in
`README.md` / `notes/results_summary.md`, and the calibration claims in `notes/calibration.md` wherever they
conflict. Status labels: **SUPPORTED** · **DIAGNOSTIC_ONLY** · **DROPPED** · **RETRACTED** · **OPEN**.

Branch of record: `exp/lpc-cmi`. All audit artifacts committed under `results/` + `notes/A0*`, `notes/FREEZE*`.

## Claims

| # | Claim | Status | Basis |
|---|---|---|---|
| 1 | LPC reduces **extractable conditional domain information** I_w(Z;D\|Y) (multi-probe, perm-null ≈0, beats CDANN) | **SUPPORTED** *(as a measurement; rename — NOT "precise CMI")* | leakage audit; `cmi-empirical-findings` |
| 2 | LPC is a **deployment mechanism** (regulariser you ship) | **DROPPED** | P1.5: at every frozen λ the leakage reduction is entangled with representation collapse beyond the pre-registered utility/eff-rank gates (`results/p15_audit*`) |
| 3 | "**CMI-screened CITA**" — the CMI/residual-CMI gate screens deployment | **DROPPED** as a method component → **DIAGNOSTIC_ONLY (report-only)** | gate falsification; CMI/density are wrong-signed for adaptation harm |
| 4 | **CMI domain-density gate** as a deployable harm-controller | **DIAGNOSTIC_ONLY** (closed) | A0: density/CMI anti-aligned with adaptation harm (weakly aligned with base difficulty) |
| 5 | **Batch rollback eligibility** via uncertainty/separability | **RETRACTED** | A0′-R: the A0′ positive was a P0 target-label-leakage artifact; whole-batch label-blind score → signal collapses (g_unc ρ→−0.40) |
| 6 | **Sample abstention** via post-alignment `s_sep` reduces deployed loss | **DIAGNOSTIC_ONLY** (closed) | A0-PILOT closed-loop: retained NLL WORSE than random; net-protection ≈ random vs oracle +6; harm-flip AUROC 0.7 does not translate to loss reduction |
| 7 | **CITA as a distinct positive method** | **RESOLVED → no novel method; STOP positive-method line** | survivor matrix (`results/survivor_matrix/`): CITA-no-LPC **IS** plain matched-CORAL (serialized-equiv \|Δbacc\|=0); vs **SPDIM** same accuracy (58.4 vs 58.6, CI [−2.0,+1.2]); +2.6 over ERM is the generic transductive lever (CI touches 0). matched-CORAL is better-calibrated than SPDIM TTA (NLL 1.80 vs 2.93, CI [−2.69,0.00]) + better worst-cohort (52.0 vs 49.8) — a *supporting* note (matched-CORAL > overfit TTA on calibration), not a project contribution. → contribution = the measurement→control gap |
| 8 | **PMCT** (prior-matched) beats matched-CORAL | **DROPPED** | PMCT ≈ matched-CORAL on real EEG → demoted to a prior-robustness ablation |
| 9 | **LPC calibration** ("principled confidence regulariser") | **DROPPED** → temperature/compression side-effect | deconfound (`results/calibration_deconfound/`, 130 datasets, TUAB excluded): a single **oracle temperature** on ERM beats LPC NLL on **123/130** (LPC wins 7/130, median +0.254 NLL worse); LPC beats *raw* ERM (115/130) but trivial rescaling does MORE on 115/130, with acc ≈ ERM. So the effect is global confidence rescaling, not structured recalibration. Caveat: oracle-T is the temperature UPPER BOUND (target-fit; a deployable source-only T was untestable — no source-val in the saved preds), which makes the "not principled/structured" conclusion conclusive but leaves "is rescaling deployable under shift" unanswered |
| 10 | **TUAB** as a pre-registered disjoint lockbox holdout | **RETRACTED** | TUAB exposure audit (`notes/TUAB_EXPOSURE_AUDIT.md`): TUAB was run as a full LOSO method comparison in root commit `fb2a878` BEFORE `TUAB_LOCKBOX.md` (`1de7a12`); numbers fed calibration + the SCPS scorecard; the lockbox still freezes the now-dropped LPC selector + residual-CMI gate |

## Net scientific finding (most consistent statement)
Within the **frozen CITA-no-LPC representation, matched-CORAL adaptation, the two pre-registered controlled shift
families, and the pre-registered source-free scores (uncertainty, separability, support, CMI-proxy), NO source-free
controller stably improves deployed loss in closed loop.** The oracle's large advantage shows harm is in principle
selectively avoidable — what fails is the mapping from unlabeled observables to actual loss magnitude. The chain is
a clean **measurement→control gap**: detecting geometric change ≠ ranking harmful flips ≠ reducing closed-loop loss.
Specifically: LPC reduces extractable domain info but past the collapse/utility bound; density/CMI-proxy reflect
geometry/difficulty but invert vs adaptation harm; outcome-conditioned aggregation fabricated a batch-rollback
signal; a 0.7 harm-AUROC still worsened retained NLL by ignoring harm magnitude.

## Naming corrections (binding for any external write-up)
- The leakage measure: **"extractable conditional domain information"**, not "precise CMI / I(Z;D\|Y)".
- The deployed branch: internal lineage name `CITA-no-LPC` is fine, but the EXTERNAL algorithm description must be
  **"source-only-selected, source-free matched-CORAL + reliability-gate alignment"**; CMI/residual-CMI appears
  **only as a report-only diagnostic**, never as a screen/gate/safety component. Do NOT write "CMI-screened",
  "safety gate", or "protective abstention".

## Standing constraints
No new gate / LPC / coverage / cohort / score search (per the frozen A0-PILOT rule). Deployment control uses no
target labels and no source examples. TUAB stays sealed pending the exposure audit's disposition.

**SURVIVOR AUDIT COMPLETE (steps 1–4 done; NO open items).** LPC's three pillars all collapsed (leakage = via
representation collapse, calibration = a single-temperature side-effect, accuracy = the generic matched-CORAL
transductive lever, not CMI/LPC). The harm-gate/rollback/abstention direction is closed (DIAGNOSTIC_ONLY,
closed-loop). CITA-no-LPC is plain matched-CORAL with no distinct advantage over SPDIM. **There is no positive-method
contribution.** The defensible contribution is the **measurement→control gap**: *source-free adaptation diagnostics
are not deployment controllers* — leakage reduction can be collapse, shift/density detection inverts vs adaptation
harm, outcome-conditioned evaluation fabricates safety signals, and a 0.7 harm-ranking AUROC need not improve
closed-loop risk. Strengthening that = pre-registered cross-adapter / cross-task replication + identifiability /
counterexample theory (a NEW phase, not a seventh gate score). TUAB requires the exposure-audit disposition (demote
to external benchmark, or a hash-proven disjoint split on the CURRENT method) before any use.
