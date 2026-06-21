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
| 7 | **CITA accuracy** (+3.0 over ERM cross-site) | **RETRACTED** (magnitude) → **OPEN** (scoped) | deflated to ~+1.8/+2.3 (2-seed, deterministic); the lever is **matched-CORAL**, not CMI/LPC; CITA ties SPDIM. Needs the frozen survivor matrix |
| 8 | **PMCT** (prior-matched) beats matched-CORAL | **DROPPED** | PMCT ≈ matched-CORAL on real EEG → demoted to a prior-robustness ablation |
| 9 | **LPC calibration** (ECE/NLL improvement, "principled confidence regulariser") | **OPEN** | not yet deconfounded vs source-only temperature scaling / logit shrinkage / accuracy-matched controls; P1.5 shows LPC trades representation compression — calibration may be a compression side-effect |
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
target labels and no source examples. TUAB stays sealed pending the exposure audit's disposition. Open items to
resolve next: #9 (calibration deconfound) and #7 (survivor matrix).
