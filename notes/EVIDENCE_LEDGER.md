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

## ACAR (Direction 2) — authoritative claim status (added 2026-06-29; `acar` branch)

ACAR is the leak-proof successor to the closed A0 line. Estimand = action-conditional paired incremental risk
`ΔR_a(B)=R_B(f_a)−R_B(f_0)` (predict negative transfer, not shift). Status labels as above.

| # | Claim | Status | Basis |
|---|---|---|---|
| A1 | The ACAR **estimand** (action-specific incremental harm `ΔR_a`, paired, label-free at deployment) is well-posed and informative | **SUPPORTED** | v2 + v3; not shift magnitude / not absolute accuracy |
| A2 | **Label-free action-conditional paired observables predict negative transfer** out-of-fold on BOTH PD and SCZ | **SUPPORTED (measurement)** | ACAR v2 G1=True (`acar-v2-protocol @ 9b2f0c1`; result `1528a94`); v3 DEV reproduces SCZ harm-AUROC 0.68–0.74 |
| A3 | ACAR v2 is a **deployable safe router** | **DROPPED → MEASUREMENT_ONLY** | v2 G2=False (router NLL not below best-fixed/random at usable coverage); SCZ coverage diagnostic missed nominal (201/225=0.8933) |
| A4 | The v3 **HSCR redesign** (mean/scale/CQR DeepSets + subject-clustered joint conformal) **passes the pre-registered development gate** | **DROPPED → `DEV_STOP / NO_LOCKBOX_CONSUMED`** | v3 DEV run #002 (`acar-v3-dev-design-v1 @ 817b04f`; result `9f4e83f`, tag `acar-v3-dev-run002-dev-stop`): no C1/C2/C3 passes S2/S4 — adaptation coverage ~0.6–1.1 % (≪15 %; conformal `q`≫`|ΔR|`) and PD center-AUROC 0.525–0.570 (<0.60) |
| A5 | Any **external Arm-B / held-out lockbox** ACAR result (binding G2, site-local coverage, harmful-rate, two-site) | **OPEN — NOT RUN; NOT AUTHORIZED** | v3 stopped at the DEV gate; lockbox NOT consumed; external freeze never reached |
| A6 | The v4 **control-first redesign** (CURB: calibrate the EXECUTED policy's subject risk on a finite λ grid, not the all-action conformal upper bound) yields a usable DEV router | **EXPLORATORY_CANDIDATE — DEV-only; SUPERSEDED by A7 (did NOT transfer)** | v4 DEV exploration #001 (`acar` @ `e9760e6`; `results/acar_v4_dev_exploration_001/`): `V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE`. 14/90 both-disease configs pass pre-registered G0–G6 OOF: coverage 16–86 %, deployed NLL reduction beating the v2-replay comparator (macro 0.0985, == v3 C0) on BOTH diseases. **Caveat: model-selection over 90 configs (selection bias); DEV-only; harm rate 15–46 % of adapted batches; no lockbox/external. NON-BINDING — read now as a NEGATIVE PRIOR (see A7).** |
| A7 | The fixed v4 candidate **transfers to the regenerated all-DEV (external-compatible) substrate** (the prerequisite for external Arm B) | **REFUTED → `SUBSTRATE_COMPATIBILITY_FAIL / NO_EXTERNAL / NO_LOCKBOX_CONSUMED`** | v4 substrate-compatibility replay (substrate `b99fa4f`, compat `5237378`/C5b; result `c605e24`, `notes/ACAR_V4_C5B_COMPAT_REPLAY_RESULT.md`; SLURM A40 877665). The fixed candidate (`shift_margin+benefit_ranked+harm_indicator`) on the B1b all-DEV substrate: PD coverage 0.024 / red −0.0022 / harm-among-adapted 0.73; SCZ coverage 0.049 / red −0.019 / harm-among-adapted 1.00; fails coverage≥0.15, red>0, red>v2_replay (v2 evaluable both). It barely adapts and is net-harmful where it does → does NOT transfer. External Arm B foreclosed; **no post-replay tuning** (NEW dated protocol only). Closeout: `notes/ACAR_V4_CLOSEOUT.md` |
| A8 | **ACAR-V5 closes the label-free adaptation deployment gap on PD/SCZ DEV** (a safe+beneficial routing candidate exists in the frozen 22-policy universe under the pre-registered G1–G5 gates) | **REFUTED → `DEV_STOP / NO_CANDIDATE_SELECTED`** | v5 Stage-2B real DEV selection (`acar-v5-protocol @ 4278435`; substrate run `acar-v5-stage1b-c4412b4-r1`; selection run `acar-v5-stage2b-ba09777-r1` @ impl `ba09777`; result commit `d287635`; SLURM 885395, 8.18 h, rc=0). **0/22 candidates eligible; CAL cert 0/22 on BOTH diseases.** Failure mode = **harm control, not coverage** (unlike v3): G1 coverage passes PD 13/22 · SCZ 19/22, but **G4 harm_among_adapted UCB ∈ [0.61,0.87] ≫ 0.30 on all 42 evaluable cells** and G3 L_harm_all UCB > 0.10 almost everywhere (PD 1/22, SCZ 0/22); EVAL `red` = −12.12 .. +0.01 (adaptation increases NLL loss); v2_replay_red PD −0.079 / SCZ −0.049 (comparator also harmful). Identity/source-state LDA `f_0` dominates. 1 non-evaluable candidate (V5-P4-002 both diseases; Holm family fixed 132). Note independently re-verified (all 9 claims + full 22-row table CONFIRMED). Consequence: **no Stage-4, no external, lockbox sealed, no rerun/tuning/reselection** — NEW dated protocol only. Closeout: `notes/ACAR_V5_STAGE2B_CLOSEOUT.md`, `notes/ACAR_V5_CLOSEOUT.md` |
| A9 | **ACAR-V5 built and admitted an external-compatible, hash-bound Stage-1B DEV substrate + a guarded, label-firewalled Stage-2 selection pipeline that runs to a clean pre-registered verdict** | **SUPPORTED (engineering/protocol only — NOT efficacy)** | 30/30 substrates admitted; registry / FINALIZED package verified (registry_sha256 `2bbe55f4…bbcbb76d`); single external-compatible source-state per fold; repair/montage/channel provenance recorded (`notes/ACAR_V5_STAGE1B*.md`). Stage-2 pipeline = FIT thresholds / CAL Holm-132 / EVAL G1–G5, with `stable_matched_coral_v1` (bounded rank-aware CORAL) + forced-tail identity-only correction, each adversarially reviewed + label-free stress-tested on the real package across two distinct nodes (`…STAGE2B2P_*`, `…STAGE2B3P_*`); the binding run completed rc=0 with the forced-tail contract holding (0/405 forced tails routed). **Scope: this is the machinery/recovery success (the object v4 never had), NOT a claim that the router works — see A8.** |

**ACAR net statement (updated 2026-07-06):** v2 (`MEASUREMENT_ONLY`) and v3 (`DEV_STOP`, coverage collapse) established the
measurement→control gap; v4's control-first redesign (CURB) found a DEV-only candidate that did NOT transfer to the regenerated
external-compatible substrate (A7, `SUBSTRATE_COMPATIBILITY_FAIL`, before external Arm B). **v5 (the substrate-robust
constrained-utility router) RECOVERED the machinery v4 lacked — a hash-bound external-compatible Stage-1B substrate + a guarded,
label-firewalled Stage-2 pipeline that runs to a clean pre-registered verdict (A9, `SUPPORTED`, engineering-only) — and then
produced the sharpest negative yet: Stage-2B real DEV selection = `DEV_STOP / NO_CANDIDATE_SELECTED` (A8, `REFUTED`), driven by
HARM control (G3/G4), not coverage. On a clean external-compatible substrate with proper harm gates, no policy in the frozen
22-candidate universe is safe or beneficial; the identity/source-state LDA `f_0` dominates.** The measurement→control gap is
**NOT closed** across v2→v5. **Authoritative ACAR claim status: measurement replicated (v2/v3); v5 substrate+pipeline engineering
SUPPORTED but efficacy REFUTED on DEV; calibrated safe/beneficial control is NOT achieved by any ACAR version.** Binding for
write-up: do NOT claim a safe deployable router, an external lockbox result, a verified coverage theorem on an external site, that
any DEV candidate (v4 or v5) is usable, or that SCZ's local signal supports deployment. Split the two v5 results cleanly:
engineering/protocol success (A9) is NOT router success (A8). No threshold/δ/seed/candidate search to chase a pass — any
continuation is a NEW dated, separately-tagged protocol (ACAR v6, a new hypothesis), never an in-place edit of `817b04f` /
`b99fa4f` / `5237378` / `4278435` / `ba09777` or these results.

## Standing constraints
No new gate / LPC / coverage / cohort / score search (per the frozen A0-PILOT rule). Deployment control uses no
target labels and no source examples. TUAB stays sealed pending the exposure audit's disposition. **ACAR: v3 is
terminated at `DEV_STOP` (coverage collapse); v4 is terminated at `SUBSTRATE_COMPATIBILITY_FAIL` (A7); v5 is terminated at
`DEV_STOP / NO_CANDIDATE_SELECTED` (A8, harm-control failure) after recovering the substrate+pipeline (A9). The lockbox is NOT
consumed, external / held-out / ASZED is NOT authorized, and Stage-4 (S1/S2/S3) is NOT run (it needs a selected candidate — there
is none). No tuning / reselection / rerun of the candidate universe / score family / policy / G3-G4 gates / λ-threshold grid /
CAL-EVAL interpretation / v2-replay comparator / batch size / MIN_BATCH; any continuation is a NEW dated protocol (ACAR v6, a new
hypothesis), never an in-place edit of `4278435` / `ba09777` or these results. See `notes/ACAR_V5_CLOSEOUT.md` +
`notes/ACAR_V5_STAGE2B_CLOSEOUT.md`.**

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
