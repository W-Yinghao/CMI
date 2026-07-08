# Project B: Refusal-First Safe EEG Adaptation

*Draft auto-assembled by `scripts/project_b_step3d_paper_update.py` (Step-3D) from frozen Step-2G synthetic tables, the Step-3A real-EEG bridge smoke, and the Step-3C bounded real benchmark expansion. No experiment was re-run.*

## 1. Problem Statement
Test-time adaptation (TTA) can help or harm an EEG decoder at deployment, and whether it helps is not knowable from source data alone. Project B is **not** another EEG TTA loss; it is a **deployment router** that, for an unlabelled target, chooses among `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and emits an auditable reason. Target labels are used only **post-hoc** for evaluation, never to decide.

## 2. Deployment Action Space
`REFUSE` (emit no decode), `IDENTITY` (source-only prediction), `OFFLINE_TTA` (batch transductive class-conditional affine adaptation), `ONLINE_TTA` (streaming). The router is refusal-first: the default is REFUSE, and a non-refusal action must clear explicit support and calibrated-risk gates.

## 3. Refusal-First Router
Action-specific blockers: support/stability/diagnostic failures block *every* action (including IDENTITY); TTA-evidence and ACAR-harm failures block only the TTA actions. Selection is safe-beneficial-then-identity: a beneficial admissible TTA can win; otherwise a support-valid IDENTITY; otherwise REFUSE.

## 4. TOS / Support-Aware Diagnostics
A vector (target size, effective sample size, class-conditional density NLL, transform norm, condition number, prediction disagreement) rather than a single OOD scalar. Too-few-target / low-ESS / support-mismatch / unstable-transform each map to a distinct OACI reason.

## 5. Prior-Decoupled Support Protocol
Support is measured under both the source prior and an estimated target prior; a label-prior shift with intact target-prior density is recorded as audit-only info, **not** a refusal. The support threshold is source-only (§8): baseline = q95 of in-source-unit target-prior NLL; the nested variant adds a scale-normalised held-out-unit *excess* to the base scale.

## 6. ACAR: Action-Conditional Conformal Adaptation Risk
Per action, a split-conformal upper bound on *error* (eligibility) and *harm* (allowed-to-adapt), calibrated on externally-supplied risk predictions over source pseudo-targets. ACAR explicitly represents `available / degenerate / unavailable`: when source pseudo-target harm gains are single-class (degenerate) or too few (unavailable), no harm bound is produced and TTA is blocked.

## 7. OACI Reason Codes
Every decision carries reason codes, separated into blocking vs audit-only, and into action-level blockers vs top-level decision reasons. A TTA blocker never reads as "IDENTITY is unsafe".

## 8. Synthetic Protocol
A controllable EEG simulator with orthogonal shift knobs and a hierarchical site/subject/session DAG. Three locked worlds: **R2** (recoverable), **HF3** (harmful / concept-shift), **H-OOD** (target-only stress). Full training (no fast config). Two source-only support modes: `in_source_subject_q95` (baseline) and `nested_site_excess_q95`.

## 9. Synthetic Results
- **R2** (strict 0.855): raw offline TTA helps on average (+0.071). The in-source support threshold over-refused (coverage 0.00); nested source-site excess fixed this (coverage 0.83, accepted bAcc 0.880). TTA stays blocked because ACAR-harm is degenerate, so R2's benefit is a knowing **missed benefit**.
- **HF3** (strict 0.568): raw offline TTA is harmful on average (-0.071); the router blocks OFFLINE_TTA under ACAR-harm degeneracy. Nested mode accepts some IDENTITY domains (coverage 0.60); a **concept-degraded identity** can pass source-only support (accepted bAcc 0.569).
- **H-OOD** (strict 0.579): raw offline TTA is harmful (-0.231). After nested widening the density `SUPPORT_MISMATCH` clears, but **LOW_ESS** remains the active blocker and OFFLINE_TTA is never selected.

## 10. Real-EEG Evidence
The real-EEG evidence has two tiers: a Step-3A **bridge smoke** and a Step-3C **bounded real benchmark expansion**. Both are source-only and label-safe; target labels are used only **post-hoc**. Neither is a full benchmark.

### 10.1 Step-3A bridge smoke
BNCI2014_004 bridge smoke ran on subjects 1–4, targets 1–2. Raw offline TTA was harmful on both targets: target 1 d_bAcc = -0.101; target 2 d_bAcc = -0.063. The router accepted support-valid IDENTITY and selected no OFFLINE_TTA. This is a **bridge smoke**, not a full benchmark. On this small bridge the nested source-subject excess was 0, so nested == baseline.

### 10.2 Step-3C bounded real benchmark expansion
The **bounded real benchmark expansion** evaluates BNCI2014_004 with `max_subjects = 6`, target subjects 1–4, eval units subject and session, both source-only support modes (`in_source_subject_q95` and `nested_source_subject_excess_q95`), 8 epochs, and `max_nested_folds = 2`.

- Raw offline TTA was harmful on BNCI2014_004: mean d_bAcc = -0.140 across targets. **no OFFLINE_TTA** was selected (offline_tta_rate = 0.00).
- Subject-level routing: coverage = 0.50, identity_rate = 0.50, accepted_bAcc = 0.618, refused/identity = 2/2 target domains.
- Session-level routing: coverage = 0.40, identity_rate = 0.40, accepted_bAcc = 0.614, refused/identity = 12/8 session domains.
- The dominant top-level refusal driver was `SUPPORT_MISMATCH`. **LOW_ESS was not active** on this real run (real subjects have ample trials).
- **nested support calibration was inert**: nested_excess ≈ 0, so the nested mode matched baseline decisions on real BNCI2014_004 (held-out source subjects were not above the in-source support boundary).

These results support harm avoidance and refusal/identity routing on real EEG, **not** an accuracy improvement over identity. See Tables 5–7.

## 11. Claim Boundary
Claimable:
1. Default Project B v1 never selects OFFLINE_TTA when ACAR-harm calibration is degenerate/unavailable.
2. Nested source-site support excess calibration fixes the Step-2E all-refuse failure on R2.
3. Project B can allow support-valid IDENTITY while still refusing low-ESS targets.
4. OACI reason codes expose whether refusal came from support, ESS, ACAR degeneracy, or TTA evidence.
5. The real-EEG bridge runs end-to-end on BNCI2014_004 under label-safe LOSO.
6. The bounded BNCI2014_004 real-EEG expansion shows raw offline TTA was harmful across the evaluated targets and the router selected no OFFLINE_TTA under degenerate ACAR-harm calibration.
7. In the bounded BNCI2014_004 expansion, the router produced heterogeneous support-valid IDENTITY vs REFUSE decisions at subject and session granularity with OACI reason audit.

NOT claimable:
1. Source-only ACAR-harm is not shown to be generally identifiable.
2. v1 does not recover R2's raw TTA benefit (TTA blocked under harm-calibration degeneracy).
3. Support-valid identity is not shown accurate under concept shift.
4. The density support threshold alone does not catch H-OOD after nested widening; LOW_ESS catches only a subset.
5. No target-label-tuned thresholds; all thresholds are source-only and label-safe.
6. This is not a full MOABB benchmark.
7. The bounded real benchmark does not establish that Project B improves accuracy over identity; it establishes refusal/identity routing and TTA harm avoidance under the observed harmful-TTA regime.
8. The bounded real benchmark does not show beneficial-TTA recovery on real EEG; raw TTA was harmful in the evaluated BNCI2014_004 targets.

## 12. Limitations
- ACAR-harm is frequently degenerate/unavailable source-only, so v1 forgoes beneficial TTA (R2 **missed benefit**).
- Source-only support cannot detect concept-shift accuracy loss (**concept-degraded identity** passes on HF3).
- Nested widening can clear the density support signal; **LOW_ESS** is then the only active support blocker (H-OOD).
- The real-EEG evidence is a bridge smoke plus a bounded real benchmark expansion (few subjects/targets, 8 epochs, ≤2 nested folds), **not a full benchmark**.
- The bounded real benchmark ran in a **harmful-TTA regime**: raw TTA was harmful on every evaluated BNCI2014_004 target, so it demonstrates harm avoidance, not beneficial-TTA recovery.

## 13. Next Benchmark Expansion
More subjects and targets, additional datasets (BNCI2014_001 / Lee2019_MI, GPU run), keeping the source-only, label-safe protocol. Whether ACAR-harm ever becomes calibratable and beneficial TTA is recoverable at scale is an empirical question, not an assumption.
