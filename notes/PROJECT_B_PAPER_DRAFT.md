# Project B: Refusal-First Safe EEG Adaptation

*Draft auto-assembled by `scripts/project_b_step3b_paper_package.py` from frozen Step-2G synthetic
tables and the Step-3A real-EEG bridge outputs. No experiment was re-run.*

## 1. Problem Statement
Test-time adaptation (TTA) can help or harm an EEG decoder at deployment, and whether it helps is not
knowable from source data alone. Project B is **not** another EEG TTA loss; it is a **deployment
router** that, for an unlabelled target, chooses among `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA`
and emits an auditable reason. Target labels are used only **post-hoc** for evaluation, never to decide.

## 2. Deployment Action Space
`REFUSE` (emit no decode), `IDENTITY` (source-only prediction), `OFFLINE_TTA` (batch transductive
class-conditional affine adaptation), `ONLINE_TTA` (streaming). The router is refusal-first: the
default is REFUSE, and a non-refusal action must clear explicit support and calibrated-risk gates.

## 3. Refusal-First Router
Action-specific blockers: support/stability/diagnostic failures block *every* action (including
IDENTITY); TTA-evidence and ACAR-harm failures block only the TTA actions. Selection is
safe-beneficial-then-identity: a beneficial admissible TTA can win; otherwise a support-valid IDENTITY;
otherwise REFUSE. This avoids a least-interventional self-lock while never adapting on unsafe grounds.

## 4. TOS / Support-Aware Diagnostics
A vector (target size, effective sample size, class-conditional density NLL, transform norm, condition
number, prediction disagreement) rather than a single OOD scalar. A too-few-target / low-ESS /
support-mismatch / unstable-transform condition each maps to a distinct OACI reason.

## 5. Prior-Decoupled Support Protocol
Support is measured under both the source prior and an estimated target prior; a label-prior shift with
intact target-prior density is recorded as audit-only info, **not** a refusal. The support threshold is
source-only (§8): baseline = q95 of in-source-unit target-prior NLL; the nested variant adds a
scale-normalised held-out-unit *excess* to the base scale.

## 6. ACAR: Action-Conditional Conformal Adaptation Risk
Per action, a split-conformal upper bound on *error* (eligibility) and *harm* (allowed-to-adapt),
calibrated on externally-supplied risk predictions over source pseudo-targets. Critically, ACAR
explicitly represents `available / degenerate / unavailable`: when the source pseudo-target harm gains
are single-class (degenerate) or too few (unavailable), no harm bound is produced and TTA is blocked.

## 7. OACI Reason Codes
Every decision carries reason codes, separated into blocking vs audit-only, and into action-level
blockers vs top-level decision reasons. A TTA blocker (e.g. `ACAR-harm` degeneracy, negative TTA
evidence) never reads as "IDENTITY is unsafe".

## 8. Synthetic Protocol
A controllable EEG simulator with orthogonal shift knobs and a hierarchical site/subject/session DAG.
Three locked worlds: **R2** (recoverable), **HF3** (harmful / concept-shift, source-calibratable
attempt), **H-OOD** (target-only stress). Full training (no fast config). Two source-only support
modes: `in_source_subject_q95` (baseline) and `nested_site_excess_q95`.

## 9. Synthetic Results
- **R2** (strict 0.855): raw offline TTA helps on average
  (+0.071). The Step-2E in-source support threshold
  **over-refused** (coverage 0.00). Nested source-site excess calibration
  fixed this: coverage 0.00 -> 0.83, accepted bAcc
  0.880. TTA remained blocked because ACAR-harm is
  degenerate, so raw-TTA's benefit is a knowing **missed benefit**, not a policy bug.
- **HF3** (strict 0.568): raw offline TTA is harmful on average
  (-0.071); the router blocks OFFLINE_TTA under ACAR-harm
  degeneracy. Nested mode accepts some IDENTITY domains (coverage 0.60); a
  **concept-degraded identity** can pass source-only support checks (accepted bAcc
  0.569).
- **H-OOD** (strict 0.579): raw offline TTA is harmful
  (-0.231). After nested widening the density
  `SUPPORT_MISMATCH` clears (4 -> 0
  domains), but **LOW_ESS** remains the active blocker for 2 of the target
  domains, and OFFLINE_TTA is never selected.

## 10. Real-EEG Bridge Result
On real **BNCI2014_004** (X shape [2860, 3, 385], classes ['left_hand', 'right_hand'], subjects [1, 2, 3, 4], targets [1, 2]), the bridge ran end-to-end under label-safe LOSO. Target labels were used only **post-hoc**.
  - target 1 (in_source_subject_q95): strict bAcc 0.601, raw offline TTA dbAcc -0.101, action(s) identity:1, ACAR-harm unavailable.
  - target 1 (nested_source_subject_excess_q95): strict bAcc 0.601, raw offline TTA dbAcc -0.101, action(s) identity:1, ACAR-harm unavailable.
  - target 2 (in_source_subject_q95): strict bAcc 0.563, raw offline TTA dbAcc -0.063, action(s) identity:1, ACAR-harm unavailable.
  - target 2 (nested_source_subject_excess_q95): strict bAcc 0.563, raw offline TTA dbAcc -0.063, action(s) identity:1, ACAR-harm unavailable.
This is a **bridge smoke, not a full benchmark**: the router posture (block TTA when harm is
uncalibratable, accept support-valid IDENTITY) reproduces on real EEG. On this small bridge the nested
source-subject excess was 0, so nested == baseline (real subjects were less OOD than synthetic held-out
sites).

## 11. Claim Boundary
Claimable: (i) the router prevents TTA when source-only harm calibration is degenerate/unavailable;
(ii) nested source-held-out support excess calibration fixes the synthetic all-refuse over-refusal on
R2; (iii) it allows support-valid IDENTITY while refusing low-ESS targets; (iv) OACI reason codes
separate the refusal sources; (v) the real-EEG bridge runs end-to-end under label-safe LOSO.
NOT claimable: we do not claim source-only ACAR-harm is generally identifiable; we do not claim v1
recovers beneficial TTA under harm-calibration degeneracy; we do not claim support-valid IDENTITY is
accurate under concept shift; we do not claim density support alone catches H-OOD after nested
widening; we do not claim a complete MOABB benchmark from Step-3A; thresholds are source-only, never
tuned on target labels.

## 12. Limitations
- ACAR-harm is frequently degenerate/unavailable source-only, so v1 forgoes beneficial TTA (R2
  **missed benefit**).
- Source-only support cannot detect concept-shift accuracy loss (**concept-degraded identity** passes
  on HF3).
- Nested widening can clear the density support signal; **LOW_ESS** is then the only active support
  blocker (H-OOD).
- The real-EEG evidence is a bridge smoke (few subjects/targets, coarse eval unit), **not a full
  benchmark**.

## 13. Next Benchmark Expansion
More subjects and targets, session-level deployment routing, and additional datasets
(BNCI2014_001 / Lee2019_MI), keeping the source-only, label-safe protocol. Threshold and harm
calibration behavior at scale is an empirical question, not an assumption.
