# Project B: Refusal-First Safe EEG Adaptation

*Consolidated manuscript draft, auto-assembled by `scripts/project_b_s5_manuscript_package.py` from frozen Step-2G/3C/S0/S1/S2/S2B/S3/S4 outputs. No experiment was re-run.*

**Project B is not a new TTA optimizer. It is a refusal-first deployment router.**

## 1. Problem Statement
Test-time adaptation (TTA) can help or harm an EEG decoder at deployment, and whether it helps is not knowable from source data alone. Project B is a **refusal-first deployment router** that, for an unlabelled target, chooses among `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and emits an auditable reason. Target labels are used only post-hoc for evaluation, never to decide.

## 2. Deployment Action Space
`REFUSE` (emit no decode), `IDENTITY` (source-only prediction), `OFFLINE_TTA` (batch transductive class-conditional affine adaptation), `ONLINE_TTA` (streaming). The default is REFUSE; a non-refusal action must clear explicit support and calibrated-risk gates.

## 3. Router Architecture
Action-specific blockers: support / stability / diagnostic failures block **every** action (including IDENTITY); TTA-evidence and ACAR failures block only the TTA actions. Selection is safe-beneficial-then-identity: a beneficial admissible TTA can win; else a support-valid IDENTITY; else REFUSE. This is a **refusal-first router**, not a least-interventional self-lock.

## 4. OACI Reason Codes
Every decision carries reason codes, separated into blocking vs audit-only and into action-level blockers vs top-level reasons. A TTA blocker never reads as "IDENTITY is unsafe". This makes refusal / identity / TTA-blocking decisions auditable.

## 5. Support-Aware Prior-Decoupled Diagnostics
A diagnostic vector (target size, ESS, class-conditional density NLL, transform norm, condition number, prediction disagreement) rather than a single OOD scalar. Support is measured under both source and estimated-target priors; a label-prior shift with intact target-prior density is audit-only, not a refusal. The support threshold is source-only.

## 6. ACAR-Harm and the Degeneracy Finding
ACAR gives per-action split-conformal upper bounds on error (eligibility) and harm (allowed-to-adapt), with an explicit `available / degenerate / unavailable` state. Source-only harm calibration is frequently degenerate (single-class or too-few pseudo-target harm gains), so TTA is blocked without a fabricated bound. **ACAR-harm degeneracy** is a real non-identifiability, not an implementation failure.

## 7. ACAR-Error for Output Eligibility
Because ACAR-harm is often degenerate, we add an **optional ACAR-error** layer: a source-only, cross-fitted identity-error predictor + split-conformal upper bound that gates IDENTITY output. It is OPTIONAL (used only when the error layer is available; otherwise the router falls back to support-only), never turning an unavailable layer into all-refuse.

## 8. Unified Non-Identifiability Boundary
Without a representativeness assumption linking source pseudo-target domains to the deployment target, source-only calibration cannot identify either action harm OR identity error under arbitrary concept shift: identical source observations + identical unlabeled target diagnostics + different target label mechanism force identical router decisions but different true target risk. ACAR-error repairs the source-representative / observable regime, but the target-only **H-OOD boundary persists**.

## 9. Synthetic Experiments
- **R2** (strict 0.855): raw offline TTA helps (0.071); nested source support calibration fixes over-refusal (coverage 0.83, accepted bAcc 0.880); TTA stays blocked under degenerate ACAR-harm — a knowing missed benefit.
- **HF3** (strict 0.568): raw TTA harmful (-0.071); a support-valid concept-degraded identity can pass v1 — motivating ACAR-error.
- **H-OOD** (strict 0.579): target-only stress (-0.231); support/ESS help but do not complete; the boundary persists.

## 10. Real EEG Experiments
On real BNCI2014_004 (bounded LOSO), raw offline TTA is harmful (mean d_bAcc -0.140) and OFFLINE_TTA is never selected (rate 0.00); the router routes support-valid IDENTITY or refuses OOD subjects. Across BNCI2014_004 / BNCI2014_001 / Lee2019_MI (S1 phase map), **no real benefit phase** is observed (max target gain 0.020 < 0.05). Worse, the source-fold TTA-gain predictor rank-transfers (BNCI2014_004 corr 0.90) but its OFFSET does not: it would select harmful TTA. Verdict: `no_real_benefit_phase_observed`.

## 11. PRIOR_ONLY Action Study
`PRIOR_ONLY` (freeze encoder/density/classifier; re-estimate only the target class prior) is the lowest-harm adaptation action (harm 0.54 vs OFFLINE_TTA 0.71) but does NOT recover R2 missed benefit (R2 prior_only gain -0.045 while OFFLINE_TTA gain 0.143): the recoverable benefit is covariate-driven, not prior-driven. `prior_only` is therefore **deferred** — not integrated into the router.

## 12. Backend Comparison with CBraMod
Under a common source-only downstream (identical z-score+PCA+Gaussian head+affine-TTA), the pretrained CBraMod foundation encoder applied zero-shot to MI is a **weaker** identity representation than the source-trained h2cmi encoder: identity Δ -0.096 (BNCI2014_004), -0.040 (BNCI2014_001), -0.084 (Lee2019_MI). Its lower support-mismatch reflects a more diffuse density (accepting worse predictions), not better support. Its one source-predictable TTA gain is a **weak-baseline artifact**: CBraMod+TTA absolute bAcc stays below the best identity baseline on every dataset. Verdict: `cbramod_weaker_representation_benefit_is_artifact`. Zero-shot foundation representation does not create a deployable benefit phase (fine-tuning is future work).

## 13. EEGAgent Integration Roadmap
Project B is a **safety governor** that an EEGAgent-style LLM workflow can call as a risk-routing tool. The hard boundary: EEGAgent may call Project B and explain its OACI reason codes, schedule other tools (artifact/PSD/symmetry checks) on a REFUSE, and generate reports — but it must NOT override Project B's refusal / no-TTA decision. See `PROJECT_B_AGENT_INTEGRATION_ROADMAP.md`.

## 14. Claim Boundary
See `PROJECT_B_NEXT_CLAIM_BOUNDARY.md`. Headline: Project B prevents unsafe adaptation, routes support-valid identity, optionally filters identity by ACAR-error, and audits via OACI. It does not guarantee TTA improvement, does not solve concept shift, and does not claim full-benchmark superiority.

## 15. Limitations
ACAR-harm frequently degenerate; support cannot detect concept-shift accuracy loss; the H-OOD target-only boundary persists; real evidence is bounded (few subjects/targets); PRIOR_ONLY is low-harm but non-recovering; zero-shot CBraMod is a weaker MI representation (fine-tuning untested); EEGAgent integration is a roadmap, not evaluated.

## 16. Conclusion
On real EEG motor imagery, no deployable OFFLINE_TTA benefit phase exists — for the native h2cmi backend or a zero-shot foundation backbone — and source-only calibration cannot identify target-only harm or error. Project B's honest contribution is therefore **refusal-first harm-avoidance, support-valid IDENTITY, optional ACAR-error output eligibility, and OACI auditability** — a safety governor, not a selective-TTA accuracy booster.
