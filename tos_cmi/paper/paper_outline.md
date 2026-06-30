# Paper outline — TOS-CMI as a measurement-to-control study

**Status:** writing mode (no new compute). Spine + abstract + section plan. Companion files:
[figures_manifest.md](figures_manifest.md), [claim_evidence_table.md](claim_evidence_table.md).

## Title
Primary: **Selective Conditional Invariance in Task-Orthogonal EEG Subspaces: A Measurement-to-Control Study**
Alt (gap-forward): **When Conditional Invariance Does Not Generalize: Measuring and Certifying Domain Leakage in EEG Representations**
Subtitle option (keep Direction-3 flavor): *From Leakage Localization to Safe Refusal*

## Central claim (three sentences)
1. **Measurement works** — score-Fisher / projection diagnostics localize conditional domain leakage I(Z;D|Y) in EEG latent spaces.
2. **Control is not guaranteed** — removing measured leakage is often insufficient, unsafe, or useless for target generalization.
3. **Certification matters** — weak global or selective penalties can manufacture misleading success; a conservative gate should **abstain** when deletion cannot be certified.

## Abstract (skeleton — to draft from claim_evidence_table)
Conditional invariance is widely used to improve EEG domain generalization, but global invariance
penalties are often unstable and can destroy task information. We study a selective alternative: identify
domain-rich, task-light subspaces via conditional **score-Fisher geometry**, and delete them **only when a
source-risk certificate permits**. On synthetic controls we show geometry alone is insufficient —
task-orthogonal domain directions can still carry conditional task information and weak gates unsafe-accept
them — and introduce a measurement-to-control certification procedure with explicit **abstention**. On
BCI-IV-2a the framework reveals a gap between leakage *measurement* and leakage *control*: in TSMNet,
subject leakage is high-dimensional and redundant — low-rank deletion preserves the task but barely removes
leakage, while global LPC removes leakage only by **collapsing the representation**; in EEGNet, low-rank
deletion removes much more leakage, yet target accuracy does not improve. Conditional domain leakage is
thus a measurable property of EEG representations but **not a sufficient control target** for cross-subject
generalization; safe selective invariance should be treated as a **certified intervention with refusal**,
not an always-on regularizer.

## Section spine

### 1. Motivation
Open from three failures of scalar conditional-invariance objectives: (a) TSMNet + global LPC **collapse**
at high λ; (b) **λ-sensitivity / fragility**; (c) leakage may be task-entangled, redundant, or non-causal.
Core question: *before controlling leakage we must measure where it lives and certify whether deleting it
is safe.* (Local/channel-wise leakage = background motivation only, NOT a main claim.)

### 2. Method — TOS-CMI as measurement → control (four layers, NOT one regularizer)
- **2.1 Score-Fisher leakage localization.** G_Y, G_{D|Y}; generalized eig G_{D|Y}v = ρ(G_Y+ηM)v.
  Emphasize: a *score-Fisher conditional leakage proxy*, not first-moment scatter, not direct CMI.
- **2.2 Direct-sum task protection.** Principle from the synthetic line: require RV=0 *and* RT=T (not
  "orthogonality is enough"). Key line: *direct-sum geometry is necessary for algebraic task preservation
  but not sufficient for conditional task safety.*
- **2.3 Conditional task-risk gate.** Δ_Y(k) = I(Y; P_kZ | (I−P_k)Z); plug-in cross-fitted log-ratio
  estimator + power certificate (why a weak nested critic is not enough).
- **2.4 Certified refusal.** *If the gate cannot certify deletion, the correct action is identity, not
  weaker invariance.* (Directly answers global-λ sensitivity.)

### 3. Synthetic certification (Phase 1) — the measurement chain is *necessary*
Covariance-only leakage (mean-scatter no-op, score-Fisher detects); geometry-safe-but-conditionally-unsafe
synergy (direct-sum alone would delete, task gate refuses); weak nested critic unsafe-accepts → plug-in
improves; certification at moderate n stays conservative (honest negative on default-on deletion). Source:
notes/PHASE131_CERTIFICATION.md, PHASE13_DIAGNOSIS.md.

### 4. EEG measurement-to-control (BCI-IV-2a, LOSO, frozen pilot)
- **4.1 TSMNet (LogEig/SPD, 210-d).** Subject decode ≈1.00. TOS localizes low-rank subject-preferential
  directions, but deleting them only slightly reduces subject decode while preserving task → leakage
  high-dimensional & redundant, low-rank deletion insufficient. Global LPC "solves" leakage only at high λ
  via collapse of the representation to the origin (Z→0); prevent the scaling pathology (warm-ramp /
  scale-invariant) and LPC removes nothing. (Phase 2.0/2.1/2.2.)
- **4.2 EEGNet (conv, 16-d).** Same score-Fisher deletion removes a large (linear ~67%) fraction of
  subject leakage with negligible task cost → low-rank removability is **representation/capacity-dependent**.
  But a substantial nonlinear residual remains, and neither selective nor global removal improves target
  accuracy. (Phase 3.0.)
- **4.3 Unified interpretation.** Conditional domain information in EEG representations is measurable, but
  its causal relation to cross-subject generalization is weak / representation-dependent: removing it can
  collapse the representation, leave redundant leakage behind, or remove leakage without improving transfer.

### 5. Related work (draft_related.md — to draft after abstract lock)
Four groups, 1–2 paragraphs each, positioning by difference (not a survey): EEG DG / cross-subject
invariance; conditional invariance / CMI / adversarial domain removal; selective erasure / concept removal /
task-preserving projection; certification / abstention / safe-intervention framing. Positioning line: *we do
not claim a stronger invariant regularizer; we show a measurement→control gap and fold refusal into the method.*

### 6. Discussion and limitations (draft_discussion.md — §6.1–§6.4)
§6.1 what the measurement→control gap means; §6.2 why leakage removal need not improve DG; §6.3 limitations
(single dataset; two backbones; EEGNet dim↔type confound; no end-to-end TOS training; certified deletion
mostly abstains; leakage removal not shown causal for DG); §6.4 implications for EEG invariance methods.

### 7. Conclusion (draft_conclusion.md)
1–2 short paragraphs: selective conditional invariance as a certified intervention with refusal; the honest
finding that conditional domain leakage is measurable, sometimes removable, but not a sufficient control
target for EEG DG on 2a.

> **Final section order is LOCKED in [section_order.md](section_order.md).** Limitations live in §6.3 (not a
> standalone §5); Related work is §5 (after Results); Conclusion is §7.

## Hard rules for drafting (from the decision)
- No new compute. Only figure polishing, result-table consistency, notes/README alignment.
- Every claim must map to a row in claim_evidence_table.md (allowed/forbidden wording enforced).
