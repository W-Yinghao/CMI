# §6 Discussion and limitations (draft)

## 6.1 What the measurement-to-control gap means
Across two EEG representations, conditional domain leakage is **measurable** by the score-Fisher
diagnostics but is **not a sufficient control target** for cross-subject generalization. The way control
fails is representation-dependent: in a high-dimensional SPD/LogEig latent the leakage is redundantly
re-encoded and is not low-rank removable, and a global penalty removes it only by collapsing the
representation; in a compact convolutional latent the leakage *is* low-rank removable (linearly), yet
removing it does not improve target accuracy. The common thread is a gap between **localizing/measuring**
leakage and **controlling** it: neither localization nor removal is, by itself, evidence of a
generalization benefit. This reframes selective conditional invariance as a **certified intervention with
refusal** — delete only when a source-risk certificate permits, and abstain otherwise.

A secondary, mechanistic observation concerns *how* the global penalty fails at large λ. The collapse is an
**objective-scaling / training-basin pathology** — the latent contracts to the origin, which trivially
satisfies the penalty — not a gradient explosion and not a smooth geometric over-compression (the apparently
stable effective rank is a scale-invariant-metric artifact). It is preventable by warm-up scheduling, but
preventing it returns the leakage to baseline, so the apparent de-domaining was an artifact of the collapse.

## 6.2 Why leakage removal need not improve generalization
The EEGNet result is the cleanest case: leakage is genuinely (if partially) removable at no task cost, yet
mean LOSO target accuracy is flat-to-worse and uncorrelated with the amount of leakage removed. Two
readings are consistent with our evidence. First, conditional domain leakage I(Z;D|Y) may be **only weakly
coupled to the component of cross-subject error that matters** on this paradigm — subject identity is
*present and decodable* without being the *cause* of the transfer gap. Second, removing a linearly-decodable
slice of identity can leave a **nonlinear residual** (we observe one) and can also remove incidentally
task-correlated variance, so the net effect on the target task is neutral-to-negative. In either reading,
leakage decodability is a property of the representation, not a certificate of a causal generalization
lever — which is exactly why we do not equate removal with improvement.

## 6.3 Limitations
- **Single dataset.** All EEG results are on BCI-IV-2a (BNCI2014_001); we do not claim generality to other
  paradigms, montages, or session structures.
- **Two backbones only.** TSMNet and EEGNet; the cross-representation contrast rests on two points.
- **EEGNet dim/type confound.** TSMNet and EEGNet differ in *both* architecture and latent dimension
  (210 vs 16), which are collinear here. We establish that low-rank removability is representation-dependent
  and capacity-mediated, but do **not** isolate the causal factor (convolutional inductive bias vs latent
  compression); a clean test needs a high-dimensional convolutional or a low-dimensional SPD latent.
- **No end-to-end TOS training.** The EEG study is a frozen-feature pilot: it measures whether *removing*
  localized leakage from a fixed representation helps; it does not show that end-to-end training with a
  certified penalty cannot help.
- **Certified deletion mostly abstains.** On synthetic controls the power-certified gate is conservative
  (frequent abstention); on EEG it abstains on TSMNet and is only diagnostic on EEGNet. By design — we
  report a certification framework with refusal, not a deletion method that is on by default.
- **Leakage removal is not shown to be a causal DG mechanism.** Even where removal succeeds (EEGNet), target
  accuracy does not improve; we make no causal claim linking conditional domain leakage to the transfer gap.
- *(Secondary)* Probe/estimator dependence: the nonlinear residual is reported under an MLP probe; the
  per-epoch encoder gradient is a between-epoch diagnostic proxy; the Fisher/LDA analysis is capped at 7
  directions by 8 source subjects under LOSO.

## 6.4 Implications for EEG invariance methods
Three practical implications follow. (i) **Report the mechanism, not just the metric:** a global
conditional-invariance penalty that "removes leakage" should be checked for representation collapse (feature
norm, not effective rank), because a falling leakage proxy can coincide with a degenerate representation.
(ii) **Treat selective deletion as a certified intervention with refusal:** localize, test the conditional
task risk, and default to the identity map when a safe, useful deletion cannot be certified at the available
sample size. (iii) **Do not treat leakage decodability as a generalization target:** measure whether
removing it actually moves target accuracy before claiming a DG benefit. The natural next steps —
disentangling capacity from representation type, and folding the certificate into end-to-end training with
refusal — are deferred to future work.
