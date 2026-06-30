# §6 Discussion & Limitations (draft)

## 6.1 What the results mean
Across two EEG representations, conditional domain leakage is **measurable** by the score-Fisher
diagnostics but is **not a sufficient control target** for cross-subject generalization. The way control
fails is representation-dependent: in a high-dimensional SPD/LogEig latent the leakage is redundantly
re-encoded and is not low-rank removable, and a global penalty removes it only by collapsing the
representation; in a compact convolutional latent the leakage *is* low-rank removable (linearly), yet
removing it does not improve target accuracy. The common thread is a **measurement-to-control gap**: neither
localizing leakage nor removing it is, by itself, evidence of a generalization benefit. This reframes
selective conditional invariance as a **certified intervention with refusal** — delete only when a
source-risk certificate permits, and abstain otherwise.

A secondary observation is mechanistic: the global-LPC failure at large λ is an **objective-scaling /
training-basin pathology** (a collapse of the latent to the origin that trivially satisfies the penalty),
not a gradient explosion and not a smooth geometric over-compression. It is preventable by warm-up
scheduling, but preventing it returns the leakage to baseline — i.e. the apparent de-domaining was an
artifact of the collapse.

## 6.2 Limitations (owned explicitly)
- **Single dataset.** All EEG results are on BCI-IV-2a (BNCI2014_001). We do not claim generality to other
  paradigms, montages, or session structures.
- **Two backbones only.** TSMNet and EEGNet; the cross-representation contrast rests on two points.
- **Dimensionality–type confound.** TSMNet and EEGNet differ in *both* architecture and latent dimension
  (210 vs 16), which are collinear in this design. We establish that low-rank removability is
  representation-dependent and capacity-mediated, but we do **not** isolate the causal factor (convolutional
  inductive bias vs latent compression). A clean test needs a high-dimensional convolutional latent or a
  low-dimensional SPD latent.
- **No end-to-end TOS training.** Our EEG study is a frozen-feature pilot: it measures whether *removing*
  localized leakage from a fixed representation helps. It does not show that end-to-end training with a
  certified penalty cannot help; it shows that leakage removal *per se* does not buy generalization here.
- **Certified deletion mostly abstains.** On synthetic controls the power-certified gate is conservative
  (frequent abstention), and on EEG the certified path abstains on TSMNet and is only diagnostic on EEGNet.
  This is by design, but it means we report a certification framework with refusal, not a deletion method
  that is on by default.
- **Leakage removal ≠ DG improvement.** Even where removal succeeds (EEGNet), mean LOSO target accuracy is
  flat-to-worse (paired-t), so we make no domain-generalization claim from leakage control.
- **Probe- and estimator-dependence.** The nonlinear residual after deletion (EEGNet) is reported under an
  MLP probe; the per-epoch encoder gradient is a between-epoch diagnostic proxy; the Fisher/LDA analysis is
  capped at 7 directions by having 8 source subjects under LOSO.

## 6.3 Future work
- **Break the dimensionality–type confound** with a high-dimensional convolutional latent and/or a
  low-dimensional SPD latent on the same dataset.
- **End-to-end certified training**: fold the conditional task-risk certificate into training (with
  refusal), rather than diagnosing a frozen representation.
- **Beyond one dataset/paradigm**: cross-dataset and cross-paradigm replication of the measurement-to-control gap.
- **Causal analysis** of why removing conditional domain leakage does not improve transfer on 2a — i.e.
  which component of cross-subject error, if any, conditional domain leakage actually explains.

## 6.4 Conclusion
Conditional domain leakage in EEG representations can be localized, and sometimes removed, but neither
measurement nor removal is sufficient evidence of a domain-generalization benefit. The practical takeaway is
to treat selective conditional invariance as a certified intervention that is allowed — and often correct —
to refuse.
