# §5 Related work (draft)

*Positioning, not a survey: 1–2 paragraphs per group, each ending on the difference from this paper.*
**Citations are placeholders `[CITE: …]` — attach real bibkeys; we do not invent references.** Method names
below are canonical and safe to cite; quantitative claims about prior work are intentionally avoided.

## 5.1 EEG domain generalization and cross-subject invariance
Cross-subject transfer is a long-standing problem in EEG/BCI decoding, where subject- and session-specific
statistics dominate. Riemannian/SPD pipelines align second-order statistics on the manifold
(`[CITE: tangent-space / MDM / Riemannian alignment]`), and batch/domain alignment layers such as the SPD
batch-normalization in TSMNet target calibration-free cross-subject decoding (`[CITE: TSMNet/SPDNet]`);
convolutional architectures such as EEGNet and ShallowConvNet are standard decoders
(`[CITE: EEGNet, ShallowFBCSPNet]`). These methods aim to *produce* an invariant or aligned representation.
**Difference:** we do not propose another aligned encoder; we take a *fixed* representation and ask, by
direct measurement, whether its residual conditional domain information is localizable, removable, and
worth removing — and find that removability and benefit are representation-dependent.

## 5.2 Conditional invariance, CMI penalties, and adversarial domain removal
A large body of domain-generalization work penalizes dependence on the domain, either marginally or
conditionally on the label: adversarial domain removal (`[CITE: DANN/CDAN]`), moment matching
(`[CITE: CORAL/MMD]`), invariance penalties (`[CITE: IRM, V-REx]`), and conditional-mutual-information /
LPC-style penalties that target I(Z;D|Y) (`[CITE: CMI/LPC, MINE-style estimators]`). These are typically
applied as always-on training regularizers tuned by a strength hyperparameter.
**Difference:** we show that on a high-dimensional SPD latent a global conditional penalty reduces the
leakage *proxy* mainly by collapsing the representation (an objective-scaling pathology), and that where it
does reduce leakage without collapse (a compact latent) target accuracy does not improve. We therefore
treat the conditional-invariance objective as something to *certify before applying*, not to apply by default.

## 5.3 Selective erasure, concept removal, and task-preserving projection
Removing a specific attribute from a representation while preserving task information is studied as concept
erasure / guarding — iterative nullspace projection and closed-form linear erasure
(`[CITE: INLP, RLACE, LEACE]`) — and more broadly as subspace/feature removal. These methods typically
*erase* a targeted concept and report that it is no longer (linearly) decodable.
**Difference:** our localization is a conditional **score-Fisher** operator (second-order in the
log-likelihood scores, so it sees covariance/synergy leakage a first-moment statistic misses), and — more
importantly — we do not treat successful erasure as the end goal. We add a conditional task-risk certificate
and show empirically that even successful (linear) erasure can leave a nonlinear residual and need not
improve downstream generalization, so erasure is necessary-evidence but not sufficient-evidence of benefit.

## 5.4 Certification, abstention, and safe intervention
Reliable ML increasingly couples decisions to calibrated uncertainty and the option to abstain
(`[CITE: selective prediction / reject option, conformal/coverage guarantees]`), and causal/effect
estimation uses cross-fitted, one-step (doubly-robust) estimators with explicit power considerations
(`[CITE: cross-fitting / one-step / debiased estimation]`).
**Difference:** we import this discipline into representation intervention. Rather than always deleting a
"domain-rich" subspace, we estimate the conditional task risk it carries with a cross-fitted plug-in
log-ratio gate and a power floor, and make **refusal (the identity map) a first-class control decision**
when a safe, useful deletion cannot be certified at the available sample size.

## Positioning sentence (use in the section's intro or conclusion)
> We do not claim a stronger invariant regularizer; we show a measurement-to-control gap for conditional
> domain leakage in EEG representations, and fold explicit certification and refusal into the method.
