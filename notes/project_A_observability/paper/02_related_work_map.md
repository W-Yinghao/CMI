# Related Work — Map

This is a **map**, not the final bibliography. It names the areas Project A sits between, what each
establishes, what gap remains, and how Project A relates. Anchor references are limited to
well-established, widely-cited works; the full citation list is finalized at manuscript time and no
citation is invented here. Where a specific attribution is not yet verified, the entry names the
*concept* rather than a paper.

## EEG/BCI transfer and cross-subject adaptation

- **Establishes.** Cross-subject and cross-session variability is the dominant obstacle in
  motor-imagery BCI; standardized benchmarking exists (the MOABB framework) and strong backbones are
  established (EEGNet; deep/shallow ConvNets; Riemannian tangent-space / covariance methods).
- **Gap.** These works optimize and report *target-labeled* performance; they do not formalize what is
  identifiable when target labels are unavailable at deployment.
- **Project A.** Uses these backbones/benchmarks as the substrate but asks a different question:
  under a stated observation regime and contract, *what target quantity is even identifiable?*

## Domain adaptation / generalization theory and impossibility

- **Establishes.** Classical DA generalization bounds relate source and target error through a
  divergence term (the Ben-David et al. line); a body of results shows unsupervised domain adaptation
  is impossible without assumptions, and that invariant-representation objectives can provably fail
  when label distributions differ across domains (the invariant-representation-lower-bound line).
- **Gap.** These are bounds/impossibility results for a *fixed* estimand (usually target risk); they
  do not provide a regime-indexed, contract-indexed account of *which* estimands become identifiable
  as observation or assumptions change.
- **Project A.** OACI reframes this as identifiability on compatibility sets `K_{R,C}`: TOS-1 is a
  clean source-only impossibility instance, and MONO-1 / contract-strength monotonicity make explicit
  how identifiability changes with observation vs assumption.

## Label shift, GLS, and target-prior estimation

- **Establishes.** Under label shift / generalized label shift, the target label prior can be
  estimated from unlabeled target data given an invertible confusion/mixture operator (black-box shift
  estimation and related moment-matching methods).
- **Gap.** The identifiability precondition (full-rank mixture / invertible confusion) is often assumed
  implicitly; the boundary between "reported prior estimate" and "identified prior" is blurred.
- **Project A.** TU-1 states the precondition as explicit contracts C1∧C2∧C3 and ships CE-R1-2 as the
  exact failure certificate when the mixture is rank-deficient; the audit layer only *admits* a target
  prior claim under TU-1, otherwise it is reported-but-not-identified.

## Domain-invariant representation learning and its limits

- **Establishes.** Adversarial / discrepancy-based invariance (domain-adversarial training and
  descendants) reduces measured domain dependence of representations.
- **Gap.** Reduced *measured* leakage does not certify reduced *reliance* or improved target risk — a
  measurement→control gap documented elsewhere in this repository.
- **Project A.** Treats conditional leakage `I(Z;D|Y)` as a **diagnostic** only (never an accuracy or
  safety guarantee), and PD-1 clarifies when the encoder-vs-decoder leakage trade-off does or does not
  bind after prior decoupling.

## Causal invariance and DG failure modes

- **Establishes.** Invariant-mechanism / invariant-risk approaches seek predictors stable across
  environments; a parallel literature catalogs when such objectives fail.
- **Gap.** Causal-invariance assumptions are strong and often unfalsifiable from the available regime.
- **Project A.** Encodes invariance-type assumptions as *contracts* with an explicit R0/R1/R2
  checkability column (checkable vs assumed), rather than as global modeling axioms.

## Test-time adaptation and source-free adaptation

- **Establishes.** Test-time adaptation (entropy minimization / batch-norm adaptation, e.g. the TENT
  line) and source-free adaptation update a model on unlabeled target data at deployment.
- **Gap.** TTA is typically justified by target-labeled evaluation; whether a given TTA *helps or
  harms* a specific target is not identifiable from what TTA actually observes.
- **Project A.** Places TTA metrics under R1 as oracle/evaluation-only (`identifiable_estimand=null`);
  the audited grids empirically show TTA frequently *harms* the oracle target metric (offline-TTA
  harm-rate ≈ 0.83 across the two ok datasets), reported as fragility evidence, not as a theorem.

## EEG foundation models and adaptation instability

- **Establishes.** Large pretrained EEG encoders are emerging; adaptation/fine-tuning stability is an
  active concern.
- **Gap.** Reporting conventions for what adaptation identifies under deployment information are not
  yet standardized.
- **Project A.** Offers a reusable, machine-checkable observability contract that such models can adopt.

## Positioning of Project A

Project A is **not** a new adaptation method and **not** a SOTA entry. It is an identifiability +
reporting layer *above* existing CMI/TTA machinery: it says, for each (regime, contract), which target
quantity is identifiable, ships an exact counterexample for every non-identifiability claim, and
enforces the boundary with an executable audit. The related work supplies the objects it audits
(backbones, TTA, label-shift estimators, invariance objectives); Project A supplies the contract that
bounds what any of them may claim.

> Citation status: anchor references above are canonical and will be cited precisely at manuscript
> time; no bibliographic entry is fabricated in this map.
