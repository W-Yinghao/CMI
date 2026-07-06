# FSR_09 — Related Work Map

**Project FSR — Phase 3A.** Maps FSR onto the literature that grounds its judgment calls. Each theme lists the load-bearing prior work, how FSR relates, and the specific delta. (Citations are by author/year for drafting; fill BibTeX in the manuscript.)

## 1. Probing shows what is *encoded*, not what is *used*
- **Prior:** amnesic probing / counterfactual behavioral tests (Elazar et al. 2020); the broader "probing accuracy ≠ usage" critique of representation probing.
- **FSR relation:** this is exactly the L1→L5 distinction. A subject probe (L1) establishes *detectability*, not *reliance* (L5). Our head-replay `R3` is the EEG analogue of an amnesic/counterfactual test at the representation level.
- **Delta:** we operationalize encoded-vs-used as a measurable ladder on real EEG and show, in a frozen graph-CMI diagnostic, that raw leakage magnitude is *anti-correlated* with reliance while task-head alignment is correctly signed.

## 2. Concept erasure removes a signal; it does not confer benefit
- **Prior:** LEACE closed-form linear concept erasure (Belrose et al. 2023); INLP iterative nullspace projection (Ravfogel et al. 2020); RLACE relaxed adversarial erasure (Ravfogel et al. 2022); Scatter Component Analysis / ISR / SPLINCE for scatter-/moment-based invariant subspaces.
- **FSR relation:** these are our L3 operators. LEACE is the strong closed-form baseline; INLP is the over-erasure control; RLACE the adversarial variant; random-`k` the falsifier.
- **Delta:** we deploy source-fitted erasers to a held-out target and show *erasable ≠ beneficial* — 0/40 cells certify a proven target benefit, and erasure can harm the target (over-erasure/task-collapse; binary-latent entanglement). LEACE's strength (linear erasure to chance) does not transfer to a DG gain.

## 3. Domain generalization requires rigorous selection and strong baselines
- **Prior:** *In Search of Lost Domain Generalization* / DomainBed (Gulrajani & Lopez-Paz 2020) — with careful model selection and a strong ERM baseline, many invariance methods do not beat ERM.
- **FSR relation:** the direct warning against inferring "generalization improved" from a moving invariance proxy. Our closed CMI-control premise (C8) is the EEG instance of this; FSR deliberately does **not** claim a DG method.
- **Delta:** we go one level deeper than "does invariance help accuracy" to "does a measured leakage/erasure proxy even predict the reliance/consequence it is assumed to control" — and answer no.

## 4. EEG subject identity is strong and entangled, not simple noise
- **Prior:** the "Identity Trap" audit of EEG foundation models (subject-identity features are highly decodable and can entangle with label/cohort structure).
- **FSR relation:** motivates the whole project — because subject signal is strong, the temptation to equate "decodable subject" with "harmful shortcut" is strong and wrong. FSR separates measurable subject signal (L1) from harmful task-coupled reliance (L4/L5/L6).
- **Delta:** we provide the audit that distinguishes the two on multiple EEG datasets/backbones and show the distinction is load-bearing (e.g. LEACE drives subject to chance yet yields no target benefit; binary-latent erasure harms task).

## 5. Decompose a joint number into distinct intervention pathways
- **Prior:** the project's own prior-decoupled TTA line (h2cmi): a single unlabeled joint-TTA delta must be decomposed into geometry vs decision-prior/prevalence pathways (exact algebraic decomposition; harm is the prior, not geometry).
- **FSR relation:** the same discipline applied to a different object — leakage × erasure × task-coupling × functional-reliance × target-consequence must not be collapsed into one "shortcut" scalar.
- **Delta:** FSR is the leakage-side realization of the measurement-then-decompose stance; it is cited as background, not as FSR leakage/reliance evidence.

## 6. Conditional invariance / CMI as a control objective (closed premise)
- **Prior:** conditional-invariance and CMI-style penalties for domain-invariant representations (IRM-adjacent, CDAN/CDANN, conditional MI regularizers).
- **FSR relation:** the repository's CIGL/FCIGL/dCIGL/MetaCMI/CITA lines are exactly these, and are a **closed** control premise (C8): the proxy moves but functional reliance and target generalization do not.
- **Delta:** FSR reframes the closed control result as the measurement→control gap and builds an audit around it, rather than proposing another control objective.

## Where FSR sits
FSR is a **measurement/audit** contribution at the intersection of (1) probing-limitation and (2) concept-erasure, disciplined by (3) DG-selection rigor, motivated by (4) EEG identity strength, and methodologically aligned with (5) pathway decomposition — explicitly not a (6) new conditional-invariance method.
