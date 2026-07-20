# Introduction

Domain-generalization (DG) methods for EEG and other physiological signals are frequently justified by a
**source-side control signal**: a quantity computable from the source domains alone — reduced
conditional-domain leakage (invariance), improved source robustness, or a better held-out source endpoint —
that is expected to translate into better performance on an unseen target subject or site. Such signals are
appealing precisely because they need no target labels. That same property makes them easy to *trust before
they are tested*: a penalty that improves a source-side signal, and a model selected to optimize it, can look
successful without any evidence that the improvement reaches the target.

Under strict target isolation, these signals can be **audited before they are trusted**. Two obstacles make a
careful audit necessary. First, under **domain–class support mismatch** — some `(domain, class)` cells having
little or no source support — conditional-invariance diagnostics that average over all cells are not
well-identified, so leakage must be measured only on estimable cells with an honest permutation null. Second,
a single point estimate on a selection split conflates a real effect with selection-time optimism, and a
method's failure to beat a baseline can always be blamed on a poor selector or a poor selection split.

We propose a **falsification battery** that tests whether a source-side control signal transfers to
**target worst-domain endpoints**, and that closes the usual escape hatches. The battery combines: a
support-aware, source-only extractable-leakage measurement; a selection→audit optimism check; a
multiplicity-controlled held-out leakage test (K1); a reproducible worst-domain endpoint test (K2); a
**source-audit oracle replay** that removes selector and selection-split quality as explanations; and a
**source→target anti-transfer** diagnostic for source-endpoint objectives. Every gate reads only source
information for its decisions; the target is used strictly for evaluation.

We instantiate the battery once, on BNCI2014-001 leave-one-subject-out motor imagery with a ShallowConvNet
backbone, and use it to falsify two plausible source-side control mechanisms as case studies: a
conditional-domain leakage-minimization method (OACI) and a source-robust endpoint objective (SRC). The
battery localizes the failure — selection leakage reductions that do not survive audit, held-out signals that
do not survive multiplicity, no reproducible endpoint gain, an oracle that cannot rescue the trajectory, and a
source endpoint objective that *anti-transfers* to the target. The point of the paper is not that these two
methods lose to ERM; it is that a support-aware, target-isolated falsification protocol makes *why* they
should not be trusted **localizable, reproducible, and reusable**.

**Contributions.**
1. A falsification battery (G0–G5) for source-side DG control signals under strict target isolation, built on
   support-aware conditional-leakage measurement, held-out leakage/endpoint gates, source-audit oracle replay,
   and source→target anti-transfer diagnostics.
2. Two case-study falsifications on BNCI2014-001 LOSO — OACI (leakage-control) and SRC (endpoint-control) —
   each localized to a specific gate, with byte/numeric-exact replay identity and pre-registered decisions.
3. A stance and a set of retained instruments: support-aware leakage and the K1/K2/oracle/anti-transfer
   diagnostics are kept as **measurement and falsification** tools, distinct from the control objectives they
   falsify. We are explicit about scope: these are single-dataset, single-backbone results; the battery has so
   far only returned "falsified," and its discriminative validity is future work.

We deliberately avoid the over-generalizations that the evidence does not support: we do not claim that DG or
EEG transfer is impossible, that support-aware invariance is useless, or that BNCI2014-001 by itself
establishes a support-mismatch regime.
