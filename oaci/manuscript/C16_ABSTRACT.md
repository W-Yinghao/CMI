# Abstract

Source-side invariance and robustness penalties are widely used in domain generalization (DG), but their
control signals — reduced conditional-domain leakage, improved source robustness, or better held-out source
endpoints — are often trusted before they are falsified against held-out *target* endpoints. We present a
falsification battery for EEG domain generalization that audits source-side control mechanisms under strict
target isolation. The battery combines support-aware conditional-leakage measurement (restricted to estimable
domain–class cells, with a grouped permutation null), a selection→audit optimism check, a
multiplicity-controlled held-out leakage test, a reproducible worst-domain endpoint test, a non-deployable
source-audit oracle replay, and a source→target anti-transfer diagnostic. Applied to BNCI2014-001
leave-one-subject-out motor imagery with a ShallowConvNet backbone, the battery localizes the failure of two
source-side control hypotheses. A conditional-domain leakage-control method reduces leakage at selection time
on every fold (Δ = −0.326, 54/54) but yields no multiplicity-stable held-out leakage evidence (0 of 54
Benjamini–Hochberg survivors) and no reproducible target worst-domain endpoint gain, and a non-deployable
source-audit oracle — replayed identity-exact over the method's own trajectory — does not rescue it. A
separate source-robust endpoint objective improves the source worst-domain endpoint (by ≈ 1 nat) under the
tested configuration while worsening the target worst-domain endpoint in every active cell, exposing
source→target anti-transfer. These results do not show that domain generalization, EEG transfer, or
support-aware invariance is impossible; they show that, under this protocol, the tested source-side control
signals should be retained as **falsification and measurement instruments** rather than trusted as deployable
control objectives. We release the battery and the fully identity-checked evidence chain so that a proposed DG
penalty can be audited before it is believed.
