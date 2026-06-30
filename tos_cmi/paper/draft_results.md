# §4 Results — EEG measurement-to-control (draft)

*Scope:* BCI-IV-2a (BNCI2014_001), LOSO, domain = subject, frozen-feature pilot. Numbers traced to
[claim_evidence_table.md](claim_evidence_table.md) (C4–C10) and [figures_manifest.md](figures_manifest.md);
allowed wording enforced. Figures: Fig 3 (collapse), Fig 4 (TSMNet), Fig 5 (EEGNet), Table 1.

## 4.0 Setup (one paragraph)
We freeze a trained backbone per LOSO fold and dump its penultimate latent Z on the source subjects (the
held-out target subject is used for reporting only, never in selection, gating, or probing). On the frozen
Z we (i) localize a candidate domain-rich, task-light subspace V_D with the score-Fisher generalized
eigenproblem (§2.1), (ii) delete it by the M-oblique direct-sum projector (§2.2) and measure subject and
task decodability on the complement RZ=(I−P_N)Z with linear and MLP probes, against a same-k random-removal
control, and (iii) sweep a global LPC penalty (CE + λ·I(Z;D|Y)) to compare selective deletion with the
standard global objective. We study two representations: TSMNet (LogEig/SPD tangent latent, z_dim=210) and
EEGNet (convolutional penultimate latent, z_dim=16).

## 4.1 TSMNet: subject leakage is high-dimensional and redundant; global LPC "works" only by collapse
**Measurement is positive.** In the TSMNet latent, subject identity is almost perfectly decodable from the
full Z (MLP 0.997, chance 0.125; Fig 4A), and the score-Fisher selector returns a compact candidate
subspace (nDcand ≈ 3 of 210; Fig 4D).

**Control is negative (deletion).** Deleting V_D preserves the task (decode 0.75→0.75; Fig 4B) but barely
reduces subject decode (0.997→0.96 MLP), essentially matching same-k random removal (≈0.997; Fig 4A,C).
Even deleting the full Fisher-discriminative subspace (the LDA cap of 7 directions for 8 source subjects)
leaves subject decode at 0.92–0.98 — subject identity is **redundantly re-encoded** across the
high-dimensional latent, so **low-rank selective deletion is insufficient** [C4].

**Control is negative (global penalty).** The global LPC penalty *appears* to remove leakage at high λ, but
the per-epoch curves (Fig 3A–C) show this is a sharp, λ-tied **collapse of the representation to the
origin**: task cross-entropy rises to ln 4 (chance), the feature norm and top singular value go to 0, and
the penalty itself goes to 0 (trivially satisfied by Z→0). This is an objective-scaling pathology, not a
gradient explosion (the diagnostic encoder gradient is ~10× smaller than in healthy low-λ training and
never non-finite) and not a smooth geometric over-compression (the "effective rank stays high" is a
scale-invariant-metric artifact) [C5]. The collapse is *fixable* by a warm-up schedule (or, at λ=1, a
scale-normalized penalty), confirming its optimization origin — but every collapse-free, task-preserving
variant leaves subject decode at the ERM level (Fig 3D), i.e. **once the collapse is prevented, the global
penalty removes no leakage** [C6]. In the tested TSMNet/2a setting, global LPC's apparent de-domaining is a
collapse artifact.

## 4.2 EEGNet: low-rank deletion removes leakage (linearly), but removal yields no generalization benefit
**Removability is representation-dependent.** In the compact EEGNet latent the *same* score-Fisher deletion
behaves very differently (Fig 5A–C, Table 1). Deleting V_D (nDcand ≈ 5 of 16) preserves the task
(0.64→0.64) and **selectively** reduces subject decode far below same-k random removal: linear 0.82→0.35
(random 0.73), MLP 0.88→0.54 (random 0.81). The informed-vs-random selectivity is ~0.35–0.55, roughly an
order of magnitude larger than on TSMNet (0.04–0.08), and same-k random deletion does *not* reproduce it —
so this is genuine subspace selectivity, not an artifact of deleting a larger fraction of a small latent
[C7]. The reduction is **partial**: a substantial nonlinear residual survives (RZ MLP 0.54, well above
chance 0.125), so we report it as *linearly reducible with a persistent nonlinear residual*, not as
elimination [C8].

**But removable ≠ beneficial.** Sweeping the global LPC penalty on EEGNet reduces subject decode
monotonically (0.89→0.19) *without* collapse (the feature norm never goes to 0; Fig 5D), yet target/LOSO
accuracy is flat-to-worse (0.36→0.39 across λ; statistically not improved, and uncorrelated with leakage
reduction, Pearson −0.14). Removing the measured leakage — by selective deletion or by a collapse-free
global penalty — does **not** improve cross-subject generalization on EEGNet/2a [C9].

## 4.3 Unified interpretation
Across both representations, conditional domain leakage is a **measurable** property of the EEG latent that
the score-Fisher diagnostics localize, but it is **not a sufficient control target** for cross-subject
generalization (Table 1). Removing it can (i) collapse the representation (TSMNet global LPC), (ii) leave
redundant high-dimensional leakage essentially intact (TSMNet low-rank deletion), or (iii) genuinely
remove leakage yet not improve transfer (EEGNet). Whether the leakage even *sits* in a low-rank removable
subspace is representation-dependent and, in our two-backbone design, **capacity-mediated**: a high-capacity
210-d latent redundantly re-encodes subject identity outside any low-rank subspace, while a compact 16-d
latent cannot. The correct operating principle is therefore a **certified intervention with refusal**: the
framework should delete only when a source-risk certificate permits, and abstain otherwise — which on
TSMNet means abstain (deletion insufficient) and on EEGNet means a diagnostic deletion that removes leakage
but is not claimed to improve generalization [C10].

> Limitation flagged here (full text in §5): representation type and latent dimensionality are collinear in
> this design (SPD↔210, conv↔16); Phase 3 establishes representation dependence, not its causal factor.

## Numbers locked (cross-check vs Table 1 / claim_evidence_table)
- TSMNet: subj 0.997; RZ 0.96 ≈ random 1.00; task 0.75→0.75; LPC collapse feat→0, tgt n/a; collapse-free LPC removes ~0.
- EEGNet: subj 0.88; RZ linear 0.35 / MLP 0.54 ≫ random 0.73/0.81; task 0.64→0.64; LPC subj 0.89→0.19, tgt 0.36→0.39.
