# §2 Methods — TOS-CMI as measurement → certified control (draft)

*Organized by the Results spine, not code history.* Four layers (localize → protect → gate → refuse) plus
the evaluation protocol. Wording follows [claim_evidence_table.md](claim_evidence_table.md); the method is
a **measurement-and-certification** procedure, not an always-on regularizer.

## 2.1 Problem setup and conditional leakage
We observe a representation Z = f(X) of EEG trials with task labels Y and a domain (subject) variable D.
Conditional domain leakage is the residual dependence of Z on D given the task, I(Z;D|Y): a representation
can be domain-invariant in aggregate yet still encode subject identity *within* each class. Unlike a global
marginal penalty, we treat I(Z;D|Y) as something to first **localize** in Z and then **delete only if
certified safe** for the task. Throughout, the held-out target domain is never used for selection, gating,
or calibration — only for final reporting (§2.5).

## 2.2 Score-Fisher localization
We localize the directions of Z that are domain-rich but task-light using a **conditional score-Fisher**
construction. Let G_Y be the label score-Fisher matrix (outer products of ∇_z log p(Y|z)) and G_{D|Y} the
conditional-domain score-Fisher matrix (∇_z log q(D|z,Y), formed on the class-conditional measure). Both are
estimated by cross-fitting (out-of-fold scores) under a whitening metric M = (Σ_W + ε Σ_ref)⁻¹. The
candidate nuisance subspace V_D is read off the leading generalized eigenvectors of

  G_{D|Y} v = ρ (G_Y + η M) v,

i.e. directions that maximize conditional-domain score energy relative to task score energy (large ρ).
This is a **score-Fisher proxy** for conditional leakage — second-order in the log-likelihood scores, so it
detects covariance/synergy structure that a first-moment (mean-scatter) statistic is blind to — not the
exact conditional mutual information [C1]. The number of retained directions nDcand is chosen by an
eigengap/ρ-threshold rule.

## 2.3 Direct-sum projection and the low-rank deletion diagnostic
Given V_D we delete it with an **M-oblique direct-sum projector** P_N onto V_D along the task carrier, and
study the complement RZ = (I − P_N) Z. The projector is constructed to satisfy the direct-sum conditions

  R V_D = 0   (the nuisance subspace is removed)   and   R T = T   (the task carrier is preserved),

rather than mere orthogonality. We then **measure removability** by decoding D and Y from RZ (linear and
MLP probes) and comparing against (i) the full Z and (ii) a same-k *random* removal control. The deletion
is informative only if it reduces domain decode well below random-k while leaving task decode unchanged.
Crucially, **direct-sum geometry is necessary for algebraic task preservation but not sufficient for
conditional task safety** [C2]: a domain-rich direction can still carry conditional task information, which
the geometry cannot see — motivating the gate.

## 2.4 Conditional task-risk gate and certified refusal
Before deletion is permitted, we certify that removing V_D does not destroy task-relevant information by
bounding the conditional task risk it carries,

  Δ_Y(k) = I(Y; P_k Z | (I − P_k) Z),

the task information in the deleted subspace beyond what the complement already holds. Δ_Y is estimated by a
**cross-fitted, one-step plug-in log-ratio** estimator with a power certificate (a minimum-detectable-effect
floor); a weak nested critic is insufficient because it can **unsafe-accept** deletions whose true Δ_Y is
large (Fig 2B). The gate combines an upper confidence bound on Δ_Y (task safety) with a lower bound on the
domain-gain (that the subspace is genuinely domain-rich). **Certification is allowed to abstain, and
abstention is a valid control decision**: if the gate cannot certify that deletion is both safe (Δ_Y small)
and useful (domain-rich) at the available sample size, the correct action is the **identity map**, not a
weaker invariance penalty [C3]. On synthetic controls this certified gate produces zero unsafe-accepts but
is deliberately conservative (Fig 2D) — an honest negative for default-on deletion, by design.

## 2.5 Frozen-feature EEG evaluation protocol
We evaluate on BCI-IV-2a (BNCI2014_001) under leave-one-subject-out (LOSO), domain = subject. For each
fold we train a backbone, **freeze** it, and dump its penultimate latent Z on the source subjects; all
diagnostics (§2.2–2.4) run on the frozen Z, and the held-out subject is used only for reporting target/LOSO
accuracy. We compare two representations to test generality: TSMNet (LogEig/SPD tangent latent, z_dim=210)
and EEGNet (convolutional penultimate latent, z_dim=16). As the standard baseline for "global" conditional
invariance we sweep an LPC penalty (CE + λ·I(Z;D|Y)) over λ, with per-epoch instrumentation (task CE,
feature norm, penalty, encoder gradient, effective rank) to diagnose *how* it changes the representation
(§4.1). The instrumentation is read-only and flag-gated (default off), with verified zero effect on
training dynamics. Probes (subject/task decode) use a 50/50 trial split with linear and MLP heads; the
random-k removal control deletes the same number of M-orthonormal directions as V_D.

> Note (frozen-feature scope): this protocol measures whether *removing* localized leakage from a fixed
> representation helps; it does not retrain the encoder with a certified penalty. The end-to-end variant is
> out of scope here and flagged in §5.
