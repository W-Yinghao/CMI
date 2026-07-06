# Project A — Repository Audit for EEG Adaptation Observability and Information Contracts

> **Status:** Step 1 deliverable (repository audit + design skeleton). Doc-only; no
> training code touched. Branch: `project/observability-contracts` (off `main` @ `c8fce20`).
> Method: 24 source files read in full (six grouped deep-readers), critical theory quotes
> verified byte-exact against source. Every "verbatim" line below is copied from the file,
> not paraphrased.

## 0. Goal

Project A asks what can be identified under three information regimes:

- **R0 — source-only:** multi-source `(X_s, Y_s, D_s)`; no target `X_T`, no target `Y_T`.
- **R1 — target-unlabeled:** source labeled + target unlabeled `X_T` (target `Y_T` forbidden).
- **R2 — minimal-paired:** a few target labels / paired source↔target sessions / montage
  anchors / pre→post calibration pairs.

It is **not a new loss first.** It is an observability- and information-contract layer above
the existing CMI / H²-CMI codebase. For each regime and each declared contract it decides
whether an estimand `T(P)` is *identifiable* — i.e. whether every world consistent with what
the regime observes and with the contract yields the same `T` — and, when it is not, it
exhibits two indistinguishable worlds as the certificate of non-identifiability.

The repo already contains most of the machinery Project A needs (a signed hierarchical-CMI
estimator, a mechanism simulator with orthogonal shift knobs, a source-only safety gate, a
three-regime harness, and an exact chain-rule identity). It also contains a **live internal
inconsistency**: the `notes/theory/` manuscript pillars still assert three claims that the
sibling `h2cmi/THEORY.md` explicitly retracts (P0-2/P0-3/P0-4). Resolving that inconsistency
*is* the seed of Project A's identifiability layer — see §7.

---

## 1. Existing repo layers

### 1.1 Old `cmi/` layer (real-EEG pipeline)

**Main method.** LPC-CMI: learn `Z=f_θ(X)` that maximises task predictability while
minimising the **conditional domain leakage** `I(Z;D|Y)` — domain (subject/session/device/
site) information remaining in `Z` after the task label `Y` is known.

**Main estimand & its exact caveat.** The trained criterion is a **posterior-KL plug-in
surrogate**, *not an upper bound*:

```
I(Z;D|Y) = E_{z,y} KL( p(D|z,y) ‖ p(D|y) )          # true CMI, π_y(D)=p(D|Y)
R_CMI(ψ) = E_{z,y} KL( q_ψ(D|z,y) ‖ π_y(D) )         # our estimator (L_CMI)
```

`regularizers.py` header, verbatim: *"NOT an upper bound … it EQUALS I(Z;D|Y) only when
q_ψ = p(D|z,y) (Step-A convergence); for a sub-optimal critic it generally UNDER-estimates
… e.g. q_ψ = π_y(D) gives L_CMI = 0 while I(Z;D|Y) can be > 0. So: a consistent plug-in
estimator, tight at convergence, not a Barber–Agakov-style bound."* This under-estimation
property is **the identifiability crux** Project A must formalize; `stepA_dom_acc` (how well
`q_ψ` predicts `D` from `(Z,Y)`) is the only exposed critic-convergence proxy.

**Ablation family (the estimand lattice).** `marginal = I(Z;D)` (label erasure),
`chain/FMCA = I(Z;(D,Y))` (Y erasure), `lpc_uniform = KL→Uniform` (CDANN target,
mis-specified under imbalance), `lpc_prior = KL→π_y` (the full method). Decoder side (dual
methods): `I(Y;D|Z) = H(Y|Z) − H(Y|Z,D) = CE_q − CE_h`.

**Main trainer/eval files:**
- `cmi/methods/regularizers.py` (LPC-CMI core): `DomainPosteriors` critic bank
  `q(D|Z,Y)/q(D|Z)/q(S|Z)/h(Y|Z,D)/intercept-h0`; `kl_to_prior`; three prior estimators
  `empirical/subject/effective`; `reg(method,…)` dispatch. **P0-5 already applied here:**
  `p_d_ref = p_d` (raw domain marginal), *not* `π_y.mean(0)`.
- `cmi/train/trainer.py`: two-step alternation — **Step A** fit posteriors on *detached* `Z`
  (`n_inner=2`); **Step B** update backbone + task head with `CE + λ_t·L_CMI`, `λ` warm-up
  (`lam_t = lam·min(1, ep/warmup)`). `_label_shift_weights` = GLS `w_i = π*(y_i)/π_{d_i}(y_i)`.
  Sampler discipline: `classbal` preserves `p(D|Y)` (correct for conditional MI); `domainbal`
  uniformizes it (mis-specified). GraphCMI/CIGL path adds `r_graph / r_node / r_edge`.
- `cmi/eval/metrics.py`: frozen-encoder held-out audit statistics — `leakage_probe`
  (residual `I(Z;D|Y)` + `leakage_advantage = dom_acc − prior_dom_acc`),
  `marginal_leakage_probe` (`I(Z;D)`), `decoder_leakage_probe` (`I(Y;D|Z)` with a
  **within-class permutation null** + `decoder_valid` span guard), `label_separability`.
- `cmi/run_loso.py` (Protocol A LOSO / B cross-session) and `cmi/run_cross_dataset.py`
  (Protocol C, leave-one-dataset-out): both fit the audit probes on a **70/30 source-only
  split** (`pi`/`ei`) — *no target labels used*.

**Assumptions / regime dependence.** MCPS (multi-class-per-subject; motor imagery, emotion):
`π_y ≈ uniform`, conditional ≈ marginal, so the conditional term is barely identifiable.
SCPS (single-class-per-subject; clinical, `subject ≈ label`): marginal removal erases the
label, so the `π_y` correction is essential — *and* the decoder term degenerates
(`D=subject ⇒ Y=g(D) ⇒ I(Y;D|Z)=H(Y|Z)`).

**Known risks / limitations (from the docs themselves):** accuracy on balanced MI is a
field-wide **wash**; ADFTD SCPS wins are **seed-sensitive** (+3.7/−5.0/−0.3); EA gains are
**100% transductive** (`ea_strict 41.8 < none 43.2` on 2a); leakage reduction magnitude is
**not** a quality signal (a marginal method removes *more* leakage by destroying the label;
`lpc_prior` only removes *training-induced* leakage toward the random-encoder floor —
no-free-lunch on irreducible subject info). All numbers are **preliminary**, pending a
unified-preprocessing re-run. DomainBed baselines are self-reimplemented first-pass `λ`.

### 1.2 New `h2cmi/` layer (post-review redesign, simulator-validated)

**Why it exists.** An isolated, post-review (ICLR-direction) redesign that (a) fixes four
theory bugs the review raised (P0-2…P0-5) at the *code* level and (b) **hard-codes three
prior NEGATIVE boundaries** as architecture rather than penalties. It runs end-to-end
**without real EEG** on a controllable mechanism simulator; the whole pipeline is exercised
by one smoke test (which **passes** — see §8 evidence). Nothing imports-with-side-effects
from `cmi/`.

**Corrected theoretical claims (verified byte-exact against `h2cmi/THEORY.md`):**

- **P0-2** — the CMI regularizer is a *neural conditional-entropy* estimator, **not** a
  posterior-KL upper bound. `R = E KL(q_ψ(d|z,y) ‖ p(d|y))` is not an upper bound (set
  `q_ψ=p(d|y)` → `R=0` while true CMI > 0); it is a posterior plug-in exact only at
  `q_ψ=p_θ(d|z,y)`. **Corrected estimand:**
  ```
  I(Z; D | Y) = H(D | Y) − H(D | Z, Y)
  ```
  with the encoder **maximising** the critic-optimal conditional cross-entropy (a min–max /
  conditional-GRL objective); the envelope theorem at `ψ*(θ)` makes the frozen-critic Step-B
  gradient `−λ·∂CE/∂z` the correct profile gradient. This **contradicts** `03_estimators.md`,
  which calls the same object an "upper bound" and the scheme explicitly "not adversarial."
- **P0-3** — there is **no** "joint CMI = 0 ⇔ zero Bayes error" theorem. Counterexample:
  `A,B` independent, `D=A`, `Y=(A,B)`, `Z=A` ⇒ `D→Y→Z` holds and
  `I(Z;D|Y)=I(Y;D|Z)=0`, yet `H(Y|Z)=H(B)>0`. The correct DPI-equality condition is
  `D ⊥ Y | Z` (Z sufficient for the D-information in Y), **not** `Y=f(Z)`. This contradicts
  the **A3** theorem asserted across `01/02/03/04`.
- **P0-4** — `I(Y;D|Z)=H(Y|Z)−H(Y|Z,D)` is a **conditional predictive-insufficiency
  diagnostic, not "genuine concept shift."** It can arise from a true `p(y|x,d)` change *or*
  from `Z` discarding task info, misspecification, incomplete label-shift correction,
  calibration/annotation noise, or thin per-domain class support. Clinical degeneracy:
  `D=subject ⇒ Y=g(D) ⇒ I(Y;D|Z)=H(Y|Z)` — the probe collapses to label predictability.
  This contradicts the "concept shift" reading in `02/04`.
- **P0-5** — the GLS reference domain marginal was coded wrong (`p_d_ref = mean_y p(d|y)`);
  in general `(1/|Y|)Σ_y p(d|y) ≠ p(d)`. **Corrected:**
  ```
  p_ref(d) = normalise( Σ_i  w_i · 1[d_i = d] )   with   w_i = π*(y_i)/p(y_i|d_i)
  ```
  and marginal alignment must be under a **fixed reference prior**
  `p_d*(z)=Σ_y π*(y) p_d(z|y)` (align class-conditionals, never raw `p_d(z)`).

**Modules available for reuse (review-section → module):** `domains/dag.py`
(domain-factor DAG: `site→subject→session`, `handling ∈
{invariant, random_effect, conditional, label_mechanism}`, per-factor `budget` in nats,
`determines_label` SCPS flag); `data/eeg_simulator.py` (orthogonal `ShiftSpec` knobs
`cov/prior/concept/concept_site_frac/montage/noise/label_mechanism_rho`; exposes both
observed `y` and latent `ystar`; `train_target_split` holds out whole target sites);
`cmi/hierarchical.py` (**confirmed from code** — signed per-factor `Î_j = H_ref_j − CE_j =
H(D_j|Y,Pa) − H(D_j|Z,Y,Pa)`, primal-dual budgets, conditional-GRL min–max);
`align/reference_marginal.py` (`gls_weights`, `gls_reference_domain_marginal`,
`ReferenceMarginalAlignment` — P0-4/P0-5); `tta/class_conditional.py` (near-identity affine
`(A,b)` + target-prior EM at the frozen source density, **identity fallback** below
`min_target`/`min_effective_classes`); `gate/safety_gate.py` (source-only learned harm
predictor over an 8-feature panel, `should_adapt`); `eval/harness.py` (three-regime
`strict_dg / offline_tta / online_tta`, domain-clustered bootstrap, selective-risk panel);
`eval/leakage.py` (**cross-fitted signed** `I_hat_j` + within-`(Y,Pa)` permutation null,
negatives kept).

**What remains simulator-only.** *Everything* validated in `h2cmi/` is
`data/eeg_simulator.py`-generated; `README` explicitly disclaims real-EEG SOTA and defers to
review §10.4 (unified preprocessing, 5–10 seeds, subject/site-clustered inference, external
lockbox). The modules take ordinary `(X[n,chans,times], y, DomainLabels)` arrays and are
*claimed* real-EEG-capable via the `cmi/data` loaders, but **no real-EEG run is reported**.

---

## 2. Theory anchors already present

### `notes/theory/01_tension.md` — Pillar 1, the Tension Theorem (~290 lines)
- **Exact identity (A1), verbatim:** `I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D).` Proved via
  chain rule `I((Y,Z);D)=I(Y;D)+I(Z;D|Y)=I(Z;D)+I(Y;D|Z)`. Numerically verified to
  `2.22e-15` over 3000 random discrete joints (`verify_tension.py`).
- **What it proves:** A2 — setting `I(Z;D|Y)=0` forces `I(Y;D|Z)=I(Y;D)−I(Z;D)>0` under
  label shift (encoder invariance ⇒ decoder leakage). A4 — GLS reweighting resolves it.
- **What Project A can reuse:** A1 is the load-bearing **observability accounting** — it
  exactly partitions the domain-information budget `I((Y,Z);D)` into encoder vs decoder
  attributions. Assumption-light; safe.
- **What Project A must NOT overclaim:** A3 ("both CMIs 0 ⇔ `Y=f(Z)` / zero Bayes error") is
  **retracted by P0-3** (`D⊥Y|Z` is the true condition). The `D→Y→Z` Markov assumption is a
  *contract*, not a fact. Everything here is simulator-only.

### `notes/theory/02_resolution.md` — Pillar 2, the GLS Resolution (~176 lines)
- **GLS / reference-prior correction, verbatim:** `w_d(y) = π*(y) / π_d(y)`,
  `p̃(z,y,d) ∝ w_d(y)·p(z,y,d)`. **Lemma 2.1** (exact algebra): under `p̃`,
  `p̃(Y=y|D=d)=π*(y) ∀d ⇒ Y⟂D ⇒ Ĩ(Y;D)=0`. **Corollary 2.2 (★):**
  `Ĩ(Z;D|Y) = Ĩ(Y;D|Z) + Ĩ(Z;D)` — all non-negative, so zeroing the encoder term forces
  both others to 0. **No forced trade-off.**
- **Identifiable under which contract:** reweighting *requires* per-domain `π_d(y)=p(y|D=d)`;
  the file adopts GLS's target-free confusion-matrix estimator `w = C⁻¹μ` — **a UDA
  (target-unlabeled) tool, not a source-only DG tool.** So the decoupling into three
  separately-zeroable terms is available **only under the GLS contract** (C7 below).
- **What stays unidentifiable without target info:** the concept residual — a *control*
  experiment in `verify_resolution.py` shows reweighting alone leaves `Ĩ(Y;D|Z)=0.145>0`.
- **Overclaim to avoid:** presenting reweighting as freely available source-only; the "A3
  dissolved" prose inherits the retracted A3. Per P0-4, the residual is *not* certified
  concept shift.

### `notes/theory/03_estimators.md` — Pillar 3, the two estimators (~165 lines)
- **Estimator claims:** encoder term via posterior-KL, **claimed an UPPER bound** on
  `I(Z;D|Y)`, "tight iff `q_ψ=p(D|z,y)`", scheme declared **"not adversarial."** Decoder term
  `Î(Y;D|Z)=CE_q−CE_h`, honestly flagged two-sided.
- **Conflict with `h2cmi/THEORY.md` (P0-2):** the "upper bound" is **false** (`q_ψ=π_y ⇒ R=0`
  while CMI > 0); the corrected estimator is a **min–max** `H(D|Y)−H(D|Z,Y)` with critic
  maximization + envelope-theorem gradient — the opposite of "not adversarial." The file's
  numpy check only perturbs `q` *away* from the optimum (always `≥` exact) and never tests
  the adversarial `q_ψ→π_y` direction that breaks the bound.
- **How Project A should phrase estimator vs estimand:** treat `03`'s "upper bound" as
  retracted; adopt `H(D|Y)−H(D|Z,Y)` as the *estimand* and the cross-fitted, permutation-
  nulled, non-truncated `eval/leakage.py` output as the *reporting estimator*. The
  estimator's fidelity is itself a contract (C5: critic sufficiency).

### `notes/theory/04_positioning.md` — Pillar 4, Positioning & Novelty (~209 lines)
- **Novelty:** three-shift↔three-term map (covariate `= I(Z;D|Y)`; label `=` reweighting;
  concept `=` residual `I(Y;D|Z)`); tension-theorem framing; concept shift as a first-class
  estimated quantity for psychiatric cross-site EEG.
- **Closest prior works:** Zhao19 (ICML'19 joint-error lower bound — marginal invariance is
  *harmful* under label shift; the "wall we route around"); GLS/Combes'20 (NeurIPS'20 — we
  adopt its reweighting as term 2); entropyDG/Zhao'20 (NeurIPS'20 conditional-entropy
  regularizer; assumes balanced classes → dissolves under label shift).
- **Project A extension:** the file *gestures at* but never *formalizes* the identifiability
  question. It self-flags the core risk verbatim: *"If the empirical concept-shift story is
  weak, the paper collapses to 'we re-derived GLS and added an entropy term.' Pillar 4's
  viability is hostage to the experiments."* Project A's contribution is to make the
  **observability question** (what is source-only identifiable) the first-class object,
  relabel `I(Y;D|Z)` as a predictive-insufficiency residual (P0-4), and supply the
  indistinguishable-worlds certificates the pillars lack.

### `h2cmi/THEORY.md` — the authoritative corrections file (~134 lines)
Not under `notes/theory/` but **overrides** the four pillars on three load-bearing points
(P0-2 estimator, P0-3 A3, P0-4 concept). This is the file Project A treats as authoritative
for the **estimator-vs-estimand** distinction. Caveat: its assertion *"none of the disputed
claims appear in code"* is a claim about the code — independently confirmed by this audit for
`cmi/hierarchical.py` (signed `H_ref−CE`), `align/reference_marginal.py` (weighted bincount),
and `eval/leakage.py` (permutation null, no truncation).

**Verification scripts:** `verify_tension.py` (A1 over random discrete joints),
`verify_resolution.py` (GLS resolution + the concept-residual control). Both discrete/synthetic.

---

## 3. Observability regimes

| Regime | Observed | Not observed | Identifiable **without** extra contract | Identifiable **only with** contract | **Non-identifiable** |
|---|---|---|---|---|---|
| **R0 source-only** | multi-source `(X_s,Y_s,D_s)`, DAG `site→subject→session`; source class-conditionals `p_S(z|y)`; per-domain `π_d(y)` | target `X_T`, `Y_T`, `π_T(y)`, target concept/transform | source-side residual `I(Z;D|Y)` (up to C5 critic quality); `I(Z;D)`, `I(Y;D|Z)` on source (up to C6 span); source LOSO harm↔gain map; `label_sep`; the A1 identity among source terms | source→target **risk bound** (needs C1+C2); GLS decoupling into (★) on source (needs C7 per-domain `π_d(y)`, which *is* on source) | **target adaptation gain**, target `π_T(y)`, target concept `p_T(y|z)`, target harm — **the TOS ceiling (A1-thm)** |
| **R1 target-unlabeled** | R0 **+** target unlabeled `X_T` ⇒ `p_T(z)` | target `Y_T` | target feature marginal `p_T(z)` | target prior `π_T(y)` via `p_T(z)=Σ_y π_T(y)p_ref(z|y)` **under C2+C1+C3** (mixture separability / full-rank `C`); GLS `w=C⁻¹μ` | target **accuracy** (needs labels); target **concept shift** `p_T(y|z)` — **A3-thm**; whether adaptation *helped* |
| **R2 minimal-paired** | R1 **+** few target labels / paired sessions / montage anchors / pre→post pairs | the bulk of target `Y_T` | direct target risk on the small labeled slice; observed pairing | low-dim transport `(A,b)` **under C8** (low-dim, invertible, full-rank, overlap); label-mechanism residual; tighter risk **bound** | high-dim / unconstrained transport; full concept mechanism without enough anchors |

**Information monotonicity (A6).** `R0 ⊂ R1 ⊂ R2`: more observation ⇒ smaller set of
compatible target worlds ⇒ more identifiable objects ⇒ tighter bounds. But the *type* of
information matters — **more source subjects cannot substitute for target-unlabeled `X_T` or
paired target anchors** (they shrink source-side variance, not the target-world ambiguity).

---

## 4. Candidate contracts

Each contract is a declared assumption an adaptation method *depends on*; Project A's job is
to say, per regime, whether it can be **falsified from data** or must be **assumed**.

### C1 — Class support overlap
**Statement:** `supp p_T(z|y) ⊆ supp p_S(z|y)` for every class `y` (no target mass off the
source class-conditional manifold). **Needed for:** any transport/risk transfer; the TTA EM.
**Failure mode:** target class in a region the source density never modelled → density-NLL
misleads. **Falsifiable source-only?** No. **Target-unlabeled?** Partially (target `Z` mass
vs source support). **Minimal-paired?** Yes, on the paired slice.

### C2 — Shared class-conditional feature geometry
**Statement:** `p_T(z|y) = p_S(z|y)` (covariate/acquisition shift only; the class→feature
rule is invariant). **Observed evidence:** matched class-conditional moments on paired data.
**Needed for:** target-prior identifiability (R1), GLS. **Failure mode:** concept shift — a
rotation of the class→source-power map (`ShiftSpec.concept`) breaks it. **Falsifiable
source-only?** No. **Target-unlabeled?** No (indistinguishable from a prior change — the A3
world pair). **Minimal-paired?** Yes.

### C3 — Label-prior identifiability (mixture separability / full-rank confusion)
**Statement:** the source class-conditional densities are linearly independent and the
confusion operator `C` is full-rank, so `π_T(y)` is the unique solution of
`p_T(z)=Σ_y π_T(y)p_ref(z|y)` (equivalently `w=C⁻¹μ` well-posed). **Needed for:** R1 prior
estimation; GLS weights. **Failure mode:** near-degenerate classes → ill-conditioned `C`,
`π_T` non-identifiable. **Falsifiable source-only?** Yes (source `C` conditioning is
observable). **Target-unlabeled?** Yes. **Minimal-paired?** Yes.

### C4 — Acquisition-vs-label-mechanism factor separation
**Statement:** each domain factor `D_j` is either an *acquisition* nuisance (site/montage/
session; encoder-invariance-eligible, `determines_label=False`) or a *label-mechanism* factor
(rater/site label channel `p(Ỹ|Y*,D)`; excluded from encoder invariance). **Observed
evidence:** the DAG `handling` annotation. **Needed for:** legitimacy of applying invariance
at all (P0-4). **Failure mode:** applying invariance to a `determines_label` factor erases the
label (SCPS degeneracy `Y=g(D)`). **Falsifiable source-only?** Partially (test whether `D_j`
predicts `Y` deterministically). **Target-unlabeled?** No. **Minimal-paired?** Yes.

### C5 — Critic sufficiency / Step-A convergence (the estimator-vs-estimand contract)
**Statement:** `q_ψ → p_θ(D|Z,Y)` (Step-A converged), so the plug-in `L_CMI` equals the true
`I(Z;D|Y)` rather than under-estimating it. **Observed evidence:** `stepA_dom_acc`, and the
in-loop-KL vs frozen-probe-KL gap. **Needed for:** any claim that a *measured* leakage value
is the *true* leakage. **Failure mode:** sub-optimal critic reads `L_CMI≈0` while true
CMI > 0 (verbatim: `q_ψ=π_y ⇒ R=0`). **Falsifiable source-only?** Only *bounded*: the probe
gives a lower bound; the gap is diagnosable but the true CMI is not point-identified. This is
the sharpest non-identifiability *inside* a single dataset and the reason P0-2 switches to the
`H(D|Y)−H(D|Z,Y)` min–max form.

### C6 — Domain-class span (decoder validity)
**Statement:** each domain spans `≥2` classes, so `I(Y;D|Z)` is a meaningful concept probe
rather than a single-class-prior artifact. **Observed evidence:** `domain_class_span_stats`,
`decoder_valid`. **Needed for:** any concept-shift reading of `I(Y;D|Z)`. **Failure mode:**
`D=subject` (SCPS) ⇒ `I(Y;D|Z)=H(Y|Z)` (P0-4 degeneracy). **Falsifiable source-only?** Yes
(directly checkable). **Target-unlabeled/minimal-paired?** Same check applies.

### C7 — GLS within-domain reweighting availability
**Statement:** per-domain `π_d(y)=p(y|D=d)` is known (or estimable via `w=C⁻¹μ`), enabling
`Ĩ(Y;D)=0` and the (★) decoupling. **Observed evidence:** source labels give source `π_d(y)`;
target needs C3 + target `X_T`. **Failure mode:** source-only DG has no target `π_T`, so the
*target-side* decoupling is unavailable — a frequent overclaim. **Falsifiable source-only?**
Source side yes; target side **no** (that is exactly what R1 buys and R0 lacks).

### C8 — Low-dimensional invertible transport with overlap
**Statement:** the acquisition/montage/session transform lies in a low-dim invertible family
(e.g. near-identity affine `A=I+UVᵀ`, `‖A−I‖` small) with sufficient class overlap.
**Needed for:** minimal-paired transport identifiability (A4). **Failure mode:** high-dim or
non-invertible transform → only a bound, not point identification; the TTA trust region
(`τ‖A−I‖²`) *encodes* this contract. **Falsifiable source-only?** No. **Minimal-paired?** Yes.

### C9 — Source→target gain transfer (the safety-gate contract)
**Statement:** the inner-LOSO source-domain distribution of adaptation gain
`Δ=bAcc_adapt−bAcc_identity` transfers to unseen real targets, so a source-only harm
predictor generalizes. **Needed for:** `gate/safety_gate.py` validity. **Failure mode:** the
**measurement→control gap** — the project has repeatedly found this NEGATIVE (see memory
`cigl-r2-gate-result`, `cmi-gate-falsification`). **Falsifiable source-only?** Only within
source (inner-LOSO AUROC); its *target* transfer is precisely what R0 cannot certify — a
prime Project A non-identifiability target.

### C10 — Bounded Bayes error / label-noise regime (A3 escape)
**Statement:** `H(Y|Z)=0` (zero Bayes error) — the only regime in which both CMIs can vanish
jointly *without* GLS. **Failure mode:** clinical EEG has irreducible label noise `H(Y|Z)>0`,
so the escape is unreachable; invoking A3 as if C10 held is the retracted-P0-3 overclaim.
**Falsifiable:** `H(Y|Z)>0` is estimable source-only (task Bayes-error lower bound).

---

## 5. Candidate theorem ledger

Proposed theorems (the user's A0–A6). For each: statement · regime · required contracts ·
proof strategy · file where the proof should live · whether a simulator counterexample is
needed. **Bold** marks where this audit already found supporting or contradicting evidence.

- **A0 — OACI identifiability definition.** `T(P)` is identifiable under regime `R` and
  contracts `C` iff for all worlds `P₁,P₂` with `Observed_R(P₁)=Observed_R(P₂)` and both
  satisfying `C`, `T(P₁)=T(P₂)`. *Regime:* all. *Contracts:* none (it is the frame). *Proof
  strategy:* definitional; instantiate `Observed_R` for R0/R1/R2 concretely against the DAG +
  simulator. *File:* `06_oaci_identifiability.md` (prerequisite for A1). *Counterexample:* n/a.

- **A1 — TOS source-only ceiling.** Under R0, target risk, `π_T(y)`, target concept shift, and
  target adaptation gain are **non-identifiable.** *Contracts:* holds even granting C4–C7 on
  the source. *Proof strategy:* fix the source joint exactly; build two target worlds — one
  with source class-conditional geometry, one with a rotated class→source map (`concept>0`) or
  a shifted `π_T` — that are **observationally identical to a source-only observer** yet have
  opposite-sign adaptation gain. *File:* `03_tos_source_only_ceiling.md`. **Counterexample
  NEEDED — buildable now** on `eeg_simulator.py` (tune `concept`/`prior` with matched source
  sites; `meta` carries oracle params for the ground-truth check). **Supported by** the
  existing negatives (EA-transductive, gate falsification, measurement→control gap).

- **A2 — Target-unlabeled prior identifiability under mixture/GLS contract.** Under R1, `π_T(y)`
  is identifiable iff **C2 ∧ C1 ∧ C3** (shared class-conditionals, support, full-rank `C`).
  *Proof strategy:* uniqueness of the mixture decomposition `p_T(z)=Σ_y π_T(y)p_ref(z|y)`;
  reduce to `w=C⁻¹μ` (Lipton'18 / Combes'20). *File:* `02_resolution.md` extension +
  `06_oaci_identifiability.md`. **Counterexample needed:** two `π_T` giving the same `p_T(z)`
  when C3 fails (degenerate classes).

- **A3 — Concept-shift non-identifiability from unlabeled target.** Under R1 (no target labels,
  no label-mechanism anchor), `p_T(y|z)` is **not** identifiable: two label mechanisms induce
  the same `p_T(z)`. *Proof strategy:* explicit pair via `label_mechanism_rho` vs a matching
  prior change. *File:* `07_counterexample_catalog.md`. **Counterexample NEEDED.** **Strongly
  supported by P0-4** (the residual is a predictive-insufficiency diagnostic) and the C6
  degeneracy — this is arguably Project A's most defensible theorem.

- **A4 — Minimal-paired transport identifiability.** Under R2, if the transform family is
  low-dim, invertible, full-rank, with sufficient overlap (**C8**), the acquisition/montage/
  session transform is identifiable to statistical error; otherwise only a bound. *Proof
  strategy:* moment-matching / procrustes identifiability on the paired anchors; degrade to a
  bound when C8 relaxes. *File:* `04_prior_decoupled_theory.md` or a new `08`-transport note.
  **Counterexample needed:** high-dim transform under-determined by `k` pairs.

- **A5 — Prior-decoupled CMI.** Under the reference prior `π*` and GLS reweighting (**C7**),
  `Ĩ(Y;D)=0` and the A1 identity becomes the all-positive additive relation
  `Ĩ(Z;D|Y)=Ĩ(Y;D|Z)+Ĩ(Z;D)` (★), so encoder and decoder leakage no longer trade off.
  *Proof strategy:* **already proved** — Lemma 2.1 + Corollary 2.2 in `02_resolution.md`
  (exact algebra, verified `≤8.3e-17`). *File:* `04_prior_decoupled_theory.md` (import + state
  the contract C7 explicitly, drop the A3-dependent prose). **No new counterexample**; reuse
  the `verify_resolution.py` concept-residual control.

- **A6 — Information monotonicity.** `Observed`-set inclusion `R0⊆R1⊆R2` ⇒ the compatible-
  target-world set shrinks ⇒ identifiable-estimand set grows and bounds tighten; but the added
  information *type* is not interchangeable (source breadth ≠ target-unlabeled ≠ paired).
  *Proof strategy:* monotonicity of the "consistent worlds" preimage under refinement of
  `Observed_R`; a separating example that more source subjects leave the target world
  ambiguous. *File:* `01_information_regimes.md`. **Counterexample needed:** two target worlds
  separated only by target `X_T`, provably unshrinkable by adding source subjects.

---

## 6. Reusable code map

| Project A need | Existing file | Reuse as-is? | Modify? | Notes |
|---|---|:--:|:--:|---|
| domain-factor DAG / contract annotations | `h2cmi/domains/dag.py` | ✅ | ➖ crossed factors | `handling`+`budget`+`determines_label` = the per-factor contract; canonical DAG is strict **nested** — crossed worlds need hand-built multi-parent `DomainFactor`s (representable, untested) |
| mechanism / indistinguishable worlds | `h2cmi/data/eeg_simulator.py` | ✅ | ➕ crossed sampler, matched-world builder | orthogonal `ShiftSpec` knobs; exposes `y` **and** `ystar`; `meta` has oracle params. Sinusoidal sources ⇒ results are about *this* generative family |
| corrected CMI estimator (estimand) | `h2cmi/cmi/hierarchical.py` | ✅ | ➖ | **confirmed** `Î_j=H(D_j|Y,Pa)−H(D_j|Z,Y,Pa)`; only `invariant`/`random_effect` factors penalised (excludes `conditional`/`label_mechanism` — the observability seam) |
| observability estimator (reporting) | `h2cmi/eval/leakage.py` | ✅ | ➕ wire into harness | cross-fitted signed `I_hat_j` + within-`(Y,Pa)` permutation null, no truncation. **Not** currently called by `harness.py` |
| prior-decoupled alignment | `h2cmi/align/reference_marginal.py` | ✅ | ➖ | `gls_weights`, `gls_reference_domain_marginal` (P0-5), `ReferenceMarginalAlignment` (P0-4) — the identifiability-correct reference measures |
| TTA / identifiability boundary | `h2cmi/tta/class_conditional.py` | ⚠️ | ➕ | identity-fallback reasons `too_few_target`/`single_class_target` are ready **non-identifiability sentinels**; but `cmi_residual` is **hardcoded 0.0** (dead feature) and online TTA never adapts its transform |
| execute/refuse contract | `h2cmi/gate/safety_gate.py` | ⚠️ | ➕ | `GATE_FEATURE_KEYS` + `should_adapt`; validity rests on **C9** (source→target gain transfer) = the non-identifiability Project A must certify; leakage feature is inert until wired |
| three-regime reporting | `h2cmi/eval/harness.py` | ✅ | ➕ add leakage panel | `run_three_settings` keeps `strict_dg`/`offline_tta`/`online_tta` separate; domain-clustered bootstrap + selective-risk |
| real-EEG leakage audit primitives | `cmi/eval/metrics.py` | ✅ | ➖ | `leakage_probe`, `marginal_leakage_probe`, `decoder_leakage_probe` (+ within-class permutation null), `decoder_valid`, `label_separability` — the R0 identifiability tests on real data |
| estimator + prior lattice | `cmi/methods/regularizers.py` | ✅ | ➖ | `DomainPosteriors`, `kl_to_prior`, `empirical/subject/effective` priors; the "consistent plug-in, not upper bound" caveat = the C5 contract statement |
| two-step estimator harness / GLS weights | `cmi/train/trainer.py` | ✅ | ➖ | `_label_shift_weights` (GLS `w_i`), `stepA_dom_acc` (C5 proxy), `classbal` sampler discipline |
| R0 observability harnesses | `cmi/run_loso.py`, `cmi/run_cross_dataset.py` | ✅ | ➕ null in cross-dataset | 70/30 source-only probe split; `_imbalance_subsample` = controlled `p(D|Y)`-skew generator; cross-dataset lacks the decoder permutation null |

**Stale-map warning:** `CODE_INVENTORY.md` predates the active CIGL direction and does **not**
list `cmi/methods/graph_regularizers.py` / `cmi/models/gnn.py` — read those directly, not via
the inventory, if Project A touches the graph layer.

---

## 7. Red flags — claims Project A must not repeat

1. **"LPC-CMI is a posterior-KL upper bound."** `README.md:59` still says this; retracted by
   `PROJECT_SUMMARY.md` **and** `h2cmi/THEORY.md` P0-2 (`q_ψ=π_y ⇒ R=0` while CMI > 0). Use
   "consistent plug-in, exact only at Step-A convergence, can under-estimate."
2. **"Both CMIs zero ⇔ zero Bayes error (A3)."** Asserted across `01/02/03/04`; **retracted by
   P0-3** (`D⊥Y|Z` is the real condition; counterexample `D=A,Y=(A,B),Z=A`). The four pillars
   were never updated — a **live internal inconsistency** to resolve, not inherit.
3. **"`I(Y;D|Z)>0` means concept shift."** P0-4: it is a predictive-insufficiency diagnostic
   confounded by insufficient `Z`, misspecification, incomplete reweighting, thin support, and
   it **degenerates to `H(Y|Z)`** under `D=subject`. Requires C2+C5+C6 to *even be evidence*.
4. **Treating the source-only safety gate as target-gain identification.** C9 / the
   measurement→control gap; the project's own gate lines are NEGATIVE. Source inner-LOSO AUROC
   is not target certification.
5. **Treating an unlabeled target feature marginal as target concept identification.** A3:
   `p_T(z)` is compatible with many `p_T(y|z)`.
6. **Treating a single-class target as class-conditional-transport identifiable.** The TTA
   identity-fallback (`min_effective_classes`) exists precisely because it is not; don't route
   around it.
7. **Treating leakage-reduction magnitude as an accuracy/quality signal.** No-free-lunch: a
   marginal method removes *more* leakage by destroying the label; `lpc_prior` only removes
   *training-induced* leakage toward the random-encoder floor.
8. **Folding EA / transductive gains into a strict-DG claim.** `ea_strict 41.8 < none 43.2` —
   the entire EA gain is target-unlabeled (R1), not R0. Keep `strict_source_only_DG` and
   `transductive_TTA` in separate tables (the repo's own discipline).
9. **Claiming the gate/TTA uses a leakage/CMI residual.** `cmi_residual` is **hardcoded 0.0**
   in `tta/class_conditional.py`; `eval/leakage.py` is **not wired** into `eval/harness.py`.
   The leakage-based contract term must be *added* before it can be claimed.
10. **Claiming online adaptation.** `fit_online` / `evaluate_online_tta` only EMA the prior;
    the affine transform stays at identity — "online TTA" is prior-only.
11. **Mixing training surrogate with identifiable estimand.** The optimized object (min–max CE
    / plug-in KL) is not the estimand (`I(Z;D|Y)`); C5 governs the gap. State them separately.
12. **Presenting simulator results as real-EEG evidence.** *All* of `h2cmi/` and all four
    theory pillars are simulator/discrete-synthetic; `cmi/` real numbers are **preliminary**
    and DomainBed baselines are self-reimplemented (baseline-strength risk).

---

## 8. Evidence attached to this audit

- **Smoke test (`conda run -n icml python -m h2cmi.tests.test_smoke`): PASSED (exit 0).** Full
  `h2cmi` pipeline runs end-to-end on the simulator:
  ```
  strict-DG  bAcc=0.667  worst-dom=0.667  ece=0.198
  offline-TTA  d_bAcc(adapt)=-0.333  d_bAcc(selective)=-0.333  coverage=1.00
  online-TTA  bAcc=0.667
  leakage I_hat: {'site': 0.222, 'subject': 1.398, 'session': 2.13}
  SMOKE TEST PASSED
  ```
  (Tiny CPU config: `offline-TTA Δ<0` is not meaningful — it confirms the harness, bootstrap,
  selective-risk, gate, and cross-fitted signed leakage all produce well-formed outputs.)

## 9. Proposed next files

Priority order, respecting the A0→A1 and A1→counterexample dependencies:

1. **`01_information_regimes.md`** — formalize R0/R1/R2 and the `Observed_R` operator; state
   **A0** (OACI identifiability definition) and **A6** (monotonicity). This is the frame every
   later theorem cites.
2. **`03_tos_source_only_ceiling.md`** — the flagship **A1** proof, with its indistinguishable-
   worlds construction spelled out and *executed* on `eeg_simulator.py` (matched source sites,
   divergent target concept/prior; oracle check via `meta`).
3. **`07_counterexample_catalog.md`** — the reusable simulator constructions: the A1 target-
   world pair, the **A3** label-mechanism-vs-prior pair, and the A2/C3 degenerate-mixture pair.
   A1 and A3 are only *certificates* once these worlds exist, so this file is the empirical
   spine of the ledger.

(Deferred to later steps: `02_contract_taxonomy.md`, `04_prior_decoupled_theory.md` — mostly
an import of the proved A5/(★), `05_csc_shift_calculus.md`, `06_oaci_identifiability.md` if A0
outgrows `01`, `08_experimental_protocol.md`.)
