# FSR_36 — Shortcut Recoverability Theory (Phase 7A)

**Project FSR — Phase 7A.** Elevates the empirical repair-scope boundary (4F repairable, 4G not, 4B refused)
into a **recoverability characterization**: *what determines whether a verified harmful shortcut can be repaired
from the available information?* This is a theory of **deployable, target-$X$-only repair operators** grounded in
the frozen controlled positive controls (PC1/4F/4G) and the natural audit (4B); it is a characterization of the
controlled constructions, **not** a claim of a universal theorem. No experiments are run for this document.

## 0. Setup and the class of operators we characterize
A frozen encoder produces branch latents $z\in\mathbb{R}^d$; a frozen head $h$ maps them to logits. A **shortcut
corruption** on a target batch $Z_T=\{z_i\}$ is a per-sample map $C:z_i\mapsto z_i + p_i$ (the injected token, or
the learned nuisance component). A **repair** $R$ maps $Z_T$ to $\hat Z_T$; harm is $R_{\text{harm}} =
\text{bAcc}(h;Z_T^{\text{clean}}) - \text{bAcc}(h; C(Z_T))$; repair succeeds if $R$ raises target bAcc toward the
clean level **without target labels**.

We characterize the **deployable** class actually used in FSR: **target-$X$-only batch-statistic repair
operators**, which apply the *same* map to every sample in a batch, with parameters measurable from the batch's
empirical moments and source references:
$$R(z_i) = A\,z_i + b,\qquad A=A(\hat\mu_T,\hat\Sigma_T,\mu_S,\Sigma_S),\ \ b=b(\hat\mu_T,\hat\Sigma_T,\mu_S,\Sigma_S).$$
E4 (mean alignment) is the translation case ($A=I$, $b=-\lambda(\hat\mu_T-\mu_S)$); E4b (excess-variance
shrinkage) is the linear case ($A=I-\lambda\sum_k(1-r_k)q_kq_k^\top$, $b$ absorbs the mean). This class is exactly
what "deployable, target-unlabeled" buys: no per-sample side information, no target labels.

## 1. The recoverability boundary (a characterization *within* the batch-affine operator + affine-in-$z$ corruption classes)
> **A target-$X$-only batch-statistic affine operator can invert a corruption $C:z_i\mapsto z_i+p_i$ iff $C$ is an
> *affine function of $z$* — i.e. $p_i$ is (approximately) constant across the batch (a translation, undone by
> $b=-\delta$), or a fixed *full-rank* linear map $p_i=Mz_i$ (undone by $A=(I+M)^{-1}$). It cannot invert an
> additive per-sample perturbation that is *independent of $z$* (a genuine stochastic bump), because the
> per-sample realization is not present in the batch statistics.** (Two qualifiers make the "iff" literally true
> *as scoped*: the deterministic map must be **full-rank** — a rank-deficient deterministic map such as a
> projection $I-vv^\top$ is not invertible; and "not invertible" on the right applies to the **batch-affine**
> operator class, see §3.)

The reason is structural: an operator that applies one $(A,b)$ to all samples can only undo a corruption that is
itself an affine function of $z$ — a constant offset (a translation) or a fixed full-rank linear map (a matrix
undoes a matrix). **Crucially, this hinges on independence:** if the per-sample scalar were a function of $z$,
e.g. $p_i=\langle w,z_i\rangle v$, then $C=(I+vw^\top)z_i$ is a *fixed full-rank linear map*, $\hat\Sigma_T$ carries
a $z$–$p$ cross-covariance term, and a batch-affine operator inverts it exactly — so a "per-sample-varying"
corruption can be batch-invertible. The non-invertible case is specifically an additive $p_i=\varepsilon_i v$
with $\varepsilon_i$ i.i.d. and **independent of $z_i$**: it requires per-sample $b_i=-\varepsilon_i v$, which is
**not measurable** from $\{\hat\mu_T,\hat\Sigma_T\}$; the map is information-destroying at the sample level (the
aggregate moments retain only a variance bump $\sigma^2 vv^\top$ along $v$, not the realization).

### Proposition 1 (first-moment deterministic corruption is recoverable) — evidence: 4F
If $C:z_i\mapsto z_i+\delta$ with $\delta$ a **batch-constant** offset, and a source reference mean $\mu_S$ is
available, then $\hat\mu_T - \mu_S = (\mu_T^{\text{clean}}-\mu_S) + \delta$, and mean alignment $R(z)=z-\lambda
(\hat\mu_T-\mu_S)$ removes $\delta$ (exactly at $\lambda=1$ up to the genuine domain-mean shift). At $\lambda=1$,
E4 removes $\delta$ **while additionally performing a source-directed mean shift** $-(\mu_T^{\text{clean}}-\mu_S)$;
it is a *pure* inverse of the injection only when $\mu_T^{\text{clean}}=\mu_S$.
- **Caveat baked into the theory (why 4F is narrow).** Identifiability of $\delta$ requires it be separable from
  the genuine domain shift $\mu_T^{\text{clean}}-\mu_S$; they share the first moment, so alignment repairs the
  offset **and** performs generic mean-TTA. This is why 4F's recovery is $73\%$ a mechanical identity (the
  operator literally inverts the injection by construction), why it must be netted against a clean-target arm to
  isolate the token-specific part, and why it is dataset-carried and **fails leave-one-dataset-out** (the genuine
  shift is dataset-specific). Recoverability $\ne$ generality.

### Proposition 2 (second-moment stochastic corruption is not invertible by a batch-affine operator) — evidence: 4G
If $C:z_i\mapsto z_i+\varepsilon_i v$ with $\varepsilon_i$ **i.i.d., $\mathbb{E}[\varepsilon_i]=0$,
$\mathrm{Var}(\varepsilon_i)>0$, and independent of $z_i$** (so $\hat\Sigma_T$ gains only $\sigma^2 vv^\top$ with
**no $z$–$\varepsilon$ cross-covariance**), then
$\hat\mu_T$ is unchanged (mean-null) and $\hat\Sigma_T$ gains $\sigma^2 vv^\top$ along $v$. A batch-statistic
operator can shrink the variance along $v$ (E4b), but cannot recover each $\varepsilon_i$; if harm arises from
**per-sample margin crossings** ($h(z_i+\varepsilon_i v)$ misclassifies while $h(z_i)$ does not), no batch-level
transform restores the individual samples. Consequently E4b does not beat a random-direction shrinkage control,
and — decisively — **even an oracle** that shrinks the *true* $v$ is sub-threshold, while the estimator recovers
$v$ (overlap $0.71$): the failure is a **genuine non-invertibility**, not mis-estimation. A direction-specific
advantage appears only when the injection dominates the covariance (a near-tautology excluded by the source-only
strength rule).
- **Corollary.** Batch-affine target-$X$-only repair is confined to **affine-in-$z$** (constant offset / fixed
  full-rank linear) corruptions; **additive per-sample noise independent of $z$** is outside its reach for any
  batch-affine operator, regardless of directional knowledge. Note a corruption *correlated with $z$* (an affine
  function of $z$) **is** invertible and is **not** class R1; R1 is specifically additive noise independent of
  $z$.

### Proposition 3 (erasure is invalid for task-coupled shortcuts) — evidence: 4B, PC1
Projection erasure $R(z)=(I-P_S)z$ removes *all* variance along the nuisance subspace $S$. If $S$ is not
(approximately) orthogonal to the task-relevant subspace $T$, erasure removes task signal too, and target bAcc
drops. This holds regardless of identifiability of $S$: in the natural spatial branch (4B) the subject subspace
is task-entangled, so erasing it hurts the target; in PC1 even an *oracle* subspace removal fails because the
injected harm rides a class-directed (task-coupled) direction. **Erasability of a subspace does not imply a valid
repair** — a valid erasure repair requires the **functional** condition that the task label stays decodable at the
head after $I-P_S$. Geometric $S\perp T$ is a **sufficient** proxy ($S\perp T\Rightarrow$ the functional
condition holds), **not necessary**: if the task information in $S\cap T$ is redundant (also present in
$S^\perp$), erasure can be harmless even without orthogonality. The "approximately orthogonal" hedge should be
read as "the functional criterion nearly holds."

## 2. Recoverability classes (non-exclusive obstruction types)
Grounding the propositions, deployable repairability classifies shortcuts by **obstruction type**. These are
**not mutually exclusive** — they sit on different axes (moment-order/determinism, task-geometry, identifiability),
and a corruption can hold several at once (e.g. $\varepsilon_i v$ with $v\in T$ is R1 **and** R2). When several
hold, the licensed action is set by the **binding (most-restrictive) obstruction**: *refuse* if any of
{stochastic, task-coupled, under-identified} holds. **R0 is the unique cell where none hold** (batch-affine,
task-separable, and identified). Full table: FSR_37.
- **R0 — affine-in-$z$, task-separable, identified:** batch-constant offset (or a fixed full-rank linear map),
  separable from domain shift. *Recoverable* by target-$X$-only affine repair (E4). *Evidence:* 4F (constant
  branch only, narrowly, construction-matched). The fixed-linear sub-branch is **algebraic, with no controlled
  FSR evidence** (E4b was only run on the *stochastic* second moment, class R1).
- **R1 — additive per-sample noise independent of $z$:** zero-mean per-sample perturbation, uncorrelated with $z$.
  *Not recoverable* by a batch-affine operator (even oracle direction). *Evidence:* 4G. *Would need* per-sample
  side information. (A per-sample scalar *correlated with $z$* is affine-in-$z$ and is R0, not R1.)
- **R2 — task-coupled:** nuisance subspace overlaps the task subspace. *Not recoverable by erasure*; needs a
  task-preserving inverse that may not exist. *Evidence:* 4B (natural), PC1 (oracle-subspace erase fails).
- **R3 — under-identified from $X$:** learned reliance / concept shift. *Not identifiable* from target-$X$ alone;
  needs labels, paired trials, or a randomized/acquisition contract. *Evidence pointers:* PC2 (future), and the
  sibling directions (concept-shift certification requires paired/minimal information; source-only competence does
  not transport — the repository's measurement$\to$control gap).

## 3. What the theory licenses (claim hygiene)
Two claims must be kept distinct: **(i) exact inversion** — impossible for *any* operator without per-sample side
information when the noise is independent of $z$ (a nondegenerate posterior over $z_i$; information-theoretic); vs
**(ii) harm reduction** — shown impossible *only for the batch-affine class* (4G falsifies E4b even with oracle
$v$; a per-sample non-affine denoiser using a clean prior is outside the class and is *not* excluded by our
evidence).
- **Allowed:** "A batch-constant (first-moment deterministic) shortcut is recoverable by a target-$X$-only mean
  alignment, up to conflation with the genuine domain-mean shift; a per-sample (second-moment) shortcut
  *independent of $z$* is not invertible by any target-$X$-only batch-statistic **(affine)** operator, even with
  the true direction; erasure is a valid repair only when the task label stays decodable after removal.
  Repairability is a *class* property of the corruption × operator class × available information, not a property of
  the shortcut's mere detectability."
- **Forbidden:** "FSR solves shortcut repair"; "all identifiable shortcuts are repairable"; "second-moment
  shortcuts cannot be repaired by *any* deployable operator" (our evidence covers **batch-affine** operators only;
  a per-sample denoiser is out of scope); "second-moment shortcuts are unconditionally unrepairable" (per-sample
  side information / training-time intervention is out of scope, not proven impossible); presenting the
  characterization as a proven universal theorem rather than one grounded in the controlled constructions.

## 4. Why this deepens the paper
The empirical Result 4 ("first-moment repairable, second-moment not") becomes a **principled boundary**: a
deployable target-unlabeled repair inverts only batch-deterministic corruptions that are task-separable, and the
open cases (per-sample stochastic, task-coupled, under-identified) are exactly where FSR *refuses* or requires a
**stronger information contract**. This is the same discipline as the sibling directions — information available
$\to$ what is identifiable $\to$ what action is licensed — now stated for shortcut repair. It reframes the
contribution from "we scoped a repair" to "we characterized the recoverability boundary of functional shortcuts."
