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

## 1. The recoverability boundary (central result)
> **A target-$X$-only batch-statistic affine operator can invert a corruption $p_i$ iff $p_i$ is a
> batch-*deterministic* function of the moment it matches — i.e. $p_i$ is (approximately) constant across the
> batch, or a fixed low-rank map with a known direction. It cannot invert a *per-sample stochastic* corruption,
> because the per-sample realization is not present in the batch statistics.**

The reason is structural: an operator that applies one $(A,b)$ to all samples can only undo a corruption that is
itself constant (a translation removes a constant offset) or a fixed linear map (a matrix undoes a matrix). A
per-sample $p_i=\varepsilon_i v$ with $\varepsilon_i$ random requires per-sample $b_i=-\varepsilon_i v$, which is
**not measurable** from $\{\hat\mu_T,\hat\Sigma_T\}$ — the map $z_i\mapsto z_i+\varepsilon_i v$ is
information-destroying at the sample level; the aggregate moments retain only its *distribution* (a variance bump
along $v$), not its *realization*.

### Proposition 1 (first-moment deterministic corruption is recoverable) — evidence: 4F
If $C:z_i\mapsto z_i+\delta$ with $\delta$ a **batch-constant** offset, and a source reference mean $\mu_S$ is
available, then $\hat\mu_T - \mu_S = (\mu_T^{\text{clean}}-\mu_S) + \delta$, and mean alignment $R(z)=z-\lambda
(\hat\mu_T-\mu_S)$ removes $\delta$ (exactly at $\lambda=1$ up to the genuine domain-mean shift). E4 is the inverse
operator of the corruption in the *same moment* it lives in.
- **Caveat baked into the theory (why 4F is narrow).** Identifiability of $\delta$ requires it be separable from
  the genuine domain shift $\mu_T^{\text{clean}}-\mu_S$; they share the first moment, so alignment repairs the
  offset **and** performs generic mean-TTA. This is why 4F's recovery is $73\%$ a mechanical identity (the
  operator literally inverts the injection by construction), why it must be netted against a clean-target arm to
  isolate the token-specific part, and why it is dataset-carried and **fails leave-one-dataset-out** (the genuine
  shift is dataset-specific). Recoverability $\ne$ generality.

### Proposition 2 (second-moment stochastic corruption is not invertible from batch moments) — evidence: 4G
If $C:z_i\mapsto z_i+\varepsilon_i v$ with $\mathbb{E}[\varepsilon_i]=0$, $\mathrm{Var}(\varepsilon_i)>0$, then
$\hat\mu_T$ is unchanged (mean-null) and $\hat\Sigma_T$ gains $\sigma^2 vv^\top$ along $v$. A batch-statistic
operator can shrink the variance along $v$ (E4b), but cannot recover each $\varepsilon_i$; if harm arises from
**per-sample margin crossings** ($h(z_i+\varepsilon_i v)$ misclassifies while $h(z_i)$ does not), no batch-level
transform restores the individual samples. Consequently E4b does not beat a random-direction shrinkage control,
and — decisively — **even an oracle** that shrinks the *true* $v$ is sub-threshold, while the estimator recovers
$v$ (overlap $0.71$): the failure is a **genuine non-invertibility**, not mis-estimation. A direction-specific
advantage appears only when the injection dominates the covariance (a near-tautology excluded by the source-only
strength rule).
- **Corollary.** Deployable target-$X$-only repair is confined to **batch-deterministic** (constant / fixed
  low-rank) corruptions; **per-sample stochastic** corruptions are outside its reach for any batch-statistic
  operator, regardless of directional knowledge.

### Proposition 3 (erasure is invalid for task-coupled shortcuts) — evidence: 4B, PC1
Projection erasure $R(z)=(I-P_S)z$ removes *all* variance along the nuisance subspace $S$. If $S$ is not
(approximately) orthogonal to the task-relevant subspace $T$, erasure removes task signal too, and target bAcc
drops. This holds regardless of identifiability of $S$: in the natural spatial branch (4B) the subject subspace
is task-entangled, so erasing it hurts the target; in PC1 even an *oracle* subspace removal fails because the
injected harm rides a class-directed (task-coupled) direction. **Erasability of a subspace does not imply a valid
repair** — a valid erasure repair additionally requires $S \perp T$.

## 2. Recoverability classes (the map)
Grounding the propositions, deployable repairability partitions shortcuts into classes (full table: FSR_37):
- **R0 — batch-deterministic, task-separable:** constant / fixed low-rank offset, separable from domain shift.
  *Recoverable* by target-$X$-only affine repair (E4). *Evidence:* 4F (narrowly; construction-matched).
- **R1 — per-sample stochastic:** zero-mean per-sample perturbation. *Not recoverable* by any batch-statistic
  operator (even oracle direction). *Evidence:* 4G. *Would need* per-sample side information.
- **R2 — task-coupled:** nuisance subspace overlaps the task subspace. *Not recoverable by erasure*; needs a
  task-preserving inverse that may not exist. *Evidence:* 4B (natural), PC1 (oracle-subspace erase fails).
- **R3 — under-identified from $X$:** learned reliance / concept shift. *Not identifiable* from target-$X$ alone;
  needs labels, paired trials, or a randomized/acquisition contract. *Evidence pointers:* PC2 (future), and the
  sibling directions (concept-shift certification requires paired/minimal information; source-only competence does
  not transport — the repository's measurement$\to$control gap).

## 3. What the theory licenses (claim hygiene)
- **Allowed:** "A batch-constant (first-moment deterministic) shortcut is recoverable by a target-$X$-only mean
  alignment, up to conflation with the genuine domain-mean shift; a per-sample (second-moment stochastic) shortcut
  is not invertible by any target-$X$-only batch-statistic operator, even with the true direction; erasure is a
  valid repair only when the nuisance subspace is task-separable. Repairability is therefore a *class* property of
  the corruption × available information, not a property of the shortcut's mere detectability."
- **Forbidden:** "FSR solves shortcut repair"; "all identifiable shortcuts are repairable"; "second-moment
  shortcuts are unconditionally unrepairable" (they are un-repairable *by batch-statistic operators from
  target-$X$*; per-sample side information or training-time intervention is out of scope, not proven impossible);
  presenting the characterization as a proven universal theorem rather than a characterization grounded in the
  controlled constructions.

## 4. Why this deepens the paper
The empirical Result 4 ("first-moment repairable, second-moment not") becomes a **principled boundary**: a
deployable target-unlabeled repair inverts only batch-deterministic corruptions that are task-separable, and the
open cases (per-sample stochastic, task-coupled, under-identified) are exactly where FSR *refuses* or requires a
**stronger information contract**. This is the same discipline as the sibling directions — information available
$\to$ what is identifiable $\to$ what action is licensed — now stated for shortcut repair. It reframes the
contribution from "we scoped a repair" to "we characterized the recoverability boundary of functional shortcuts."
