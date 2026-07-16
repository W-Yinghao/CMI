# CMI-Trace erasure-oracle — claim boundary

Reframes TOS from a "task-orthogonal selective erasure method" into a **source-estimated approximation to a
safe-erasure existence oracle**. Exploratory; does not weaken the confirmed P0/P1 or ladder results.

## CONFIRMED claims (real, cluster-CI)
1. **Safe removable, functionally-unused subject leakage EXISTS** (exact-head nullspace oracle, both datasets).
   A subspace inside the classifier's exact nullspace ker(W_c) carries **~71–78%** of the conditional subject
   leakage (Δ_D +0.83/+0.80 on BNCI2014/2015 ERM), is **subject-specific** (beats same-rank random by +0.81/
   +0.77, CIs far from 0), and is removable with the classifier's softmax/predictions **algebraically
   unchanged** (replay err ~1e-14, task bAcc unchanged at every fold). Evidence:
   `results/cmi_trace_erasure_oracle/exact_head_null_summary.csv`.
2. **This explains the measurement→control gap**: most CMI-measured leakage sits in the head's nullspace
   (functionally unused), so reducing it cannot change the head's behaviour or reliance.
3. **CIGL does not clear the nullspace**: CIGL's residual leakage is less removable in ker(W_c) (Δ_D 0.40/0.24
   vs ERM 0.83/0.80) — it lowers total CMI but does not specifically move leakage out of the safe subspace.
4. **The transductive ladder benefit is not from source subject directions**: the EEGNet subset oracle (target-
   label upper bound) finds only +0.003 (BNCI2014) / null (BNCI2015) among source-fit subject-direction
   subsets; the ladder's +0.019 came from fitting LEACE with TARGET geometry. rank-ordered PREFIX underperforms
   arbitrary SUBSET where a benefit exists (ordering is imperfect).
5. **The Bayes oracle behaves correctly**: separable synthetic DGP → finds the pure-subject axis (Δ_Y*=0);
   entangled DGP → returns IDENTITY (correct answer, not a failure).

## Manuscript-safe wording
> There exists a subspace of the frozen representation, lying entirely in the task classifier's algebraic
> nullspace, that carries most of the label-conditional subject leakage and can be removed while leaving the
> classifier's predictions exactly unchanged (a subject-specific effect, beyond same-rank random removal). This
> demonstrates that a large fraction of the measured leakage is functionally unused by the deployed classifier,
> which is why reducing it does not, by itself, change classifier reliance or transfer.

## PENDING
- **Version 1 Train-Through-Erasure** (GPU pilot running): does forcing the head to train THROUGH the erasure
  let the encoder route the task out of the deleted subspace? Success is pre-frozen as kept-branch CMI down +
  source task retained ≥−0.02 + informed beats random, NOT target accuracy alone.

## FORBIDDEN wording
- "CMI erasure solves EEG DG" / "TOS improves DG" / "subject erasure generally helps or fails".
- "the exact-head-null oracle improves target accuracy" (it does NOT change predictions — that is the point).
- "safe removable leakage implies deployable benefit" (existence ≠ utility).
- "oracle" results are NOT deployable where they use target labels (subset oracle) — clearly labeled.
