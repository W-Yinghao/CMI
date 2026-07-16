# CMI-Trace erasure ORACLE — execution note

Redefines TOS around an EXISTENCE oracle for safely-removable subject leakage, answering the question the
prior three TOS results (high-dim redundancy / global collapse / linear-removable-but-target-flat) never did:

> Does a subspace exist that carries conditional subject leakage AND can be removed with bounded task loss?

Branch `agent/cmi-trace-erasure-oracle` (base `agent/cmi-trace-relaxation-ladder`). Env: `c84c-eeg2025-v3`
(CPU oracles + tests), `eeg2025` (GPU DGCNN training for Version 1 TTE). Reuses P0/P1 DGCNN audit npzs (stored
verified head) + regenerated EEGNet dumps. Do NOT touch H2CMI/OACI.

## The three oracles

    P*_{k,delta} = argmax_P  I(PZ;D|Y,(I-P)Z)   s.t.  I(Y;PZ|(I-P)Z) <= delta,  rank(P)<=k
    Delta_D(P) = I(deleted;D|Y,kept) = I(Z;D|Y) - I((I-P)Z;D|Y)      (subject leakage removed)
    Delta_Y(P) = I(Y;deleted|kept)  = H(Y|kept) - H(Y|Z)            (task information lost)

| Oracle | info used | question | module |
|--------|-----------|----------|--------|
| Bayes-safe (synthetic) | true joint | does a task-safe leakage subspace exist in theory? | `tos_cmi/eval/erasure_oracle_bayes.py` |
| **exact-head nullspace** (real) | source rep + stored head | is there a direction the classifier does NOT use but that carries subject leakage? | `tos_cmi/eeg/erasure_oracle.py` |
| target (upper bound) | target calibration labels | is there a target-beneficial deletion among candidates? | `exhaustive_subset_oracle` |

## Results

### 1. exact-head nullspace oracle — HEADLINE (the clean positive)
For a stored linear head, softmax(Wz+b) depends only on logit DIFFERENCES, so removing any subspace of
ker(W_c) [W_c = class-centered head] leaves probabilities + predictions ALGEBRAICALLY unchanged. Fitting the
label-conditional subject subspace INSIDE ker(W_c) and removing it (subject/fold-cluster 95% CI, both datasets):

| stratum | Δ_D removed | ~% of CMI | specific vs same-rank random | softmax err | task unchanged |
|---------|-------------|-----------|------------------------------|-------------|----------------|
| BNCI2014 ERM | +0.832 [+0.794,+0.872] | ~71% | +0.809 [+0.769,+0.850] | 2.0e-14 | ✓ (all folds) |
| BNCI2015 ERM | +0.803 [+0.777,+0.835] | ~78% | +0.765 [+0.739,+0.798] | 2.5e-15 | ✓ |
| BNCI2014 CIGL | +0.395 [+0.351,+0.439] | ~66% | +0.374 | 9.5e-15 | ✓ |
| BNCI2015 CIGL | +0.236 [+0.183,+0.295] | ~69% | +0.224 | 1.7e-15 | ✓ |

**PROVEN on real EEG, both datasets: a subspace inside the classifier's exact nullspace carries ~71-78% of
the conditional subject leakage, is subject-SPECIFIC (>>same-rank random, CIs far from 0), and is removable
with the classifier's softmax/predictions ALGEBRAICALLY unchanged (err ~1e-14).** This is "safe removable,
functionally-UNUSED subject leakage exists" — NOT "erasure improves DG". It directly EXPLAINS the confirmed
P0/P1 measurement→control gap: most CMI-measured leakage sits in the head's nullspace, so reducing it cannot
change the head's behaviour or reliance. CIGL's residual leakage is LESS removable in ker(W_c) (0.40/0.24 vs
ERM 0.83/0.80): CIGL lowers total CMI but does not specifically clear the nullspace.

### 2. EEGNet exhaustive subset oracle — the rank/existence question
Over all 2^r subsets of the source-fit subject basis, target-utility best subset (selected on T_select,
reported on T_eval). Aggregate (subject-cluster): BNCI2014 full-basis −0.011, best-SUBSET +0.003, best-PREFIX
+0.001, random −0.003 (best beats random 67% of folds); BNCI2015 full-basis over-removes, best-subset ~null.
=> Even the target-label ORACLE finds only a tiny (+0.003) benefit on BNCI2014 and none on BNCI2015: the
transductive benefit seen in the ladder (+0.019) came from fitting LEACE with TARGET geometry, not from any
source-estimated subject-direction subset. rank-ordered PREFIX underperforms arbitrary SUBSET -> generalized-
eigenvalue ordering is imperfect where a benefit exists.

### 3. synthetic Bayes leakage oracle — existence in theory
Exact Δ_D*, Δ_Y* on known Gaussian mixtures + constrained search. Separable DGP → oracle finds the pure-subject
axis (Δ_Y*=0, Δ_D*=0.88); entangled DGP → oracle returns IDENTITY (correct, not a failure). Gives the
(Δ_Y*, Δ_D*) Pareto frontier and the oracle-REGRET / safety-VIOLATION metrics for TOS.

## TOS reframed: a source-estimated approximation to the safe-erasure oracle
TOS is now evaluated by **oracle regret** Regret_D = Δ_D*(P*) − Δ_D*(P_TOS) and **safety violation**
Violation_Y = max(0, Δ_Y*(P_TOS) − delta), not first by target accuracy. On synthetic true-DGP both are exact;
on real EEG the exact-head-null oracle is the task-safety lower benchmark and the target subset-oracle the
utility upper bound, with TOS in between.

## Version 1 Train-Through-Erasure (in progress)
Post-hoc deletion may fail only because the head trained on the FULL Z. TTE inserts (I-P_0) BEFORE the head
and fine-tunes the top block + a fresh head THROUGH the erasure (lower encoder frozen), so the encoder must
route the task out of the deleted subspace. GPU pilot 898821/898822 RUNNING (DGCNN, both datasets, seed 0,
arms {full, exact_head_null, subject, random}; metrics kept-branch CMI + subject residual + target/source
bAcc). CPU smoke (3+3 ep) already shows informed arms drop kept-branch CMI 0.185→0.14 + subject residual
0.667→0.55 vs random 0.18/0.66. Success is pre-frozen as kept-CMI down + source retained ≥−0.02 + informed
beats random, NOT target accuracy alone.

## Version 1 TTE — RESULT (both datasets complete, subject/fold-cluster CI)
| dataset | arm | Δ target bAcc | Δ kept-branch CMI | Δ source bAcc |
|---------|-----|---------------|-------------------|---------------|
| BNCI2014 (9) | exact_head_null | −0.003 [−0.010,+0.005] | **−0.428** [−0.457,−0.400] | +0.000 |
| BNCI2014 (9) | subject | +0.001 | −0.337 | +0.001 |
| BNCI2014 (9) | random | −0.000 | −0.027 | +0.001 |
| BNCI2015 (12) | exact_head_null | +0.004 [−0.001,+0.008] | **−0.481** [−0.498,−0.464] | −0.001 |
| BNCI2015 (12) | subject | −0.000 | −0.300 | −0.001 |
| BNCI2015 (12) | random | −0.000 | −0.045 | −0.001 |

**VERDICT: METHOD SUCCESS on the pre-frozen criteria, but NOT a DG gain.** Training THROUGH the erasure
(exact_head_null) yields a deployed kept-branch with ~43–48% LESS conditional subject leakage (CIs far from 0),
FULLY retaining source task (Δsource ≈ 0), subject-SPECIFICALLY (informed −0.43/−0.48 vs random −0.03/−0.05,
~10–16×). This proves the encoder CAN route the task out of the deleted subspace during training — so the
post-hoc ladder failure was NOT "no safe complement exists". But target bAcc is FLAT (CIs include 0): the
deployable cleaning does not translate to a DG improvement — consistent with the confirmed measurement→control
gap. Honest positive: safe removable leakage exists AND can be suppressed during training, task-safely, but
utility ≠ removability.
