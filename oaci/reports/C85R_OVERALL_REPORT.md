# C85R Synthetic Contract Semantic Repair Overall Report

## 1. Final Disposition

C85R completed the additive semantic-satisfiability repair and stopped at:

```text
C85_SYNTHETIC_CONTRACT_V2_SEMANTICALLY_REPAIRED_READY_FOR_C85T_PM_REVIEW
```

The precise status is:

```text
SEMANTICALLY_VALIDATED_NOT_SCIENTIFICALLY_EXECUTED
```

This report completes the C85R lifecycle record. The previously committed
readiness, red-team, and regression Markdown files were complete on disk, but
C85R lacked the project-standard standalone overall Markdown/JSON/hash bundle.
This document fills that reporting gap. It does not alter the C85R protocol,
V2 generator contract, semantic tables, theorem statuses, or scientific scope.

## 2. Scope And Hard Boundary

C85R repairs whether the locked synthetic scenarios define executable
mathematical objects. It does not execute the future scientific benchmark.

Allowed work was limited to:

```text
exact finite risk enumeration for S10;
exact support/moment/allocation/variance identities for S9;
exact covariance and pairwise-scale identities for S6/S7;
schema, seed, hash, chronology and OPEN-status replay;
static no-real-data and no-active-acquisition audits.
```

It did not perform:

```text
S0-S10 4,096-replicate scientific simulations;
proof completion or theorem-status transition;
EEG, direct label-view, candidate-array or checkpoint access;
selector, empirical inference, training or forward calls;
GPU work or active acquisition;
C85T/C85E authorization;
new data/model-zoo or manuscript work.
```

## 3. Repository Chronology

```text
accepted C85P final HEAD:
  00146d25fc15c56c771c87062bb1f2d2f262f59a

C85R protocol-before-V2 commit:
  03bb684e59e3432ae6f484c8c8a537213f52a6cd

C85R implementation commit:
  360c422f3110d61cfef5d09fead78562bb52c497

C85R readiness-report commit:
  9b1aaf07a7e73278b5d5308b8e51ce9c4586e201
```

The repair protocol, sidecar and timing audit were committed and pushed before
the V2 generator, validator, semantic tables or tests existed.

Authoritative hashes:

```text
C85P protocol:
  af4c2cb35a6b6555d6c9ded3105eb7ad4f061ba237d3e8cc3ed6f5a18aede006

historical V1 generator:
  c87fec6a6572291fad8849f6c08bea2cb3f49467e243ded1d44c1f38e3d0b297

C85R repair protocol:
  e37bb444fdd174ba4ca1f95e91d9193378f11dd0ef2aeac3e03cbf6249a34b68

operative V2 generator:
  e055c2a785374a3067ce90746a5941b39847b88a4f33e4ff8da5ca8adfde355a
```

## 4. Historical Preservation

The following objects remain immutable:

```text
oaci/reports/C85_TPAMI_DECISION_THEORY_PROTOCOL.json
oaci/reports/c85p_tables/synthetic_generator_contract.json
all 32 C85P CSV registries, 193 total rows
oaci/reports/C85P_PROTOCOL_READINESS.md
oaci/reports/C85P_FINAL_REPORT_RED_TEAM.md
oaci/reports/C85P_REGRESSION_VERIFICATION.md
```

The V1 generator is recorded as a historical schema-valid but semantically
blocked object. It was not edited in place or relabelled as if it had always
contained the V2 semantics.

## 5. Exact Blocker Classification

### 5.1 S10 contradiction

The V1 criterion required the richer experiment's registered-policy risk to
strictly exceed the coarse registered-policy risk. Exact enumeration shows
both were `11/40`; the strict criterion was false.

### 5.2 S9 incomplete experiment

The V1 metadata supplied two stratum masses and pairwise SDs, but not a joint
four-action loss vector. It therefore could not generate one query, determine
the population-optimal action, or evaluate the registered estimator.

### 5.3 S6/S7 incomplete stochastic law

The V1 geometry supplied utilities, `epsilon`, `tau`, pairwise scale and sample
size, but did not define random action-utility errors. The selection
distribution and dependence among pairwise comparisons were undefined.

These are contract-semantic blockers, not failed random realizations and not
empirical outcomes.

## 6. S10 Exact Risk Derivation

The state prior is `(1/4,1/2,1/4)`. Statewise regrets for actions 0, 1 and 2
are:

| State | action 0 | action 1 | action 2 |
|---|---:|---:|---:|
| theta0 | `0` | `3/5` | `1` |
| theta1 | `7/10` | `0` | `4/5` |
| theta2 | `1` | `1/2` | `0` |

Under the coarse channel, the joint prior/observation masses are:

```text
y0:
  theta0 = 1/4
  theta1 = 1/4

y1:
  theta1 = 1/4
  theta2 = 1/4
```

The unnormalized action-risk contributions are:

| Observation | action 0 | action 1 | action 2 | Bayes action |
|---|---:|---:|---:|---:|
| y0 | `7/40` | `3/20` | `9/20` | 1 |
| y1 | `17/40` | `1/8` | `1/5` | 1 |

Therefore:

```text
coarse Bayes risk:
  3/20 + 1/8 = 11/40
```

The historical rich registered rule always selected action 1:

```text
theta0 contribution:
  (1/4)(3/5) = 3/20

theta1 contribution:
  0

theta2 contribution:
  (1/4)(1/2) = 1/8

historical rich registered risk:
  3/20 + 1/8 = 11/40
```

Thus the historical strict reversal was contradicted exactly:

```text
11/40 == 11/40
```

V2 changes only the rich registered rule to always action 0:

```text
theta0 contribution:
  0

theta1 contribution:
  (1/2)(7/10) = 7/20

theta2 contribution:
  (1/4)(1) = 1/4

V2 rich registered risk:
  7/20 + 1/4 = 3/5

rich unrestricted risk:
  0

rich policy approximation gap:
  3/5

registered-policy reversal:
  3/5 - 11/40 = 13/40
```

The utility table, prior, coarse/rich channels, garbling witness, loss, sample
size, theorem targets and criterion are unchanged. The reversal is a
restricted/non-nested policy-class phenomenon only.

## 7. S9 Full-Information Loss Law

### 7.1 Population and support

The strata are `L/H`, with masses `4/5` and `1/5`. Conditional on a stratum,
`R` is Rademacher with support `{-1,+1}` and equal probabilities. One label
query reveals all four losses:

```text
loss_0 = 3/10
loss_1 = 3/10 + 1/20 + sigma_h R
loss_2 = 13/20
loss_3 = 17/20

sigma_L = 1/50
sigma_H = 1/5
```

Exact support:

| Stratum | R | Joint probability | loss 0 | loss 1 | loss 2 | loss 3 | loss1-loss0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L | -1 | `2/5` | `3/10` | `33/100` | `13/20` | `17/20` | `3/100` |
| L | +1 | `2/5` | `3/10` | `37/100` | `13/20` | `17/20` | `7/100` |
| H | -1 | `1/10` | `3/10` | `3/20` | `13/20` | `17/20` | `-3/20` |
| H | +1 | `1/10` | `3/10` | `11/20` | `13/20` | `17/20` | `1/4` |

Every support value lies in `[0,1]`.

### 7.2 Population action identity

Population mean losses are:

```text
action 0: 3/10
action 1: 7/20
action 2: 13/20
action 3: 17/20
```

Action 0 is uniquely optimal. Within both strata,
`E[loss1-loss0]=1/20`. The exact pairwise SDs are `1/50` in L and `1/5` in H.

### 7.3 Fixed allocations

For budget 64, passive proportional ideals are `256/5` and `64/5`.
Largest-remainder rounding gives:

```text
passive:
  n_L=51, n_H=13
```

Neyman weights are:

```text
p_L sigma_L = (4/5)(1/50) = 2/125
p_H sigma_H = (1/5)(1/5) = 1/25
```

Their normalized shares are `2/7` and `5/7`. Largest-remainder rounding gives:

```text
Neyman:
  n_L=18, n_H=46
```

### 7.4 Analytic variance

For the unbiased stratified estimator,

```text
Var(D_hat) = sum_h p_h^2 sigma_h^2/n_h
```

Exact values are:

| Design | Exact variance | Decimal |
|---|---:|---:|
| passive | `1327/10359375` | `0.00012809653092006034` |
| Neyman | `317/6468750` | `0.000049004830917874396` |

Neyman allocation has lower variance in this one locked scenario. This is not
a universal active-testing theorem and does not authorize active acquisition.

## 8. S6/S7 Stochastic Error Law

Both scenarios now use:

```text
u_hat_a = u_a + xi_a
xi_a iid Normal(0,pairwise_sigma^2/2)
selection = argmax_a u_hat_a
tie rule = first canonical action index
```

Since `pairwise_sigma=1/50`:

```text
Var(xi_a) = 1/5000
Var(xi_i-xi_star) = 1/2500
SD(xi_i-xi_star) = 1/50
```

For distinct non-optimal `i,j`:

```text
Cov(xi_i-xi_star, xi_j-xi_star)
  = Var(xi_star)
  = 1/5000

Corr(xi_i-xi_star, xi_j-xi_star)
  = (1/5000)/(1/2500)
  = 1/2
```

The pairwise errors share `xi_star` and are not independent. This dependence
does not invalidate a union bound, which needs no independence.

The following eight outputs are locked for each future C85T scenario:

```text
epsilon-near-optimal set;
Hill-2 effective size;
entropy effective size;
top-1 probability;
probability selected outside A_epsilon;
mean regret;
registered T7 probability bound;
Monte Carlo standard error.
```

All 16 output-contract rows have `computed_in_C85R=0`.

## 9. T7 Bound Supersession

Under the convention

```text
E exp(lambda(xi_i-xi_star))
  <= exp(lambda^2 sigma_i^2/2),
```

the open proof target is based on:

```text
selecting i implies xi_i-xi_star >= Delta_i.
```

The V2 primary target is:

```text
P(selected outside A_epsilon)
  <= min(
       1,
       sum_{i:Delta_i>epsilon}
         exp[-Delta_i^2/(2 sigma_i^2)]
     ).
```

The historical expression

```text
min(1,sum exp[-(Delta_i-epsilon)^2/(2 sigma_i^2)])
```

is retained as a looser candidate diagnostic. C85R proves neither expression;
T7 remains `OPEN` and C85T must audit event inclusion and constants.

## 10. Proof-Obligation Precision

### T3 randomized policy collapse

The required premise is equality of action kernels for
`P_theta^E`-almost every observation under every state/prior included in the
risk claim. Equality of one coupled random draw is insufficient.

### T4 two-state lower bound

The future theorem requires either unique different optimal actions or
disjoint optimal-action sets with an explicit decoder. Every nonoptimal action
in state `j` must have regret at least `Delta`. The prior is equal, observation
space common, randomized rules allowed, and total-variation convention fixed.

### T6 mean/tail non-equivalence

C85T must derive the exact CVaR alpha region for S5. The confidence parameter
must remain strictly inside `(0,1)`; endpoints are inadmissible.

No proof status changed.

## 11. Scenario Preservation Matrix

| Scenario | V2 disposition |
|---|---|
| S0 | unchanged |
| S1 | unchanged |
| S2 | unchanged |
| S3 | unchanged |
| S4 | unchanged |
| S5 | unchanged |
| S6 | additive error law and future-output schema only |
| S7 | additive error law and future-output schema only |
| S8 | unchanged |
| S9 | generative-law completion; historical query/policy/SD fields retained |
| S10 | rich registered policy only, plus audit metadata |

All theorem statuses remain:

```text
T1 OPEN
T2 OPEN
T3 OPEN
T4 OPEN
T5 OPEN
T6 OPEN
T7 OPEN
```

## 12. Semantic Preflight

Sixteen semantic checks passed:

```text
historical, protocol and V2 hash replay;
unaffected-scenario identity;
S6/S7 additive-only completion;
S10 minimal-policy repair and exact risk identities;
S9 support, moments, allocations and variances;
S6/S7 scale and covariance identities;
T7 primary-target supersession;
T3/T4/T6 precision;
deterministic seed replay;
T1-T7 OPEN replay;
zero scientific execution.
```

The seed function was evaluated as a deterministic identity only:

```text
low64(SHA256(C85_SYNTHETIC_V1|scenario_id|replicate_id))
```

No RNG was instantiated and no scientific sample was drawn.

## 13. Complete Table Manifest

| Table | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `S10_historical_exact_risk_audit.csv` | 6 | 709 | `65a215a86f349ec621b9453115d5879a57256cae117b403b31467855ee8300fd` |
| `S6_S7_noise_coupling_contract.csv` | 2 | 410 | `f26a61ee93daa38b2b063174c630a9747cd56d6bf7507ddfe0a47cb6705b83cb` |
| `S6_S7_output_contract.csv` | 16 | 1,125 | `5a7c4f416b7f16a133fb627dad685f0a55ad7358492423470b1a7e2f735c59e2` |
| `S9_allocation_contract.csv` | 2 | 313 | `28d383b4188b00de296d3e117e4657d6dcd214077c10897cc6250ea80d5db8d3` |
| `S9_analytic_variance_contract.csv` | 2 | 383 | `2b5dc8ba176922cd69619e531e20d3aefd58c5f86d3dd5192a4fe7ea72cf3b30` |
| `S9_loss_vector_law.csv` | 4 | 775 | `be85c6bf9ba409111455124ceb241e14afb0c2772e42d84a427793a27489f879` |
| `T7_bound_supersession.csv` | 2 | 338 | `9077a6eff26f484978e8dab6c51fb9bdae65530c6fc4b062ee26ac38e23569a4` |
| `artifact_identity_replay.csv` | 4 | 636 | `6762566a7cc1290fc62388adcf7a6b7e80e9cab9b1fe30a4c947e1dfe3f2c27d` |
| `deterministic_seed_replay.csv` | 11 | 1,431 | `850c39a2a49aad609141ac232d02c8f1a3ee6da9d11b81c98f874390ff0fec4b` |
| `failure_reason_ledger.csv` | 1 | 165 | `66262295378d03b5ce9de645e07efb114fbffa2bb600e08f4040ef3890076612` |
| `historical_contract_supersession.csv` | 4 | 794 | `262e710cfdac451328b410f4a923fea88589e563f699bd15cc0f2a1fa2133828` |
| `proof_obligation_precision_addendum.csv` | 3 | 884 | `10b64d6f25f4e18ccfeb48299190a215543f8890ac1c495b0c3540f7394c54f8` |
| `risk_register.csv` | 12 | 1,129 | `262324edd7d755dae7d0def125fe16cec971f61e28608f8a08f9fe72bd92a56d` |
| `semantic_satisfiability_validation.csv` | 16 | 1,584 | `3f0beacad448b93781bad324e54a108ca4121fa1250189ef2b535ae1739fe804` |
| `theorem_status_replay.csv` | 7 | 238 | `7e3924bb86b663b4d695cee9a56aa7d0228054c223d0ecd65ad0bf9ad5c38194` |

Totals:

```text
CSV tables: 15
CSV rows:   92
V2 JSON/sidecar: 2 files
```

## 14. Implementation And Test Coverage

The validator is:

```text
oaci/theory/c85r_synthetic_semantic_repair.py
```

It uses Python standard-library exact fractions and imports no NumPy, SciPy,
PyTorch, MNE, MOABB, training, selector, model or empirical-analysis module.

The two C85R test files contain 30 tests covering:

```text
historical S10 equality;
V2 S10 exact risks and minimality;
S9 support, means, SDs, allocations and variances;
S6/S7 scale and shared-star dependence;
T7 expression identity;
T3/T4/T6 precision;
all T1-T7 OPEN statuses;
historical object preservation;
protocol-before-implementation chronology;
no real imports, active authorization or C85T result;
suite inclusion and table replay.
```

## 15. Regression Evidence

All accepted runs used:

```text
implementation commit:
  360c422f3110d61cfef5d09fead78562bb52c497

environment:
  /home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact

Python:
  3.13.7

GPU:
  0
```

| Suite | Result | Seconds | stdout SHA-256 | stderr |
|---|---|---:|---|---:|
| focused | 321 passed | 6.99 | `5bf6e56ac6a143ef7e7467457effa6bf0ac2380b94bc8fadae452ca60d9060f9` | 0 bytes |
| C65 | 932 passed, 1 skipped, 3 deselected | 74.39 | `9f9a97c603362a7d1787541228d889ecf8f63e16e80af586f3deae0e77a31e39` | 0 bytes |
| C23 | 1,343 passed, 1 skipped, 3 deselected | 104.33 | `26a9ce1ec21b8a6134227338ee921b96907b53cc7110b1085732f93b2240bdfa` | 0 bytes |
| full | 2,267 passed, 1 skipped, 3 deselected | 315.50 | `673dee5461dfde332ebee8507e4ea0996e746ac6a4a9fecf7c8bff794fe121f9` | 0 bytes |

Every stderr SHA-256 is the empty-file digest:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The skip is the finalized C78F red-team test. The three deselections are the
standing C79P unauthorized-adapter checks. `squeue` showed no active
C84/C85/OACI job; `sacct` was not used.

## 16. Red Team And Failure Ledger

The final red team passed `64/64` checks with zero blockers. It covered
chronology, historical preservation, exact S10/S9/S6/S7 semantics, T7 and
proof-precision contracts, theorem statuses, execution isolation,
authorization boundaries, regressions and Git hygiene.

The risk registry contains 12 rows, all `CLOSED_AT_READINESS`. The failure
ledger contains one explicit `NONE_OPEN` row and no blocking reason.

## 17. Protected Counters

```text
S0-S10 scientific simulations:    0
project proofs completed:          0
theorem-status transitions:        0
EEG/direct-label reads:             0 / 0
candidate-array/checkpoint reads:   0 / 0
selector/empirical inference:       0 / 0
new empirical p-values:             0
training/forward/GPU:               0 / 0 / 0
active acquisition executions:      0
C85T/C85E authorizations:            0 / 0
new data/model-zoo actions:          0
manuscript actions:                  0
```

## 18. Claim Contract

Supported statements:

```text
the historical S10 strict-reversal criterion was internally contradicted;
the V2 policy-only repair produces the intended exact restricted-policy
  reversal while preserving utilities and experiments;
S9 V2 is a complete full-information loss-vector experiment;
S6/S7 V2 have an explicit coupled Gaussian estimation-error law;
the V2 contract is semantically satisfiable and ready for C85T PM review.
```

Forbidden statements:

```text
T1-T7 is proved or invalidated;
S0-S10 has produced a scientific synthetic result;
Neyman allocation is universally superior;
the T7 bound is established;
the repair authorizes C85T, C85E, active acquisition, new data/model zoos, or
  manuscript work.
```

## 19. Final Gate And Next Stage

The complete C85R disposition is:

```text
C85_SYNTHETIC_CONTRACT_V2_SEMANTICALLY_REPAIRED_READY_FOR_C85T_PM_REVIEW
```

C85T is only a possible next milestone. It requires separate PM review and
authorization and must bind the exact V2 generator hash. C85E, active
acquisition, new datasets/model zoos and manuscript work remain unauthorized.

