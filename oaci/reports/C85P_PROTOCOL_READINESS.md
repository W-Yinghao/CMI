# C85P TPAMI Statistical-Decision Theory Protocol Readiness

## Final Gate

```text
C85_TPAMI_DECISION_THEORY_PROTOCOL_LOCKED_READY_FOR_PROOF_AND_SYNTHETIC_EXECUTION
```

C85P is protocol/readiness work only. It creates no C85 theorem result,
synthetic scientific result, empirical gate, active-acquisition lock, real-data
authorization, or manuscript text.

## Chronology And Identity

```text
C84A accepted HEAD:
  218d19e2ef1b4b5e7517008cbcb60d4b6bbc3e42

C85 protocol-before-implementation commit:
  2449be1c24e313922688b5e957ce6d19cb75d9d6

C85 implementation commit:
  73844601d82037cfe9b8f31cb21bc53bd9b5f334

C85 protocol SHA-256:
  af4c2cb35a6b6555d6c9ded3105eb7ad4f061ba237d3e8cc3ed6f5a18aede006

locked S0-S10 generator-contract SHA-256:
  c87fec6a6572291fad8849f6c08bea2cb3f49467e243ded1d44c1f38e3d0b297
```

The protocol, timing audit, PM addendum, and exact generator contract were
committed and pushed before any `oaci/theory` implementation existed. Git
ancestry and file-history tests enforce this order.

## C84A PM Addendum

The additive addendum records the frozen U11/MaNo action equivalence to B1:

| Dataset | B1-equivalent contexts | Total | Fraction |
|---|---:|---:|---:|
| Lee2019_MI | 175 | 176 | 0.9943181818181818 |
| Cho2017 | 160 | 160 | 1.0 |
| PhysionetMI | 607 | 608 | 0.9983552631578947 |

Cho's registered MaNo formula retains its immutable Q1/Q2 pass and Cho remains
category A. Its realized action map is nevertheless exactly the fixed B1 map
on the frozen 160 contexts, so that result does not establish incremental
realized target-unlabeled information value over B1. It does not imply that the
experiment contains no information or that another rule could not use it.

The addendum also fixes two boundaries:

- COTT Q2 passes do not imply a qualified one-label frontier.
- `C84-L4` does not imply that labels contain no value.

The frozen C84-D/C84-L4 result is unchanged.

## Formal Decision Object

C85 locks a common state/action/loss representation:

```text
theta in Theta
A = {1,...,M}
u_theta(a) in [0,1]
a*(theta) in argmax_a u_theta(a)
ell(theta,a) = max_j u_theta(j) - u_theta(a)
E = (Z,{P_theta^E})
delta: Z -> probability distributions over A
```

The finite implementation validates Markov kernels, bounded utility tables,
deterministic and randomized-rule risk, observation-only garbling, exact finite
unrestricted Bayes risk, registered-policy risk, policy gaps, and total
variation. It is a contract utility, not a completed theorem proof.

## Information And Policy Value

The protocol distinguishes:

```text
D(E):       all measurable randomized rules
Delta(E):   a registered restricted rule class
R*(E):      infimum over D(E)
R_Delta(E): infimum over Delta(E)
G_Delta(E): R_Delta(E) - R*(E)
```

This prevents an observed reversal among fixed policies from being called a
reversal of Blackwell information value. The experiment comparison and policy
approximation/optimization problem are separate objects.

## Realized Policy Use

The registered quantities are action divergence, action entropy, and
incremental fixed-policy risk value relative to `delta_0`. T3 states a future
proof obligation: almost-sure equality of realized action rules gives equal
fixed-policy loss/risk and zero realized incremental value. Its explicit
non-implication is experiment equivalence or absence of information.

## Robust Risk

The distinct risk functionals are:

```text
mean risk:
  E[L]

worst-group risk:
  sup_g E[L|g]

upper-loss-tail CVaR:
  inf_eta {eta + E[(L-eta)_+]/(1-alpha)}
  for symbolic alpha in (0,1)

distributionally robust risk:
  sup_Q E_Q[L]
```

No C84 alpha is selected. CVaR does not replace the C84 Q1 floor, and worst
target, quantile, CVaR, and worst-group risk remain distinct.

## Partial Identification

For an observation `z`, compatibility assumptions define a nonempty utility
set `U(z)`. Randomized finite-action minimax regret is locked as the LP:

```text
minimize t over q,t
q >= 0
sum_a q_a = 1
sum_a q_a(max_j u_j^k-u_a^k) <= t
for every registered extreme point k.
```

Point identification means a singleton set; a nontrivial identified set has
positive infinity-norm diameter. C85P creates no empirical utility interval.

## Theorem Registry

| ID | Target | C85P status |
|---|---|---|
| T1 | Blackwell unrestricted-risk monotonicity | OPEN |
| T2 | Restricted-policy finite nonmonotonicity counterexample | OPEN |
| T3 | Policy-collapse equivalence | OPEN |
| T4 | Two-state Le Cam regret lower bound | OPEN |
| T5 | Finite multi-candidate Fano extension | OPEN |
| T6 | Mean/tail non-equivalence counterexample | OPEN |
| T7 | Sub-Gaussian near-optimal-set selection bound | OPEN |

Classical literature status and project proof status occupy separate columns.
In particular, T4 records
`(Delta/2)(1-TV(P0,P1))` only as a candidate form. C85T must derive the exact
constant, testing reduction, randomized-action convention, and action-gap
condition. T5 may remain open. No theorem is marked `PROVED`,
`PROVED_FINITE_MODEL_ONLY`, or `COUNTEREXAMPLE` in C85P.

## Near-Tie Geometry

The protocol registers the epsilon-near-optimal set, gap-softmax weights,
Hill-2 effective size, and entropy effective size. T7's candidate union bound
is explicitly `OPEN`; it requires sub-Gaussian pairwise error scales and makes
no silent independence assumption. Raw candidate count cannot replace the
gap-weighted summaries.

## Costly Labels

One C85 label query is defined as revealing the loss vector of all frozen
candidates for one target trial. This is full-information testing, not a
one-arm-per-pull bandit. Passive stratification, active testing,
pairwise-difference variance design, disagreement acquisition, stopping, and
importance weighting are advisory future classes only. Every active-method row
has `authorized=0`, and eight prerequisites block execution.

## Synthetic Contract

S0-S10 are fixed with exact rational utilities/channels, state or group laws,
policy classes, risk functionals, seeds, sample sizes, and pass/fail criteria.
The deterministic stream is:

```text
low64(SHA256(C85_SYNTHETIC_V1|scenario_id|replicate_id))
```

The 11 validation rows state `CONTRACT_VALIDATED_NOT_EXECUTED`. The three
counterexample checks state `NOT_EXECUTED_C85T_REQUIRED`. C85P tests schemas,
seeds, finite probability contracts and fail-closed behavior with independent
unit fixtures; it does not execute S0-S10 or publish synthetic findings.

## Literature Audit

Fourteen primary or canonical sources were verified for exact scope. The
registry includes Blackwell's experiment comparison, Le Cam's decision-theory
monograph, Yu and Yang-Barron on minimax lower bounds, Rockafellar-Uryasev on
CVaR, Manski on partial identification/minimax regret, Hill on effective
diversity, Fisher-Rudin-Dominici on Rashomon sets, Kaufmann-Cappe-Garivier on
best-arm identification, Madani-Lizotte-Greiner on active model selection,
Sagawa et al. on group DRO, and Kossen et al. on active testing.

No theorem condition is imported from memory without a source row. Citation
verification does not complete the corresponding C85 project proof.

## Artifact Inventory

```text
theory modules:                 5 plus package init
materialized CSV contracts:    32
materialized contract rows:    193
locked synthetic scenarios:    11
theorem targets:                7, all OPEN
literature sources:             14
risk-register rows:             15, all closed at readiness
open failure-ledger rows:       0
focused C85 tests:              35
```

The C85 protocol, implementation, tables, tests, and reports total about
149 KiB (excluding ignored bytecode). No raw EEG, label file, logits,
checkpoint, weight, optimizer state, cache, or generated scientific result is
tracked.

## Protected Boundary

```text
EEG arrays:                       0
direct label roots:               0
candidate logits/utilities:       0
model checkpoints:                0
selector/Q0/inference calls:      0
new empirical p-values:           0
training/forward/GPU calls:       0 / 0 / 0
active acquisition execution:     0
C85 real-data locks/authorization: 0
manuscript changes:               0
```

## Verification

At implementation commit `73844601d82037cfe9b8f31cb21bc53bd9b5f334` in
Python 3.13.7:

```text
focused: 291 passed
C65:     902 passed, 1 skipped, 3 deselected
C23:   1,313 passed, 1 skipped, 3 deselected
full:  2,237 passed, 1 skipped, 3 deselected
```

All accepted stderr files are empty. The initial component-focused attempt
with the wrong hard-coded full protocol-commit SHA is preserved in the
regression report and was corrected without changing any protocol object.
`squeue` showed zero active C84/C85 jobs. `sacct` was not used.

## Next Boundary

C85T may begin only after PM review and must use the exact protocol and
S0-S10 contract. It may complete proofs, counterexamples, and synthetic
validation, but remains no-real-data work. C85E, active acquisition, new data,
new model zoos, and manuscript work remain unauthorized.
