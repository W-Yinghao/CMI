# OACI EEG-DG Project Memory Through C80P

## Current gate

```text
C80_LABEL_BUDGET_FRONTIER_PROTOCOL_LOCKED_READY_FOR_PI_AUTHORIZATION
```

C80P completed a protocol, synthetic-calibration, implementation, and
readiness milestone. It did not execute C80E or compute a real-data
label-budget statistic. C80P is designed after C79E, prospective only to the
new C80 budget computations, and retrospective with respect to the already
observed seed-3/seed-4 fields and evaluation outcomes. It is not independent
confirmation, new-subject replication, target-population confirmation, or
external validation.

## Binding objects

```text
C79E accepted base HEAD:      dadd166
C80P protocol commit:         f5d83b3
C80P protocol SHA-256:        c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85
C80P synthetic implementation: c98e084
C80P analysis lock:           972f47c
C80P analysis-lock SHA-256:   05a99e7ccc357b90b6675756caa680fd16541358bb697fdded89351f1e7ae4a8
C80P pre-execution red team:  17946fa
C80P readiness result:        1b02454
scientific registry:          5 paths x 16 categories = 80 / 80
C80E authorization received:  false
real-data budget statistics:  0
```

The analysis lock binds the two existing 1,296-unit primary fields, physical
view identities, exact C79 P1 selector, budget grid, Q0 acquisition policy,
Monte Carlo streams, dependence and simultaneous-inference rules, endpoints,
materiality gates, and report schemas.

## Accepted C79E base

C79E closed the training-seed replication layer with strict compound
nonreplication and directional concordance:

```text
P1-M reliability:                 fail
  effect:                         0.756456
  Holm p:                         0.070039
P1-A material actionability:      pass

P2-L local association:           fail
  effect:                         0.210137
  positive trajectory cells:      32 / 32
  worst-control / Holm p:          0.092 / 0.368
P2-T transport qualification:     fail again
  fixed-kernel LOTO R2:           -0.098497
  fixed-kernel LORO R2:           -0.032944

H2R exact model:                  did not qualify again
H4R strict-source F2:             did not qualify again
H5R target-unlabeled F4:          did not qualify again
H6R:                              family-wise inactive
```

P1-A passed on both seeds, while the seed-4 compound P1 failed because P1-M
did not pass the locked family. The local P2 effect retained its positive
direction but did not pass the seed-4 blocked-control family; fixed-kernel
LOTO/LORO qualification failed on both seeds. This is gate-level training-seed
heterogeneity, not reversal, absence of all signal, or proof that construction
labels are useless.

## Frozen existing-field universe

```text
primary targets:                  [1,2,3,5,6,7,8,9]
levels:                           [0,1]
candidates per target-level-seed: 81 = 1 ERM + 40 OACI + 40 SRC
seed-3 primary units:             1,296
seed-4 primary units:             1,296
target 4:                         engineering-only, excluded from C80 primary paths
same-label oracle:                closed
construction/evaluation overlap: 0
```

Seed 3 and seed 4 share targets and raw trial identities. They are repeated
training factors, not independent subject populations. Trial rows, checkpoint
rows, Monte Carlo chains, and budget points are not independent scientific
samples.

## Availability-only budget decision

The requested finite grid was `[1,2,4,8,16,32,64]` labels per class plus
`FULL`. Before protocol hashing, an allowed availability-only audit found:

```text
minimum construction labels in any primary target-class cell: 61
budget 64 feasible for all cells:                              false
candidate/evaluation outcomes used in this decision:           0
```

Budget 64 was therefore removed mechanically before the protocol hash. The
locked primary grid is:

```text
[1, 2, 4, 8, 16, 32, FULL]
```

This is not a scientific frontier result.

## Locked policy and decision frontier

Q0 is nested, class-stratified uniform sampling without replacement. Within
each seed x target x level field, C80E must use the exact C79 P1 candidate
score:

```text
mean(midrank(bAcc), midrank(-NLL), midrank(-ECE))
```

Selection may use construction labels only. Evaluation labels may be read
only by the locked scoring stage after selection is frozen. Target ID, level,
seed, candidate identity, trial ID, and row order are keys or dependence
groups only, never predictive features.

The primary endpoint is expected held-evaluation standardized regret. A
budget is actionability-qualified only if all locked conditions pass:

```text
regret-reduction margin:          at least 0.05
simultaneous inference:           exact target sign-flip max-T across budgets
positive-target consistency:      at least 6 / 8
catastrophic-target threshold:    no target below -0.10
frontier closure:                 budget and every larger budget qualify
```

`B*_s` is the smallest seed-specific budget satisfying the closure rule.
Cross-seed stability requires both B* values, distance no greater than one
grid step, direction/materiality concordance, and the locked paired-target
heterogeneity condition.

Reliability, top-1/top-5/top-10, coverage, regime composition, and target-level
effects are mandatory secondary outputs. Reliability is not a logical
precondition for actionability. The exact top-gap/effective-multiplicity path
is descriptive moderation only and cannot rescue H2/H2R. Active acquisition,
adaptive stopping, policy search, new feature/kernel/model work, and
seed-specific tuning are outside C80.

## Monte Carlo and synthetic calibration

The smallest allowed chain count satisfying both pre-locked precision rules is
2,048:

```text
chains   Hoeffding bound   empirical p95   decision
256      0.08488           0.08203         fail
512      0.06002           0.06250         fail
1024     0.04244           0.04199         fail
2048     0.03001           0.03076         pass
```

Synthetic-only calibration passed:

```text
registered scenarios:             9 / 9
B* seed-scenario cells:           18 / 18
minimum B* recovery:              0.964844
max-T family-wise error:          0.044922
target-cluster coverage:          0.953125
correct target-level FPR:         0.033203
naive row-i.i.d. negative FPR:    0.914062
pseudoreplication trap:           detected
```

The first synthetic attempt is retained in the failure ledger. It failed
known-ground-truth signal-strength and percentile-bootstrap coverage checks.
The additive repair changed only synthetic calibration construction and the
registered target-bootstrap implementation; it changed no scientific
registry, accessed no real outcome, and introduced no outcome-dependent
decision.

## Validation

```text
C79 accepted-input hashes:        10 / 10
C79 field manifest rows:           4 / 4
C79 view-isolation rows:            4 / 4
C79 scientific replay rows:       11 / 11
C79 cross-seed replay rows:         6 / 6
pre-execution red team:            36 / 36
blocking risks:                     0
same-label-oracle access:           0
real-data budget computations:      0

focused regression:               29 passed
C65-C80P:                         344 passed, 1 conditional skip
C23-C80P:                         751 passed, 1 conditional skip
full OACI:                      1,679 passed, 1 conditional skip
```

The three C79P preauthorization-state tests were explicitly deselected because
their inverse state is covered by accepted C79E execution tests. The single
conditional skip is the finalized C78F red-team condition. No C80 registry
path was skipped: all real C80 paths remain deliberately unexecuted.

## Authorization and stop boundary

C80P authorizes nothing beyond readiness. C80E requires direct PI
authorization bound to:

```text
protocol commit:       f5d83b3
protocol SHA-256:      c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85
analysis lock commit:  972f47c
exact seed3/seed4 field and view manifests
```

If authorized, C80E is read-only existing-field analysis under Q0. It does not
authorize training, forward/re-inference, GPU use, seed 5, target-4 primary
use, BNCI2014_004, same-label-oracle work, active acquisition, feature/kernel/
model search, checkpoint recommendations, external-validity claims, or
manuscript drafting.
