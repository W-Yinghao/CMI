# C80P - Cross-Seed Label-Budget Frontier Protocol Readiness

## Final gate

```text
C80_LABEL_BUDGET_FRONTIER_PROTOCOL_LOCKED_READY_FOR_PI_AUTHORIZATION
```

Primary taxonomy:

```text
C80P-A_post_C79_existing_field_label_budget_protocol_locked_complete
```

C80P is protocol/readiness work only. It is designed after C79E, prospective
to new C80 budget computations, and retrospective with respect to the existing
seed-3/seed-4 fields and evaluation outcomes. It is not independent
confirmation, new-subject replication, target-population confirmation, or
external validation.

## Locked objects

```text
protocol commit:             f5d83b3
protocol SHA-256:            c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85
synthetic implementation:    c98e084
analysis execution lock:     972f47c
analysis lock SHA-256:       05a99e7ccc357b90b6675756caa680fd16541358bb697fdded89351f1e7ae4a8
registry:                    5 paths x 16 categories = 80 / 80
C80E authorization received:false
real-data budget statistics: 0
```

The analysis lock binds both existing 1,296-unit primary fields, their physical
view hashes, the exact C79 P1 selector, all dependence and simultaneous
inference rules, the Monte Carlo streams, and future report schemas.

## Availability-only grid decision

The requested finite grid was `[1,2,4,8,16,32,64]` labels per class plus
`FULL`. Before protocol hashing, the permitted availability-only audit found a
minimum construction class count of 61. Budget 64 is infeasible for targets
2, 5, 6, 7, and 9 and was mechanically removed without computing a candidate
score or evaluation outcome.

The locked primary grid is therefore:

```text
[1, 2, 4, 8, 16, 32, FULL]
```

## Locked scientific object

The primary C80E object, if separately authorized, is expected held-evaluation
standardized regret under the exact C79 P1 score:

```text
mean(
  midrank(bAcc),
  midrank(-NLL),
  midrank(-ECE)
)
```

within each seed x target x level field of 81 candidates. Selection labels
come only from the physically separate construction view. Evaluation labels
are read only after selection is frozen. Target 4 and the same-label oracle are
unreachable from all primary paths.

Q0 uses nested, class-stratified uniform sampling without replacement. Active
learning, adaptive stopping, policy search, seed-specific tuning, and
evaluation-informed tie breaking are excluded.

## Frontier and inference lock

The seed-specific frontier is the target-cluster expected standardized-regret
curve. A budget must satisfy the locked 0.05 regret-reduction margin, exact
target max-T family test, at least 6/8 positive targets, no target below the
-0.10 catastrophic threshold, and all-larger-budget closure. `B*` is the first
budget satisfying that closure.

Cross-seed stability requires both seed-specific B* values, at most one grid
step of distance, materiality/direction concordance, and the locked paired
target heterogeneity condition. Seeds share targets and raw trials and are
repeated training factors, not independent populations.

Reliability, top-k, coverage, regime composition, and target-level effects are
mandatory secondary outputs. Reliability is not an actionability precondition.
S3 geometry moderation is descriptive only and cannot rescue H2/H2R.

## Monte Carlo and synthetic calibration

The smallest candidate meeting both the pre-locked Hoeffding and simultaneous
eight-cell empirical precision criteria is 2,048 chains:

```text
N=256:  bound 0.08488, empirical p95 0.08203  fail
N=512:  bound 0.06002, empirical p95 0.06250  fail
N=1024: bound 0.04244, empirical p95 0.04199  fail
N=2048: bound 0.03001, empirical p95 0.03076  pass
```

Synthetic calibration passed:

```text
frontier scenarios:             9 / 9
B* seed-scenario cells:        18 / 18
minimum B* recovery:            0.964844
max-T family-wise error:        0.044922
target-cluster coverage:        0.953125
correct target-level FPR:       0.033203
naive row-i.i.d. negative FPR:  0.914062
pseudoreplication trap:         detected
```

The first synthetic attempt is retained in the failure ledger. It failed only
known-ground-truth signal-strength and percentile-bootstrap calibration; the
repair changed no scientific registry and accessed no real data.

## Provenance and red team

```text
C79 accepted-input hashes:       10 / 10
C79 field manifest rows:          4 / 4
C79 view isolation rows:          4 / 4
C79 scientific replay rows:      11 / 11
C79 cross-seed replay rows:       6 / 6
pre-execution red-team:          36 / 36
blocking risks:                   0
same-label oracle access:         0
real-data budget computations:    0
```

## Regression

```text
focused:    29 passed
C65-C80P:  344 passed, 1 conditional skip
C23-C80P:  751 passed, 1 conditional skip
full OACI: 1,679 passed, 1 conditional skip
```

All suites have zero failures and three explicit C79P pre-authorization-state
deselections. No C80 primary registry path was skipped; real C80 paths remain
unexecuted by protocol.

## Authorization boundary

C80P does not authorize C80E. Future direct PI authorization must bind:

```text
protocol commit f5d83b3
protocol SHA-256 c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85
analysis execution-lock commit 972f47c
seed3 and seed4 field/view manifests
```

No C80P gate authorizes BNCI2014_004, seed 5, target 4 primary use,
same-label-oracle analysis, active acquisition, new feature/kernel/model search,
or manuscript drafting.
