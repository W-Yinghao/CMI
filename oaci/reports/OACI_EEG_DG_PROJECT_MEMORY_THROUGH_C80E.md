# OACI EEG-DG Project Memory Through C80E

## Current gates

```text
C80-A_stable_low_regret_label_budget_frontier_across_training_seeds
C80E_COMPLETE_C81_PROTOCOL_REVIEW_REQUIRED
```

C80E is complete. C81 is not authorized. No seed 5, BNCI2014_004,
same-label-oracle analysis, active acquisition, new feature/kernel/model search,
checkpoint recommendation, deployment claim, or manuscript drafting is
authorized.

## Authorization governance

The PI's direct statement `我明确授权C80E了` was accepted as authoritative.
No token and no repeated protocol or lock hash were required. The executor
automatically bound that statement to the repository's single operative C80R
protocol, lock, and field/view manifest set. This policy is recorded at
`3d9dd76`.

Authorization remains scope-specific. A direct statement authorizes the named
milestone and its unique current operative objects; it does not waive protocol
completeness, view isolation, or scientific claim boundaries and does not
authorize a later milestone.

## Protocol and repair chronology

The original C80P objects were safely blocked at `6c18fd4` before any budget
outcome because they lacked complete taxonomy precedence, used a mismatched
authorization-guard schema, and did not bind a real-data adapter. C80R repaired
those defects additively:

```text
C80R repair protocol:          e88a244
protocol SHA-256:              2d72eb5119056a6520fd33fc0ac14ee6270bfd573b59c36b74be6aa3dc25fe39
C80R complete adapter:         e5cb41a
C80R initial complete lock:    f19acd8
C80E authorization policy:     3d9dd76
C80E repaired preflight:       7937740
```

Authorized job `894641` froze construction-only selections, then stopped before
evaluation access because a generic descriptor verifier incorrectly required
32-cell arrays and the seven-budget label vector to have the same first
dimension. The failure had zero evaluation-label reads, zero evaluation
outcomes, zero oracle access, and zero target-4 primary use. The frozen
selection payload was not inspected.

The additive repair was locked before evaluation access:

```text
repair protocol:               c19ef34
exact descriptor verifier:     37e38d0
replacement analysis lock:     0797599
replacement lock SHA-256:      2149895865bd44b4ab8358c76848bb6774abb59d4a203b261864be0ec599ff62
authorization binding refresh: 2f1c559
```

No budget, selector, RNG stream, score, threshold, dependence rule, registry
entry, taxonomy, or report schema changed. Successful CPU job `894646` reused
the frozen construction selections without recomputation and completed all five
registered paths in four seconds.

## Frozen scientific design

```text
primary targets:  [1,2,3,5,6,7,8,9]
levels:           [0,1]
training seeds:   [3,4]
candidates/cell:  81 = 1 ERM + 40 OACI + 40 SRC
budgets/class:    [1,2,4,8,16,32,FULL]
policy:           nested stratified uniform sampling without replacement
Monte Carlo:      2,048 chains used only for numerical integration
selector:         exact C79 P1 midrank(bAcc,-NLL,-ECE) construction rule
primary cluster:  target
```

`FULL` is cell-specific and ranges from 61 to 81 labels per class. It is not a
universal numeric budget. Target 4 is excluded from every primary object; the
same-label oracle remained closed; construction/evaluation views remained
physically disjoint; trial ID and row order remained keys/clusters only.

The frontier gate is source-relative: regret reduction at least 0.05, exact
seven-budget max-T p at most 0.05, at least 6/8 positive targets, no target
below -0.10, and all-larger-budget closure.

## Primary result

All five paths ran unconditionally. Both seeds qualify at every registered
budget, giving:

```text
B*_seed3:                         1 label/class
B*_seed4:                         1 label/class
ordinal distance:                 0
registered cross-seed stability: PASS

seed3 B=1 regret:                 0.353383
seed4 B=1 regret:                 0.373705
seed3 B=1 source-relative gain:   0.426093
seed4 B=1 source-relative gain:   0.423742
B=1 max-T p, each seed:           0.042802
positive targets:                8/8 at every budget, both seeds
catastrophic targets:            0

seed4-minus-seed3 B=1 gain:      -0.002350
paired simultaneous 95% CI:      [-0.072244, 0.067543]
```

The cross-seed rule passes, but the interval is not an equivalence interval
contained inside the materiality margin. Do not call the seeds statistically
equivalent.

## Secondary results and interpretation

At B=1, top-1 is only `0.037842`/`0.038391`, top-5 is
`0.159210`/`0.164825`, and top-10 is `0.280792`/`0.294128` for seeds 3/4.
The registered source baseline is unusually weak: regret
`0.779476`/`0.797447`, worse than random expectation
`0.481982`/`0.497584`. C80-A therefore identifies a stable improvement over
the registered source baseline, not low absolute regret or reliable top-1
checkpoint recovery.

Reliability and actionability rise together under the fixed policy, but this is
concordance, not causality or a reliability prerequisite. Four seed-3 targets
and three seed-4 targets have nonmonotone curves. Every one of 16
seed-by-left-out-target analyses changes B* from 1 to 2 or 4. The exact
one-label boundary is therefore small-target sensitive.

S3 does not show stable moderation: effective-multiplicity and top-gap
correlation signs reverse across seeds, and raw candidate count is constant.
This does not rescue the twice-nonqualified H2/H2R model and does not change B*.

## Epistemic boundary

C80 was designed after C79 and is prospective only to the locked budget
computations. It reuses observed seed-3/seed-4 targets, raw trials, candidate
fields, and the fixed construction/evaluation split. It is a retrospective
existing-field design study.

The result does not establish:

```text
universal one-label sufficiency
few-label deployment
absolute low-regret checkpoint recovery
source-only or target-unlabeled selection
active acquisition value
new-subject, target-population, cohort, or dataset generality
a representation or multiplicity mechanism
an OACI or SRC rescue
```

## Integrity and regression

```text
machine result freeze:       a43aa27
scientific red-team:         be3e5c7, 32/32 PASS
final narrative report:      212d864
final-report red-team:       212d864, 30/30 PASS
lifecycle regression repair: ebc6afe
regression/memory delivery:   42a9f72

focused:   54 passed
C65-C80E:  369 passed, 1 conditional skip, 3 historical deselections
C23-C80E:  776 passed, 1 conditional skip, 3 historical deselections
full OACI: 1,704 passed, 1 conditional skip, 3 historical deselections
final stderr bytes: 0 for all four accepted jobs
```

The lifecycle regression repair changed no scientific object. Superseded
wrong-worktree and obsolete-lifecycle test attempts remain disclosed in
`c80e_tables/regression_attempt_ledger.csv`.

## Next control point

Stop for PM review. C81 may review a future external-study protocol, but C80E
does not authorize it. Any next real-data execution requires a new explicit
scope and must preserve the existing-field, small-target, source-relative, and
non-deployable interpretation of C80-A.
