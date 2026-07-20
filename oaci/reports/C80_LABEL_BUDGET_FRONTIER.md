# C80 - Existing-Field Cross-Seed Label-Budget Frontier

## Final scientific gate

```text
C80-A_stable_low_regret_label_budget_frontier_across_training_seeds
```

Project-control gate:

```text
C80E_COMPLETE_C81_PROTOCOL_REVIEW_REQUIRED
```

Under the exact locked Q0 policy and source-relative materiality gate, both
training seeds have `B*=1` label per class and the registered cross-seed
stability rule passes. The precise result is a stable source-relative
regret-reduction frontier in the existing fields. It is not a universal
one-label sufficiency result and does not imply low absolute regret or reliable
top-1 checkpoint recovery.

## Epistemic status

C80 was designed after C79 outcomes and is prospective only to the locked
budget computations. It reuses the same targets, raw trials, candidate fields,
and construction/evaluation partition from seeds 3 and 4. It is a retrospective
existing-field design study, not independent confirmation, new-subject
replication, target-population confirmation, or external validation.

The primary universe contains targets `[1,2,3,5,6,7,8,9]`, two levels, two
training seeds, and 81 candidates per cell. Target 4 is excluded from every
primary object. The same-label oracle remained closed.

## Authorization and execution

The PI directly stated `我明确授权C80E了`. That statement was accepted without
a token or repeated hashes. The executor automatically bound it to the unique
operative protocol, lock, and manifest set and recorded the binding in commit
`3d9dd76`.

The first authorized job, `894641`, froze the construction-only selection
payload and then stopped before evaluation because a generic shard verifier
incorrectly required the 32-cell arrays and seven-budget label array to share
one first dimension. Evaluation-label reads and evaluation outcomes were zero.

The additive repair sequence was:

```text
c19ef34  repair protocol before evaluation access
37e38d0  exact selection field/shape/dtype verifier
0797599  replacement analysis lock
2f1c559  automatic direct-authorization binding refresh
```

No scientific registry entry, score, budget, RNG stream, threshold,
dependence rule, or taxonomy changed. Job `894646` reused the frozen selection
payload without recomputation and completed the evaluation and all registered
paths in four CPU seconds. Both jobs used CPU only; stderr for the successful
job was empty.

## Locked design

```text
budgets per class: [1,2,4,8,16,32,FULL]
policy:            nested stratified uniform sampling without replacement
Monte Carlo:       2,048 chains as numerical integration
selector:          exact C79 P1 midrank(bAcc,-NLL,-ECE) rule
primary cluster:   target
candidate field:   1 ERM + 40 OACI + 40 SRC per target x level x seed
frontier gate:     mean regret reduction >= 0.05,
                   exact seven-budget maxT p <= 0.05,
                   >= 6/8 positive targets,
                   no target below -0.10,
                   all-larger-budget closure
```

`FULL` is cell-specific. Construction counts range from 61 to 81 labels per
class; FULL is not a universal numeric budget and no interpolation between 32
and FULL is used.

## P1 - Seed-specific frontiers

| Budget | Seed 3 regret | Seed 3 reduction | Seed 3 maxT p | Seed 4 regret | Seed 4 reduction | Seed 4 maxT p |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.353383 | 0.426093 | 0.042802 | 0.373705 | 0.423742 | 0.042802 |
| 2 | 0.278209 | 0.501267 | 0.035019 | 0.300202 | 0.497246 | 0.035019 |
| 4 | 0.216602 | 0.562874 | 0.011673 | 0.232480 | 0.564968 | 0.011673 |
| 8 | 0.166310 | 0.613166 | 0.007782 | 0.181128 | 0.616319 | 0.007782 |
| 16 | 0.134536 | 0.644940 | 0.007782 | 0.144624 | 0.652823 | 0.007782 |
| 32 | 0.111871 | 0.667605 | 0.007782 | 0.117686 | 0.679762 | 0.007782 |
| FULL | 0.082794 | 0.696682 | 0.007782 | 0.110667 | 0.686781 | 0.007782 |

Every budget has positive regret reduction in 8/8 targets for both seeds, and
no catastrophic target occurs. Every budget directly qualifies, so the
all-larger-budget closure yields:

```text
B*_seed3 = 1
B*_seed4 = 1
```

The FULL regret and top-k values replay the accepted C78S/C79 full-construction
results, providing an endpoint consistency check.

## P2 - Cross-seed stability

```text
B* ordinal distance:                    0
seed4-minus-seed3 reduction at B=1:    -0.002350
paired simultaneous 95% CI:            [-0.072244, 0.067543]
direction concordance:                  PASS
registered heterogeneity gate:          PASS
registered cross-seed stability:        PASS
```

The interval contains zero but is not contained inside the materiality margin.
Accordingly, this is a pass under the exact registered rule, not an equivalence
claim about training seeds.

## S1 - Reliability and actionability

Reliability and source-relative decision utility increase together under this
fixed policy:

| Budget | Seed 3 reliability | Seed 3 reduction | Seed 4 reliability | Seed 4 reduction |
|---:|---:|---:|---:|---:|
| 1 | 0.396898 | 0.426093 | 0.375551 | 0.423742 |
| 8 | 0.685906 | 0.613166 | 0.683663 | 0.616319 |
| 32 | 0.797382 | 0.667605 | 0.796861 | 0.679762 |
| FULL | 0.828524 | 0.696682 | 0.832255 | 0.686781 |

This is concordance, not evidence that reliability causes actionability or is
a necessary actionability precondition.

## S2 - Top-k and target heterogeneity

| Seed | Budget | Top-1 | Top-5 | Top-10 | Joint-good coverage at top-10 |
|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 0.037842 | 0.159210 | 0.280792 | 0.652313 |
| 3 | 32 | 0.145630 | 0.522156 | 0.735382 | 0.903900 |
| 3 | FULL | 0.125000 | 0.687500 | 0.750000 | 0.937500 |
| 4 | 1 | 0.038391 | 0.164825 | 0.294128 | 0.699707 |
| 4 | 32 | 0.146820 | 0.459839 | 0.622375 | 0.881897 |
| 4 | FULL | 0.125000 | 0.500000 | 0.687500 | 0.875000 |

The 81-candidate random top-1/top-5/top-10 baselines are
`0.012346/0.061728/0.123457`. B=1 improves on those baselines, but top-1 remains
about 3.8%, so the result is not deterministic checkpoint identification.

The registered source baseline selects the evaluation top-1/top-5/top-10 in
zero primary cells. Its mean standardized regret is `0.779476` on seed 3 and
`0.797447` on seed 4, compared with random-policy expectations `0.481982` and
`0.497584`. The early frontier therefore reflects strong improvement over a
weak source baseline. At B=1, the selected candidate beats random expected
regret with probability `0.677063` and `0.668030`.

Four seed-3 targets and three seed-4 targets have nonmonotone target-level
curves. The raw curves and registered closure rule are retained; no smoothing
or interpolation was applied.

## S3 - Registered geometry moderation

S3 does not show stable cross-seed moderation. Raw candidate count is fixed at
81 and therefore has undefined correlation. Across budgets, effective
multiplicity correlations are mostly negative for seed 3 but positive for seed
4, while top-gap correlations are positive for seed 3 and negative for seed 4.
This descriptive sign reversal does not qualify a model, rescue H2/H2R, or
change the primary frontier.

## Sensitivity and precision

The eight-target primary result is directionally broad, but the minimal budget
is small-target sensitive. Leaving out any target changes B* from 1 to 2 or 4
in all 16 seed-by-left-out-target analyses. The B=1 maxT p-value is `0.042802`.

Thus the registered C80-A decision is valid for the locked full eight-target
field, while the exact one-label boundary is not robust to target omission.
This is a design point for a later untouched study, not a universal minimum.

## Claim boundary

C80 establishes only that the fixed construction-label policy has a stable,
source-relative material frontier in these existing fields. It does not
establish:

```text
source-only or target-unlabeled selection
few-label deployment
absolute low-regret checkpoint recovery
active acquisition value
new-target or target-population generality
external-dataset validation
a representation or multiplicity mechanism
an OACI or SRC rescue
```

No C80 result authorizes C81, seed 5, BNCI2014_004, active-learning search,
same-label-oracle analysis, checkpoint recommendations, or manuscript
drafting.
