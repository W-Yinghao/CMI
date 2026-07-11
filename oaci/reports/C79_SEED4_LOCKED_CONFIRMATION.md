# C79E - Seed-4 Post-Seed-3 Prospective Replication

## Final gate

```text
C79-E_seed4_does_not_replicate_either_core_pattern
```

C79E is an explicitly outcome-informed, post-seed-3 protocol that was locked
before every protected seed-4 checkpoint outcome. It tests training-seed
robustness over the same targets and raw trials. It is not pre-C78S
confirmation, new-subject replication, target-population confirmation, or
external validation.

## Field and isolation

The complete locked seed-4 engineering field passed its freeze barrier:

```text
complete units:                  1,458
primary units:                   1,296
target-4 engineering units:        162
ERM / OACI / SRC:             18 / 720 / 720
strict-source rows:            6,718,464
target-unlabeled rows:           839,808
construction / evaluation:   2,235 / 2,373
construction-evaluation overlap:       0
failed retained units:                   0
training target rows or labels read:     0
outcome-driven retention or retry:       0
same-label oracle access:                 0
```

Checkpoint, sidecar, semantic state, optimizer, genealogy, cadence, and
instrumentation replay passed for all 1,458 units. All registered numerical
identity maxima, including Wz+b/logit, softmax, hook-z, repeat-logit, and
repeat-z errors, were exactly zero. The successful phase ledger records
`8.715907` GPU-hours; the frozen external payload is `30,109,676,522` bytes
(about `28.04 GiB`).

## Seed-4 decisions

| Path | Seed-4 result | Locked decision |
|---|---:|---|
| P1-M reliability | 0.756456; raw p=0.011673; Holm p=0.070039 | inactive |
| P1-A actionability | top-1/5/10 = 0.1250/0.5000/0.6875; regret=0.110667 | material |
| P1 compound transition | measurement gate failed; actionability passed | does not replicate |
| H2R exact model | deviance reduction -8.717406; p=0.862 | does not qualify |
| P2-L local association | 0.210137; 32/32 positive; worst-control p=0.092; Holm p=0.368 | inactive |
| P2 transport | LOTO R2=-0.098497; LORO R2=-0.032944 | neither path qualifies |
| P2 compound local/nontransport | local gate failed; transport remains unqualified | does not replicate |
| H4R strict-source F2 | incremental R2=-0.096288; adjusted p=1.0 | does not qualify |
| H5R target-unlabeled F4 | incremental R2=0.010450; adjusted p=1.0 | does not qualify |
| H6R positive control | incremental R2=0.415635; raw p=0.011673; Holm p=0.070039 | family-wise inactive |

### P1: measurement and actionability remain separate

The construction/evaluation reliability estimate is large and positive, with a
target-bootstrap 95% interval of `[0.681795, 0.833763]`, but it does not pass
the locked six-path Holm family. The construction score is nevertheless
materially actionable inside this target-label diagnostic information class:

```text
construction top-1 / top-5 / top-10: 0.1250 / 0.5000 / 0.6875
random top-1 / top-5 / top-10:       0.0123 / 0.0617 / 0.1235
source standardized regret:           0.7974
construction standardized regret:     0.1107
regret reduction:                      0.6868
```

The compound P1 replication requires both P1-M and P1-A, so it is inactive.
This does not turn construction labels into a source-only, target-unlabeled,
zero-label, OACI, SRC, or deployable selector. The selected-regime composition
(12/16 OACI and 4/16 SRC, with no ERM anchor) is diagnostic only and contains
no checkpoint recommendation.

### P2: positive local structure does not pass the strict family

The local target-unlabeled association remains positive in all 32 registered
trajectory cells. Five blocked controls have max-stat p=0.002, while the
identity-only matched control yields the worst registered p=0.092. The locked
family-wise result is therefore inactive (`Holm p=0.368`). Fixed-kernel LOTO
and LORO increments remain negative and unqualified. Thus P2 does not
replicate, even though the descriptive local effect remains positive.

This is not evidence that the true association is zero. It is also not a
causal representation mechanism, W-versus-z origin result, validated target
gauge, universal nontransport theorem, or deployable control.

## Cross-seed robustness

All registered aggregate effects retain their seed-3 direction, and all gates
except P2-L/P2-overall are concordant. P2 local association changes from
`0.242656` to `0.210137`; its paired target difference is `-0.032519` with a
95% bootstrap interval `[-0.061609, 0.004581]`. The effect-difference interval
includes zero, but the locked gate changes from pass to fail. C79 therefore
records gate-level training-seed heterogeneity without claiming a significant
effect-size difference.

Seed-3 and seed-4 p-values were not combined. Shared targets and trial IDs were
handled as paired/repeated objects, and the two seeds were not treated as
independent target populations.

## Registered candidate boundaries

H2R, H4R, and H5R did not qualify again under their exact registered models.
These are candidate-specific nonqualification results. They do not establish
universal irrelevance of multiplicity, universal failure of strict-source or
target-unlabeled functions, equivalence to zero, or absence of representation
information. H6R is descriptively large but remains inactive under the fixed
family-wise correction.

## Taxonomy

```text
C79-E_seed4_does_not_replicate_either_core_pattern
C79-S3_P1_measurement_not_training_seed_robust
C79-S5_H2_exact_model_does_not_qualify_again
C79-S8_P2_local_association_collapses
C79-S10_H4_F2_does_not_qualify_again
C79-S12_H5_F4_does_not_qualify_again
C79-S15_H6_familywise_inactive_on_seed4
C79-S16_cross_seed_effects_directionally_concordant
C79-S17_material_training_seed_heterogeneity
C79-S18_target4_excluded_from_primary
C79-S19_same_label_oracle_remains_closed
C79-S20_training_seed_robustness_not_target_population_confirmation
C79-S21_no_unregistered_cross_seed_rescue
C79-S22_no_new_feature_or_kernel_campaign
C79-S23_C80_not_automatically_authorized
```

`C79-S3` and `C79-S8` denote failure of the locked seed-4 gates, not proof of a
zero underlying effect. `C79-S17` is driven by P2 gate discordance; the paired
effect-difference interval itself includes zero.

## Provenance and repairs

Direct authorization bound protocol commit `ec4834c`, protocol SHA-256
`e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587`,
field lock `35d0c65`, and analysis lock `7cebf2e` before seed-4 access.
Six additive engineering repairs are retained in the retry/repair ledger. None
changed the scientific registry or locked implementation, accessed a seed-4
scientific outcome, or introduced an outcome-dependent decision.

Authorization/lock preflight passed 24/24 checks, the scientific-result red
team passed 17/17 checks, target 4 remained outside every primary estimand,
null, and multiplicity family, and the same-label oracle remained closed.

## Stop boundary

C79E changes the training-seed robustness evidence only. It does not authorize
C80, seed 5, BNCI2014_004, new targets, a new feature/kernel/model campaign,
same-label-oracle analysis, checkpoint recommendations, deployability claims,
or manuscript drafting.

