# S2P Phase C cross-task gate results

This document records the gate-first path validation for the frozen CBraMod cross-task audit. It is an
experimental record, not a manuscript result. Both gates used the immutable Phase-B representation contracts,
frozen encoders, source-only downstream fitting, and final-score-only target labels.

## SEED-V gate

The five gate objects completed on native 62-channel trial means. The released reference cleared the random
initialization floor:

```text
random target Kappa:    0.1382978723
released target Kappa:  0.2393617021
released - random:     +0.1010638298
gate verdict:           PASS
```

The screening checkpoints were `H200_s0=0.1382978723`, `H1000_s0=0.2925531915`, and
`H2000_s0=0.2180851064`. These single-seed gate values authorize the fleet but do not establish a budget trend.
Not every rank-4 geometry object passed the held-out captured-energy stability check; unstable object geometry
must remain uninterpretable rather than being silently removed.

## ISRUC_S3 gate

The five gate objects completed all ten official Cohort-III rotations and all three fixed downstream-head seeds.
The released reference cleared the random initialization floor:

```text
random target Kappa:    0.5270727602
released target Kappa:  0.6840139876
released - random:     +0.1569412274
gate verdict:           PASS
```

The screening checkpoints were `H200_s0=0.4341833306`, `H1000_s0=0.5767054622`, and
`H2000_s0=0.5983238376`. ISRUC remains a ten-subject directional replication; epochs are not independent
biological units and its evidence must not be pooled with SEED-V or FACED into a global p-value.

## Launch boundary

The two passing gates authorize only the corresponding ten-object frozen-readout fleets. They do not authorize
encoder pretraining, encoder fine-tuning, H4000, CodeBrain Stage-2, another downstream dataset, Phase B2, or
manuscript work. Task-gated L5 uses 200 source-validation-energy-matched random orthobases per interpretable
object and applies Holm correction across all task-gated pretrained cells within each dataset.
