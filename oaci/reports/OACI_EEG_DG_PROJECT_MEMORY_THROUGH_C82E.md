# OACI EEG-DG Project Memory Through C82E

## Current Scientific Gate

```text
C82-D_zero_label_comparison_training_seed_method_identity_or_target_heterogeneous
```

C82 is complete and awaits PM review. It does not authorize C83 or any new
scientific execution.

## Historical Boundary

C82 is a post-C81-outcome-access recovery using the exact C81 selection payload
frozen before C81 evaluation access. It is prospective only to the new C82
computation and result freeze. It does not retroactively repair C81:

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```

C80E also remains unchanged. C82 is not independent confirmation, new-subject
evidence, new-dataset evidence, external validation, or target-population
replication.

## Operative Objects And Chronology

```text
C81 final report:                 d64f16ba4cd04ac6e716b0eb2522e47ef3c8522c
C81 additive GitHub audit:        6f73bbc0ecdbe61db07e6d57cffabb98faab468d
C82 protocol:                     8b0df50b3707dbb3af4a459b6dc6de36c97d562f
C82 protocol SHA-256:             9f58c7a8e6b495a6d8f510c0d72d24ede4485908ef94bc078abe8f124b03a8f3
C82 applicability narrowing:      dc3362f
C82 implementation:               7f107f9
C82 pre-lock validation:          192b4d6
C82 lock replay tests:            5f5ba08
C82 analysis lock:                6c6739c61d362bc33df6d8b016e4cda724772a62
C82 analysis-lock SHA-256:        d5de6d6ff242b9f3d7f9c318cbdd6e1e16c509060bc14cca59292b738a75f5ce
C82P readiness:                   b1cfa00
direct C82E authorization:        5644157
pre-adapter attempt ledger:       24f5ee1
machine result freeze:            ce0564d
scientific red team:              61b3fe2
main report:                      d4c035d
regression verification:          b88c757
final-report red team:            35f87a2
```

## Frozen Selection And Views

```text
selection manifest self SHA-256:
  4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519

selection payload SHA-256:
  1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257

payload bytes:              415,284
contexts:                   32
selection methods:          19
selection recomputed:       false
field/view manifest digest: 6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
```

The successful C82 job opened 16 target-evaluation views containing 4,746
label rows only after authorization consumption and frozen-selection replay.
Construction-label content was not reopened; C80 Q0 artifacts supplied the
labeled comparators. Target 4 and the same-label oracle were not accessed.

## Execution

The direct post-readiness PI statement `授权 C82E` was automatically bound to
the unique protocol, lock, selection, and field/view digest under policy
`3d9dd76`.

Job `895213` was rejected in zero seconds by a mistyped shell-level HEAD guard
before the adapter started. It opened no payload or view and did not consume
authorization. The attempt and wrapper-only correction are preserved.

Job `895214` was the only scientific execution:

```text
state / exit:       COMPLETED / 0:0
runtime:            4 seconds
allocation:         48 CPU / 96 GiB / GPU 0
stderr:             0 bytes
selection replay:   exact
atomic publication: PASS
```

## Result Identity

```text
result SHA-256:
  d8060e6636adf7fcca7a0ace0e47bb7043676b7681569e09fb8705dcb8d5a8b7

artifact-manifest SHA-256:
  910e2ff1d8445dae262be82d417140cd44fc48be1306f2bbe5a439ec3549f0a2

method-context rows:       672 / 672
canonical fields:           16 / 16
registered tables:          23 / 23
table hashes:               23 / 23
table row counts:           23 / 23
```

## Primary Decision

```text
seed 3 category: B
  Q1: COTT / U13 passes
  Q2: no method passes

seed 4 category: C
  Q1: no method passes
  Q2: no method passes

A intersection: none
B intersection: none
LOTO preserved: 7 / 16
required LOTO:  12 / 16
final taxonomy: C82-D
```

C82-A and C82-B require the same fixed zero-label method across both seeds.
C82-C requires the registered stability condition. None applies before C82-D
precedence.

## Q1 Results

The strict-source primary comparator S1 had regret:

```text
seed 3: 0.779476
seed 4: 0.804823
```

COTT was the only Q1 pass:

```text
                         seed 3      seed 4
COTT regret             0.338641    0.465335
improvement over S1     0.440835    0.339488
simultaneous lower      0.283446    0.182099
max-T p                 0.015564    0.101167
favorable targets       8 / 8       7 / 8
worst target            0.166885   -0.076885
decision                PASS         FAIL
```

ATC, NuclearNorm, MaNo, SND, and ALine did not pass Q1 on either seed.

## Q2 Results

Frozen Q0 B=1 regret remained:

```text
seed 3: 0.353383
seed 4: 0.373705
```

No zero-label primary passed noninferiority. For COTT:

```text
                         seed 3      seed 4
COTT minus Q0 B=1      -0.014743    0.091630
simultaneous upper      0.144528    0.250901
max-T p                 0.517510    1.000000
favorable targets       6 / 8       4 / 8
Q2 catastrophic targets 2 / 8       3 / 8
decision                FAIL         FAIL
```

The favorable seed-3 pooled mean cannot substitute for simultaneous
target-cluster noninferiority.

## Objective And Measurement Separation

COTT top-k localization was:

```text
                         top1      top5      top10
seed 3                  0.1250    0.3125    0.3750
seed 4                  0.0000    0.2500    0.3125
```

COTT mean Spearman was `0.276605 / 0.184232`, and pairwise ordering accuracy
was `0.600160 / 0.568720`. These measurement relationships coexist with Q1
success only on seed 3 and Q2 failure on both seeds.

The U16 diagnostic remained secondary. Its incremental R2 was negative in all
32 contexts, with means `-0.667244 / -0.769685`. LORO is formally nonoperative;
C82 makes no cross-regime selector-transport claim.

## Target Sensitivity

All eight seed-3 leave-one-target panels lost the full-panel COTT method
identity even when another method preserved category B. Seed 4 preserved C in
seven panels; leaving out target 2 changed the category to B. This yields 7/16
preserved panels and establishes the registered target-composition instability
for the C82 comparison.

## Accepted Interpretation

The frozen-field audit supports only:

1. COTT materially improves over strict source on seed 3, but not under the
   exact seed-4 gate.
2. No registered zero-label primary is noninferior to Q0 B=1 under the locked
   simultaneous target-cluster rule on either seed.
3. No fixed zero-label method supports a common A/B result across seeds.
4. The result is method-, seed-, and target-composition heterogeneous.
5. Measurement, regret, noninferiority, localization, and target robustness
   remain distinct evidence levels.

It does not prove universal zero-label impossibility, universal one-label
sufficiency, external validity, causal mechanism, cross-regime transport,
generator rescue, or deployability.

## Verification

```text
scientific red team:   59 / 59 PASS
final-report red team: 50 / 50 PASS

focused C82: 43 passed                         job 895215
C65-C82:     460 passed, 1 skip, 3 deselected job 895221
C23-C82:     871 passed, 1 skip, 3 deselected job 895222
full OACI: 1,795 passed, 1 skip, 3 deselected job 895218
```

All accepted stderr files are empty. Diagnostic regression attempts remain in
the ledger but are not counted as accepted.

## Stop Rule

C82 is complete for PM review. No C83, seed 5, BNCI2014_004, target 4, oracle,
active acquisition, new method, new training, or manuscript experiment is
authorized.
