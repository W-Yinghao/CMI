# C84SL Final Report Red Team

## Decision

All 64/64 blocking red-team checks passed. The audit was performed against the
operative V2 analysis lock at commit
`33075c97afd87f05d2856463c43be3246d83f95c`, SHA-256
`94c896f0f00c53441095da6225f9ac574eb4a9baa904821a5dab3f11ea76f75c`.

No real target label, real selector score, or scientific statistic was read or
computed during C84SL.

## Check matrix

| Category | Passed | Blocking evidence |
|---|---:|---|
| Protocol chronology and provenance | 8/8 | C84F report and manifest identities replay; scientific V3 preserved; operationalization, end-to-end repair, and frontier clarification precede their implementations; method registry fixed; historical initial lock preserved and marked non-operative |
| Frozen field identity | 8/8 | 1,944 target artifacts, 1,944 sidecars, 76,464 slices, 118 subjects, 9,621 registry rows, raw-manifest identity, trial-registry identity, and 3,888/3,888 artifact hashes replay |
| Label-view and process isolation | 10/10 | Label-only provisioner; exact trial alignment; class-stratified hash split; disjoint physical roots; no EEG in label files; no candidate input to provisioning; no evaluation descriptor in selection; no construction path in held scoring; immutable selection barrier; oracle path absent |
| Selection and Q0 | 10/10 | Frozen method registry; six fixed zero-label representatives; S1 fixed; score direction/tie rules fixed; 2,048 PCG64 chains; paired nested subsets; chains excluded from N; deterministic FULL across chains; construction-only Q0; complete score/rank/digest freeze |
| Evaluation and inference | 10/10 | Canonical 18,608-row schema; held-evaluation utility exact; target is principal cluster; eight-context equal weighting; six-method max-T family; 65,536 draws; Q1 gates exact; Q2 gates exact; no pooled dataset p-value; measurement endpoints cannot substitute for decisions |
| Heterogeneity and taxonomy | 8/8 | Level decisions precede averaging; panel/seed gates retained; same-method LOTO; exact 17/15/57 thresholds; same method across datasets; C84-E/D/A/B/C precedence; no hidden level-specific C under C84-C; C84-L1--L4 closure/stability exact |
| End-to-end result freeze | 6/6 | Production S0--S20 execution 21/21; all required table writers registered; coverage/regime/catastrophic outputs executable; exact schema and finite checks; manifest cross-validation; injected partial write leaves no final root |
| Authorization and repository hygiene | 4/4 | Fresh C84S authorization required; authorization record absent; training/forward/GPU/checkpoint paths forbidden; no raw EEG, labels, weights, or oversized payload added to Git |

## Additive repair audit

The initial lock was not accepted merely because its components existed. A
pre-label end-to-end audit found five readiness gaps:

1. no unified production Stage-C table/result/manifest entrypoint;
2. S12, S13, and S15 were declarative rows instead of executable production
   tests;
3. Q0 `FULL` used all trials but its order and digest were chain-dependent;
4. coverage, selected-regime, and catastrophic-failure outputs lacked complete
   standalone production paths;
5. label-frontier target/level stability semantics required an executable
   method-free budget clarification.

These gaps were found before target-label access and before authorization. The
initial lock remains immutable and unconsumed. The repair protocols were
committed first, the implementation was replaced additively, and the V2 lock
binds the complete production path.

## Isolation findings

Static inspection passed 16/16 checks over the 12 lock-bound implementation
files. It proved:

```text
label provisioner -> no candidate-artifact import
selection          -> no evaluation-view descriptor or import
evaluation         -> no selection mutation callable
analysis           -> immutable selection identity required
C84S modules       -> no training, optimizer, checkpoint-forward, or GPU path
oracle             -> no reachable same-label implementation
```

Synthetic adversarial fixtures rejected construction/evaluation overlap,
evaluation-before-freeze, target-row duplication, Monte Carlo pseudoreplication,
trial-order drift, unknown/missing schema fields, and partial final publication.

## Dependence and claim audit

The implementation treats target subject as the scientific cluster. Trial rows,
candidate units, panel/seed/level repeats, and Monte Carlo chains cannot inflate
sample size. Dataset-specific Q1/Q2 families remain separate, and no pooled
three-dataset p-value exists.

The final taxonomy preserves method identity. Different methods in different
datasets cannot support C84-A or C84-B. Level disagreement, panel/seed failure,
LOTO failure, or dataset mismatch activates C84-D before A/B/C. C84-C requires
no hidden level-specific Q1 pass.

No report text claims universal EEG validity, exact replication of the
four-class BNCI2014_001 field, universal zero-label impossibility, universal
one-label sufficiency, deployability, or causal mechanism.

## Protected-state audit

```text
target construction labels accessed: 0
target evaluation labels accessed:   0
real target-y operations:             0
real selector scores:                 0
real scientific statistics:           0
training / forward / GPU:             0 / 0 / 0
same-label oracle:                    0
C84S authorization:                  absent
C85 authorization:                   absent
```

## Final gate

```text
C84S_MULTIDATASET_LABEL_VIEWS_SELECTION_INFERENCE_IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION
```

