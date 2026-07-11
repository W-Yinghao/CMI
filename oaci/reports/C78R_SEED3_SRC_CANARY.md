# C78R — Seed-3 SRC Instrumented Canary / Full Seed-3 Expansion Gate

**Final gate:** `SRC_CANARY_EXECUTED_AND_VALIDATED_FULL_SEED3_READY_BUT_NOT_AUTHORIZED`

**Primary:** `C78R-A_SRC_canary_executed_and_validated`

**Secondary active:** `C78R-S1 + C78R-S2 + C78R-S3 + C78R-S4 + C78R-S5 + C78R-S6 + C78R-S7 + C78R-S8 + C78R-S9 + C78R-S11`

## Protocol and scope

```text
protocol commit:       99f710d
execution lock commit: 750cb38
protocol SHA-256:      8f0a2dd57420a5b9aa66c4174eb10477b93e46946122c6952660aceab4d0d29d
training job:          892951
dataset/target/seed:   BNCI2014_001 / 4 / 3
regime/temperature:    SRC / 0.1
levels:                0 + 1
```

The historical C11 SRC objective/engine/plan files replay byte-exactly at commit `2555b36`. C78R did not train ERM or OACI. It loaded the two protocol-locked C78 ERM checkpoints read-only because historical SRC is a stage-2 objective initialized from ERM; OACI weights and target outcomes were unavailable to the worker.

## Training and instrumentation

```text
SRC checkpoints:                 80 / 80
level 0 / level 1:               40 / 40
checkpoint + optimizer replay:   80 / 80
target rows/labels in training:   0 / 0
source-audit rows in training:    0
GPU:                              Tesla V100-PCIE-32GB
GPU wall hours:                   0.438256
peak GPU memory:                  8.064 GiB
external payload:                 1.441 GiB

strict-source rows:              368,640
target-unlabeled rows:           46,080
Wz+b / logits max error:         0.0
softmax / hook / repeat error:    0.0 / 0.0 / 0.0
failed units:                     0
```

The source, target-unlabeled, construction, evaluation, and oracle views remain physically separated. C78R linked the existing C78 content-addressed trial inputs only after all SRC checkpoints were frozen; primary instrumentation never received label/oracle descriptors.

## Compatibility and resources

C78 and C78R match exactly on the registered `checkpoint_Wb`, strict-source, and target-unlabeled schemas. This is infrastructure compatibility, not cross-regime scientific replication.

Measured phase costs:

```text
C78 level 0 ERM+OACI: 956.852 s
C78 level 1 ERM+OACI: 938.617 s
C78R level 0 SRC:     762.718 s
C78R level 1 SRC:     754.249 s
```

The remaining 48-phase plan is phase-based, not checkpoint-count runtime extrapolation:

```text
base GPU estimate:      7.583 h
25% safety envelope:    9.479 h
storage estimate:       24.928 GiB
25% storage envelope:   31.160 GiB
C78/C78R fixed bytes:   177973609 / 239416
```

C78 did not separately time ERM and OACI; their measured context cost is therefore retained as a combined upper-bound component. CPU peak RAM is unavailable because Slurm accounting is unavailable; no estimate is substituted.

## Red team and regression

Independent red-team passed `59/59` blocking checks with `7` documented repairs/caveats. C78 artifacts replay unchanged and no report/raw payload exceeds 50 MiB.

Regression: focused_C78R 15 green (job 892991), C65_C78R 182 green (job 892992), C23_C78R 589 green (job 892993), full_OACI 1517 green (job 892994).

## Decision

C78R closes the SRC execution/instrumentation compatibility blocker. Target 4 now has `2 ERM + 80 OACI + 80 SRC = 162` retained units, and the technical full-field path is ready.

This does not authorize the remaining 1,296 units or 48 phases. It does not establish multi-regime science, measurement-control replication, SRC transfer, representation transport, an escape hatch, checkpoint actionability, selector behavior, or deployability. C78F requires a separately locked scientific/compute protocol and explicit authorization. Seed 4 remains reserved for C79 and untouched.
