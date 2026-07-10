# C78 — Seed-3 OACI+ERM Instrumented Training Pilot / Full-Field Expansion Gate

**Final gate:** `PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD`

**Primary:** `C78-A_seed3_OACI_ERM_pilot_executed_and_validated`

**Secondary active:** `C78-S1 + C78-S2 + C78-S3 + C78-S4 + C78-S5 + C78-S6 + C78-S7 + C78-S8 + C78-S9 + C78-S11`

## Dual-mode provenance

```text
no-auth baseline commit:  67bca01
no-auth gate:             PILOT_READY_BUT_NOT_AUTHORIZED
authorized worker commit: 4ac865f (determinism repair 44781eb)
successful training job:  892832
```

The no-auth baseline remains evidence that prompt prose cannot trigger execution. The later exact CLI token authorized only the locked 82-unit field. Job `892830` failed its synthetic deterministic canary before any data load; the failure was retained and repaired prospectively before job `892832`.

## Execution result

```text
planned / actual units:       82 / 82
ERM anchors:                   2
OACI trajectory checkpoints: 80
SRC units:                     0
levels completed:              0 + 1
checkpoint hash replay:       82 / 82
optimizer hash replay:        82 / 82
GPU:                           Tesla V100-PCIE-32GB
GPU wall hours:                0.543639
peak GPU memory:               8.065 GiB
external payload:              1.675 GiB
```

CPU peak RAM is unavailable because the Slurm accounting database refused the post-completion query. No estimate is substituted. GPU runtime/memory, process CPU time, storage bytes, checkpoint counts, and cache rows are measured.

## Target isolation

The training process loaded exactly source-training subjects `[1,2,3,7,8,9]` (`3456` rows). It loaded zero target rows, zero target labels, and zero source-audit rows. All 82 retention decisions, checkpoint hashes, optimizer hashes, and sidecars were frozen before target/source-audit provisioning.

Post-freeze views are physically separate:

```text
strict-source input rows:        4,608
target-unlabeled input rows:       576
construction label rows:           261
evaluation label rows:             315
same-label-oracle rows:             576
```

The primary instrumentation descriptor contains no target label, split-role, evaluation, or oracle path.

## Instrumentation

```text
instrumented units:             82 / 82
strict-source cache rows:       377,856
target-unlabeled cache rows:     47,232
Wz+b/logit max error:                 0
softmax max error:                    0
hook-z max error:                     0
repeat logits/z max error:            0
failed units:                          0
```

The registered C75/C76 source and target-unlabeled functional/architecture blocks are computable. C78 does not test their predictive qualification or reopen representation-feature mining.

## Smoke-only observations

Target endpoints were opened only after freeze for pipeline and future-power sanity. No best checkpoint ID or recommendation was emitted.

```text
level 0: candidate M=41, top-two bAcc gap=0.001431, epsilon-optimal count=2
level 1: candidate M=41, top-two bAcc gap=0.016221, epsilon-optimal count=1
random top-1 baseline: 0.024390
```

The trajectory stress is material to interpretation:

```text
level 0 OACI source-risk feasible: 23/40; lambda max 20.0
level 1 OACI source-risk feasible: 23/40; lambda max 20.0; surrogate min -49.694
```

These are finite pipeline outputs, not evidence of training stability, measurement-control replication, or target control.

## Red team

Independent authorized red-team passed `52/52` blocking checks, with four nonblocking stress/caveat checks and `10` recorded repairs. Key repairs:

- `R1_no_auth_vs_authorized_provenance`: dual-mode ledger preserves commit 67bca01 and authorized jobs separately
- `R2_GPU_determinism_gate`: prospective lock repaired with CUBLAS_WORKSPACE_CONFIG=:4096:8; job 892832 passed; failed attempt retained
- `R3_target_process_isolation`: training loaded source-train subjects only; target was provisioned only after FIELD_FROZEN and inference received an X/ID-only NPZ
- `R4_dummy_vs_real_identity`: authorized instrumentation checked 425088 real trial-unit rows over 82 checkpoints with all maxima zero
- `R5_trajectory_stress`: reported as pipeline smoke stress; no stability, replication, or control claim
- `R6_CPU_peak_RAM`: CPU peak RAM is marked unavailable; no estimate substituted; GPU peak/runtime and storage remain measured
- `R7_ERM_OACI_asymmetry`: all tables and report keep anchors and trajectories separate
- `R8_SRC_gap`: full seed-3 expansion is not ready or authorized; final gate requires PM-reviewed SRC canary/path proof
- `R9_smoke_target_outcomes`: smoke emits no checkpoint ID, best flag, or recommendation and carries diagnostic-only fields
- `R10_isolation_boolean_semantics`: the rerun checks unsafe visibility fields are false and physical-separation fields are true; the failed review attempt is retained

Regression: focused_C78 29 green (job 892863), C65_C78 167 green (job 892866), C23_C78 574 green (job 892864), full_OACI 1502 green (job 892865).

## Decision

C78 validates the exact historical OACI+ERM seed-3 training and instrumentation path for one target and two deletion levels. It does not constitute multi-regime replication, measurement-control replication, cross-regime transport, source/target-unlabeled escape-hatch evidence, representation mechanism evidence, seed-level confirmation, a selector, or checkpoint control.

SRC was not exercised. Therefore the 1,458-unit full seed-3 field is not ready and not authorized. PM review must choose a prospective SRC canary or demonstrate that SRC shares the exact validated execution/instrumentation path before any expansion.
