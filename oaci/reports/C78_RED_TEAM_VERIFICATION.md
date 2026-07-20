# C78 Authorized Red-Team Verification

Final status: `PASS`

- Blocking checks passed: `52/52`.
- Nonblocking disclosed stress/caveat checks: `4`.
- Authorized final report existed before this red-team: `false` (the canonical result was still the no-auth schema).
- Field: `82/82`; checkpoint/optimizer independent replay: `82/82`; instrumentation failed units: `0`.
- Real identity maxima (`Wz`, softmax, hook, repeat): `0 / 0 / 0 / 0`.
- Training target rows/labels, seed 4, BNCI2014_004, SRC units: `0 / 0 / 0 / 0 / 0`.

## Repairs and claim limits

- **R1_no_auth_vs_authorized_provenance**: C78 has both a no-auth readiness baseline and a later exact-token execution Resolution: dual-mode ledger preserves commit 67bca01 and authorized jobs separately
- **R2_GPU_determinism_gate**: job 892830 failed before data because CuBLAS deterministic workspace was unset Resolution: prospective lock repaired with CUBLAS_WORKSPACE_CONFIG=:4096:8; job 892832 passed; failed attempt retained
- **R3_target_process_isolation**: the generic MOABB loader returns labels Resolution: training loaded source-train subjects only; target was provisioned only after FIELD_FROZEN and inference received an X/ID-only NPZ
- **R4_dummy_vs_real_identity**: P0 dummy identity was insufficient Resolution: authorized instrumentation checked 425088 real trial-unit rows over 82 checkpoints with all maxima zero
- **R5_trajectory_stress**: both levels have 23/40 source-risk-feasible OACI points, lambda reaches 20, and level-1 surrogate reaches -49.694 Resolution: reported as pipeline smoke stress; no stability, replication, or control claim
- **R6_CPU_peak_RAM**: Slurm accounting DB refused the post-completion query Resolution: CPU peak RAM is marked unavailable; no estimate substituted; GPU peak/runtime and storage remain measured
- **R7_ERM_OACI_asymmetry**: ERM has one anchor while OACI has 40 trajectory points per level Resolution: all tables and report keep anchors and trajectories separate
- **R8_SRC_gap**: SRC execution/instrumentation path remains unexercised Resolution: full seed-3 expansion is not ready or authorized; final gate requires PM-reviewed SRC canary/path proof
- **R9_smoke_target_outcomes**: post-freeze bAcc/NLL/ECE could be misread as checkpoint selection Resolution: smoke emits no checkpoint ID, best flag, or recommendation and carries diagnostic-only fields
- **R10_isolation_boolean_semantics**: red-team job 892850 incorrectly required every isolation-ledger boolean to be true, including safe negative fields such as target_unlabeled_contains_labels=false Resolution: the rerun checks unsafe visibility fields are false and physical-separation fields are true; the failed review attempt is retained

## Verdict boundary

C78-A is supported only as an OACI+ERM training/instrumentation canary. C78 does not establish multi-regime replication, measurement-control replication, representation mechanism, source or target-unlabeled escape hatch, checkpoint control, or seed-level confirmation. SRC remains unexercised, so the 1,458-unit field is neither ready nor authorized.
