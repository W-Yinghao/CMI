# C77 — Independent Multi-Regime Instrumented Replication Protocol

**Final gate:** `SEED3_MULTIREGIME_INSTRUMENTED_PILOT_READY_BUT_NOT_AUTHORIZED`

**Primary:** `C77-A_multiregime_seed3_seed4_replication_protocol_ready`

**Secondary active:** `C77-S1 + C77-S2 + C77-S3 + C77-S4 + C77-S5 + C77-S6 + C77-S7 + C77-S8`

## Gate-first result

- Protocol lock commit: `23f549d`; C77 protocol SHA-256 `cf3ee46645a91d7dbfb25a4ebf6fdca2641932f9e5853fdd573d4e1d7b33ea5a`.
- C78 seed-3 protocol SHA-256: `ad6f4e034318b879755ca46a719d39cfd3d3c36d7ee8478771d08778a8b71afc`.
- Real training / EEG forward / re-inference / GPU / seed-3 / seed-4 / BNCI2014_004 access: `0 / 0 / 0 / 0 / 0 / 0 / 0`.
- Exact primary regime identities recovered: `ERM + OACI + SRC`.
- Comparable 40-checkpoint trajectories per level: `OACI + SRC`; ERM is a one-checkpoint shared stage-1 anchor.
- Registered levels: `0 + 1`; full field per seed: `1,458` checkpoint-target-level units.
- Seed-3 pilot: SHA-selected target `4`, regime `OACI`, both levels, shared ERM anchors, `82` units.

## Historical regime boundary

ERM and OACI are original pre-C14 regimes. SRC is not presented as an untouched method candidate: it was introduced after C10, fixed at `smooth_temperature=0.1` before C12, and C12 falsified its source-to-target transfer. C77 uses it only as a pre-existing, target-isolated negative-control trajectory. `global_lpc` and `uniform` are recoverable but excluded prospectively to avoid unnecessary regime multiplicity.

All seven regime/engine/manifest blobs checked by red team are byte-identical to their historical commits. Historical C11 evidence also replays `target_fit_ids_empty=true` and `selector target_read=false`.

## Synthetic power

The pre-committed 486-cell grid used 400 replicates per cell in 8 content-disjoint Slurm shards:

```text
null association FPR:                   0.023627
stable local-association power:          0.979676
transport drop under heterogeneity:      0.103704
actionability drop at high multiplicity: 0.007500
```

The final contrast passes only the registered directional gate. Its `0.0075` magnitude is small and is not called material. Seed-3 must recalibrate this design effect before any seed-4 protocol is finalized.

## Compute and storage

```text
C78 pilot:        82 units, 1.458 GiB cache, 8–24 GPU-hour budget range
seed-3 full:      1458 units, 25.928 GiB cache, 108–324 GPU-hour budget range
seed-3 + seed-4: 2916 units, 51.856 GiB cache
```

The GPU numbers are conservative unmeasured planning ranges, not observed runtime. C78 P1 must measure and re-gate runtime before P2. Future GPU primary is the historically used `V100`; CPU instrumentation/analysis uses `cpu-high`, 48 cores. Availability is not authorization.

## Red team

Independent red team passed `60/60` blocking checks. It blocked one intermediate run because analysis mutated a protocol-hash-locked failure ledger; the locked bytes were restored and dynamic outcomes moved to `analysis_failure_reason_ledger.csv`. It also enforced:

- `R1_level_dimension`: C78 locks levels 0 and 1; 1458 units/seed
- `R2_ERM_trajectory_symmetry`: ERM labeled one-point shared anchor; OACI/SRC are comparable trajectories
- `R3_SRC_history`: SRC disclosed as post-C10, pre-C14 negative control; C12 falsification retained
- `R4_synthetic_multiplicity_effect`: passes locked direction only; no material-effect claim; C78 must recalibrate
- `R5_training_runtime`: compute table reports a budget range, with mandatory P1 recalibration before P2
- `R6_locked_failure_ledger`: restored locked bytes; post-compute outcomes moved to analysis_failure_reason_ledger.csv
- `R7_living_scaffold_manifest`: preserve both historical rows but replay current bytes for the other 62 immutable artifacts; C77 excludes living scaffolds from its artifact manifest
- `R8_artifact_index_cycle`: artifact_manifest.csv and large_artifact_scan.csv are index files excluded from the indexed payload set

The one nonblocking failure is the synthetic multiplicity materiality caveat above. Regression: focused_C77 23 green (job 892770), C65_C77 138 green (job 892757), C23_C77 545 green (job 892758), full_OACI 1473 green (job 892759).

## Decision

Independent instrumented training is now scientifically justified for checkpoint-field replication and cross-regime transport testing. It is **not authorized in C77**. No EEG hypothesis has replicated yet; no representation mechanism, target-population generalization, selector, checkpoint recommendation, or deployable control is claimed.

The pre-committed JSON at `C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.json` is intentionally not overwritten after compute. Post-compute gate evidence is in `C77_REPLICATION_PROTOCOL_RESULT.json` and the C77 tables.
