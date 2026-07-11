# STAR_01A Blind Six-Cell Training Completion Readout

Report date: 2026-07-11

This is a training-integrity, provenance, and immutable-checkpoint-closure readout.
All six approved STAR_01A continuation cells completed their fixed final step.
No FACED target sample, target label, target class distribution, or target metric was read or computed.
No checkpoint was selected using source_val or target information.
No scientific efficacy, transfer, or mechanism result is reported here.
The separate source-only task audit has not yet run.
STAR_01B all-cell target scoring remains blocked pending PM review.

## 1. Formal status

```text
STAR_00_PROJECT_CHARTER: PASS
STAR_00A_DESIGN_AND_RED_TEAM_PREFLIGHT: PASS
STAR_H200_ARTIFACT_SUPPLY: AVAILABLE_IMMUTABLE
STAR_00B_REAL_PATH_PREFLIGHT: PASS
STAR_00C_LAUNCH_LOCK_AND_PERSISTENCE: PASS

STAR_01A_BLIND_SIX_CELL_TRAINING: COMPLETE
STAR_01A_EXECUTION_INTEGRITY: PASS
STAR_01A_FINAL_CHECKPOINT_IMMUTABLE_CLOSURE: PASS
STAR_01A_SOURCE_ONLY_TASK_AUDIT: NOT_RUN_SEPARATE_AFTER_CLOSURE

STAR_01B_ALL_CELL_TARGET_SCORING: BLOCKED
STAR_MANUSCRIPT_CLAIM: FORBIDDEN
S2P_PHASE_B: INDEPENDENT / UNCHANGED
H2CMI_AND_OACI: PROTECTED / UNCHANGED
```

The `PASS` classifications in this report apply only to execution integrity and immutable artifact closure. They are not scientific positive gates.

## 2. Executive disposition

- Six of six approved cells completed exactly 3,750 optimizer updates.
- Six of six cells persisted exactly 3,750 telemetry rows; the total is 22,500 rows.
- Six of six fixed final-step checkpoints passed strict reload.
- Every recorded loss, gradient norm, and parameter-delta norm was finite.
- The paired immutable H200 start SHA remained unchanged before and after every run.
- All six training error logs and the closure error log are empty.
- The `afterok` closure completed and published six SHA-named, mode-`0444` checkpoint payloads.
- B/C/D preserved the frozen optimizer-update, batch-count, and scheduler-step contract.
- B/C/D were not strict FLOP-matched, wall-clock-matched, or GPU-hardware-matched.
- FACED source_val and target_test sample reads were zero in every training cell.
- Target tensors created and target metrics computed were zero.

Therefore STAR_01A is ready for PM review of training completion and for the separately gated source-only audit. It is not authorized for target scoring by this report.

## 3. Scope actually executed

The array contained only the six approved continuation cells:

| Array task | Cell | Seed | Continuation allocation |
|---:|---|---:|---|
| 0 | `H200_SSL_CONT_s0` | 0 | 3,750 full-reconstruction SSL updates |
| 1 | `H200_SSL_CONT_s1` | 1 | 3,750 full-reconstruction SSL updates |
| 2 | `H200_STAR_TRUE_s0` | 0 | 3,000 common SSL + 750 true-label anchor updates |
| 3 | `H200_STAR_TRUE_s1` | 1 | 3,000 common SSL + 750 true-label anchor updates |
| 4 | `H200_STAR_SHUFFLED_s0` | 0 | 3,000 common SSL + 750 frozen shuffled-label anchor updates |
| 5 | `H200_STAR_SHUFFLED_s1` | 1 | 3,000 common SSL + 750 frozen shuffled-label anchor updates |

`H200_BASE_s0/s1` were not retrained. H500, H1000, H2000, released, and random references were not trained or scored. No additional variant, ratio, learning rate, label budget, layer policy, task head, or checkpoint-selection rule was introduced.

## 4. Execution authority and dependency provenance

| Field | Frozen value |
|---|---|
| Project branch | `project/star-task-anchor` |
| Approved execution commit | `0cdcbfed1bb8f783f295e252476a6432b8eb6ba6` |
| Remote execution HEAD at launch/closure | `0cdcbfed1bb8f783f295e252476a6432b8eb6ba6` |
| Frozen S2P merge-base | `a9134eb5eb7f8486a5e1ee41831823dab39381ed` |
| Attempt | `attempt_01` |
| Approval phase | `STAR_01A_BLIND_TRAINING` |
| Approval manifest hash | `b8e7a4f2f90c8043b616d8b6209a065769cf573825b5bd5acbbe0a08d2e45b67` |
| Approval file SHA-256 | `01b8118be1ec08a3b32660693e8ced471d27a88e419ac5836f444476b1242eb9` |
| Array job | `893038`, tasks `0-5` |
| Closure job | `893039` |
| Closure dependency | `afterok:893038` |
| Launch receipt hash | `3ee68baadfe389e957d655ecf3c2a3283aed1cac1a5ed80d7abcc8233178a548` |
| Launch receipt file SHA-256 | `ba983707e897aa440ebb9ee61dd154852b127ab703c087b9eb33d2f3ba8e9b39` |
| Target scoring at launch | `BLOCKED`; no target job submitted |

The approval bound the exact six cells, fixed `attempt_01`, 3,750 optimizer steps, final checkpoint step 3,750, the clean execution commit, and the input/code artifact hashes. All six execution manifests report the approved branch, commit, approval hash, clean tracked worktree, and blocked target-scoring state.

The STAR branch remains based on the frozen S2P dependency commit. The execution-commit diff from that dependency contains only `star_eeg/` and `results/star/` paths. It contains no `docs/S2P_*`, `results/s2p_*`, `h2cmi/`, or `oaci/` modification.

## 5. Frozen training configuration

| Setting | Executed value |
|---|---|
| Optimizer steps | 3,750 per cell |
| Primary checkpoint | Fixed final step 3,750 |
| Batch size | 64 |
| Optimizer | Fresh AdamW |
| Base learning rate | `5e-4` |
| Weight decay | `5e-2` |
| Scheduler | Step-wise cosine, eta-min `1e-5` |
| Gradient clipping | Norm `1.0` |
| SSL mask ratio | `0.5` |
| SSL loss | Masked MSE, mean reduction |
| Precision | FP32; mixed precision disabled |
| Model mode | Train |
| `zero_grad` | `set_to_none=True` |
| CBraMod parameter count | 4,924,000 |
| Temporary task-head parameters | 57,609 (`6400 -> 9`) |
| Checkpoint cadence | 750, 1500, 2250, 3000, 3750 |

The temporary head was instantiated under the same registry in B/C/D. It remained unchanged in both B cells, was updated in C/D, and is marked for discard before frozen evaluation.

## 6. Per-cell runtime record

| Task | Cell | Host | Slurm partition / GPU | Steps | Wall time | Peak GPU memory | Result |
|---:|---|---|---|---:|---:|---:|---|
| 0 | `H200_SSL_CONT_s0` | `node32` | A40 / NVIDIA A40 | 3,750 | 3,824.706 s (63.75 min) | 14,745,200,128 B (13.73 GiB) | COMPLETE |
| 1 | `H200_SSL_CONT_s1` | `node32` | A40 / NVIDIA A40 | 3,750 | 3,856.095 s (64.27 min) | 14,745,223,680 B (13.73 GiB) | COMPLETE |
| 2 | `H200_STAR_TRUE_s0` | `node51` | L40S / NVIDIA L40S | 3,750 | 3,195.026 s (53.25 min) | 14,816,915,968 B (13.80 GiB) | COMPLETE |
| 3 | `H200_STAR_TRUE_s1` | `node52` | L40S / NVIDIA L40S | 3,750 | 3,402.124 s (56.70 min) | 14,816,915,968 B (13.80 GiB) | COMPLETE |
| 4 | `H200_STAR_SHUFFLED_s0` | `node55` | A100 / NVIDIA A100-PCIE-40GB | 3,750 | 3,590.857 s (59.85 min) | 14,820,925,440 B (13.80 GiB) | COMPLETE |
| 5 | `H200_STAR_SHUFFLED_s1` | `nodeaudible01` | A100 / NVIDIA A100-SXM4-40GB | 3,750 | 7,427.194 s (123.79 min) | 14,818,451,968 B (13.80 GiB) | COMPLETE |

The summed per-cell wall time is 25,296.002 seconds, or approximately 7.03 GPU-hours. This is an operational accounting number across heterogeneous GPUs, not a FLOP estimate or a matching criterion. The longer D seed-1 wall time did not produce an integrity failure and was not used for selection.

### 6.1 Scheduler amendment and hardware boundary

The approved Slurm source file, SHA-256 `57a8714a7fc0053c78611df977af3490f0831b3469038ffad35147eb1bfc6ee3`, originally requested the A40 partition. After task 0 had completed and task 1 was running, the PM explicitly directed that A40, A100, H100, and L40S all be eligible and that Slurm choose among them. The still-pending tasks 2-5 were updated in the Slurm controller to:

```text
Partition=A40,A100,H100,L40S
```

No task was canceled, requeued, resubmitted, overwritten, or moved to a new attempt. The approved `.sbatch` source was deliberately not edited while the array and closure were active, so every runner continued to validate the same approval-bound source hash. Slurm selected L40S for C and A100 for D; H100 was eligible but not selected.

This operational amendment did not change data, optimizer, schedule, update count, model scope, checkpoint endpoint, or firewall behavior. It does mean that this execution is not GPU-hardware-matched: B ran on A40, C on L40S, and D on A100. Hardware allocation therefore happened to align with variant. The frozen protocol did not require hardware matching, and FP32 was used throughout, but this fact must remain visible in any later scientific review; this report does not claim cross-hardware bitwise equivalence.

## 7. Compute, schedule, and stream contract verification

The executed comparison is optimizer-update-, batch-count-, and scheduler-step-matched. It is not strict FLOP-matched:

- B executed 3,750 full-reconstruction updates per seed.
- C/D each executed 3,000 full-reconstruction updates and 750 anchor updates per seed.
- Every cell executed 3,750 encoder updates and 3,750 optimizer steps.
- The learning-rate trace over all 3,750 steps was byte-identical in all six cells; its derived trace hash is `bc982451f9e1324179a705ec25b0bb0f0421f1588377d91a905fa6c9410e2812`.
- B's semantic schedule hash is `4908a1445454a3ea1fe393b626e324b41dd48207ff2f0241ba2eae954568056f` because every anchor slot is replaced by SSL.
- C/D share semantic schedule hash `831f133c7fcd49a22be01dbc78d2bce7757768e1d77ffa4e9d134812a35d3237`.
- The CBraMod model-update scope hash is equal across B/C/D within each seed.
- The initial model-state hash is equal across B/C/D within each seed:
  - seed 0: `f9c9261308eef1918cd3b4aa68be4ba21a7eb89bfa7cbedf80aaa60fbf0f9be9`
  - seed 1: `00dba9d6f2602b4bdc499a1cbc2dec7aca7a2fa3629d673f802fa6c0ece11408`
- Common SSL batch IDs and tensor hashes are equal across B/C/D within seed.
- The aggregate stream hash is equal across B/C/D within seed:
  - seed 0: `e4b0e4a0b0ff296c07a9b0596f7eb7fb8cbd4174fc0ff4b50b4c86125eac81ac`
  - seed 1: `eabf5bd3fd3ce68d6d4e5f32986e3c51ff9865b27b1976328a818a946deb77ee`
- B contains exactly 750 replacement-SSL batches per seed.
- C and D contain exactly 750 anchor batches per seed.
- C/D anchor sample IDs, batch boundaries, order, and X-tensor hashes match for all 750 slots per seed.
- C/D anchor-label hashes differ in all 750 of 750 anchor slots per seed.
- B's temporary head remained unchanged; C/D temporary heads changed through finite gradients and are excluded from final frozen evaluation.

These checks establish the frozen allocation and semantic-control contract. They do not establish a scientific benefit for any variant.

## 8. FACED and target firewall audit

| Variant cells | FACED source_train reads per cell | source_val reads | test/target reads | Target tensors | Target metrics |
|---|---:|---:|---:|---:|---:|
| B: `H200_SSL_CONT` | 0 | 0 | 0 | 0 | 0 |
| C: `H200_STAR_TRUE` | 48,000 | 0 | 0 | 0 | 0 |
| D: `H200_STAR_SHUFFLED` | 48,000 | 0 | 0 | 0 | 0 |

Each C/D read count is exactly 750 anchor batches times batch size 64. Every loader access audit reports `PASS`, with zero non-source reads. No source_val sample was used for gradients or checkpoint selection. No target process was submitted in the blind training chain.

The source-only task audit is a separate post-closure process and is recorded in the completion matrix as:

```text
NOT_RUN_SEPARATE_AFTER_CLOSURE
```

This pending audit status does not invalidate training completion, but it blocks task-gate reporting and any L4/L5/L6 interpretation. A future task-gate failure may suppress mechanism interpretation for a cell; it must not remove that cell from the frozen one-shot all-cell target-scoring universe. Only an integrity or firewall failure may block scoring of a cell.

## 9. Per-cell integrity and immutable checkpoint closure

All six `completion.json` payloads report every required check as true:

```text
all_gradients_finite
all_losses_finite
all_parameter_deltas_finite
checkpoint_telemetry_hash_verified
exact_telemetry_rows
final_checkpoint_strict_reload
final_step_exact
formal_final_step_3750_if_scientific
no_target_data_access
no_temporary_files
run_summary_hash_verified
source_checkpoint_sha_unchanged
telemetry_hash_verified
```

| Cell | Immutable H200 start SHA-256 | Final/immutable SHA-256 | Bytes | Mode | Strict reload |
|---|---|---|---:|---:|---|
| `H200_SSL_CONT_s0` | `64977656005c6ac848af317caa48215eb50c780c869e8cebc930cc6bc5c15e63` | `596ef720bc41849f441e57c143c4c0ada409238810f10d8323fe7be3af4df110` | 59,559,580 | `0444` | PASS |
| `H200_SSL_CONT_s1` | `125d0c393e16a5d782125117d0521b79a615552d2ffc30e7a2ddb7a272103736` | `ac6a786b3e78c92c41d5b1ae8bb1e2ee46c19d1a6f52aef0c4ba01e85badfb44` | 59,559,580 | `0444` | PASS |
| `H200_STAR_TRUE_s0` | `64977656005c6ac848af317caa48215eb50c780c869e8cebc930cc6bc5c15e63` | `f689e23c58a79aab036041904ae226b1a748bc8b284890c9226921c04c7f328c` | 60,022,096 | `0444` | PASS |
| `H200_STAR_TRUE_s1` | `125d0c393e16a5d782125117d0521b79a615552d2ffc30e7a2ddb7a272103736` | `fe7028941b73025eb08758f7e6d7595904c725b8917de49d351046d3f669757d` | 60,022,096 | `0444` | PASS |
| `H200_STAR_SHUFFLED_s0` | `64977656005c6ac848af317caa48215eb50c780c869e8cebc930cc6bc5c15e63` | `d4ca6609cb5dd09fbdd4aa9f8571059ce75f6be951049f71ea834396182b5ec1` | 60,022,096 | `0444` | PASS |
| `H200_STAR_SHUFFLED_s1` | `125d0c393e16a5d782125117d0521b79a615552d2ffc30e7a2ddb7a272103736` | `ca088ae5f94b977884c80f7e5a0771f49dca617775348b27343f3232848979ac` | 60,022,096 | `0444` | PASS |

The closure independently verified that source SHA values remained stable during copy, destination SHA values matched the training final checkpoints, all payloads strict-reloaded, and no temporary file remained.

Closure status:

```text
phase: STAR_01A_FINAL_CHECKPOINT_IMMUTABLE_CLOSURE
status: PASS
closed_cells: 6/6
all_3750_steps: true
all_3750_telemetry_rows: true
all_copy_sources_stable: true
all_final_checkpoints_strict_reload: true
all_immutable_shas_match: true
all_integrity_finite: true
all_sources_unchanged: true
no_target_data_used: true
no_temporary_files: true
target_metrics_computed: false
target_scoring_allowed: false
```

The closure go/no-go canonical hash is `87f34d5f4099157f7e4a25cbb234bf84cac2ed9ba99a6759d643bc0020dae911`.

## 10. Full per-cell provenance hashes

The config hash includes the intended variant and model seed, so it is expected to differ across cells. The stream hash is intentionally shared across B/C/D within a seed.

### `H200_SSL_CONT_s0`

```text
telemetry_sha256:          abc1bb13ae4a76bffbb12032fb9a6ec6a7d8c796b9e8c8a268b5421d82ef5fc3
run_manifest_hash:         b7c112e8ce45e938d598bacbbab798a05866fc8a479138c47eb911faf881ae42
execution_manifest_hash:   f8bf4528396616ad86f173c3c13bc050a7cceeec645e726740a3c5b06ff44dce
run_summary_hash:          c1f6250588178ceda4fb4c14a5a9873310c0a66e812fc8fe7819baa6bb6d3daf
completion_hash:           d975ac0555078ff8f10756c3ac5ca9560b1cb66f711a9f04d64329fd0d680896
config_hash:               6029ffd3ad7581bdce4d1659c952a6f5246fbf4fc9a3e932b98d972f7acb0f32
stream_hash:               e4b0e4a0b0ff296c07a9b0596f7eb7fb8cbd4174fc0ff4b50b4c86125eac81ac
```

### `H200_SSL_CONT_s1`

```text
telemetry_sha256:          b058f2b69772329c82a257375259a5118f375e1e2f676cc9685713d82a60087f
run_manifest_hash:         94f027eb43600d73a4223d9f663bcb1ae6c4357d2acd0fbdd1b2407d13cda1a6
execution_manifest_hash:   14b7a1b8872ace7d982b304855906e23b22657dddb03aa8822c248ac6b16ba3b
run_summary_hash:          f3b4c9251a7e61ff1f8f755cf84c7b75da11554dbea4e71a4ba428ba69bba2ae
completion_hash:           170e853b22440e906a100ae97e2c7f31484d28bf3666d799e36df00e2c887ddc
config_hash:               3831fd6f58b2fd890ca18ffcd66c90ebba27f4c4545061ce2cce6b7d2f406277
stream_hash:               eabf5bd3fd3ce68d6d4e5f32986e3c51ff9865b27b1976328a818a946deb77ee
```

### `H200_STAR_TRUE_s0`

```text
telemetry_sha256:          e8a2aa8b49be30569b4d1f9ed230fe20cc617a9d768d8bca537743aa94180277
run_manifest_hash:         beedc7d6dedf577d17bd8c6f58e5828f8058e920278018e0d7ba056502eb6299
execution_manifest_hash:   8cc475bf53af40f7bf4374167125f685f4a093c04e8a42a1d8d1e8330e93b634
run_summary_hash:          0a064413480efc02200c9222a9d9b359fd46cd8fdd6b26c0ae797896e126e240
completion_hash:           f8deae2990aad02664c584c70bdd0634a5cf968fce696a606a71910fd5b86ce0
config_hash:               4bc2260179a90dd0ba2a9a3d4d81f2b41a26a35cc7a246236f40b7e469f8a627
stream_hash:               e4b0e4a0b0ff296c07a9b0596f7eb7fb8cbd4174fc0ff4b50b4c86125eac81ac
```

### `H200_STAR_TRUE_s1`

```text
telemetry_sha256:          b598feec4ade79e5eda2f9f49ccb6931571845cd712cf682cc17f57b9c95b163
run_manifest_hash:         987bff045dd3022186d87ae7c73db999f68639231b346f4efa6e873c7569976d
execution_manifest_hash:   37bcd965a4d5c6edf61c712270311c2fed27ee690e7de08b325294d8471edd06
run_summary_hash:          805e0333f8041a365bd58460f64f1791c1cb064065837f5e7a907725943ad74f
completion_hash:           773468f0665b17cc800adf064d5747e4ea2311f8fd09b1edb708c2db252f7e6d
config_hash:               ba3bb5f05880b0672647187a62ccd5542cd7c325e3041c4481964dd4d15285ec
stream_hash:               eabf5bd3fd3ce68d6d4e5f32986e3c51ff9865b27b1976328a818a946deb77ee
```

### `H200_STAR_SHUFFLED_s0`

```text
telemetry_sha256:          fa7412baef6e8de8dd008ac9b1c50fd20bc2c412963ab32bf52c6315e51c4e0e
run_manifest_hash:         2487e3aae2f959b3e13e321bfb082ec6169205e0e7d9e5f8d6f876d409f8f74b
execution_manifest_hash:   3b347774eff45f0187d847d6f23dc5b2253f9f84b73dc8f645371ec4adb2e90a
run_summary_hash:          39145db8ecd03def9d8977ea22905cdc12629ec5d8549eee937ac70c8ea83ce2
completion_hash:           e895f6a30c5e5898952fb429449e4f85f12fd0aa651a136f6d7868a7f611fff6
config_hash:               04cda67b69bed7619e06ef49388c4230fc97bf67f1761cfe4064990fd87d7249
stream_hash:               e4b0e4a0b0ff296c07a9b0596f7eb7fb8cbd4174fc0ff4b50b4c86125eac81ac
```

### `H200_STAR_SHUFFLED_s1`

```text
telemetry_sha256:          2b6fb5b947205a8567e9b0783a69f0c043af1510068fd7539c1a9cf4fe9e6bb5
run_manifest_hash:         5a5d78d8adfdb8e59f1d8b62a5111c4c88adc3730ac62ec60bdfb63bfe9ed53f
execution_manifest_hash:   7a9cad7d7c8ab4ef191a046202dda5471f39a0397bad11b7f4a2a01f3c3bfbce
run_summary_hash:          23d6a0f6d4413515b9cf8cd1cc3bb46554441809177202cfc7015e61f9329162
completion_hash:           f554207495a07bbd44d33acc37a881102caa8cfd96e37928673286df437b6d11
config_hash:               3cb757ec21f1321af15f615083add72e0908d8ca52d17f3afae8b8dbaf1d05d8
stream_hash:               eabf5bd3fd3ce68d6d4e5f32986e3c51ff9865b27b1976328a818a946deb77ee
```

## 11. Closure-level artifact hashes

| Artifact | File SHA-256 |
|---|---|
| `star01_final_checkpoint_manifest.json` | `0ba53af4b3e9f3aee84217127af1570963bcf9ee9a0c3d3283532a339207ae34` |
| `star01_training_completion_matrix.csv` | `6369abc44b4143e7b07508cbcc86ea47714fecaf2516c10a171d304b3fbab5da` |
| `star01_closure_go_nogo.json` | `26239c3615636491b9099d26a4971eff9d035efa4f86b035ba8f08620cd84307` |
| Closure `completion.json` | `f09f5019c58717aee2c0ba251853de114d6ae94985edc8797ed2a5e9864458a1` |

Authoritative runtime paths:

```text
/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01/<cell>/attempt_01/
/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01_closure/attempt_01/star01_final_checkpoint_manifest.json
/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01_closure/attempt_01/star01_training_completion_matrix.csv
/home/infres/yinwang/CMI_AAAI_star_runtime/results/star01_closure/attempt_01/star01_closure_go_nogo.json
```

The closure payloads and manifests are read-only. The original training attempt directories are also frozen after completion. A rerun, if ever separately authorized for infrastructure reasons, must use a new attempt directory and may not overwrite `attempt_01`.

## 12. What this report does not establish

This report contains no FACED target Kappa, balanced accuracy, weighted F1, bootstrap confidence interval, random-floor comparison, released-reference comparison, or H500/H1000/H2000 comparison. It does not compute L1, L4, L5, or L6. It does not evaluate any positive scientific gate.

In particular, this report does not establish any of the following:

- that C is better than B;
- that C is better than D;
- that source-task semantics improve transfer;
- that task anchoring reorganizes subject/task roles;
- that H200 STAR equals or exceeds H500 or any other S2P budget;
- that subject identity is harmless;
- that STAR is a validated general method.

Even a future positive result would initially be only a two-seed positive screen under one frozen semantic permutation.

## 13. Required next gate

The next permitted operation is the separate source-only integrity/task-gate audit on the immutable final-step checkpoints. That audit may use the frozen source-only path and source_val under the existing protocol, but it may not alter a checkpoint, choose a replacement checkpoint, tune a protocol value, or omit a cell from future scoring.

After the source-only audit, the PM must review the STAR_01A completion package. Only an explicit PM approval may open `STAR_01B_ALL_CELL_TARGET_SCORING`. If opened, target evaluation must be a single all-cell operation covering A/B/C/D × seeds 0/1 plus the frozen random, released, H500, H1000, and H2000 references. Partial or seed-by-seed target screening remains forbidden.

## 14. Readout verification record

The following read-only checks were performed against the authoritative runtime tree before this report was finalized:

- `squeue` returned no live row for array `893038` or closure `893039`.
- Six cell-level `completion.json` files were present and each reported `status: COMPLETE`.
- All six telemetry payloads contained exactly 3,750 rows.
- All 13 cell-level completion checks were true for every cell.
- All six training `.err` files and the closure `.err` file had size zero.
- Six SHA-named final checkpoint payloads existed under the closure tree with mode `0444`.
- The closure go/no-go reported `status: PASS` and all ten closure checks were true.
- The closure manifest, completion matrix, go/no-go, and completion file SHA values matched the values recorded above.
- All six run summaries reported finite loss/gradient/delta state, unchanged source SHA, strict final reload, no target data use, and no target metric.
- Cross-cell stream checks confirmed paired starts, common SSL identity, C/D anchor-X identity, and true/shuffled label difference.
- The repository merge-base with the frozen S2P dependency was exactly `a9134eb5eb7f8486a5e1ee41831823dab39381ed`.
- The execution-commit diff contained only STAR paths and no protected S2P, H2CMI, or OACI path.

## 15. Final training-completion verdict

```text
Six-cell submission:                         PASS (6/6 together)
Fixed optimizer endpoint:                   PASS (3750/3750 each)
Persistent telemetry:                       PASS (3750 rows each)
Loss/gradient/parameter-delta integrity:     PASS
Immutable H200 starts unchanged:             PASS
Final checkpoint strict reload:              PASS (6/6)
SHA-named read-only final closure:            PASS (6/6, mode 0444)
FACED source-only training firewall:          PASS
Target access / target metrics:               NONE / NONE
Source-only post-closure task audit:          PENDING
STAR_01A execution-integrity disposition:     PASS
STAR_01 scientific outcome:                   NOT EVALUATED
STAR_01B target scoring:                      BLOCKED
```

STAR_01A blind training and immutable closure are complete. The evidence chain is sufficient for training-completion review. No scientific claim is authorized by this readout.
