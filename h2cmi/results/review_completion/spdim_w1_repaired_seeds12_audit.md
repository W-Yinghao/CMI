# SPDIM W1 Repaired Seeds 1/2 Execution and Three-Seed Merge Audit

- status: `pass`
- final label: Official SPDIM W1 repaired-split three-source-seed same-split baseline.
- accepted array: `892389`
- excluded zero-result array: `892385` on P100
- final_squeue_absent: `True`

## Submission and Partition History

The committed launcher default was `#SBATCH --partition=H100,L40S`. The effective accepted submission was:

```bash
sbatch --partition=A40 h2cmi/results/review_completion/slurm/spdim_w1_repaired_seeds12_8task.slurm
```

The pending array was then expanded in place, without cancellation or resubmission:

```bash
scontrol update JobId=892389 Partition=H100,A100,L40S,A40
scontrol update JobId=892389 Partition=H100,A100,L40S,A40,V100
```

- launch commit: `493f649911f354d4deaa307c80be3cb367149fce`
- working directory: `/home/infres/yinwang/CMI_AAAI_spdim_clean_a8b9368`
- array and concurrency: `0-7%4`
- resource-only override: runner, controller, config, manifest, methods, seeds, hyperparameters, and target specs were unchanged.

## Accepted Result-Carrying Tasks

| task | job | seed | shard | partition | GPU | capability | rows | stdout | stderr |
|---:|---:|---:|---|---|---|---|---:|---|---|
| 0 | 892464 | 1 | seed1_shard0 | V100 | Tesla V100-PCIE-16GB | 7.0* | 116 | pass | known_harmless_warnings_only |
| 1 | 892465 | 1 | seed1_shard1 | V100 | Tesla V100S-PCIE-32GB | 7.0* | 116 | pass | known_harmless_warnings_only |
| 2 | 892466 | 1 | seed1_shard2 | V100 | Tesla V100S-PCIE-32GB | 7.0* | 116 | pass | known_harmless_warnings_only |
| 3 | 892467 | 1 | seed1_shard3 | V100 | Tesla V100S-PCIE-32GB | 7.0* | 112 | pass | known_harmless_warnings_only |
| 4 | 892842 | 2 | seed2_shard0 | V100 | Tesla V100S-PCIE-32GB | 7.0* | 116 | pass | known_harmless_warnings_only |
| 5 | 892883 | 2 | seed2_shard1 | A40 | NVIDIA A40 | 8.6* | 116 | pass | known_harmless_warnings_only |
| 6 | 892957 | 2 | seed2_shard2 | V100 | Tesla V100S-PCIE-32GB | 7.0* | 116 | pass | known_harmless_warnings_only |
| 7 | 892389 | 2 | seed2_shard3 | A100 | NVIDIA A100-SXM4-40GB | 8.0* | 112 | pass | empty |

*Compute capability is mapped from the recorded GPU model because the runtime controller did not emit `torch.cuda.get_device_capability`; compatibility is additionally gated by absence of unsupported-architecture, kernel-image, and CUDA-initialization failures.

All tasks recorded the `icml` executable, Python 3.9.25, PyTorch 2.8.0+cu128, CUDA 12.8, MOABB 1.2.0, MNE 1.8.0, frozen checksums, seed, shard, and expected rows.

## Excluded P100 Launch

Array `892385` was canceled after its task logs reported unsupported Tesla P100 sm_60. It produced `0` accepted rows. This is a real compatibility failure and is not classified as harmless.

| task | stderr verdict | stdout sha256 | stderr sha256 |
|---:|---|---|---|
| 0 | unsupported_gpu_architecture_real_failure | `1a6fad83bfc8cece8f379c5000d5554043b96c4ef8ab29b12ca7944009b2dffc` | `2c22cd820e0e9cd6824224f0299cb9c94b4c6904ef1460b350683a284c1fbc4e` |
| 1 | unsupported_gpu_architecture_real_failure | `75e2219a1f6fead5ee98fafc3d45d730784958a3d0be27975326de116c2b53b2` | `29ba97aaccd0caba9c8340407d377411bbb0ef5150b28c3f6952592ed390f027` |

## Row and Merge Gates

- seeds 1/2 rows: `920/920`
- seed 1 rows: `460`
- seed 2 rows: `460`
- seeds 1/2 result SHA-256: `0d4cd2031a6ec213ad1312021e5fefca349673732cc94ee92238eb036dcaf452`
- seed-0 source SHA-256: `118ec37f3a195d50c24abf24b4c61048cdbc0ffff7d9c0f0bf51c83f7f69229c`
- final rows: `1380/1380`
- final result SHA-256: `95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3`
- final duplicate keys: `0`
- single-class adaptation rows: `0`
- single-class evaluation rows: `0`
- adapt/eval overlap failures: `0`
- missing prediction hashes: `0`
- missing logits hashes: `0`

P8 seed-0 CSV bytes were not modified. Its 460 parsed rows match the seed-0 subset of the deterministic final merge exactly; no seed-0 model was rerun.

## Aggregation Gate

- bootstrap replicates: `10000`
- bootstrap seed: `20260710`
- cluster unit: dataset x target subject
- dataset-stratified resampling preserves 9/52/54 target counts
- seeds averaged before aggregation and bootstrap
- methods and paired contrasts preserved within sampled subjects

## Internal Validation Review

- Eight accepted shard keys cover each seed/subject/method exactly once.
- P100 logs are retained separately and no P100 partial row is accepted.
- Resource partition changes are fully disclosed and do not alter scientific configuration.
- The final label is enabled only because every execution, row, seed-preservation, and inference gate passes.
- No extra seeds, methods, TeX edits, H2CMI reruns, geometry stress, or orthogonal-score work is included.
