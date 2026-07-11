# P9 Three-Seed Independent Red-Team Review

## Verdict

No blocker. The packet is suitable for the frozen label:

`Official SPDIM W1 repaired-split three-source-seed same-split baseline.`

## Independent Checks

- Parsed `920/920` seeds-1/2 rows and `1380/1380` final rows; all
  dataset x target x seed x method keys are unique.
- Verified the eight accepted shard counts (`116/116/116/112` per seed),
  shard checksums, exact shard union, and raw CSV/summary byte identity with
  the external run cache.
- Verified the accepted logs contain only the disclosed MOABB zero-buffer or
  Matplotlib font-cache warnings, or are empty. No accepted log contains an
  unsupported-architecture, kernel-image, CUDA-initialization, or traceback
  failure.
- Verified seed-0 SHA-256 remains
  `118ec37f3a195d50c24abf24b4c61048cdbc0ffff7d9c0f0bf51c83f7f69229c`
  and all 460 seed-0 rows are preserved exactly in the final merge.
- Cross-checked all 1,380 rows against the frozen repaired-split manifest.
  Split hashes, trial counts, subject/session identities, class presence, and
  adapt/eval disjointness agree. Leakage, target-performance selection,
  official-pretrained-weight, and vendoring flags are all false.
- Independently reconstructed the 460 seed-averaged subject-method units.
  Method and paired-contrast point estimates match the committed tables.
- Independently reran the 10,000-replicate dataset-stratified paired
  subject-cluster bootstrap with seed `20260710`; all reported confidence
  intervals match.
- Verified output dimensions and arithmetic: method CI `40`, contrast CI
  `50`, harm `60`, and seed stability `24` rows, all with unique keys.
- Verified the digest makes no SPDIM-over-RCT claim. Subject-weighted bAcc is
  `-0.0027407` for SPDIM geodesic minus RCT and `-0.0040097` for SPDIM bias
  minus RCT.

## Adversarial Findings and Resolution

1. The first completion-gate implementation used the wrong Slurm format
   fields for array identity. It was corrected to `%i|%F|%K`, then rerun;
   accepted array `892389` is absent from `squeue`.
2. The first P100 zero-row check used a fixed row count. It was replaced with
   evidence attribution: started P100 jobs `892386` and `892387` have real
   sm_60 incompatibility and cancellation records, and no retained result
   summary is attributed to either job. Accepted P100 rows are therefore zero.
3. Generated validation sections were renamed from red-team review to
   internal validation review so they are not presented as independent
   evidence.

## Residual Risk

GPU compute capabilities are mapped from recorded device models because the
runtime controller did not emit `torch.cuda.get_device_capability`. This does
not affect the accepted execution gate: each accepted task recorded its actual
GPU model and completed without architecture, kernel, or CUDA failures.
