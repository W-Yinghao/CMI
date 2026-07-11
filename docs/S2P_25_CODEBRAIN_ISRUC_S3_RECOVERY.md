# S2P_25 - CodeBrain ISRUC_S3 Asset Recovery

**Status:** CLOSED / PASS for raw and processed sequence contracts. No model work was run.

## Source

The local machine contained ISRUC Group-I assets and a four-fold flat epoch LMDB, but no authoritative Group-III
tree. Official ISRUC Cohort III assets were therefore recovered from the University of Coimbra dataset host:

```text
10 raw subject RAR archives
10 extracted-channel MAT files
official download bytes: 5,315,115,956
```

Every file is pinned by Content-Length and SHA256 in
`isruc_s3_recovery/isruc_s3_subject_asset_manifest.csv`. A download-resume canary exposed server Range behavior that
could append a full response to a partial file; strict byte checks rejected those staging payloads before extraction.
Clean redownloads were validated and promoted to `${ISRUC_RAW_ROOT}`; corrupted staging duplicates were
never used and were removed only after the source and processed authorities both passed. Both authority trees are
read-only, with content hashes remaining the scientific immutability contract.

## Raw contract

All ten subjects pass:

```text
archive contains <subject>/<subject>_1.txt
MAT contains F3_A2/C3_A2/O1_A2/F4_A1/C4_A1/O2_A1
each MAT epoch has 6000 samples at 200 Hz
hypnogram length = MAT epochs + 30
labels map from {0,1,2,3,5} to five classes {0,1,2,3,4}
remaining epochs can be trimmed to complete 20-epoch sequences
```

The ten-fold split is fixed as the upstream rotating 8:1:1 subject contract: fold subject is validation, the next
subject is test, and the remaining eight are training. The complete mapping is
`isruc_s3_recovery/isruc_s3_split_manifest.csv`.

## Processing contract

The recovered assets were processed with the upstream semantics:

```text
six-channel order fixed
scipy resample to 6000 samples
MNE FIR 0.3-35 Hz
50 Hz notch
exclude last 30 hypnogram epochs
map stage 5 to 4
trim only the final incomplete 20-epoch block
save paired sequence/label files by exact stem
```

Filtering was vectorized across epochs only after an exact canary against the upstream per-epoch loop:

```text
probe shape: [2,6,6000]
max_abs_diff: 0
exact_equal: true
```

## Authority

```text
output: ${ISRUC_PROCESSED_ROOT}
subjects: 10
paired sequences: 425
usable epochs: 8500
sequence shape: [20,6,6000]
label shape: [20]
tree SHA256: b223c03cb811eb45e247e320e7c7ef43279076dcb653aeb08633673e7ac7b9e6
```

The bounded metadata preflight recomputes this tree hash and verifies all 425 stem pairs, shapes, dtypes, label
domain, ten subject directories, channel order, and sequence length. A locally repeated metadata preflight reports
all three downstream asset contracts PASS.

## Boundary

This closes only the ISRUC_S3 asset and sequence blocker. It does not validate released/random downstream behavior,
run frozen probes, run fine-tuning, launch Stage-2, or authorize the bounded experiment. Remaining launch gates are
SLURM stability and the released tokenizer temporal/frequency target canary.
