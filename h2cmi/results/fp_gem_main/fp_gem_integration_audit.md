# FP-GEM Integration Audit

- phase: `P12A_PRECOMPUTE`
- CPU dry-run status: `PASS`
- GPU smoke status: `PENDING`
- approve single smoke: `true`
- approve P12B fleet before smoke: `false`

## Static And Artifact Gates

- selected datasets: `['BNCI2014_001', 'Lee2019_MI']`
- selected source seeds: `[0, 1, 2]`
- target-seed units: `189`
- expected new rows: `378`
- expected reused P9 rows: `756`
- expected final rows: `1134`
- exact unit keys match repaired manifest: `True`
- exact unit keys match P9 source hashes: `True`
- all adaptation/evaluation IDs disjoint: `True`
- all adaptation splits have both classes in the frozen manifest: `True`
- all evaluation splits have both classes: `True`
- target-label leakage detected: `False`
- target-performance selection detected: `False`
- actual CPU feature-hook probe: `{'dataset': 'BNCI2014_001', 'feature_shape': [1, 210], 'logit_shape': [1, 2], 'semantic_max_abs_error': 0.0, 'finite': True, 'target_labels_accessed': False}`
- V100 reproduction units: `161`
- A100 reproduction units: `28`
- source checkpoint index SHA-256: `0a22c34b46f749f49de4e048971fdff3a509a0b65ca799fff3bc809a3d6c35b4`
- execution unit manifest SHA-256: `dbc4080d7d17c2d6d0cfa74901da31f2c5b79d6079b4acc37c3b73c840149326`

## Feature Hook Gate

Official TSMNet computes `logeig -> dtype/device conversion -> classifier`. The runner registers a forward pre-hook on `TSMNet.classifier`, so it captures the exact classifier input rather than reconstructing or replacing the decoder. The smoke must prove direct classifier replay within 1e-7 and a 210-dimensional feature.

## Leakage Boundary

Source labels are used only for exact P9 source training and the post-hoc source class-conditional density. Adaptation receives target X/features and a dummy-zero label tensor only for the official RCT API. Evaluation labels are first read after both GEM fits complete. The smoke does not read evaluation labels or compute performance.

## Checkpoint Availability

No P9 TSMNet checkpoint files were found or recorded. The committed P9 `source_model_sha256` column supplies one consistent expected state hash per dataset x target x seed. The runner must reproduce that state exactly before it may execute RCT or GEM.
