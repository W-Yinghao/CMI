# FP-GEM Integration Audit

- phase: `P12A_SOURCE_PROVENANCE_AMENDMENT_PRECOMPUTE`
- CPU dry-run status: `PASS`
- GPU smoke status: `PASS`
- approve single smoke: `true`
- approve P12B fleet after smoke: `true`

## Static And Artifact Gates

- selected datasets: `['BNCI2014_001', 'Lee2019_MI']`
- selected source seeds: `[0, 1, 2]`
- target-seed units: `189`
- expected new-method rows: `378`
- expected same-checkpoint control rows: `756`
- expected reused P9 rows: `0`
- P9 reference rows: `756`
- expected final rows: `1134`
- exact unit keys match repaired manifest: `True`
- exact unit keys match P9 source hashes: `True`
- every unit reference hash matches its P9 row: `True`
- every unit split hash/count matches the repaired manifest: `True`
- every unit hardware group matches the frozen P9-family mapping: `True`
- all adaptation/evaluation IDs disjoint: `True`
- all adaptation splits have both classes in the frozen manifest: `True`
- all evaluation splits have both classes: `True`
- target-label leakage detected: `False`
- target-performance selection detected: `False`
- actual CPU feature-hook probe: `{'dataset': 'BNCI2014_001', 'feature_shape': [1, 210], 'logit_shape': [1, 2], 'semantic_max_abs_error': 0.0, 'finite': True, 'target_labels_accessed': False}`
- V100 reproduction units: `161`
- A100 reproduction units: `28`
- source checkpoint index SHA-256: `587685bc9e15a853c62daf8175e2eb6533dd73aa8ae4998e653a36a664f17c91`
- execution unit manifest SHA-256: `3bb1250b3faf583ff79324326b0159b6a6dd9f8efd3a92ecc21231e31fb2c267`

## Feature Hook Gate

Official TSMNet computes `logeig -> dtype/device conversion -> classifier`. The runner registers a forward pre-hook on `TSMNet.classifier`, so it captures the exact classifier input rather than reconstructing or replacing the decoder. The smoke must prove direct classifier replay within 1e-7 and a 210-dimensional feature.

## Leakage Boundary

Source labels are used only for exact P9 source training and the post-hoc source class-conditional density. Adaptation receives target X/features and a dummy-zero label tensor only for the official RCT API. Evaluation labels are first read after official SPDIM geodesic/bias and both GEM fits complete. The smoke does not read evaluation labels or compute performance.

## Checkpoint Availability

No P9 TSMNet checkpoint files were found or recorded. The committed P9 `source_model_sha256` column supplies a provenance reference per dataset x target x seed, but not recoverable weights. Every P12 unit must persist one exact-config retrain and run all six methods from that actual hashed checkpoint. Direct P9 row reuse is prohibited when the actual state hash differs.

## GPU Smoke Gate

- job id: `893433`
- status: `PASS`
- smoke payload SHA-256: `4bdcbb27f7303bc99f642119ae996b936c4a77b56a026c393e729bc87c672fe7`
- exact P9 source-training configuration reproduced: `True`
- P9 reference state hash matched by retrain (informational, not a gate): `False`
- all six methods share the reproduced state: `True`
- feature-hook replay passed: `True`
- target performance observed: `false`
