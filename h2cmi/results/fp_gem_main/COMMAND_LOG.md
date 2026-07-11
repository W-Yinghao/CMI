# FP-GEM Command Log

- P12A CPU-only freeze generation: `python -m h2cmi.prepare_fp_gem_freeze`. The method, scope, statistical estimands, and interpretation grid were frozen before target performance observation.
- Pre-amendment smoke job `893415`: zero-result infrastructure failure because `/tmp` was not visible on the compute node.
- Pre-amendment smoke job `893416`: clean V100 exact-config source retrain; stopped on the overstrict unrecoverable P9 byte-hash gate before RCT/GEM/evaluation labels/metrics; accepted rows `0`.
- P12A source-provenance amendment: `python -m h2cmi.prepare_fp_gem_freeze`. Direct P9 row reuse is replaced by frozen same-checkpoint reruns of the four official controls; no scientific setting changed and no target performance had been observed.
