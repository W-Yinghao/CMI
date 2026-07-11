# FP-GEM Command Log

- P12A CPU-only freeze generation: `python -m h2cmi.prepare_fp_gem_freeze`. The method, scope, statistical estimands, and interpretation grid were frozen before target performance observation.
- Pre-amendment smoke job `893415`: zero-result infrastructure failure because `/tmp` was not visible on the compute node.
- Pre-amendment smoke job `893416`: clean V100 exact-config source retrain; stopped on the overstrict unrecoverable P9 byte-hash gate before RCT/GEM/evaluation labels/metrics; accepted rows `0`.
- P12A source-provenance amendment: `python -m h2cmi.prepare_fp_gem_freeze`. Direct P9 row reuse is replaced by frozen same-checkpoint reruns of the four official controls; no scientific setting changed and no target performance had been observed.

- P12A smoke submission: `sbatch --parsable -p V100 --export=ALL,FP_GEM_REPO=/home/infres/yinwang/CMI_AAAI/.codex_p12_launch_5b71ee8 scripts/fp_gem_smoke.slurm`. Job `893433` left `squeue`; artifact/hash/hook/leakage gates returned `PASS`. No target performance metric was computed.
- P12B rejected submissions: V100 arrays `0-160%6`, `0-63%6`, and `0-31%6` each returned `QOSMaxSubmitJobPerUserLimit` before job creation; accepted rows `0`.
- P12B scheduler-only sharding freeze: six V100 stride-6 tasks cover 161 group indices and two A100 stride-2 tasks cover 28 group indices exactly. Runner/config/unit manifest and scientific settings are unchanged.
