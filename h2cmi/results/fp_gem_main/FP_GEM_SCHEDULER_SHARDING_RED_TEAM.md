# FP-GEM Scheduler Sharding Red-Team Review

Status: **PASS**. Reviewed before P12B fleet submission.

- Confirmed all three oversized submissions were rejected before job creation; no job ID or result artifact exists.
- Confirmed V100 shard lengths `27,27,27,27,27,26` cover group indices `0..160` exactly once.
- Confirmed A100 shard lengths `14,14` cover group indices `0..27` exactly once.
- Confirmed maximum simultaneous array tasks is `6 + 2 = 8`.
- Confirmed a shard stops on the first failed unit and the runner only skips an existing artifact with `status=ok/pass`, preserving clean retry semantics.
- Confirmed runner SHA-256 remains `720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd`.
- Confirmed config SHA-256 remains `d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165`.
- Confirmed unit-manifest SHA-256 remains `3bb1250b3faf583ff79324326b0159b6a6dd9f8efd3a92ecc21231e31fb2c267`.
- Confirmed launcher shell syntax and no-Slurm-accounting-call gates pass.

Verdict: the stride launcher changes scheduler packing only. It does not change scientific unit identity, execution order within a unit, methods, target data, seeds, or hyperparameters.
