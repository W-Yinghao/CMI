# OACI EEG-DG Project Memory Through C84R

C84R repaired the metadata-only C84P montage blocker before any real-data access.
The exact primary interface is 20 channels with SHA-256 `988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04`;
FCz is dropped and Fz substitution/interpolation/masking are forbidden.

The historical 21-channel protocols remain preserved. V2 protocols retain the C84P
subject partitions, methods, budgets, inference and field arithmetic. All 1,944 planned
candidate IDs bind the V2 interface. The engineering-only C84C scope is 243 units and
9 phases across Lee/Cho/Physionet, panel A, seed 5, level 0.

C84C is implemented and locked at `4eaad36cafefb2645f1d5c6e393ae5a51ff33af9` / `f9cabf8f362917d663e13154910085d5b105740b265789a2323dd7bc0193222b`, but is not authorized.
C84F and C84S are neither locked nor authorized. No dataset was downloaded and no EEG,
label, training, forward, GPU, candidate or selector outcome was accessed in C84R.

Final gate: `C84_COMMON_20_CHANNEL_MONTAGE_REPAIRED_CANARY_LOCKED_READY_FOR_PI_AUTHORIZATION`.
