# W1 Repaired H2CMI Audit

- status: `pass`
- launch_commit: `ab93820db4bcf08e0a44827272d82328e86376a2`
- manifest_hash: `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`
- expected_rows: `3450`
- row_count: `3450`
- final_squeue_absent: `True`
- prediction_hash_complete: `True`
- logits_hash_available: `False`
- all_eval_both_classes: `True`
- target_label_leakage_detected: `False`

## Shards

| job_id | shard | rows | expected | stderr status | stdout complete | checksum |
|---|---|---:|---:|---|---|---|
| 890592 | h2p7-bnci-all | 270 | 270 | empty | True | `6a832ac5e899fca980b895d4b83ed45333c6920d0e5590843e6289cc628da4b8` |
| 890593 | h2p7-cho-00-17 | 540 | 540 | known_harmless_moabb_warning_only | True | `9728fc108a3914a2a3a3bbe6caaae02609b3415b58baf0df923d40592149b300` |
| 890594 | h2p7-cho-18-35 | 540 | 540 | known_harmless_moabb_warning_only | True | `d6166bdf82a31f0a4cf4ce4c29a5a2149b06c231e26328c4d09e192f2a6dac61` |
| 890595 | h2p7-cho-36-51 | 480 | 480 | known_harmless_moabb_warning_only | True | `14b446862159940b10c300cd8f35da4fb4e67deeb50bf630f7cae7334fe21e75` |
| 890629 | h2p7-lee-clean-00-17 | 540 | 540 | empty | True | `87c879869f2df27511374fe55775c0b1701b09a07d73efb27d6b81b9c86077cb` |
| 890630 | h2p7-lee-clean-18-35 | 540 | 540 | empty | True | `7527674d1510908e6deb2fbac1a7c75c5c0d47842cba4ca8e05a2ba9fdb5a7f3` |
| 890631 | h2p7-lee-clean-36-53 | 540 | 540 | empty | True | `b9192d553cd21f9be40e46ee2acda90a1911e2d242b35dcd814f09b3d0f6f863` |

## Excluded Jobs

- `890596`: dirty raw launch status after earlier shards wrote artifacts; excluded and rerun as clean job 890629
- `890597`: pending Lee shard canceled before result use; replaced by clean job 890630
- `890598`: pending Lee shard canceled before result use; replaced by clean job 890631

## Validation

- `squeue` final state: all accepted job IDs absent.
- stderr accepted statuses: empty or known harmless MOABB warning only.
- stdout exists and contains launch commit, clean porcelain block, and completion line for every accepted shard.
- CSV/JSON parse and row-count validation passed.
- No single-class eval rows remain.
- No target-label leakage flag was detected.
- Config checksum is reconstructed from launch stdout command line, manifest hash, runner checksum, and manifest checksum for each shard.

## Red Team Review

- The canceled dirty Lee launch is excluded and not used in any result artifact.
- P7B runs H2CMI only; no SPDIM, TeX, geometry stress, or orthogonal-score work is included.
- The split uses target labels only for frozen split construction; runtime adaptation rows mark labels hidden.
- The result is labeled `repaired_split_confirmatory` only because all validation gates pass.
