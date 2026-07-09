# SPDIM W1 Repaired Seed-0 Four-Shard Resubmission

Per user approval, the monolithic P8 job `891435` was cancelled and excluded from the confirmatory merge. At cancellation it had produced 56 partial rows: BNCI2014_001 36 and Cho2017 20. Those artifacts were moved outside the repository to `/home/infres/yinwang/.cache/h2cmi_training_caches/p8_monolithic_891435_partial_excluded`.

The replacement launch uses four non-overlapping target shards, source seed 0 only, and the frozen repaired split manifest hash `231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e`.

| shard | array task | target spec | expected rows |
|---|---:|---|---:|
| shard0 | 0 | `BNCI2014_001=1-9;Cho2017=1-20` | 116 |
| shard1 | 1 | `Cho2017=21-49` | 116 |
| shard2 | 2 | `Cho2017=50-52;Lee2019_MI=1-26` | 116 |
| shard3 | 3 | `Lee2019_MI=27-54` | 112 |

Total expected rows remain 460. Shard outputs are written to a repository-external cache directory keyed by launch commit so parallel GPU jobs do not dirty the launch worktree.

## Red Team Review

- `891435` partial rows are excluded and must not be merged into the confirmatory result.
- The four shard specs are non-overlapping and cover exactly 115 target subjects.
- This resubmission does not approve seeds 1/2 or a full SPDIM baseline.
- Monitoring remains `squeue` only, with final completion decided by artifact parse/count/checksum validation.
