# FP-GEM Scheduler Sharding Addendum

Status: **FROZEN BEFORE P12B FLEET**.

The scheduler rejected V100 arrays with 161, 64, and 32 submitted elements under `QOSMaxSubmitJobPerUserLimit`. All three submissions failed before job creation and produced zero rows.

P12B therefore uses eight long-lived operational shards:

| group | array | stride | frozen group indices per task |
|---|---:|---:|---:|
| V100 | `0-5%6` | 6 | 27, 27, 27, 27, 27, 26 |
| A100 | `0-1%2` | 2 | 14, 14 |

Each task invokes the unchanged runner sequentially for `task_id, task_id + stride, ...`. The 161 V100 and 28 A100 group indices are covered exactly once with no gaps or overlap. A failed shard preserves completed unit JSON/checkpoints; an exact-command retry skips completed units and resumes the remaining frozen indices.

This is a scheduler-only change. Runner, config, repaired manifest, unit manifest, methods, seeds, source training, adaptation settings, and statistics are unchanged. Maximum concurrent GPU tasks remains eight.

- launcher SHA-256: `e140a69246ae60287fb15e517150e7c45037cfc9956af4f533bb7e37c472da01`
- shell syntax gate: `PASS`
- exact coverage gate: `PASS`
