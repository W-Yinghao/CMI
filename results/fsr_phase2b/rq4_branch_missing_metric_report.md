# FSR RQ4 — branch-locality (DESCRIPTIVE; blocked, not failed)

Source: `FBCSP_F0_AGGREGATE.csv` on `project/fbcsp-lgg-spatial-cmi-fusion` @ 39c245a (backbone `FBCSPLGGGraph`, branches graph / temporal / spatial; **no separate node branch** — `permute_nodes` is a null, not a branch). Read via `git show`.

## Branch load (from ablation + gate weights)

| dataset | branch | ablation_drop (mean-zero) | gate_weight | status |
|---|---|---|---|---|
| BNCI2014_001 | graph | -0.0174 | 0.2791 | neutral_or_slightly_harmful |
| BNCI2014_001 | temporal | -0.0127 | 0.2322 | neutral_or_slightly_harmful |
| BNCI2014_001 | spatial | +0.0736 | 0.4888 | load_bearing |
| BNCI2015_001 | graph | -0.0077 | 0.2387 | neutral_or_slightly_harmful |
| BNCI2015_001 | temporal | -0.0019 | 0.1891 | neutral_or_slightly_harmful |
| BNCI2015_001 | spatial | +0.0880 | 0.5723 | load_bearing |

## What RQ4 CAN say (descriptive)
- The **spatial branch is load-bearing**: BNCI2014_001 zero_spatial drop +0.0736, gate 0.489; BNCI2015_001 zero_spatial drop +0.0880, gate 0.572.
- The **graph/temporal branches are neutral-to-slightly-harmful** on 4-class 2a and starved after fusion.
- **P6 spatial-CMI is a scaffold / not promoted**, not a confirmed spatial-CMI result.

## What RQ4 CANNOT say (blocked)
- "spatial leakage is harmful" — no per-branch leakage probe exists.
- "graph leakage is benign" — same.
- "per-branch CMI predicts reliance" — no per-branch functional-reliance (L5) measurement exists.

## Missing metrics (both HIGH, `needs_small_frozen_probe`)
| missing_metric | needed_for | status | resolution |
|---|---|---|---|
| per-branch leakage probe on `spatial_z`/`graph_z`/`node_z` | RQ4 predictor (L1 per branch) | absent on disk (0 frozen embeddings, 0 per-branch probe) | small_frozen_probe (Phase 3/4, PM-gated) |
| per-branch functional reliance (L5) | RQ4 endpoint | absent | small_frozen_probe (couples to the dump) |

**RQ4 quantitative status: `BLOCKED_MISSING_METRIC` for every branch.** No probe is run in Step 2B; this is a blocked (not failed) RQ. Producing the two metrics requires re-inference to freeze `last_spatial_z`/`graph_z`/`node_z` per fold plus a trained per-branch domain probe and a per-branch R3-style removal replay — deferred to a PM-approved Phase-3/4 frozen-probe run.
