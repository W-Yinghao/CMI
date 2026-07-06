# FSR Phase-2 outputs (Step 2A)

CPU-only, read-only re-analysis of the frozen Step-1 evidence. No GPU, no training, no CMI sweep.
Everything here is produced by `scripts/fsr/*.py` from frozen artifacts; re-run to reproduce.

## Reproduce
```bash
python scripts/fsr/validate_step1_index.py     # -> schema_validation.json (fail-closed)
python scripts/fsr/build_phase2_tables.py       # -> route_metric_table / analysis_inclusion / missing_metric
python scripts/fsr/analyze_cigl_gap.py          # -> cigl_gap_reproduction.{json,csv}, cigl_gap_bootstrap.csv
python scripts/fsr/analyze_tos_erasure.py        # -> tos_erasure_summary / tos_randomk_specificity / tos_task_safety_flags
```
Env: any Python 3.9 with `numpy`, `scipy`, `pandas`. TOS artifacts are read from git branch `tos` via `git show` (not checked out here).

## Files
| file | producer | what it is |
|---|---|---|
| `schema_validation.json` | validate_step1_index.py | 15/15 fail-closed checks on `artifact_index.csv` (rows=37, 18 cols, routes, tags) |
| `route_metric_table.csv` | build_phase2_tables.py | 37 routes normalized onto the L1–L6 ladder + RQ include flags |
| `analysis_inclusion_table.csv` | build_phase2_tables.py | per-route Phase-1 inclusion (predictor/endpoint levels, tag, reason) |
| `missing_metric_decisions.csv` | build_phase2_tables.py | 22 (route, missing level) rows with a policy token + resolution |
| `cigl_gap_reproduction.json` | analyze_cigl_gap.py | RQ1 entry-gate reproduction + acceptance flags + provenance |
| `cigl_gap_reproduction.csv` | analyze_cigl_gap.py | tidy correlation table (recomputed vs frozen), incl. dataset/seed strata |
| `cigl_gap_bootstrap.csv` | analyze_cigl_gap.py | bootstrap CI rows (point + percentiles) per recomputed correlation |
| `tos_erasure_summary.csv` | analyze_tos_erasure.py | per (dataset,backbone,eraser) subject-removal + target deltas + verdict |
| `tos_randomk_specificity.csv` | analyze_tos_erasure.py | LEACE vs random-k NLL + subject-removal specificity (non-specific flag) |
| `tos_task_safety_flags.csv` | analyze_tos_erasure.py | INLP task-collapse + LEACE/RLACE binary-EEGNet harm flags |

## Headline reproduction (entry gate — reproduction, not conclusion)
```text
align_k2 -> R3 (pooled n=126): recomputed +0.3382 [+0.167,+0.506] == frozen +0.3382   REPRODUCED
graph_kl -> R3 (seed0  n=42 ): recomputed -0.4191 (negative sign confirmed)
graph_kl -> R3 (pooled n=126): NOT recomputable (per-fold seeds 1/2 pruned); frozen -0.342 carried, flagged
difference align-graph_kl (seed0 n=42): +0.816 [+0.219,+1.333], excludes 0
```

## Phase-1 inclusion (revised gate: >=1 predictor {L1-L4} AND >=1 endpoint {L5,L6})
```text
RQ1 (leakage->reliance):        CIGL
RQ2 (erasure->target):          TOS_mean_scatter, TOS_LEACE, TOS_INLP, TOS_RLACE, TOS_random_k
RQ3 (alignment vs leakage):     CIGL
RQ4 (branch-locality):          none (per-branch leakage+reliance MISSING -> SUPPORT_ONLY; the HIGH gap)
```

## Provenance notes
- Per-fold `graph_kl` for seeds 1/2 was pruned from every branch (raw audit `.npz` + r2-gate JSONs uncommitted); only seed0 is recomputable. Recorded in `missing_metric_decisions.csv` as `artifact_missing`.
- TOS collapse-curve numeric JSONs are not committed on `tos` (only `collapse_curves.png` + the `PHASE21` / `CLAIMS_LEDGER` docs); the two in-loss boundary routes cite those docs.
- No target-label fit/use enters any RQ table: `YES_FORBIDDEN` (LPC legacy) and all `AUDIT_ONLY` rows have `include_rq*`=NO.

This is Step 2A (schema + tables + reproduction). The RQ1–RQ4 conclusions are Step 2B and await PM approval.
