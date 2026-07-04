# CSC-realEEG v1 run — INFRASTRUCTURE ABORT (not a scientific FAIL)

**Tag:** `csc-realeeg-v1` @ `39c866d` (frozen; **NOT moved, NOT deleted** — history preserved).
**SLURM job:** `881413` (cpu-high, nodecpu01), started 2026-07-04 18:14, **scancel'd 2026-07-04 22:55**.

## What happened
The v1 engine runs the 801 cohorts **strictly serially, single-core** (`run_validation` has no
`n_jobs`/parallelism; the `--jobs` arg is dead). Measured cost on one real cohort (NULL_cov, exact frozen
config + seed 21000000, single core):

| | seconds |
|---|---|
| B3 (`certify_paired_calibrated`, `n_boot=200` internal null) | **638.0** ← dominant |
| Route A (`run_frozen_protocol`, 40+120+120+240 bootstraps) | 42.8 |
| **per cohort** | **~681** |

→ serial total ≈ **801 × 681 s ≈ 6.32 days**. The partition hard cap is **5 days**
(`cpu-high MaxTime=5-00:00:00`, `881413 EndTime=2026-07-09 18:14`), and the runner writes the result JSON only
**after all 801 cohorts finish** (no checkpointing). So the job would have been SIGKILL'd at the 5-day wall at
~79 % with **zero artifact**. At kill time it had done ~25/801 cohorts (~3 %).

## Verdict for this run (per reviewer)
```
csc-realeeg-v1 run:
  infrastructure infeasible / serial runtime wall
  NOT a scientific FAIL
  no result artifact
  no endpoint evaluated
```
No TIER1/TIER2/TIER3 was computed. No PASS/FAIL. No clinical/PD claim. Synthetic tags dee8958 / 0595f64
untouched. The cache, manifests, method locks, and injection definitions are all intact and unchanged.

## Resolution (authorized): v2 = performance-only parallel engine
Same science, parallel execution. New tag `csc-realeeg-v2` (v1 stays as-is). Allowed changes: cohort-level
parallelization, canonical sort before verdict, streaming checkpoint/partial + resume, progress logging,
worker-exception→infra-fail, BLAS thread limiting, v2 provenance fields. **Forbidden:** feature, montage, cache,
injection defs, Route A / B3 method, `n_boot=200`, `cohorts_per_condition=100`, seed schedule, gates, alpha,
denominators, bootstrap-bound definition, genuine-contrast semantics. Target walltime <10 h on 24 cores
(~6.3 d / ~20). v2 freeze package must be re-audited (serial↔parallel identity + fail-closed checkpoint/resume)
BEFORE any `csc-realeeg-v2` tag or run is authorized.
