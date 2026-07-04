# CSC-realEEG v2 — performance-only parallel package + AUDIT (NO tag, NO run)

**Status: v2 freeze package BUILT + self-audited + red-teamed. NOT tagged, NOT run.** v2 is a PERFORMANCE +
artifact-safety successor to the serial, infrastructure-infeasible `csc-realeeg-v1` (see
[CSC_REALEEG_V1_INFRA_ABORT.md](CSC_REALEEG_V1_INFRA_ABORT.md)). Creating tag `csc-realeeg-v2` and running each
need a separate reviewer go. Synthetic tags dee8958 / 0595f64 untouched. No clinical/PD claim.

## What changed (allowed: performance / artifact-safety only)
1. **Cohort-level parallelism.** New `run_validation_parallel` runs the 801 independent cohorts via
   `joblib.Parallel(loky)`; the serial `run_validation` is kept as the FROZEN reference.
2. **Canonical sort before verdict.** Records are re-sorted `(condition_index, cohort_index)` == serial order
   before `evaluate_verdict`, because `_b3_rates`' cohort bootstrap resamples the fired-flag **array** whose
   order must match serial for byte-identity.
3. **Streaming checkpoint + resume.** Each cohort → `<out>.partial.jsonl`; periodic `<out>.checkpoint.json`.
   Resume is allowed ONLY when the checkpoint's `task_table_sha256` + `git_head` + cache/bank/engine hashes all
   still match (else `InfraError`).
4. **Fail-closed assembly.** `_assemble_records` requires exactly one non-errored record per task — missing /
   duplicate / worker-errored → `InfraError` → runner **exit 2, NO endpoint evaluated** (never a silent skip).
5. **BLAS thread pin** (`OMP/MKL/OPENBLAS/NUMEXPR=1`) in the sbatch + recorded in provenance — required for
   cross-process numerical identity (single-threaded reductions are deterministic) AND to avoid oversubscription.
6. **Provenance.** Payload `execution` block (mode, n_jobs, task_table_sha256, n_tasks/expected=801, blas, resumed)
   + bank `execution_provenance` documenting the performance-only scope and the unchanged-from-v1 list.
7. **Progress logging**, worker-exception capture, `n_jobs = min(SLURM_CPUS_PER_TASK, 24)`.

## What did NOT change (verified by diff vs the v1 frozen worktree)
- **Engine:** 0 lines removed from v1; all additions. All 12 frozen science functions **byte-IDENTICAL**:
  `run_validation`, `build_cohort`, `certify_b3`, `certify_A`, `frozen_A_cfg`, `evaluate_verdict`, `_b3_rates`,
  `_a_rates`, `cohort_bootstrap_upper`, `_prior_resample`, `_rotate_proba`, `_pooled_clf`.
- **Manifests:** `realeeg_b3_manifest.json`, `realeeg_routeA_manifest.json`, `realeeg_lee2019_cache_manifest.json`
  **byte-identical** to v1. `realeeg_bank_manifest.json` changed only in `engine_sha256` (re-pin
  `7ec08ad4…`→`8535b799…`) + the additive `execution_provenance` block; `run_spec` / `conditions` /
  `gating_summary` / `seed_schedule` / `pooled_boundary_note` **IDENTICAL**.
- **Locked knobs unchanged:** `b_certifier_internal_null=200`, `cohorts_per_condition=100`,
  `subjects_per_cohort=30`, `positive_decision_budgets=[20,30]`, `b_cohort_bootstrap=2000`,
  `invalid_fraction_cap=0.20`, `realeeg_base_seed=20000000`, gating set = the 4 controls, alpha, denominators,
  cohort-bootstrap definition, genuine-contrast semantics. Runner `ml` dict + `frozen_A_cfg(40/120/120/240)`
  IDENTICAL.

## Numerical-identity evidence
- `test_serial_parallel_identity_on_smoke_subset`: serial == parallel(n_jobs=1) == parallel(n_jobs=3), 0
  record mismatches on the scientific keys, identical `evaluate_verdict`, identical canonical order, stable
  `task_table_sha256`. (Identity is param-independent → the test uses tiny `n_boot` on a toy cache + test seeds,
  never the confirmatory base seed / real cache.)
- **Why identity holds:** each cohort's `rng = default_rng(seed)` is created fresh inside the serial loop and
  never carries state across cohorts, so every cohort is a pure deterministic function of (condition, seed,
  cache). Parallelism cannot change any record; the canonical sort restores serial order; BLAS=1 makes the
  certifier bit-reproducible across processes.

## Verification (all green)
`py_compile` OK; **full test suite 45/45 pass** (36 v1 + 9 new v2: identity, canonical-table, dup/missing
fail-closed, worker-exception, resume-guard incl. null/absent binding, partial≠final, unchanged-science,
execution-provenance, honest-801-breakdown); **dry-run 58/58 PASS** (adds `dry_run_task_table_is_801` +
execution-provenance + BLAS checks); `--execute` and `--execute --resume` both **REFUSE (fail-closed)** with no
`csc-realeeg-v2` tag; sbatch `bash -n` OK.

## Estimated runtime
Serial (v1, measured): ~681 s/cohort × 801 ≈ **6.3 days** (> 5-day wall → the abort). Parallel at 24 cores
(cohorts balanced at ~681 s, embarrassingly parallel): ~⌈801/24⌉ waves × ~681 s ≈ **~6.5 h pure compute,
~7–8 h with overhead** — comfortably under the 5-day wall, with checkpoint/resume as a backstop.

## Task-count closure (reviewer point 2)
Machine-readable breakdown in `execution_provenance.task_count_breakdown`: **8 multi-cohort conditions × 100 + 1
genuine descriptive × 1 = 801** (NOT 9 × 100). Honesty flag recorded: `NULL_real_session` keeps REAL labels (a
stratified pseudo-condition split), so "multi-cohort" ≠ "injected" — 7 of the 8 inject synthetic labels.
`test_task_count_breakdown_is_honest_801` asserts this and cross-checks it against the actual bank conditions.

## Parent-owned writing (reviewer point 5)
Verified: **workers only RETURN records; the parent process alone appends the JSONL** (`Parallel(...)` collects
`out`, then a single `with open(partial_path,"a")` in the parent). No concurrent worker writes → no NFS
append-corruption risk. Checkpoint is written atomically (`tmp` + `os.replace`).

## Final-artifact write conditions (reviewer points 4 & 6)
A FINAL artifact is produced only when ALL hold: `_assemble_records` sees exactly 801 unique task records (no
dup / no missing / no worker error) → else `InfraError` → runner **exit 2, no verdict**; and the sbatch finalizes
(`mv TMP→OUT`) only after its freshness check asserts `execution.n_tasks==801`, `len(per_cohort)==801`,
`schema_version==v2`, `git_head==HEAD`, `expected_code_ref==refs/tags/csc-realeeg-v2`, `synthetic_tags_untouched`,
`base_seed==20000000`, and the bank hash. A partial/checkpoint is progress evidence, never a result.

## Red-team (4 adversarial lenses → per-finding verify; 7 agents, 0 errors)
Lenses: numerical-identity, forbidden-science-change, fail-closed/result-integrity, provenance/leakage. Result:
**1 CONFIRMED finding (severity LOW), 0 uncertain, 0 on identity / science-change / provenance.**

- **[LOW, fail_closed] resume binding guard fail-OPEN on a null/absent checkpoint field** (`realeeg_engine.py`).
  `ck.get(key) is not None` skipped the mismatch check when a binding field was null/missing, so a hand-edited/
  truncated checkpoint that nulls e.g. `engine_sha256` (but keeps a matching `task_table_sha256`) could resume
  under a changed input. Verifier confirmed it is **not reachable through `execute()`** — which always stamps all
  5 binding fields non-null AND independently refuses on git-dirty / engine-hash / cache-hash / method-hash drift
  before resume is entered — so it is pure defense-in-depth, cannot admit a wrong scientific verdict on the
  guarded `--execute` path. **FIXED**: dropped the `is not None` clause → the guard now fails closed on absence
  too; added a resume test that nulls and then removes a binding field and asserts `InfraError`. Engine re-pinned
  `8535b799…`→`d5aaaf78…`.

No identity, science-change, or provenance defects were found or survived verification.

## Next (each a separate reviewer go)
1. create tag `csc-realeeg-v2` (freeze) → 2. frozen worktree `CMI_AAAI_csc_realeeg_v2_frozen` → 3. run exactly
once → 4. C6/red-team re-aggregation. Still NOT authorized here.
