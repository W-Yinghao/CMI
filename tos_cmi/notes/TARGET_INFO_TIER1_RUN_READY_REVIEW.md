# Tier-1 run-ready review — execute_real wiring (NO runs; locked)

Branch `science-target-info-v1`. The real Tier-1 executable path is now WIRED but stays locked: `--execute` HALTs
under both the default manifest (`preflight_only`) and the `TEMPLATE` manifest (`awaiting_pm_run_go`). No manifest
flip, no enable token, no run. The uploaded PDF is a stale 2a-only snapshot and is NOT the writing baseline.

## 1. What execute_real now does
- `execute_real(cfg, manifest)`: belt-and-suspenders re-check (`runs_allowed` AND `experiments_allowed` true AND
  `run_status==approved`), then `_assemble_run(cfg, _real_provider(...))` → writes two-phase outputs to
  `results/target_info/tier1_run/`.
- `_real_provider`: for each (dataset, world, fold, alpha, intervention, budget) loads the frozen dump, injects the
  world (`inject` for V2 source-invisible World A; `inject_source_rich` for source-rich source-visible World A),
  fits the eraser + a same-k random eraser on SOURCE (`v2_worlds.FACTORIES`), and hands arrays to `execute_task`.
- `execute_task` (the leak-sensitive CORE, unit-tested on dummy arrays): replays subject-seeded splits +
  nested-k subsets, runs `TARGET_LEAK_STRUCTURAL_PASS` per split, computes the DECISION from source+calibration
  only (`compute_decision_row` / `b3_sequential_decision`), then the AUDIT metric in a separate post-decision phase.

## 2. What it still cannot do without a PM token
- It cannot run: `authorize_execution` requires `run_status==approved` + `runs_allowed`/`experiments_allowed` true +
  a valid `--enable-token` (sha256-matched) + matching `approved_scope_hash`/`approved_driver_hash`. The active
  manifest is `preflight_only` and the TEMPLATE is `awaiting_pm_run_go`; both fail. A plain YAML flip of the driver
  booleans does nothing (the gate keys on the manifest + out-of-band token).
- `_real_provider` (real-dump load + world injection) is UNEXERCISED — it is the primary surface for this review.

## 3. Dataflow: source / calibration / audit
```
SOURCE (Zs, ys, z_src)        -> eraser fit, head fit, source safety, B0 source benefit          (labels allowed)
TARGET CALIBRATION (per split, nested k-subset) -> CalibrationContext -> B2/B3 ΔbAcc_cal LCB      (labels allowed)
TARGET UNLABELED (B1)         -> UnlabeledTargetContext -> triage (reject/abstain/request_labels)  (NO labels)
TARGET AUDIT (per split)      -> AuditView -> audit ΔbAcc, AFTER the decision is frozen            (labels: audit phase only)
```
`LabelAccessGuard("decision")` forbids audit labels during the decision; `LabelAccessGuard("audit")` releases them
only in the audit phase. `compute_decision_row` has no audit parameter; `calibration_delta_bacc`/`_boot_delta_samples`
refuse an `AuditView`.

## 4. Decision rows vs audit rows (two-phase output)
- `decision_rows.csv` — source/calibration-derivable ONLY: dataset, backbone, world, fold, target_subject,
  split_id, budget, k, intervention, source_safety_status, calibration_benefit_lcb, same_k_random_calibration,
  domain_gain, decision_action, decision_input_hash, calibration_idx_hash, calibration_label_hash,
  `audit_idx_hash_hash_only` (a hash-of-hash provenance — NOT audit labels/metrics). The writer RAISES if a decision
  row carries any `audit_delta*`/`audit_label*`/`audit_metric*` field.
- `audit_rows.csv` — attached AFTER the decision is frozen: audit_delta_bacc, audit_delta_nll, specificity_flag
  (accepted_specific / accepted_non_specific), linked by `decision_input_hash`.
- `joined_rows.csv` — post-hoc join for reporting; the report must state joined rows are post-hoc audit output that
  did NOT participate in any decision.

## 5. Manifest / token authorization
- `target_info_tier1_run_manifest.yaml` (ACTIVE): `preflight_only`, all flags false — locked.
- `target_info_tier1_run_manifest_TEMPLATE.yaml` (shape only): `awaiting_pm_run_go`, all flags false, token null.
- To enable a run the PM sets `run_status: approved`, both allowed flags true, fills `approved_scope_hash`
  (`scope_hash(cfg)`), `approved_driver_hash` (sha256(driver)[:16]), `approved_git_commit`, and
  `approved_enable_token_sha256 = sha256(<out-of-band token>)`, then passes `--enable-token <token>`.

## 5b. Run-readiness v2 status
```
Implemented:
  source-safety UCB path (source_task_drop_ucb): cluster-bootstrap upper bound over source SUBJECTS; underpowered
    (< min_clusters) -> ABSTAIN (reason=underpowered_safety_ucb); the accept-gate keys on the UCB, NEVER the mean.
    The old point-estimate _source_safety is REMOVED; execute_task now uses the UCB.
  provider-validation mode scaffold (--provider-validate-one-dump): exercises _real_provider on ONE real dump
    (Lee EEGNet, first fold, source-rich World A, split_id 0, B0/B2(k=4)/B4, identity/leace_baseline/random_k);
    runs the full decision+audit path internally but REDACTS every metric VALUE in the committed output.
  provider-validation manifest template (target_info_tier1_provider_validation_manifest_TEMPLATE.yaml).
  metric-redaction output contract (_redact_validation_output): safe field NAMES + redacted-metric-field COUNTS +
    shapes/hashes/contexts; NO metric VALUE, no metric field name.
Still not allowed (all HALT):
  actual Tier-1 smoke ; provider-validation run (provider_validation_allowed=false) ; manifest approval ;
  enable token ; runs_allowed/experiments_allowed flip.
Original three risks, now:
  (a) _real_provider unexercised -> provider-validation mode PREPARED (gated), not run.
  (b) point-estimate safety -> REPLACED by a subject-cluster UCB; no point-estimate accept path remains.
  (c) small-k weak LCB -> expected abstention at small k, not a blocker (documented).
```

## 6. Known risks (for review before enabling a run)
1. **`_real_provider` unexercised**: real-dump load + `inject`/`inject_source_rich` + `FACTORIES` mapping have not
   run; the world-injection dim/`m` and the `leace`→`leace_baseline` mapping should be validated on one real dump
   during the run-enable step. `execute_task` (the leak core) IS tested.
2. **cal_deltas per-split sourcing**: `execute_task` builds calibration from each split's `calibration_idx` /
   nested-k subset only; a per-split `TARGET_LEAK_STRUCTURAL_PASS` runs before compute. The run-enable review must
   confirm no cross-split pooling is introduced.
3. **Simplified statistics (smoke-level, reviewable)**: source safety is a point-estimate task-drop (a
   subject-clustered UCB is a run-ready TODO); calibration LCB is a within-split bootstrap over the k-subset trials,
   so small k (1,2) yields wide/low LCB → abstain (expected, not a bug). These are deliberate smoke simplifications
   to revisit before any accuracy claim.
4. **Small-k degeneracy**: bootstrap resamples that collapse to one class are dropped; at very small k the sample
   count can shrink — acceptable for a smoke (leads to abstain), flagged for review.

## 7. Tests passed
`test_target_info_tier1.py` — full suite green (see report). New run-ready tests: authorization-requires-approved-
manifest+token (+ positive control), guarded-context loader, decision-rows-no-audit-metrics, audit-rows-post-
decision-only, audit-permutation-invariant decision rows, calibration-permutation-can-change decision, subject-
seeded + nested-k replay in execute_task, B2-unavailable-k marks without audit reuse, B3 ascending no-future-peek,
B1-accept-impossible, B4-excluded, same-k specificity flag, assemble-run two-phase output, execute-halts.

## 8. Exact command if later approved (do NOT run now)
```
# 1. PM fills an approved run manifest (run_status=approved, flags true, hashes + approved_enable_token_sha256)
# 2. then, only with the out-of-band token:
python -m tos_cmi.eeg.run_target_info_tier1_smoke \
  --execute \
  --run-manifest <pm_approved_run_manifest.yaml> \
  --enable-token <out_of_band_token>
# scope: Lee2019_MI + Cho2017 x EEGNet x (V2 source-invisible + source-rich source-visible World A) x seed0 x
#        first5 x R=10 x k{1,2,4,8,16} x alpha{0.5,1,2}; outputs -> results/target_info/tier1_run/
```
Until then: no run, no flag flip, no token. See `TARGET_INFO_TIER1_RUN_READINESS.md`, `SOURCE_RICH_FINAL_VERDICT.md`.
