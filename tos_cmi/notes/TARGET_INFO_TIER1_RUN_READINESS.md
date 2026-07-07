# Tier-1 RUN-READINESS hardening (P0-1..P0-4) — NO runs

Branch `science-target-info-v1`. Addresses the three run-blockers disclosed by the wired-path red-team
(wf_dddd28b5-946) before any Tier-1 run. **Nothing runs**: `--execute` and `--preflight-real-splits` both HALT.

Files: `eeg/run_target_info_tier1_smoke.py`, `eeg/target_info_splits.py`, `eeg/test_target_info_tier1.py`,
`eeg/configs/target_info_tier1_smoke_driver_fixed.yaml`, `eeg/configs/target_info_tier1_run_manifest.yaml` (new).

## P0-1 — tamper-proof run authorization (not a YAML boolean)
Execution is authorized ONLY by an APPROVED run manifest (`target_info_tier1_run_manifest.yaml`), checked by
`authorize_execution(cfg, manifest, enable_token)`:
```
run_status == "approved"
manifest.runs_allowed AND manifest.experiments_allowed
enable token: sha256(--enable-token) == manifest.approved_enable_token_sha256   (enable_token_required)
approved_scope_hash   == scope_hash(cfg)          (datasets/backbones/worlds/budgets/interventions/thresholds/alpha)
approved_driver_hash  == sha256(driver_config)[:16]
approved_git_commit   in {matches HEAD, "ANY", null-soft}
```
A plain YAML flip of `runs_allowed`/`experiments_allowed` in the DRIVER config does NOT enable a run — the executor
keys on the MANIFEST, whose `run_status` is `preflight_only` and whose approved hashes/token are `null`. The CLI
requires `--execute --run-manifest <path> --enable-token <token>`. Current manifest = `preflight_only` ⇒ `--execute`
and bare both return `EXPERIMENTS_DISABLED: run manifest is preflight_only; requires separate PM go.`
Tests: `test_plain_yaml_flip_does_not_enable_execution`, `test_missing_enable_token_halts`,
`test_wrong_enable_token_halts`, `test_scope_hash_mismatch_halts`, `test_driver_config_hash_mismatch_halts`,
`test_fully_approved_manifest_authorizes` (positive control — the gate CAN authorize, it is not a brick).

## P0-2 — calibration-delta taint / provenance
- `calibration_delta_bacc` raises `TypeError` on an `AuditView` (or non-`CalibrationContext`) — cal deltas can
  come only from a `CalibrationContext`.
- `LabelAccessGuard(phase)`: `audit_labels()` raises `PermissionError` in the decision phase; audit labels are
  readable only in the audit phase (after the decision is frozen).
- `compute_decision_row` records `delta_source=calibration_only`, `calibration_idx_hash`,
  `calibration_label_hash`, `decision_input_hash` — and takes NO audit input. `finalize_decision_row` (audit
  phase) attaches `audit_idx_hash`, `audit_label_hash`, and the honest held-out metric AFTER the decision is
  frozen, so audit provenance never influences the decision.
- Per-split invariance: a decision is invariant to permuting THAT split's audit labels (the cal/audit split is
  disjoint per split). Across R>1 splits the audit UNION covers other splits' calibration trials, so "cal deltas
  must be sourced per-split from `calibration_idx`, never from the audit union" — this is the load-bearing rule
  the run executor must honor (see PM-go prerequisites below).
Tests: `test_cal_deltas_depend_only_on_calibration_labels`,
`test_audit_label_permutation_changes_only_audit_metrics_not_decision` (decision fixed, audit metric moves),
`test_calibration_label_permutation_can_change_decision`, `test_audit_view_cannot_be_passed_to_calibration_delta`,
`test_decision_rows_store_calibration_and_audit_hashes`, `test_label_access_guard_phase_separation`.

## P0-3 — real-split preflight (split/schema only; NEVER metrics)
`--preflight-real-splits` would load real frozen-dump target labels ONLY, build stratified cal/audit splits,
per-class counts, k-availability, index/label hashes, disjointness — and write manifest/tables. It fits NO eraser
and computes NO ΔbAcc / gate action / performance metric. Pure logic in `_preflight_from_labels` (tested on dummy
arrays). It is GATED behind `manifest.preflight_allowed` (currently `false`) ⇒ HALTs with
`PREFLIGHT_DISABLED: run manifest preflight_allowed=false; requires separate PM go.` The real-dump loading branch
is UNEXERCISED until the PM enables preflight (mirrors `execute_real`).
Tests: `test_preflight_real_splits_outputs_no_metrics`, `test_unavailable_k_marked_not_shrunk`,
`test_no_audit_label_reuse_for_calibration`, `test_preflight_gated_by_manifest`.

## P0-4 — actual run still halts
`--execute` (and bare) HALT with `EXPERIMENTS_DISABLED: run manifest is preflight_only; requires separate PM go.`
`execute_real` is a locked stub with a belt-and-suspenders check on the manifest booleans; it raises rather than
touching real dumps even if reached. Test: `test_execute_still_halts_when_runs_disabled`.

## PM-GO prerequisites (must hold WHEN a run is finally authorized)
1. `execute_real` must source `cal_deltas`/`cal_random_deltas` EXCLUSIVELY from each split's `calibration_idx`
   (build a `CalibrationContext` from `calibration_idx`), NEVER from `audit_idx` or `audit_scalar`.
2. Per (subject, split): call `target_leak_structural_check` on the ACTUAL `(calibration_idx, audit_idx)` before
   any compute; do NOT pool "all labels seen so far" across the R repeats.
3. To authorize: fill the manifest with the real `approved_scope_hash`, `approved_driver_hash`,
   `approved_git_commit`, and `approved_enable_token_sha256`, set `run_status: approved` +
   `runs_allowed/experiments_allowed: true`, and pass the out-of-band `--enable-token`. The enable token is the
   out-of-band secret that a plain config edit cannot forge.

## Status
38/38 tests pass. No run. Next PM decision (per the driver report): approve `--preflight-real-splits`
(enable `manifest.preflight_allowed`) OR the actual smoke run — NOT taken here. The uploaded PDF is a stale
2a-only snapshot and is NOT the writing baseline for this branch.
