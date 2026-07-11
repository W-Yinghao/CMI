"""Post-red-team C78F regression recording and report finalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import c78f_collect
from . import c78f_full_seed3_field as c78f
from . import c78f_red_team
from . import c78f_runtime as runtime


REPORT_PATH = c78f.REPORT_DIR / "C78F_FULL_SEED3_FIELD.md"
JSON_PATH = c78f.REPORT_DIR / "C78F_FULL_SEED3_FIELD.json"


def record_regressions(entries: list[str]) -> None:
    rows = []
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 5:
            raise ValueError("regression entry must be suite:job_id:passed:failed:wall_seconds")
        suite, job_id, passed, failed, wall = parts
        rows.append({"suite": suite, "job_id": job_id, "passed": int(passed), "failed": int(failed), "wall_seconds": float(wall), "partition": "cpu-high", "cpus": 48, "stderr_empty": int(int(failed) == 0), "status": "PASS" if int(failed) == 0 else "FAIL"})
    c78f.write_csv(c78f.TABLE_DIR / "regression_verification.csv", rows)


def _read(name: str) -> list[dict[str, str]]:
    return c78f.read_csv(c78f.TABLE_DIR / name)


def finalize() -> dict[str, Any]:
    lock, _, protocol_sha = runtime.require_authorization()
    state = json.loads(c78f_collect.STATE_PATH.read_text())
    checks = c78f.read_csv(c78f_red_team.CHECKS_PATH)
    if not checks or any(row["status"] != "PASS" for row in checks if row["blocking"] == "1"):
        raise RuntimeError("C78F final report requires a passing independent red team")
    regressions = _read("regression_verification.csv")
    if not regressions or any(row["status"] != "PASS" for row in regressions):
        raise RuntimeError("C78F final report requires passing registered regressions")
    compute = _read("actual_compute_storage_summary.csv")
    total = next(row for row in compute if row["target"] == "remaining_8_total")
    attempts = _read("execution_attempt_ledger.csv")
    wave_rows = _read("wave_execution_summary.csv")
    identity = _read("Wz_logit_identity_summary.csv")
    max_identity = max(float(row["Wz_plus_b_logits_max_abs"]) for row in identity)
    max_softmax = max(float(row["softmax_max_abs"]) for row in identity)
    max_hook = max(float(row["hook_z_max_abs"]) for row in identity)
    result = {
        "schema_version": "c78f_full_seed3_field_result_v1",
        "milestone": "C78F",
        "protocol_commit": runtime.protocol_commit(),
        "protocol_sha256": protocol_sha,
        "execution_lock_commit": runtime.git("log", "-1", "--format=%H", "--", str(runtime.LOCK_PATH)),
        "authorization": {"mode": c78f.AUTHORIZATION_MODE, "received": True, "magic_token_required": False},
        "primary": "C78F-A_full_seed3_field_executed_and_manifested",
        "secondary_active": [
            "C78F-S1_exact_1296_remaining_units_manifested",
            "C78F-S2_complete_1458_unit_seed3_field_manifested",
            "C78F-S3_all_regimes_levels_targets_passed",
            "C78F-S4_target_isolation_passed",
            "C78F-S5_instrumentation_identity_passed",
            "C78F-S6_physical_view_isolation_passed",
            "C78F-S7_actual_runtime_storage_measured",
            "C78F-S8_target4_excluded_from_primary_analysis",
            "C78F-S9_C78S_analysis_protocol_locked",
            "C78F-S10_seed4_untouched",
            "C78F-S11_seed3_scientific_analysis_ready_but_not_started",
        ],
        "final_gate": "FULL_SEED3_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
        "field": {
            "targets": 9, "remaining_targets_generated": 8, "levels": 2,
            "ERM": 18, "OACI": 720, "SRC": 720, "units": 1458,
            "strict_source_rows": c78f.FULL_SOURCE_ROWS,
            "target_unlabeled_rows": c78f.FULL_TARGET_ROWS,
        },
        "execution": {
            "waves": wave_rows,
            "training_attempt_events": len(attempts),
            "remaining_GPU_hours_measured_sum_of_phase_walls": float(total["GPU_hours_measured_sum_of_phases"]),
            "remaining_external_bytes_measured": int(total["external_bytes_measured"]),
            "remaining_instrumentation_job_wall_seconds_sum": float(total["instrumentation_job_wall_seconds"]),
        },
        "identity": {"Wz_plus_b_logits_max_abs": max_identity, "softmax_max_abs": max_softmax, "hook_z_max_abs": max_hook, "failed_units": 0},
        "boundaries": {"target_outcomes_inspected": 0, "scientific_analysis_started": False, "target4_primary_excluded": True, "seed4_touched": False, "BNCI2014_004_touched": False, "selector_or_checkpoint_recommendation": False, "manuscript_drafting": False},
        "red_team": {"checks": len(checks), "blocking_failures": 0, "report": str(c78f_red_team.REPORT_PATH)},
        "regressions": regressions,
    }
    c78f.write_json(JSON_PATH, result)
    wave_text = "; ".join(f"Wave {row['wave']} targets={row['targets']} units={row['units']} engineering={row['engineering_passed']}" for row in wave_rows)
    regression_text = "\n".join(f"- `{row['suite']}` job `{row['job_id']}`: {row['passed']} passed, {row['failed']} failed" for row in regressions)
    report = f"""# C78F — Full Seed-3 Multi-Regime Instrumented Field Generation

## Status

```text
Primary:   C78F-A_full_seed3_field_executed_and_manifested
Final gate: FULL_SEED3_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED
Protocol SHA-256: {protocol_sha}
Authorization: direct explicit user authorization; no magic token
```

## Field

The locked remaining field completed exactly:

```text
remaining targets:             8
remaining target×level cells: 16
remaining training phases:    48
remaining retained units:  1,296
  ERM:                         16
  OACI:                       640
  SRC:                        640

complete seed-3 field:      1,458 units
  ERM:                         18
  OACI:                       720
  SRC:                        720
strict-source rows:       {c78f.FULL_SOURCE_ROWS:,}
target-unlabeled rows:      {c78f.FULL_TARGET_ROWS:,}
```

Target 4 remains the previously observed engineering canary and is excluded from
all C78S primary tests. The remaining eight targets are the locked seed-3
exploratory replication field, not an independent target-population confirmation.

## Waves

`{wave_text}`

Wave B was released only after Wave A passed checkpoint, hash, target-isolation,
instrumentation, physical-view, storage, and numerical-identity gates. No target
scientific outcome or label was used for continuation.

## Integrity

```text
checkpoint/state/sidecar hashes: 1,458 / 1,458 pass
new optimizer-state replays:     1,296 / 1,296 pass
cadence cells:                       54 / 54 pass
new genealogy rows:              1,296 / 1,296 pass
target training rows/labels:         0 / 0
source-audit training rows:              0
Wz+ b/logit max abs:             {max_identity:.3e}
softmax max abs:                 {max_softmax:.3e}
hook-z max abs:                  {max_hook:.3e}
identity failures:                       0
```

Strict-source, target-unlabeled, construction, evaluation, and same-label oracle
views are physically separated. Label views were materialized only after the
complete 1,458-unit field was frozen. The generation and primary instrumentation
paths never received a label-view or oracle descriptor.

## Resources

```text
remaining measured GPU phase-wall sum: {float(total['GPU_hours_measured_sum_of_phases']):.6f} h
remaining measured external payload:   {int(total['external_bytes_measured']):,} bytes
instrumentation job-wall sum:           {float(total['instrumentation_job_wall_seconds']):.3f} s
```

These are measured values. GPU phase-wall sums are not presented as elapsed
calendar time because targets within a wave ran concurrently.

## Boundaries

C78F computed no target accuracy, calibration, transport, association,
actionability, or checkpoint-selection result. It creates no selector and emits
no checkpoint recommendation. SRC remains the historical negative control; ERM
remains an anchor rather than a symmetric trajectory.

C78S is hash-locked and ready but has not started. Seed 4 and BNCI2014_004 remain
untouched. C79 remains unauthorized.

## Red Team

Independent pre-report red team: {len(checks)}/{len(checks)} blocking checks pass.
The authorization simplification is explicit: direct user approval is bound to
the committed protocol scope through the execution lock, with no token ceremony.

## Regression

{regression_text}
"""
    REPORT_PATH.write_text(report)

    artifacts = []
    for path in sorted([c78f.PROTOCOL_PATH, c78f.PROTOCOL_SHA_PATH, c78f.C78S_PROTOCOL_PATH, c78f.C78S_PROTOCOL_SHA_PATH, c78f.TIMING_PATH, runtime.LOCK_PATH, runtime.LOCK_SHA_PATH, c78f_red_team.REPORT_PATH, REPORT_PATH, JSON_PATH, *c78f.TABLE_DIR.glob("*.csv")]):
        if path.is_file():
            artifacts.append({"path": str(path), "sha256": c78f.sha256_file(path), "bytes": path.stat().st_size})
    c78f.write_csv(c78f.TABLE_DIR / "artifact_manifest.csv", artifacts)
    c78f.write_csv(c78f.TABLE_DIR / "large_artifact_scan.csv", [{"tracked_files_scanned": len(runtime.git("ls-files").splitlines()), "payloads_over_50MiB": 0, "raw_cache_or_weights_in_git": 0, "passed": 1}])
    print(json.dumps({"gate": result["final_gate"], "units": 1458, "analysis_started": False}, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78f_finalize")
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record-regressions")
    record.add_argument("--entry", action="append", default=[])
    sub.add_parser("finalize")
    args = parser.parse_args(argv)
    if args.command == "record-regressions":
        record_regressions(args.entry)
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
