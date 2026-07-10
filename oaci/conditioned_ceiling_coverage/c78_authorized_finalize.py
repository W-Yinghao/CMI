"""Finalize authorized C78 only after independent red-team and regressions."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from . import c78_authorized_collect as collect
from . import c78_authorized_common as common
from . import c78_seed3_instrumented_pilot as c78


MAIN_REPORT = c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.md"
RESULT_JSON = c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.json"
RED_TEAM = c78.REPORT_DIR / "C78_AUTHORIZED_RED_TEAM_VERIFICATION.md"
CANONICAL_RED_TEAM = c78.REPORT_DIR / "C78_RED_TEAM_VERIFICATION.md"
NO_AUTH_REPORT = c78.REPORT_DIR / "C78_NO_AUTH_BASELINE.md"
NO_AUTH_JSON = c78.REPORT_DIR / "C78_NO_AUTH_BASELINE.json"
NO_AUTH_RED_TEAM = c78.REPORT_DIR / "C78_NO_AUTH_RED_TEAM_VERIFICATION.md"
HANDOFF = Path("oaci/OACI_CODEX_HANDOFF.md")
LOG_ROOT = c78.EXTERNAL_ROOT / "logs"


def _rows(name: str) -> list[dict[str, str]]:
    with open(c78.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(c78.TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _latest_log(prefix: str) -> tuple[Path, Path, int]:
    paths = sorted(LOG_ROOT.glob(f"{prefix}_*.out"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not paths:
        raise RuntimeError(f"missing C78 authorized regression log: {prefix}")
    stdout = paths[-1]
    return stdout, stdout.with_suffix(".err"), int(stdout.stem.rsplit("_", 1)[1])


def collect_regressions() -> list[dict[str, Any]]:
    rows = []
    for suite, prefix in (
        ("focused_C78", "c78-reg-focused"),
        ("C65_C78", "c78-reg-c65"),
        ("C23_C78", "c78-reg-c23"),
        ("full_OACI", "c78-reg-full"),
    ):
        stdout, stderr, job_id = _latest_log(prefix)
        text = stdout.read_text()
        matches = re.findall(r"(\d+) passed(?:, [^\n]+)? in [0-9.]+s", text)
        stderr_text = stderr.read_text() if stderr.is_file() else "missing"
        if len(matches) != 1 or stderr_text or "failed" in text.lower():
            raise RuntimeError(f"C78 authorized regression failed: {suite} job {job_id}")
        skipped = 0
        skipped_match = re.search(r"(\d+) skipped", text)
        if skipped_match:
            skipped = int(skipped_match.group(1))
        rows.append({
            "suite": suite, "job_id": job_id, "passed_tests": int(matches[0]),
            "skipped_tests": skipped, "stderr_empty": 1, "status": "passed",
            "stdout_sha256": c78.sha256_file(stdout), "stderr_sha256": c78.sha256_file(stderr),
        })
    return rows


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def _archive_no_auth() -> None:
    NO_AUTH_REPORT.write_bytes(_git_blob(common.NO_AUTH_RESULT_COMMIT, "oaci/reports/C78_SEED3_INSTRUMENTED_PILOT.md"))
    NO_AUTH_JSON.write_bytes(_git_blob(common.NO_AUTH_RESULT_COMMIT, "oaci/reports/C78_SEED3_INSTRUMENTED_PILOT.json"))
    NO_AUTH_RED_TEAM.write_bytes(_git_blob(common.NO_AUTH_RESULT_COMMIT, "oaci/reports/C78_RED_TEAM_VERIFICATION.md"))


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timing_audit(state: dict[str, Any]) -> None:
    lock = common.load_execution_lock()
    field = common.field_frozen_path(lock)
    views = common.primary_input_gate_path(lock)
    instrumentation = common.instrumentation_gate_path(lock)
    collect_state = collect.STATE_PATH
    c78.TIMING_PATH.write_text(
        "# C78 Protocol Timing Audit\n\n"
        f"- Protocol anchor: `{c78._protocol_commit()}`.\n"
        f"- Protocol SHA-256: `{lock['protocol_sha256']}`.\n"
        f"- No-auth result commit: `{common.NO_AUTH_RESULT_COMMIT}`.\n"
        f"- Exact CLI authorization received: `{lock['authorization']['received_at_utc']}`.\n"
        "- First authorized GPU submission: job `892830` (synthetic canary failed before data).\n"
        "- Successful source-only training: job `892832`.\n"
        f"- FIELD_FROZEN materialized: `{_iso_mtime(field)}`.\n"
        f"- Post-freeze physical views materialized: `{_iso_mtime(views)}`.\n"
        f"- Instrumentation complete: `{_iso_mtime(instrumentation)}`.\n"
        f"- First target endpoint smoke collection: `{_iso_mtime(collect_state)}`.\n"
        "- Seed-4 access: `never`.\n"
        "- Full seed-3 expansion: `not authorized`.\n\n"
        "The target endpoint smoke occurred only after checkpoint retention and all physical instrumentation manifests were frozen. C78 remains a single-target pipeline canary.\n"
    )


def _main_report(state: dict[str, Any], regressions: list[dict[str, Any]]) -> str:
    trajectory = _rows("source_trajectory_sanity.csv")
    geometry = _rows("effective_multiplicity_top_gap_smoke.csv")
    runtime = {row["stage"]: row for row in _rows("training_runtime_ledger.csv")}
    storage = _rows("actual_compute_storage_summary.csv")[0]
    repairs = _rows("authorized_red_team_repair_ledger.csv")
    blocking = [row for row in _rows("authorized_red_team_checks.csv") if row["blocking"] == "1"]
    regression_text = ", ".join(
        f"{row['suite']} {row['passed_tests']} green" + (f", {row['skipped_tests']} expected skip" if row["skipped_tests"] else "") + f" (job {row['job_id']})"
        for row in regressions
    )
    return f"""# C78 — Seed-3 OACI+ERM Instrumented Training Pilot / Full-Field Expansion Gate

**Final gate:** `{state['final_gate_candidate']}`

**Primary:** `C78-A_seed3_OACI_ERM_pilot_executed_and_validated`

**Secondary active:** `C78-S1 + C78-S2 + C78-S3 + C78-S4 + C78-S5 + C78-S6 + C78-S7 + C78-S8 + C78-S9 + C78-S11`

## Dual-mode provenance

```text
no-auth baseline commit:  {common.NO_AUTH_RESULT_COMMIT[:7]}
no-auth gate:             PILOT_READY_BUT_NOT_AUTHORIZED
authorized worker commit: 4ac865f (determinism repair 44781eb)
successful training job:  892832
```

The no-auth baseline remains evidence that prompt prose cannot trigger execution. The later exact CLI token authorized only the locked 82-unit field. Job `892830` failed its synthetic deterministic canary before any data load; the failure was retained and repaired prospectively before job `892832`.

## Execution result

```text
planned / actual units:       82 / 82
ERM anchors:                   2
OACI trajectory checkpoints: 80
SRC units:                     0
levels completed:              0 + 1
checkpoint hash replay:       82 / 82
optimizer hash replay:        82 / 82
GPU:                           Tesla V100-PCIE-32GB
GPU wall hours:                {float(runtime['source_only_training']['GPU_hours_measured']):.6f}
peak GPU memory:               {int(float(runtime['source_only_training']['peak_GPU_memory_bytes'])) / 2**30:.3f} GiB
external payload:              {int(storage['actual_external_bytes']) / 2**30:.3f} GiB
```

CPU peak RAM is unavailable because the Slurm accounting database refused the post-completion query. No estimate is substituted. GPU runtime/memory, process CPU time, storage bytes, checkpoint counts, and cache rows are measured.

## Target isolation

The training process loaded exactly source-training subjects `[1,2,3,7,8,9]` (`3456` rows). It loaded zero target rows, zero target labels, and zero source-audit rows. All 82 retention decisions, checkpoint hashes, optimizer hashes, and sidecars were frozen before target/source-audit provisioning.

Post-freeze views are physically separate:

```text
strict-source input rows:        4,608
target-unlabeled input rows:       576
construction label rows:           261
evaluation label rows:             315
same-label-oracle rows:             576
```

The primary instrumentation descriptor contains no target label, split-role, evaluation, or oracle path.

## Instrumentation

```text
instrumented units:             82 / 82
strict-source cache rows:       377,856
target-unlabeled cache rows:     47,232
Wz+b/logit max error:                 0
softmax max error:                    0
hook-z max error:                     0
repeat logits/z max error:            0
failed units:                          0
```

The registered C75/C76 source and target-unlabeled functional/architecture blocks are computable. C78 does not test their predictive qualification or reopen representation-feature mining.

## Smoke-only observations

Target endpoints were opened only after freeze for pipeline and future-power sanity. No best checkpoint ID or recommendation was emitted.

```text
level 0: candidate M=41, top-two bAcc gap={float(geometry[0]['best_minus_second_bAcc_gap']):.6f}, epsilon-optimal count={geometry[0]['epsilon_optimal_count']}
level 1: candidate M=41, top-two bAcc gap={float(geometry[1]['best_minus_second_bAcc_gap']):.6f}, epsilon-optimal count={geometry[1]['epsilon_optimal_count']}
random top-1 baseline: {float(geometry[0]['uniform_random_top1']):.6f}
```

The trajectory stress is material to interpretation:

```text
level 0 OACI source-risk feasible: {trajectory[0]['OACI_risk_feasible_count']}/40; lambda max {float(trajectory[0]['lambda_max']):.1f}
level 1 OACI source-risk feasible: {trajectory[1]['OACI_risk_feasible_count']}/40; lambda max {float(trajectory[1]['lambda_max']):.1f}; surrogate min {float(trajectory[1]['train_surrogate_min']):.3f}
```

These are finite pipeline outputs, not evidence of training stability, measurement-control replication, or target control.

## Red team

Independent authorized red-team passed `{len(blocking)}/{len(blocking)}` blocking checks, with four nonblocking stress/caveat checks and `{len(repairs)}` recorded repairs. Key repairs:

""" + "\n".join(f"- `{row['item']}`: {row['resolution']}" for row in repairs) + f"""

Regression: {regression_text}.

## Decision

C78 validates the exact historical OACI+ERM seed-3 training and instrumentation path for one target and two deletion levels. It does not constitute multi-regime replication, measurement-control replication, cross-regime transport, source/target-unlabeled escape-hatch evidence, representation mechanism evidence, seed-level confirmation, a selector, or checkpoint control.

SRC was not exercised. Therefore the 1,458-unit full seed-3 field is not ready and not authorized. PM review must choose a prospective SRC canary or demonstrate that SRC shares the exact validated execution/instrumentation path before any expansion.
"""


def _handoff_section(state: dict[str, Any], regressions: list[dict[str, Any]]) -> str:
    regression = ", ".join(f"{row['suite']}={row['passed_tests']}" for row in regressions)
    return f"""## 0. Current continuation state (2026-07-10)

The detailed C23-C31 history below remains background; the authoritative tip is the authorized C78 dual-mode milestone:

```text
C78 no-auth commit: {common.NO_AUTH_RESULT_COMMIT[:7]}
C78 execution lock: 4ac865f
C78 successful job: 892832
C78 final gate:     {state['final_gate_candidate']}
C78 primary:        C78-A_seed3_OACI_ERM_pilot_executed_and_validated
```

C78 first proved the exact-token guard with a no-auth baseline, then executed the locked target-4/seed-3/levels-0+1 OACI+ERM field after exact authorization. Job 892830 failed the synthetic deterministic gate before data; the repair was committed and job 892832 produced 82/82 checkpoints, optimizer states, and sidecars from source-training subjects only. Target rows/labels and source-audit rows in training were zero.

Post-freeze CPU instrumentation produced 377,856 strict-source and 47,232 target-unlabeled rows over all 82 units. Every checkpoint/optimizer hash replayed; Wz/logit, softmax, hook, and repeat errors were zero. Authorized red-team passed 52/52 blocking checks.

This is an execution/instrumentation canary, not a multi-regime or scientific replication. Both levels show only 23/40 source-risk-feasible OACI points, lambda reaches 20, and level-1 surrogate reaches -49.694. SRC remains unexercised, so the 1,458-unit field is not ready or authorized. Seed 4 and BNCI2014_004 remain untouched.

Regression: `{regression}`.

Authoritative artifacts:

```text
oaci/reports/C78_SEED3_INSTRUMENTED_PILOT.md
oaci/reports/C78_SEED3_INSTRUMENTED_PILOT.json
oaci/reports/C78_AUTHORIZED_RED_TEAM_VERIFICATION.md
oaci/reports/C78_PROTOCOL_TIMING_AUDIT.md
oaci/reports/c78_tables/artifact_manifest.csv
```

Wait for PM review. No SRC canary, full seed-3 expansion, seed 4, or BNCI2014_004 work is authorized.

---
"""


def _update_handoff(state: dict[str, Any], regressions: list[dict[str, Any]]) -> None:
    text = HANDOFF.read_text()
    start = text.index("## 0. Current continuation state")
    end = text.index("\n## 1. What C23–C31 established", start)
    HANDOFF.write_text(text[:start] + _handoff_section(state, regressions) + "\n" + text[end:])


def _artifact_manifest() -> list[dict[str, Any]]:
    paths = [
        *sorted(c78.TABLE_DIR.glob("*")),
        *sorted(c78.REPORT_DIR.glob("C78_*.md")),
        *sorted(c78.REPORT_DIR.glob("C78_*.json")),
        *sorted(c78.REPORT_DIR.glob("C78_*.sha256")),
        *sorted(Path("oaci/conditioned_ceiling_coverage").glob("c78_*.py")),
        *sorted(Path("oaci").glob("slurm_c78_*.sh")),
        *sorted(Path("oaci/tests").glob("test_c78_*.py")),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.is_file() or path.name in {"artifact_manifest.csv", "large_artifact_scan.csv"}:
            continue
        seen.add(path)
        row_count = 0
        if path.suffix == ".csv":
            with open(path, newline="") as stream:
                row_count = sum(1 for _ in csv.DictReader(stream))
        rows.append({
            "path": str(path), "sha256": c78.sha256_file(path),
            "size_bytes": path.stat().st_size, "row_count": row_count,
            "raw_EEG_or_weights": 0, "tracked_or_to_be_tracked": 1,
        })
    return rows


def finalize() -> dict[str, Any]:
    checks = _rows("authorized_red_team_checks.csv")
    blocking = [row for row in checks if row["blocking"] == "1"]
    if len(blocking) != 52 or any(row["passed"] != "1" for row in blocking):
        raise RuntimeError("C78 authorized finalization requires 52/52 blocking red-team checks")
    if "Final status: `PASS`" not in RED_TEAM.read_text():
        raise RuntimeError("C78 authorized red-team is not PASS")
    regressions = collect_regressions()
    _write_csv("regression_verification.csv", regressions)
    state = json.loads(collect.STATE_PATH.read_text())
    _archive_no_auth()
    _timing_audit(state)
    MAIN_REPORT.write_text(_main_report(state, regressions))
    CANONICAL_RED_TEAM.write_text(RED_TEAM.read_text())
    result = {
        "schema_version": "c78_seed3_instrumented_pilot_authorized_result_v1",
        "milestone": "C78", "final_gate": state["final_gate_candidate"],
        "dual_mode_provenance": {
            "no_auth_commit": common.NO_AUTH_RESULT_COMMIT,
            "no_auth_gate": "PILOT_READY_BUT_NOT_AUTHORIZED",
            "authorized_execution_lock_commit": "4ac865f",
            "determinism_repair_commit": "44781eb",
            "successful_training_job": 892832,
        },
        "protocol": {"commit": c78._protocol_commit(), "sha256": state["protocol_sha256"], "execution_lock_sha256": state["execution_lock_sha256"]},
        "scope": {"planned_units": 82, "actual_units": 82, "ERM_anchors": 2, "OACI_checkpoints": 80, "SRC_units": 0, "target": 4, "seed": 3, "levels": [0, 1]},
        "execution_boundary": {"training_attempted": 1, "real_forward_attempted": 1, "GPU_used": 1, "target_rows_during_training": 0, "target_labels_during_training": 0, "source_audit_rows_during_training": 0, "seed4_access": 0, "BNCI2014_004_access": 0},
        "runtime": state["execution"], "instrumentation": state["instrumentation"],
        "taxonomy": state["taxonomy"], "smoke": state["smoke"],
        "red_team": {"blocking_passed": 52, "blocking_total": 52, "nonblocking_checks": 4, "repair_count": len(_rows("authorized_red_team_repair_ledger.csv"))},
        "regression": regressions,
        "claims": {
            **state["claims"],
            "SRC_exercised": False,
            "full_seed3_ready": False,
        },
    }
    RESULT_JSON.write_bytes(c78.canonical_bytes(result) + b"\n")
    _update_handoff(state, regressions)
    manifest = _artifact_manifest()
    _write_csv("artifact_manifest.csv", manifest)
    large = [{
        "path": row["path"], "size_bytes": row["size_bytes"],
        "over_50MiB": int(int(row["size_bytes"]) > c78.MAX_GIT_PAYLOAD),
        "passed": int(int(row["size_bytes"]) <= c78.MAX_GIT_PAYLOAD),
    } for row in manifest]
    _write_csv("large_artifact_scan.csv", large)
    if any(row["over_50MiB"] for row in large):
        raise RuntimeError("C78 authorized report artifact exceeds 50 MiB")
    print(json.dumps({"gate": result["final_gate"], "red_team": "52/52", "regressions": regressions, "artifacts": len(manifest)}, sort_keys=True))
    return result


if __name__ == "__main__":
    finalize()
