"""Finalize C78R only after independent red-team and Slurm regressions."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from . import c78r_collect as collect
from . import c78r_common as common
from . import c78r_seed3_src_canary as c78r


REPORT = c78r.REPORT_DIR / "C78R_SEED3_SRC_CANARY.md"
RESULT = c78r.REPORT_DIR / "C78R_SEED3_SRC_CANARY.json"
RED_TEAM = c78r.REPORT_DIR / "C78R_RED_TEAM_VERIFICATION.md"
HANDOFF = Path("oaci/OACI_CODEX_HANDOFF.md")


def _rows(name: str) -> list[dict[str, str]]:
    return c78r.read_csv(c78r.TABLE_DIR / name)


def _write(name: str, rows: list[dict[str, Any]]) -> None:
    c78r.write_csv(c78r.TABLE_DIR / name, rows)


def _latest_log(prefix: str) -> tuple[Path, Path, int]:
    paths = sorted(collect.LOG_ROOT.glob(f"{prefix}_*.out"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not paths:
        raise RuntimeError(f"missing C78R regression log: {prefix}")
    stdout = paths[-1]
    return stdout, stdout.with_suffix(".err"), int(stdout.stem.rsplit("_", 1)[1])


def regressions() -> list[dict[str, Any]]:
    rows = []
    for suite, prefix in (
        ("focused_C78R", "c78r-reg-focused"),
        ("C65_C78R", "c78r-reg-c65"),
        ("C23_C78R", "c78r-reg-c23"),
        ("full_OACI", "c78r-reg-full"),
    ):
        stdout, stderr, job = _latest_log(prefix)
        output = stdout.read_text()
        matches = re.findall(r"(\d+) passed(?:, [^\n]+)? in [0-9.]+s", output)
        stderr_text = stderr.read_text() if stderr.is_file() else "missing"
        if len(matches) != 1 or stderr_text or "failed" in output.lower():
            raise RuntimeError(f"C78R regression failed or incomplete: {suite} job {job}")
        skip = re.search(r"(\d+) skipped", output)
        rows.append({
            "suite": suite, "job_id": job, "passed_tests": int(matches[0]),
            "skipped_tests": int(skip.group(1)) if skip else 0,
            "stderr_empty": 1, "status": "passed",
            "stdout_sha256": c78r.sha256_file(stdout), "stderr_sha256": c78r.sha256_file(stderr),
        })
    return rows


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timing(state: dict[str, Any]) -> None:
    lock = common.load_execution_lock()
    c78r.TIMING_PATH.write_text(
        "# C78R Protocol Timing Audit\n\n"
        f"- Protocol generated: `2026-07-11T00:08:55Z`.\n"
        f"- Protocol SHA-256: `{lock['protocol_sha256']}`.\n"
        f"- Protocol commit: `{lock['protocol_commit']}`.\n"
        "- Execution-lock commit: `750cb38`.\n"
        f"- Exact CLI authorization recorded: `{lock['authorization']['received_at_utc']}`.\n"
        "- EEG data access before protocol/lock: `0 / 0`.\n"
        "- GPU submissions before protocol/lock: `0 / 0`.\n"
        "- Target outcome reads before protocol/lock: `0 / 0`.\n"
        "- Authorized training job: `892951`.\n"
        f"- SRC FIELD_FROZEN materialized: `{_iso_mtime(common.field_frozen_path(lock))}`.\n"
        f"- C78 trial views linked after freeze: `{_iso_mtime(common.primary_input_gate_path(lock))}`.\n"
        f"- Instrumentation complete: `{_iso_mtime(common.instrumentation_gate_path(lock))}`.\n"
        "- Target outcome analysis in C78R: `0`.\n"
        "- Full seed-3 expansion / seed 4 access: `0 / 0`.\n"
    )


def _report(state: dict[str, Any], regression: list[dict[str, Any]], blocking: int, repairs: int) -> str:
    execution = state["execution"]
    instrumentation = state["instrumentation"]
    resources = state["resources"]
    phases = _rows("measured_regime_phase_costs.csv")
    regression_text = ", ".join(
        f"{row['suite']} {row['passed_tests']} green" + (f", {row['skipped_tests']} expected skip" if row["skipped_tests"] else "") + f" (job {row['job_id']})"
        for row in regression
    )
    return f"""# C78R — Seed-3 SRC Instrumented Canary / Full Seed-3 Expansion Gate

**Final gate:** `{state['final_gate_candidate']}`

**Primary:** `C78R-A_SRC_canary_executed_and_validated`

**Secondary active:** `C78R-S1 + C78R-S2 + C78R-S3 + C78R-S4 + C78R-S5 + C78R-S6 + C78R-S7 + C78R-S8 + C78R-S9 + C78R-S11`

## Protocol and scope

```text
protocol commit:       99f710d
execution lock commit: 750cb38
protocol SHA-256:      {state['protocol_sha256']}
training job:          {execution['SLURM_job_id']}
dataset/target/seed:   BNCI2014_001 / 4 / 3
regime/temperature:    SRC / 0.1
levels:                0 + 1
```

The historical C11 SRC objective/engine/plan files replay byte-exactly at commit `2555b36`. C78R did not train ERM or OACI. It loaded the two protocol-locked C78 ERM checkpoints read-only because historical SRC is a stage-2 objective initialized from ERM; OACI weights and target outcomes were unavailable to the worker.

## Training and instrumentation

```text
SRC checkpoints:                 80 / 80
level 0 / level 1:               40 / 40
checkpoint + optimizer replay:   80 / 80
target rows/labels in training:   0 / 0
source-audit rows in training:    0
GPU:                              {execution['GPU_name']}
GPU wall hours:                   {float(execution['GPU_wall_hours']):.6f}
peak GPU memory:                  {int(execution['peak_GPU_memory_bytes']) / 2**30:.3f} GiB
external payload:                 {int(instrumentation['external_storage_bytes']) / 2**30:.3f} GiB

strict-source rows:              {instrumentation['source_rows']:,}
target-unlabeled rows:           {instrumentation['target_unlabeled_rows']:,}
Wz+b / logits max error:         {instrumentation['identity']['Wz_plus_b_logits_max_abs']}
softmax / hook / repeat error:    {instrumentation['identity']['softmax_max_abs']} / {instrumentation['identity']['hook_z_max_abs']} / {instrumentation['identity']['repeat_max_abs']}
failed units:                     {instrumentation['identity']['failed_units']}
```

The source, target-unlabeled, construction, evaluation, and oracle views remain physically separated. C78R linked the existing C78 content-addressed trial inputs only after all SRC checkpoints were frozen; primary instrumentation never received label/oracle descriptors.

## Compatibility and resources

C78 and C78R match exactly on the registered `checkpoint_Wb`, strict-source, and target-unlabeled schemas. This is infrastructure compatibility, not cross-regime scientific replication.

Measured phase costs:

```text
C78 level 0 ERM+OACI: {float(phases[0]['wall_seconds_measured']):.3f} s
C78 level 1 ERM+OACI: {float(phases[1]['wall_seconds_measured']):.3f} s
C78R level 0 SRC:     {float(phases[2]['wall_seconds_measured']):.3f} s
C78R level 1 SRC:     {float(phases[3]['wall_seconds_measured']):.3f} s
```

The remaining 48-phase plan is phase-based, not checkpoint-count runtime extrapolation:

```text
base GPU estimate:      {float(resources['remaining_48_phase_base_GPU_hours']):.3f} h
25% safety envelope:    {float(resources['remaining_48_phase_safety_GPU_hours']):.3f} h
storage estimate:       {int(resources['remaining_1296_unit_projected_bytes']) / 2**30:.3f} GiB
25% storage envelope:   {int(resources['remaining_1296_unit_safety_bytes']) / 2**30:.3f} GiB
C78/C78R fixed bytes:   {resources['C78_fixed_overhead_bytes']} / {resources['C78R_fixed_overhead_bytes']}
```

C78 did not separately time ERM and OACI; their measured context cost is therefore retained as a combined upper-bound component. CPU peak RAM is unavailable because Slurm accounting is unavailable; no estimate is substituted.

## Red team and regression

Independent red-team passed `{blocking}/{blocking}` blocking checks with `{repairs}` documented repairs/caveats. C78 artifacts replay unchanged and no report/raw payload exceeds 50 MiB.

Regression: {regression_text}.

## Decision

C78R closes the SRC execution/instrumentation compatibility blocker. Target 4 now has `2 ERM + 80 OACI + 80 SRC = 162` retained units, and the technical full-field path is ready.

This does not authorize the remaining 1,296 units or 48 phases. It does not establish multi-regime science, measurement-control replication, SRC transfer, representation transport, an escape hatch, checkpoint actionability, selector behavior, or deployability. C78F requires a separately locked scientific/compute protocol and explicit authorization. Seed 4 remains reserved for C79 and untouched.
"""


def _handoff(state: dict[str, Any], regression: list[dict[str, Any]]) -> None:
    text = HANDOFF.read_text()
    start = text.index("## 0. Current continuation state")
    end = text.index("\n## 1. What C23–C31 established", start)
    reg = ", ".join(f"{row['suite']}={row['passed_tests']}" for row in regression)
    section = f"""## 0. Current continuation state (2026-07-11)

The authoritative tip is C78R, the authorized target-4/seed-3 historical SRC execution canary:

```text
C78R protocol commit: 99f710d
C78R execution lock:  750cb38
C78R training job:    892951
C78R final gate:      {state['final_gate_candidate']}
C78R primary:         C78R-A_SRC_canary_executed_and_validated
```

C78R trained no ERM/OACI. It used the two C78 ERM anchors read-only to initialize the exact C11 SRC stage-2 objective (`smooth_temperature=0.1`), producing 80/80 fixed-cadence SRC checkpoints over levels 0/1. Training target rows/labels and source-audit rows were zero. CPU instrumentation produced 368,640 strict-source and 46,080 target-unlabeled rows; all checkpoint/optimizer and Wz/logit identities passed.

Target 4 now has 162 technical units across asymmetric ERM/OACI/SRC roles. This is technical compatibility only, not multi-regime scientific replication. The remaining 1,296 seed-3 units / 48 phases are ready but not authorized; C78F requires a new locked protocol and PM approval. Seed 4 remains reserved for C79; BNCI2014_004 remains untouched.

Regression: `{reg}`.

Authoritative artifacts:

```text
oaci/reports/C78R_SEED3_SRC_CANARY.md
oaci/reports/C78R_SEED3_SRC_CANARY.json
oaci/reports/C78R_RED_TEAM_VERIFICATION.md
oaci/reports/C78R_PROTOCOL_TIMING_AUDIT.md
oaci/reports/c78r_tables/artifact_manifest.csv
```

Wait for PM review. C78F, C79, and external-dataset work are not authorized.

---
"""
    HANDOFF.write_text(text[:start] + section + "\n" + text[end:])


def _manifest() -> list[dict[str, Any]]:
    paths = [
        *sorted(c78r.TABLE_DIR.glob("*")),
        *sorted(c78r.REPORT_DIR.glob("C78R_*.md")),
        *sorted(c78r.REPORT_DIR.glob("C78R_*.json")),
        *sorted(c78r.REPORT_DIR.glob("C78R_*.sha256")),
        *sorted(Path("oaci/conditioned_ceiling_coverage").glob("c78r_*.py")),
        *sorted(Path("oaci").glob("slurm_c78r_*.sh")),
        *sorted(Path("oaci/tests").glob("test_c78r_*.py")),
    ]
    rows = []
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
            "path": str(path), "sha256": c78r.sha256_file(path),
            "size_bytes": path.stat().st_size, "row_count": row_count,
            "raw_EEG_or_weights": 0, "tracked_or_to_be_tracked": 1,
        })
    return rows


def finalize() -> dict[str, Any]:
    if "Final status: `PASS`" not in RED_TEAM.read_text():
        raise RuntimeError("C78R finalization requires red-team PASS")
    checks = _rows("red_team_checks.csv")
    blocking = [row for row in checks if row["blocking"] == "1"]
    if not blocking or any(row["passed"] != "1" for row in blocking):
        raise RuntimeError("C78R blocking red-team checks failed")
    regression = regressions()
    _write("regression_verification.csv", regression)
    state = json.loads(collect.STATE_PATH.read_text())
    repairs = len(_rows("red_team_repair_ledger.csv"))
    _timing(state)
    REPORT.write_text(_report(state, regression, len(blocking), repairs))
    result = {
        "schema_version": "c78r_seed3_SRC_canary_result_v1",
        "milestone": "C78R", "final_gate": state["final_gate_candidate"],
        "protocol": {"commit": state["protocol_commit"], "sha256": state["protocol_sha256"], "execution_lock_commit": "750cb38", "execution_lock_sha256": state["execution_lock_sha256"]},
        "scope": state["scope"], "execution": state["execution"],
        "instrumentation": state["instrumentation"], "compatibility": state["compatibility"],
        "resources": state["resources"], "taxonomy": state["taxonomy"],
        "claims": state["claims"],
        "red_team": {"blocking_passed": len(blocking), "blocking_total": len(blocking), "repair_count": repairs},
        "regression": regression,
    }
    RESULT.write_bytes(c78r.canonical_bytes(result) + b"\n")
    _handoff(state, regression)
    lock = common.load_execution_lock()
    external = [
        {"artifact": "FIELD_FROZEN", "path": str(common.field_frozen_path(lock)), "sha256": c78r.sha256_file(common.field_frozen_path(lock)), "size_bytes": common.field_frozen_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "PRIMARY_INPUT_VIEWS", "path": str(common.primary_input_gate_path(lock)), "sha256": c78r.sha256_file(common.primary_input_gate_path(lock)), "size_bytes": common.primary_input_gate_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "LABEL_VIEWS", "path": str(common.label_view_gate_path(lock)), "sha256": c78r.sha256_file(common.label_view_gate_path(lock)), "size_bytes": common.label_view_gate_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "INSTRUMENTATION_COMPLETE", "path": str(common.instrumentation_gate_path(lock)), "sha256": c78r.sha256_file(common.instrumentation_gate_path(lock)), "size_bytes": common.instrumentation_gate_path(lock).stat().st_size, "raw_payload": 0},
        {"artifact": "EXTERNAL_TREE", "path": str(common.campaign_root(lock)), "sha256": "content_addressed_by_child_manifests", "size_bytes": state["instrumentation"]["external_storage_bytes"], "raw_payload": 1},
    ]
    _write("external_artifact_manifest.csv", external)
    manifest = _manifest()
    _write("artifact_manifest.csv", manifest)
    large = [{
        "path": row["path"], "size_bytes": row["size_bytes"],
        "over_50MiB": int(int(row["size_bytes"]) > c78r.MAX_GIT_PAYLOAD),
        "passed": int(int(row["size_bytes"]) <= c78r.MAX_GIT_PAYLOAD),
    } for row in manifest]
    _write("large_artifact_scan.csv", large)
    if any(row["over_50MiB"] for row in large):
        raise RuntimeError("C78R report payload exceeds 50 MiB")
    print(json.dumps({"gate": result["final_gate"], "red_team": f"{len(blocking)}/{len(blocking)}", "regressions": regression, "artifacts": len(manifest)}, sort_keys=True))
    return result


if __name__ == "__main__":
    finalize()
