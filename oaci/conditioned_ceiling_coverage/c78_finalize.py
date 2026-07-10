"""Finalize the C78 no-auth P0 report only after red-team and regressions."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any

from . import c78_seed3_instrumented_pilot as c78


MAIN_MD = c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.md"
RESULT_JSON = c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.json"
RED_TEAM = c78.REPORT_DIR / "C78_RED_TEAM_VERIFICATION.md"
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
        writer.writeheader()
        writer.writerows(rows)


def _latest_log(prefix: str) -> tuple[Path, Path, int]:
    paths = sorted(LOG_ROOT.glob(f"{prefix}_*.out"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not paths:
        raise RuntimeError(f"missing C78 regression log: {prefix}")
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
            raise RuntimeError(f"C78 regression did not pass cleanly: {suite} job {job_id}")
        rows.append({
            "suite": suite, "job_id": job_id, "passed_tests": int(matches[0]),
            "stderr_empty": 1, "status": "passed",
            "stdout_sha256": c78.sha256_file(stdout),
            "stderr_sha256": c78.sha256_file(stderr),
        })
    return rows


def _main_report(state: dict[str, Any], regressions: list[dict[str, Any]]) -> str:
    protocol = _rows("c78_protocol_replay.csv")
    dummy = _rows("Wz_logit_dummy_ABI.csv")[0]
    storage = _rows("c78_storage_preflight.csv")[0]
    env = next(row for row in _rows("c78_environment_preflight.csv") if "environment_prefix" in row)
    repairs = _rows("red_team_repair_ledger.csv")
    checks = _rows("red_team_checks.csv")
    regression_text = ", ".join(f"{row['suite']} {row['passed_tests']} green (job {row['job_id']})" for row in regressions)
    blocking = [row for row in checks if row["blocking"] == "1"]
    return f"""# C78 — Seed-3 OACI+ERM Instrumented Training Pilot / Full-Field Expansion Gate

**Final gate:** `{state['final_gate_candidate']}`

**Primary execution taxonomy:** `not evaluable; no P1 execution occurred`

**Secondary active:** `C78-S8 + C78-S9 + C78-S11`

## Gate-first result

```text
planned field:          82 units
ERM anchors:             2
OACI trajectory units:  80
SRC units:                0
training attempted:       0
real EEG forward:         0
real EEG rows loaded:     0
GPU requested:            0
checkpoints created:      0
raw cache rows:           0
seed-4 access:            0
BNCI2014_004 access:      0
```

The exact CLI authorization token was not passed to the C78 command. Prompt text, generic approval language, environment variables, whitespace variants, and substring matches were not accepted. This is therefore the required no-training P0 result, not a failed training run.

## Protocol and scope

- C78 protocol anchor: `{state['protocol_commit'][:7]}`.
- Full protocol SHA-256: `{state['protocol_sha256']}`.
- Accepted C77 result: `{c78.PARENT_RESULT_COMMIT[:7]}`; the protocol anchor is its prospective ancestor.
- Explicit unit manifest: `82/82` unique planned units, target `4`, seed `3`, levels `0 + 1`.
- Per level: one shared ERM stage-1 final anchor and OACI epochs `4,9,...,199` (`40` fixed-cadence records).
- The protocol's 1,458-unit `execution_matrix` was not treated as C78 authorization.

Seven historical code/config identities replay byte-exact, including ERM, OACI, the training engine, and the confirmatory manifest. ERM and OACI remain asymmetric: ERM is a shared anchor; OACI is the trajectory.

## P0 readiness

```text
locked environment SHA match: {env['environment_hash_match']}
storage free snapshot:         {float(storage['free_GiB_snapshot']):.3f} GiB
required temporary reserve:    {float(storage['required_temporary_reserve_GiB']):.3f} GiB
storage capacity pass:         {storage['capacity_passed']}
dummy Wz+b max error:          {float(dummy['Wz_plus_b_max_abs']):.3e}
dummy softmax max error:       {float(dummy['softmax_max_abs']):.3e}
dummy repeat logit/z error:    {float(dummy['repeat_logit_max_abs']):.3e} / {float(dummy['repeat_z_max_abs']):.3e}
```

The dummy ABI used CPU synthetic inputs only. Real Wz/logit identity, training determinism, target isolation, checkpoint genealogy, cadence completeness, runtime, and cache materialization remain explicit P1 runtime gates; their tables report zero checked real units rather than inheriting the dummy pass.

## Isolation boundary

Six physically separate view schemas are locked. The training process receives source training inputs only; target-unlabeled instrumentation and target label views are deferred until all 82 retention decisions and checkpoint manifests are frozen. The same-label-oracle path is unavailable to primary pilot validation. These are execution contracts, not claims that runtime isolation has already passed.

## Red team

Independent red team passed `{len(blocking)}/{len(blocking)}` blocking checks before this report was created. Its principal repairs were:

""" + "\n".join(f"- `{row['item']}`: {row['resolution']}" for row in repairs) + f"""

Regression: {regression_text}.

## Decision

C78 is ready for a separately invoked exact-token P1, but it has not trained or instrumented the 82-unit field. Consequently none of `C78-A` through `C78-E` is active. No measurement-control replication, cross-regime transport result, representation mechanism, strict-source escape hatch, selector, checkpoint recommendation, deployability, or target-population claim is made.

Even a future successful OACI+ERM P1 cannot authorize the 1,458-unit expansion. SRC was not exercised, so PM review must first choose a prospective SRC canary or prove that SRC shares the exact validated execution/instrumentation path.
"""


def _handoff_section(state: dict[str, Any], regressions: list[dict[str, Any]]) -> str:
    regression_text = ", ".join(f"{row['suite']}={row['passed_tests']}" for row in regressions)
    return f"""## 0. Current continuation state (2026-07-10)

The detailed C23-C31 history below remains background; the authoritative tip is C78 P0:

```text
C78 protocol anchor: {state['protocol_commit'][:7]}
C78 protocol SHA:    {state['protocol_sha256']}
C78 final gate:      {state['final_gate_candidate']}
C78 P1 execution:    not authorized / not attempted
```

C78 locked an explicit 82-unit target-4, seed-3, levels-0/1 OACI+ERM canary: two ERM anchors and 80 fixed-cadence OACI trajectory checkpoints. The exact CLI authorization token was not passed. Prompt/generic authorization text and environment values were rejected, so training, real forward, real-data load, GPU request, checkpoint creation, and raw cache emission all remained zero.

P0 replayed seven historical code/config identities, the locked environment, storage capacity, physical-view schemas, and a CPU synthetic Wz/logit ABI. Independent red team passed before the report was emitted. No C78 execution taxonomy case is active. Seed 4 and BNCI2014_004 remain untouched; SRC remains unexercised and blocks automatic full-field expansion.

Regression: `{regression_text}`.

Authoritative artifacts:

```text
oaci/reports/C78_SEED3_INSTRUMENTED_PILOT.md
oaci/reports/C78_SEED3_INSTRUMENTED_PILOT.json
oaci/reports/C78_RED_TEAM_VERIFICATION.md
oaci/reports/C78_PROTOCOL_TIMING_AUDIT.md
oaci/reports/c78_tables/artifact_manifest.csv
```

Next action requires PM review. A real C78 P1 must be a new command carrying the exact committed CLI token. It may execute only the 82-unit OACI+ERM field. It may not run SRC, seed 4, BNCI2014_004, or the full seed-3 field.

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
        c78.PROTOCOL_PATH, c78.PROTOCOL_SHA_PATH, c78.TIMING_PATH,
        c78.STATE_PATH, MAIN_MD, RESULT_JSON, RED_TEAM,
        Path("oaci/conditioned_ceiling_coverage/c78_seed3_instrumented_pilot.py"),
        Path("oaci/conditioned_ceiling_coverage/c78_red_team.py"),
        Path("oaci/conditioned_ceiling_coverage/c78_finalize.py"),
        Path("oaci/slurm_c78_preflight.sh"),
        Path("oaci/slurm_c78_red_team.sh"),
        Path("oaci/slurm_c78_regression.sh"),
        Path("oaci/slurm_c78_finalize.sh"),
        Path("oaci/tests/test_c78_seed3_instrumented_pilot.py"),
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
    checks = _rows("red_team_checks.csv")
    blocking = [row for row in checks if row["blocking"] == "1"]
    if not blocking or any(row["passed"] != "1" for row in blocking):
        raise RuntimeError("C78 finalization requires all blocking red-team checks")
    if "Final status: `PASS`" not in RED_TEAM.read_text():
        raise RuntimeError("C78 red-team report is not PASS")
    regressions = collect_regressions()
    _write_csv("regression_verification.csv", regressions)
    state = json.loads(c78.STATE_PATH.read_text())
    if state["final_gate_candidate"] != "PILOT_READY_BUT_NOT_AUTHORIZED":
        raise RuntimeError("C78 no-auth finalizer received a non-readiness state")
    MAIN_MD.write_text(_main_report(state, regressions))
    result = {
        "schema_version": "c78_seed3_instrumented_pilot_no_auth_result_v1",
        "milestone": "C78", "final_gate": state["final_gate_candidate"],
        "protocol": {"commit": state["protocol_commit"], "sha256": state["protocol_sha256"], "prospective": True},
        "authorization": state["authorization"], "scope": state["scope"],
        "execution_boundary": state["execution_boundary"],
        "taxonomy": state["taxonomy"],
        "preflight": state["preflight"],
        "red_team": {"blocking_passed": len(blocking), "blocking_total": len(blocking), "repair_count": len(_rows("red_team_repair_ledger.csv"))},
        "regression": regressions,
        "claims": {
            "pilot_executed": False, "multiregime_replication": False,
            "SRC_exercised": False, "full_seed3_ready": False,
            "measurement_control_replication": False,
            "representation_mechanism": False, "strict_source_escape_hatch": False,
            "selector": False, "checkpoint_recommendation": False,
            "deployable": False, "target_population_generalization": False,
            "manuscript": False,
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
        raise RuntimeError("C78 report payload exceeds 50 MiB")
    print(json.dumps({"gate": result["final_gate"], "blocking_red_team": len(blocking), "regressions": regressions, "artifacts": len(manifest)}, sort_keys=True))
    return result


if __name__ == "__main__":
    finalize()
