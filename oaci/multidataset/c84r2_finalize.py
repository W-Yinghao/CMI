"""Assemble no-real-data C84R2 readiness and red-team artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from . import c84r2_canary_runtime_repair as runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r2_tables"
LOG_DIR = Path("/home/infres/yinwang/CMI_AAAI/c84r2_regression_logs")
LOCK_COMMIT = "270fbb0d9f47f9bf6a2888ee58fd7ca6eadff0ea"
GATE = "C84C_RUNTIME_LOCK_AND_COMPLETE_ENGINEERING_REPLAY_READY_FOR_PI_AUTHORIZATION"
EXPECTED_COUNTS = {"focused": 90, "c65": 576, "c23": 987, "full": 1911}
SUMMARY_VALUE = {
    "passed": re.compile(r"(?P<value>[0-9]+) passed"),
    "skipped": re.compile(r"(?P<value>[0-9]+) skipped"),
    "deselected": re.compile(r"(?P<value>[0-9]+) deselected"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise RuntimeError(f"empty C84R2 table: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError(f"C84R2 table schema mismatch: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def parse_jobs(value: str) -> dict[str, int]:
    jobs: dict[str, int] = {}
    for item in value.split(","):
        suite, job = item.split(":", 1)
        if suite not in EXPECTED_COUNTS or suite in jobs:
            raise ValueError(f"invalid C84R2 regression job binding: {item}")
        jobs[suite] = int(job)
    if set(jobs) != set(EXPECTED_COUNTS):
        raise ValueError("all four C84R2 regression jobs are required")
    return jobs


def summary_count(text: str, name: str) -> int:
    matches = list(SUMMARY_VALUE[name].finditer(text))
    return int(matches[-1].group("value")) if matches else 0


def regression_rows(jobs: Mapping[str, int], tested_commit: str) -> list[dict[str, Any]]:
    rows = []
    for suite in ("focused", "c65", "c23", "full"):
        job = jobs[suite]
        stdout = LOG_DIR / f"c84r2-regression-{job}.out"
        stderr = LOG_DIR / f"c84r2-regression-{job}.err"
        text = stdout.read_text(encoding="utf-8")
        passed = summary_count(text, "passed")
        if passed != EXPECTED_COUNTS[suite] or " failed" in text:
            raise RuntimeError(f"C84R2 {suite} regression did not pass exactly: {passed}")
        if f"suite={suite} commit={tested_commit}" not in text:
            raise RuntimeError(f"C84R2 {suite} regression metadata mismatch")
        rows.append({
            "suite": suite,
            "job_id": job,
            "commit_under_test": tested_commit,
            "environment": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact",
            "partition": "cpu-high",
            "CPU": 48,
            "GPU": 0,
            "passed": passed,
            "failed": 0,
            "skipped": summary_count(text, "skipped"),
            "deselected": summary_count(text, "deselected"),
            "stdout_path": str(stdout),
            "stdout_bytes": stdout.stat().st_size,
            "stdout_sha256": sha256_file(stdout),
            "stderr_path": str(stderr),
            "stderr_bytes": stderr.stat().st_size,
            "stderr_sha256": sha256_file(stderr),
            "stderr_status": "EMPTY" if stderr.stat().st_size == 0 else "NONEMPTY_BLOCKER",
            "skip_reason": (
                "C78F already passed red-team and finalized"
                if summary_count(text, "skipped") else "NONE"
            ),
        })
    if any(row["stderr_bytes"] or row["failed"] for row in rows):
        raise RuntimeError("C84R2 regression failure or nonempty stderr")
    return rows


def all_values(path: Path, field: str, expected: str) -> bool:
    rows = read_csv(path)
    return bool(rows) and all(row[field] == expected for row in rows)


def tracked_payload_hygiene() -> tuple[bool, bool]:
    forbidden = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".fif", ".edf", ".gdf", ".mat"}
    tracked = [REPO_ROOT / item for item in git("ls-files").splitlines()]
    return (
        not any(path.suffix.lower() in forbidden for path in tracked),
        all(path.stat().st_size <= 50 * 1024 * 1024 for path in tracked),
    )


def red_team_rows(regressions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lock = runtime.read_json(runtime.EXECUTION_LOCK_PATH)
    lock_sha = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    bound = runtime.verify_bound_object_registry(lock)
    protocols = runtime.verify_protocol_sidecars(lock)
    montage = runtime.verify_montage_binding(lock)
    candidate = runtime.verify_candidate_identity(lock)
    environment = runtime.verify_distribution_environment(lock)
    loaders = runtime.verify_loader_source_files(lock)
    raw_clean, size_clean = tracked_payload_hygiene()
    protected = lock["protected_state_at_lock"]
    lock_names = {path.name for path in REPORT_DIR.glob("C84*EXECUTION_LOCK*.json")}
    checks = {
        "repair_protocol_hash_replays": runtime.sha256_file(runtime.REPAIR_PROTOCOL_PATH) == runtime.REPAIR_PROTOCOL_SHA_PATH.read_text().split()[0],
        "execution_lock_self_hash_replays": lock_sha == runtime.EXECUTION_LOCK_SHA_PATH.read_text().split()[0],
        "runtime_bound_objects_63_of_63": len(bound) == lock["runtime_bound_object_count"] == 63,
        "runtime_bound_SHA_and_blob_replay": all(row["replay_pass"] for row in bound),
        "protocol_sidecars_6_of_6": len(protocols) == 6,
        "montage_exact_order_and_digest": montage["montage_sha256"] == runtime.EXPECTED_MONTAGE_SHA256,
        "candidate_ID_digest_243": candidate["canary_unit_count"] == 243,
        "environment_python_exact": environment["python"] == "3.13.7",
        "environment_distributions_exact": environment["distributions"] == {"chardet": "5.2.0", "mne": "1.11.0", "moabb": "1.5.0", "torch": "2.6.0"},
        "loader_sources_4_of_4": len(loaders) == 4 and all(row["before_get_data"] for row in loaders),
        "historical_V1_lock_preserved_nonoperative": lock["historical_lock_supersession"]["operative_for_execution"] is False,
        "only_C84C_V1_and_V2_locks_exist": lock_names == {"C84C_EXECUTION_LOCK.json", "C84C_EXECUTION_LOCK_V2.json"},
        "V2_lock_status_not_authorized": lock["status"] == runtime.LOCK_READY_STATUS,
        "fresh_authorization_record_absent": not runtime.AUTHORIZATION_RECORD_PATH.exists(),
        "C84F_and_C84S_locks_absent": not any(name.startswith(("C84F_", "C84S_")) for name in lock_names),
        "subject_identity_contract_three_datasets": len(read_csv(TABLE_DIR / "exact_subject_identity_contract.csv")) == 3,
        "epoch_interface_contract_persisted": all_values(TABLE_DIR / "actual_epoch_interface_contract.csv", "persisted", "1"),
        "source_audit_contract_243": all_values(TABLE_DIR / "source_audit_instrumentation_contract.csv", "unit_count", "243"),
        "target_unlabeled_contract_243": all_values(TABLE_DIR / "target_unlabeled_instrumentation_contract.csv", "unit_count", "243"),
        "persisted_replay_contract_blocking": all_values(TABLE_DIR / "persisted_artifact_replay_contract.csv", "failure_is_blocking", "1"),
        "optimizer_load_contract": all_values(TABLE_DIR / "optimizer_replay_contract.csv", "load_required", "1"),
        "deterministic_prefix_three_datasets": len(read_csv(TABLE_DIR / "deterministic_prefix_contract.csv")) == 3,
        "attempt_ledger_all_stages_wrapped": all_values(TABLE_DIR / "attempt_ledger_contract.csv", "inside_failure_ledger", "1"),
        "complete_gate_all_components_243": all_values(TABLE_DIR / "canary_complete_gate.csv", "required", "243"),
        "synthetic_calibration_all_pass": all_values(TABLE_DIR / "synthetic_calibration.csv", "passed", "1"),
        "risk_register_no_open_blocker": all(row["blocking"] == "0" for row in read_csv(TABLE_DIR / "risk_register.csv")),
        "failure_ledger_closed_before_lock": all(row["status"].startswith("CLOSED") for row in read_csv(TABLE_DIR / "failure_reason_ledger.csv")),
        "focused_regression_exact": next(row for row in regressions if row["suite"] == "focused")["passed"] == 90,
        "C65_regression_exact": next(row for row in regressions if row["suite"] == "c65")["passed"] == 576,
        "C23_regression_exact": next(row for row in regressions if row["suite"] == "c23")["passed"] == 987,
        "full_regression_exact": next(row for row in regressions if row["suite"] == "full")["passed"] == 1911,
        "all_regression_stderr_empty": all(row["stderr_bytes"] == 0 for row in regressions),
        "zero_real_EEG_and_labels": protected["real_EEG_arrays_loaded"] == protected["real_labels_read"] == 0,
        "zero_download_training_forward_GPU": protected["dataset_downloads"] == protected["training_forward_GPU_jobs"] == 0,
        "zero_candidate_and_instrumentation_artifacts": protected["candidate_units_created"] == protected["source_audit_artifacts_created"] == protected["target_unlabeled_artifacts_created"] == 0,
        "no_raw_EEG_weight_or_cache_in_Git": raw_clean,
        "no_tracked_payload_over_50MiB": size_clean,
    }
    return [{
        "check_id": f"C84R2-RT-{index:02d}",
        "check": name,
        "status": "PASS" if value else "FAIL",
        "blocking": 1,
        "real_data_access": 0,
    } for index, (name, value) in enumerate(checks.items(), start=1)]


def render_reports(
    regressions: Sequence[Mapping[str, Any]], red_team: Sequence[Mapping[str, Any]], tested_commit: str,
) -> None:
    lock_sha = runtime.EXECUTION_LOCK_SHA_PATH.read_text().split()[0]
    rows = "\n".join(
        f"| {row['suite']} | {row['job_id']} | {row['passed']} | {row['skipped']} | {row['deselected']} | {row['stderr_status']} |"
        for row in regressions
    )
    (REPORT_DIR / "C84R2_REGRESSION_VERIFICATION.md").write_text(f"""# C84R2 Regression Verification

All four suites ran at verification commit `{tested_commit}` (which contains C84C V2
lock commit `{LOCK_COMMIT}`) in the dedicated exact CPU
environment. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, the corrected
leading-numeric suite parser, and the established three C79P deselections were fixed.

| Suite | Job | Passed | Skipped | Deselected | Stderr |
|---|---:|---:|---:|---:|---|
{rows}

The conditional skip, where present, is the finalized C78F field test. Every accepted
stderr file is empty. No GPU or real-data execution was requested.
""", encoding="utf-8")

    passed = sum(row["status"] == "PASS" for row in red_team)
    (REPORT_DIR / "C84R2_FINAL_REPORT_RED_TEAM.md").write_text(f"""# C84R2 Final Report Red Team

```text
checks:              {len(red_team)}
passed:              {passed}
failed:              {len(red_team) - passed}
real outcome access: 0
```

The audit replays all 63 lock-bound objects by SHA-256 and Git blob, six protocol
sidecars, the exact montage and 243-unit identity, the dedicated package environment,
four MOABB source files, complete source/target instrumentation contracts, persisted
artifact replay, deterministic-prefix and attempt-ledger behavior, regressions and Git
payload hygiene. The historical V1 lock remains preserved and non-operative.
""", encoding="utf-8")

    (REPORT_DIR / "C84R2_PROTOCOL_READINESS.md").write_text(f"""# C84R2 Protocol Readiness

## Result

C84R2 closes the historical C84C runtime-binding and declared-check coverage gaps.
Before output-root creation or loader import, the V2 runtime now replays the current
bytes and Git blobs of every bound executable/registry object, all protocol hashes,
montage identity, candidate-ID digest, repository identity and package metadata.

```text
repair protocol commit:  6c7e59f907431e073b2f8e580c4f25cb9e052a50
implementation commit:   ddaa6d4531f13922481f53b827f13e62280d7968
C84C V2 lock commit:     {LOCK_COMMIT}
C84C V2 lock SHA-256:    {lock_sha}
runtime objects replay:  63 / 63
protocol hashes replay:  6 / 6
canary units:            243
```

The executable canary now binds exact loaded subject sets, actual ordered 20-channel
Epochs at 160 Hz, half-open 480-sample tensors, 243 strict-source audit artifacts, 243
target-unlabeled artifacts, persisted checkpoint/optimizer/sidecar replay and a
deterministic-prefix fingerprint. Authorization consumption is followed immediately by
an attempt ledger before protected imports.

No C84C authorization record exists. C84F and C84S remain unlocked and unauthorized.
No real EEG, label, download, training, forward, GPU, candidate unit or instrumentation
artifact was accessed or created in C84R2.

```text
{GATE}
```
""", encoding="utf-8")

    (REPORT_DIR / "OACI_EEG_DG_PROJECT_MEMORY_THROUGH_C84R2.md").write_text(f"""# OACI EEG-DG Project Memory Through C84R2

C84R2 preserves the 20-channel C84R scientific interface and adds a fail-closed C84C
runtime/engineering lock. Historical C84C V1 remains preserved but is non-operative.
The operative future canary objects are V3 protocol plus V2 lock `{LOCK_COMMIT}` /
`{lock_sha}`.

The lock binds 63 repository objects, exact Python/package and MOABB loader-source
identities, exact subjects/channels/sampling, dual source-audit and target-unlabeled
instrumentation, persisted artifact replay, deterministic-prefix checks and complete
post-consumption failure ledgers. No protected real-data activity occurred.

Final gate: `{GATE}`.
""", encoding="utf-8")


def generate(jobs: Mapping[str, int]) -> dict[str, Any]:
    tested_commit = git("rev-parse", "HEAD")
    if git("rev-parse", "origin/oaci") != tested_commit:
        raise RuntimeError("C84R2 finalizer requires HEAD == origin/oaci")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", LOCK_COMMIT, tested_commit],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    if git("status", "--porcelain"):
        raise RuntimeError("C84R2 finalizer requires a clean worktree")
    regressions = regression_rows(jobs, tested_commit)
    red_team = red_team_rows(regressions)
    if not all(row["status"] == "PASS" for row in red_team):
        raise RuntimeError(f"C84R2 red team failed: {[row['check'] for row in red_team if row['status'] != 'PASS']}")
    write_csv(TABLE_DIR / "regression_attempt_ledger.csv", regressions)
    write_csv(TABLE_DIR / "final_report_red_team.csv", red_team)
    render_reports(regressions, red_team, tested_commit)
    return {
        "gate": GATE,
        "red_team": f"{len(red_team)}/{len(red_team)}",
        "regressions": {row["suite"]: row["passed"] for row in regressions},
        "authorization_consumed": False,
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "dataset_downloads": 0,
        "GPU_jobs": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="focused:ID,c65:ID,c23:ID,full:ID")
    args = parser.parse_args(argv)
    print(json.dumps(generate(parse_jobs(args.jobs)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
