"""Finalize C84FL2 readiness without executing the protected C84F runtime."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from . import c84f_runtime_guard as runtime
from . import c84fl2_lock as lock_builder
from . import c84fl2_protocol as protocol
from . import c84r_regression_suite as suites


REPO_ROOT = protocol.REPO_ROOT
REPORT_DIR = protocol.REPORT_DIR
TABLE_DIR = protocol.TABLE_DIR
LOG_DIR = Path("/home/infres/yinwang/CMI_AAAI/c84fl2_regression_logs")
REGRESSION_COMMIT = "196bb44b334c5f41d334988a00714b7c3e85c3f0"
FAILED_REGRESSION_COMMIT = "f680c762d474a6de6870d667c8e8788bc65162e1"
LOCK_COMMIT = "47bf20fbc341c136da0e3ed997a490fb0f135c49"
LOCK_SHA256 = "f9df9dcefea59b05bfea24d1b744d82bfc933d76efde3f9aececf67401ea6b05"
RECONCILIATION_SHA256 = "2ac679a5308d5d972b14d38e01cbc0d875ca6c5e547b752945fc831e38081f62"
FIELD_V7_SHA256 = "9db0219befecb11cf72386b96e28ee9d9430c3df5d7947298f102492f072b737"
FULL_FIELD_V2_SHA256 = "dafc44dbc24ea5d4d1cea61207479cbd986c9f8129b111682a00f15a44b1d15d"
REGRESSIONS = (
    ("focused", 896163, 227, 0, 0),
    ("C65", 896164, 713, 1, 3),
    ("C23", 896165, 1124, 1, 3),
    ("full", 896166, 2048, 1, 3),
)
FAILED_ATTEMPTS = (
    ("focused", 896157),
    ("C65", 896158),
    ("C23", 896159),
    ("full", 896160),
)


def _git(*args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return runtime.sha256_file(path)


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).returncode == 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise RuntimeError(f"refusing empty C84FL2 table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise RuntimeError(f"C84FL2 table schema drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runtime.canonical_bytes(value) + b"\n")


def _pytest_summary(text: str) -> dict[str, Any]:
    pattern = re.compile(
        r"(?:(?P<failed>[0-9]+) failed, )?(?P<passed>[0-9]+) passed"
        r"(?:, (?P<skipped>[0-9]+) skipped)?"
        r"(?:, (?P<deselected>[0-9]+) deselected)? in (?P<elapsed>[^\n]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise RuntimeError("pytest summary is absent")
    match = matches[-1]
    return {
        "failed": int(match.group("failed") or 0),
        "passed": int(match.group("passed")),
        "skipped": int(match.group("skipped") or 0),
        "deselected": int(match.group("deselected") or 0),
        "elapsed": match.group("elapsed"),
    }


def _log_paths(job_id: int) -> tuple[Path, Path]:
    stem = LOG_DIR / f"c84fl2-regression-{job_id}"
    return Path(f"{stem}.out"), Path(f"{stem}.err")


def _regression_rows() -> list[dict[str, Any]]:
    rows = []
    for suite, job_id, passed, skipped, deselected in REGRESSIONS:
        stdout, stderr = _log_paths(job_id)
        if not stdout.is_file() or not stderr.is_file():
            raise RuntimeError(f"C84FL2 regression log is absent: {job_id}")
        text = stdout.read_text(encoding="utf-8")
        summary = _pytest_summary(text)
        expected = {"failed": 0, "passed": passed, "skipped": skipped, "deselected": deselected}
        if any(summary[key] != value for key, value in expected.items()):
            raise RuntimeError(f"C84FL2 regression count mismatch: {suite}: {summary}")
        if f"commit={REGRESSION_COMMIT}" not in text or stderr.stat().st_size:
            raise RuntimeError(f"C84FL2 regression identity/stderr mismatch: {suite}")
        rows.append({
            "suite": suite, "job_id": job_id, "commit": REGRESSION_COMMIT,
            **summary, "environment": "c84c-eeg2025-v3-exact",
            "allocation": "cpu-high|48_CPU|96_GiB|GPU_0",
            "stdout_sha256": _sha256(stdout), "stderr_sha256": _sha256(stderr),
            "stderr_bytes": 0,
            "skip_reason": "C78F already passed red-team and finalized" if skipped else "NONE",
            "deselection_reason": "three historical C79 authorization-state tests" if deselected else "NONE",
            "status": "PASS",
        })
    return rows


def _attempt_rows() -> list[dict[str, Any]]:
    rows = []
    for suite, job_id in FAILED_ATTEMPTS:
        stdout, stderr = _log_paths(job_id)
        if not stdout.is_file() or not stderr.is_file():
            raise RuntimeError(f"C84FL2 failed-attempt log is absent: {job_id}")
        text = stdout.read_text(encoding="utf-8")
        summary = _pytest_summary(text)
        if summary["failed"] != 6 or f"commit={FAILED_REGRESSION_COMMIT}" not in text:
            raise RuntimeError(f"C84FL2 failed-attempt identity drift: {suite}/{job_id}")
        if stderr.stat().st_size:
            raise RuntimeError(f"C84FL2 failed-attempt stderr is nonempty: {suite}/{job_id}")
        rows.append({
            "suite": suite, "job_id": job_id, "commit": FAILED_REGRESSION_COMMIT,
            **summary, "stderr_bytes": 0,
            "failure_class": "historical_milestone_assertion_outlived_stage",
            "failure_detail": "six old tests still prohibited the newly required C84F lock",
            "real_data_access": 0, "training_forward_GPU": 0,
            "replacement_commit": REGRESSION_COMMIT,
            "replacement_job": next(job for name, job, *_ in REGRESSIONS if name == suite),
            "disposition": "PRESERVED_AND_REPLACED",
        })
    return rows


def _suite_file_rows() -> list[dict[str, Any]]:
    rows = []
    for suite in ("focused", "c65", "c23", "full"):
        paths = suites.suite_files(suite)
        expanded: list[Path] = []
        for path in paths:
            expanded.extend(sorted(path.glob("test_*.py")) if path.is_dir() else [path])
        if not expanded:
            raise RuntimeError(f"empty suite registry: {suite}")
        for path in expanded:
            rows.append({
                "suite": suite, "test_file": str(path.relative_to(REPO_ROOT)),
                "leading_numeric_parser": int(suite in {"c65", "c23"}),
                "C34S_included": int(path.name == "test_c34s_artifact_hygiene.py"),
                "C84FL2_focused_file": int(path.name in {
                    "test_c84fl2_full_field_lock.py",
                    "test_c84f_dual_level_training_contract.py",
                    "test_c84f_target_instrumentation_contract.py",
                }),
            })
    by_suite = {name: [row for row in rows if row["suite"] == name] for name in ("focused", "c65", "c23", "full")}
    if len(by_suite["focused"]) != 18:
        raise RuntimeError("C84FL2 focused suite is not 18 files")
    if not any(row["C34S_included"] for row in by_suite["c23"]):
        raise RuntimeError("C23 suite again omits C34S")
    for name in by_suite:
        if name != "focused" and sum(row["C84FL2_focused_file"] for row in by_suite[name]) != 3:
            raise RuntimeError(f"C84FL2 files absent from cumulative suite: {name}")
    return rows


def _repository_hygiene() -> dict[str, Any]:
    tracked = [REPO_ROOT / value for value in _git("ls-files").splitlines()]
    files = [path for path in tracked if path.is_file()]
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".fif", ".edf", ".gdf", ".mat"}
    forbidden = [str(path.relative_to(REPO_ROOT)) for path in files if path.suffix.lower() in forbidden_suffixes]
    oversized = [str(path.relative_to(REPO_ROOT)) for path in files if path.stat().st_size > 50 * 1024 * 1024]
    maximum = max(files, key=lambda path: path.stat().st_size)
    return {
        "tracked_files": len(files), "maximum_file": str(maximum.relative_to(REPO_ROOT)),
        "maximum_file_bytes": maximum.stat().st_size,
        "forbidden_payloads": forbidden, "oversized_payloads": oversized,
    }


def _red_team_rows(
    regressions: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock = runtime.read_json(runtime.EXECUTION_LOCK_PATH)
    lock_sha = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    bound = runtime.verify_bound_object_registry(lock)
    protocol_replay = runtime.verify_protocol_sidecars(lock)
    interface = runtime.verify_interface(lock)
    environment = runtime.verify_distribution_environment(lock)
    candidate = runtime.verify_candidate_and_wave_registry(lock)
    reuse = runtime.verify_dual_canary_reuse(lock)
    resources = _read_csv(TABLE_DIR / "resource_estimate.csv")
    risks = _read_csv(TABLE_DIR / "risk_register.csv")
    failures = _read_csv(TABLE_DIR / "failure_reason_ledger.csv")
    synthetic = _read_csv(TABLE_DIR / "synthetic_calibration.csv")
    remaining = _read_csv(TABLE_DIR / "remaining_paired_training_registry.csv")
    waves = _read_csv(TABLE_DIR / "wave_registry.csv")
    failed_roots = _read_csv(TABLE_DIR / "historical_failed_root_rejection.csv")
    subset = _read_csv(TABLE_DIR / "canary_subset_replay_contract.csv")
    retry = _read_csv(TABLE_DIR / "retry_policy.csv")
    full_field_protocol = runtime.read_json(
        REPORT_DIR / "C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL_V2.json"
    )
    hygiene = _repository_hygiene()
    suite_rows = _suite_file_rows()
    active = subprocess.run(
        ["squeue", "-h", "-o", "%i|%j|%T"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.splitlines()
    active_c84fl2 = [line for line in active if "c84fl2" in line.lower()]
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/oaci")
    dirty = _git("status", "--porcelain")
    historical_tree = _git("ls-tree", "-r", "--name-only", protocol.BASE_HEAD, "oaci/reports")
    checks = (
        ("RT01", "accepted_base_HEAD", protocol.BASE_HEAD == "4d2ca75b2fc149c80c3e51e93709aab12e67813a"),
        ("RT02", "C84C_result_hash", protocol.HASHES["c84c_result"] == "bec3a8b205a3d13fdb848ce1f82f71f903d05a97f746fdae25b3b4cce40e67f0"),
        ("RT03", "C84C_manifest_hash", _sha256(protocol.C84C_MANIFEST) == protocol.HASHES["c84c_manifest"]),
        ("RT04", "C84L1C_result_hash", protocol.HASHES["c84l1c_result"] == "5bcccf351704c427d148ca1f44de26ef7e0b137d8de56aa0cf9ca3f6723abaf5"),
        ("RT05", "C84L1C_manifest_hash", _sha256(protocol.C84L1C_MANIFEST) == protocol.HASHES["c84l1c_manifest"]),
        ("RT06", "operative_registry_hash", _sha256(protocol.OPERATIVE_REGISTRY) == protocol.HASHES["operative_registry"]),
        ("RT07", "protocol_precedes_implementation", _is_ancestor(lock_builder.PROTOCOL_COMMIT, lock_builder.IMPLEMENTATION_COMMIT)),
        ("RT08", "historical_C84FL2_base_had_no_field_lock", "oaci/reports/C84F_EXECUTION_LOCK.json" not in historical_tree.splitlines()),
        ("RT09", "execution_lock_hash", lock_sha == LOCK_SHA256),
        ("RT10", "execution_lock_status", lock["status"] == lock_builder.LOCK_READY_STATUS),
        ("RT11", "authorization_absent", not runtime.AUTHORIZATION_RECORD_PATH.exists()),
        ("RT12", "C84S_lock_absent", not any(REPORT_DIR.glob("C84S*EXECUTION_LOCK*.json"))),
        ("RT13", "C84S_unauthorized", lock["authorization"]["C84S_authorized"] is False),
        ("RT14", "C84F_unexecuted", lock["scope"]["real_execution_at_lock"] is False),
        ("RT15", "runtime_bound_objects", len(bound) == lock["runtime_bound_object_count"] == 79),
        ("RT16", "implementation_file_count", len(lock["implementation"]["files"]) == 38),
        ("RT17", "protocol_binding_count", len(protocol_replay) == 7),
        ("RT18", "exact_environment", environment["distributions"] == lock["environment"]["distributions"]),
        ("RT19", "exact_20_channel_interface", len(interface["channels"]) == 20),
        ("RT20", "complete_unit_count", candidate["units"] == 1944),
        ("RT21", "level_arithmetic", candidate["level0"] == candidate["level1"] == 972),
        ("RT22", "operative_unit_ID_uniqueness", len(_read_csv(TABLE_DIR / "operative_complete_unit_registry_replay.csv")) == 1944),
        ("RT23", "dual_canary_reuse_count", reuse["units"] == 486),
        ("RT24", "dual_canary_artifact_replay", reuse["artifact_files_replayed"] == 2430),
        ("RT25", "reuse_source_balance", reuse["source_counts"] == {"C84C": 243, "C84L1C": 243}),
        ("RT26", "historical_failed_roots_rejected", len(failed_roots) == 2 and all(row["authorization_reusable"] == row["artifact_reusable"] == "0" for row in failed_roots)),
        ("RT27", "remaining_unit_count", len(remaining) == 1458),
        ("RT28", "remaining_level_balance", sum(row["level"] == "0" for row in remaining) == sum(row["level"] == "1" for row in remaining) == 729),
        ("RT29", "remaining_wave_arithmetic", sum(int(row["candidate_units"]) for row in waves if row["action"] == "TRAIN_PAIRED_LEVELS") == 1458),
        ("RT30", "remaining_phase_arithmetic", sum(int(row["training_phases"]) for row in waves if row["action"] == "TRAIN_PAIRED_LEVELS") == 54),
        ("RT31", "paired_training", full_field_protocol["paired_training"]["same_model_init_across_levels"] is True),
        ("RT32", "level0_full_source_identity", full_field_protocol["paired_training"]["level0"] == protocol.LEVEL0_ID),
        ("RT33", "level1_fixed_deletion_identity", full_field_protocol["paired_training"]["level1"] == protocol.LEVEL1_ID),
        ("RT34", "model_field_barrier", lock["barriers"]["source_views_only_before_model_freeze"] is True and lock["barriers"]["new_target_loader_call_before_model_manifest"] is False),
        ("RT35", "model_field_gate_1944", lock["barriers"]["model_field_gate"]["units"] == 1944),
        ("RT36", "target_registry_after_model_freeze", full_field_protocol["target_registry"]["created_only_after_model_field_freeze"] is True),
        ("RT37", "target_label_fields_zero", full_field_protocol["target_registry"]["label_fields"] == 0),
        ("RT38", "target_contexts_complete", lock["barriers"]["complete_target_gate"]["target_contexts"] == 944),
        ("RT39", "candidate_context_slices_complete", lock["barriers"]["complete_target_gate"]["candidate_context_slices"] == 76464),
        ("RT40", "all_target_unit_artifacts", lock["barriers"]["complete_target_gate"]["all_target_artifacts"] == 1944),
        ("RT41", "target_stage_cannot_train", lock["implementation"]["target_stage_training_callable"] is False),
        ("RT42", "canary_subset_witnesses", len(subset) == 6 and all(row["complete_target_artifact_reusable"] == "0" and row["C84F_recompute_complete_target_artifact"] == "1" for row in subset)),
        ("RT43", "linear_tolerance_fixed", lock["numerical_gates"]["linear_in_memory_and_persisted_abs_max"] == 2e-5),
        ("RT44", "strict_tolerance_fixed", lock["numerical_gates"]["softmax_repeat_logits_repeat_z_abs_max"] == 1e-6),
        ("RT45", "runtime_tolerance_widening_forbidden", lock["numerical_gates"]["runtime_widening_allowed"] is False),
        ("RT46", "target_failure_cannot_retrain", any(row["failure_class"] == "target_instrumentation_failure" and row["training_callable_from_target_repair"] == "0" for row in retry)),
        ("RT47", "resources_within_envelope", len(resources) == 10 and all(row["within_envelope"] == "1" for row in resources)),
        ("RT48", "risk_register_closed", len(risks) == 30 and all(row["blocking"] == "0" for row in risks)),
        ("RT49", "failure_ledger_clear", len(failures) == 1 and failures[0]["failure_id"] == "NONE" and failures[0]["blocking"] == "0"),
        ("RT50", "synthetic_calibration", len(synthetic) == 25 and all(row["pass"] == "1" for row in synthetic)),
        ("RT51", "initial_regression_failure_preserved", len(attempts) == 4 and all(row["failed"] == 6 for row in attempts)),
        ("RT52", "replacement_regressions_pass", [row["passed"] for row in regressions] == [227, 713, 1124, 2048]),
        ("RT53", "conditional_skip_explained", all(row["skipped"] in (0, 1) for row in regressions)),
        ("RT54", "deselections_explained", all(row["deselected"] in (0, 3) for row in regressions)),
        ("RT55", "regression_stderr_empty", all(row["stderr_bytes"] == 0 for row in regressions)),
        ("RT56", "leading_numeric_C34S_included", any(row["suite"] == "c23" and row["C34S_included"] for row in suite_rows)),
        ("RT57", "new_test_files_in_all_suites", all(sum(row["C84FL2_focused_file"] for row in suite_rows if row["suite"] == name) == 3 for name in ("focused", "c65", "c23", "full"))),
        ("RT58", "tracked_raw_weight_cache_absent", not hygiene["forbidden_payloads"]),
        ("RT59", "tracked_payload_over_50MiB_absent", not hygiene["oversized_payloads"]),
        ("RT60", "active_C84FL2_jobs_absent", not active_c84fl2),
        ("RT61", "branch_oaci", branch == "oaci"),
        ("RT62", "HEAD_equals_origin", head == origin),
        ("RT63", "worktree_clean_before_report_generation", not dirty),
        ("RT64", "C84FL2_real_data_access_zero", lock["chronology"]["C84FL2_real_data_access"] == 0),
        ("RT65", "target_labels_before_lock_zero", lock["chronology"]["target_labels_before_lock"] == 0),
        ("RT66", "scientific_outcomes_before_lock_zero", lock["chronology"]["scientific_outcomes_before_lock"] == 0),
    )
    failed = [name for _, name, passed in checks if not passed]
    if failed:
        raise RuntimeError(f"C84FL2 red-team failed: {failed}")
    rows = [{
        "check_id": check_id, "check": name, "status": "PASS", "blocking": 0,
        "real_data_access": 0, "scientific_outcome_access": 0,
    } for check_id, name, _ in checks]
    context = {
        "lock": lock, "lock_sha256": lock_sha, "head": head, "origin": origin,
        "bound_objects": len(bound), "protocol_bindings": len(protocol_replay),
        "environment": environment, "candidate": candidate, "reuse": reuse,
        "resources": resources, "hygiene": hygiene, "suite_rows": suite_rows,
    }
    return rows, context


def _report_markdown(
    context: Mapping[str, Any], regressions: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], red_team: Sequence[Mapping[str, Any]],
) -> str:
    regression_lines = "\n".join(
        f"| {row['suite']} | {row['job_id']} | {row['passed']} | {row['skipped']} | "
        f"{row['deselected']} | {row['stderr_bytes']} |" for row in regressions
    )
    attempt_lines = "\n".join(
        f"| {row['suite']} | {row['job_id']} | {row['passed']} | {row['failed']} | "
        f"{row['replacement_job']} |" for row in attempts
    )
    return f"""# C84FL2 Overall Report

## Decision

C84FL2 reconciles the accepted level-0 and level-1 engineering canaries into a
single fail-closed dual-level C84F implementation. It creates the one permitted
C84F execution lock and stops before authorization or real execution.

Final gate:

```text
{protocol.SUCCESS_GATE}
```

This is a readiness gate only. C84F remains unauthorized and unexecuted. C84S
has no execution lock and is not authorized.

## Chronology And Immutable Identities

```text
accepted C84L1C base:       {protocol.BASE_HEAD}
C84FL2 protocol commit:     {lock_builder.PROTOCOL_COMMIT}
C84FL2 implementation:      {lock_builder.IMPLEMENTATION_COMMIT}
C84F lock commit:           {LOCK_COMMIT}
regression replacement:     {REGRESSION_COMMIT}
verification/report base:   {context['head']}

reconciliation SHA-256:     {RECONCILIATION_SHA256}
field V7 SHA-256:           {FIELD_V7_SHA256}
full-field V2 SHA-256:      {FULL_FIELD_V2_SHA256}
C84F lock SHA-256:          {LOCK_SHA256}
operative registry SHA-256: {protocol.HASHES['operative_registry']}
```

All older C84FL and field protocols remain historical and unchanged. The
additive protocol commit precedes implementation. C84FL2 accessed zero real EEG,
labels, training, forward, GPU, selector outcomes or scientific endpoints.

## Accepted Dual-Canary Reuse

The lock byte/hash-replays both accepted engineering fields:

```text
C84C level 0:   243 units / 9 phases
C84L1C level 1: 243 units / 9 phases
combined:       486 units / 18 phases
replayed external artifact files: 2,430
```

Reusable objects are candidate identity, checkpoint, optimizer state, sidecar,
genealogy/state descriptor and strict-source audit artifact. Failed jobs
`895366` and `895928` and their roots are explicitly rejected. The six canary
target contexts and 486 candidate-context slices are replay witnesses only;
they are not substituted for the final uniform target field.

## Complete Field And Remaining Waves

The operative field has 1,944 unique candidate units: 972 level 0 and 972 level
1, across 24 zoos and 72 phases. Level 0 keeps the full 12-subject source panel.
Level 1 uses the fixed registered subject x `left_hand` deletion. Each paired
dataset/panel/seed cell uses equal model initialization and otherwise identical
architecture, optimizer, epochs, cadence and deterministic settings.

Remaining work is exactly 1,458 units / 54 phases:

| Wave | Scope | Units | Phases |
|---|---|---:|---:|
| A | panel A / seed 6 / both levels | 486 | 18 |
| B0 | panel B / seed 5 / both levels | 486 | 18 |
| B1 | panel B / seed 6 / both levels | 486 | 18 |

Wave release can inspect engineering identity and replay only. It cannot inspect
target arrays, predictions, calibration, accuracy, selector scores, regret,
Q1/Q2 or label-budget outcomes.

## Model Freeze And Target Barrier

No new target subject may be loaded until an atomic model-field manifest proves
1,944/1,944 checkpoints, optimizer states, sidecars and strict-source artifacts,
72/72 phases, 486/486 reused units, 1,458/1,458 new units, 972/972 units per
level, and zero target rows/labels or outcome-driven retention/retry.

Only after that barrier may the unlabeled interface create a label-free trial
registry for all 118 target subjects. The structural target-y slot is never
indexed, represented, converted, hashed, summarized or logged.

## Complete Target Instrumentation

The locked target stage creates one uniform all-target artifact per unit:

```text
unit artifacts:             1,944 / 1,944
target contexts:              944 / 944
candidate-context slices:  76,464 / 76,464
target subjects: Lee 22, Cho 20, Physionet 76
```

The target module has no training callable. Instrumentation failure cannot
retrain or alter model retention. All six canary contexts must match by trial
and candidate ID and replay logits, probabilities and z. The field-wide linear
gate is fixed at `2e-5`; softmax, repeated logits and repeated z remain `1e-6`.
No tolerance can be widened at runtime.

## Runtime Lock

The lock replays {context['bound_objects']} repository objects, 38 implementation
files, seven protocol sidecars, the exact environment and loader sources, the
20-channel interface, all 1,944 IDs and all 2,430 reusable external artifacts.
It requires clean `HEAD == origin/oaci`, a fresh direct C84F authorization and
an empty content-addressed output root before protected imports or data access.

Lock status:

```text
LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED
```

No authorization record exists. The shortest future direct statement is
`\u6388\u6743 C84F`; it must be server-bound to this unique lock and protocols.

## Resource Envelope

Measured dual-canary training was 10,734.285 seconds over 18 phases. The linear
remaining projection is 8.945 GPU-hours; the complete 5x safety estimate is
59.635 GPU-hours, below the 250-hour ceiling. Download plus derived payload is
projected at 245,201,501,528 bytes, below 2 TiB. The complete target
instrumentation projection is 49,036,391,984 bytes. No runtime scope reduction
is allowed.

## Synthetic Validation And Regression Lifecycle

All 25 synthetic/contract fixtures passed, including dual-canary identity,
failed-root rejection, unit/wave arithmetic, paired initialization, target
barrier, target-y failure, full-context coverage, canary replay, numerical
gates, atomic manifests and authorization fail-closed behavior.

The first regression attempt correctly exposed six stale historical tests that
still asserted no C84F lock could exist:

| Suite | Failed job | Passed | Failed | Replacement job |
|---|---:|---:|---:|---:|
{attempt_lines}

Those tests were repaired to assert the historical no-lock state at the
historical commit and the current not-authorized/unexecuted lock at the current
commit. No runtime or scientific object changed.

| Suite | Slurm job | Passed | Skipped | Deselected | Stderr bytes |
|---|---:|---:|---:|---:|---:|
{regression_lines}

Every replacement ran on `cpu-high` with 48 CPUs, 96 GiB and GPU 0 in
`c84c-eeg2025-v3-exact`. The one cumulative skip is finalized C78F. The three
cumulative deselections are established C79 authorization-state tests. All
stderr files are empty. The leading-numeric parser includes C34S and every new
C84FL2 test in focused, C65, C23 and full as applicable.

## Red Team, Risks, And Hygiene

All {len(red_team)}/{len(red_team)} blocking red-team checks passed. All 30
registered risks are closed and nonblocking; the failure ledger is clear. Git
contains no raw EEG, model weights, optimizer states, NumPy caches or tracked
file over 50 MiB. No C84FL2 job remains active at report generation.

## Evidence Boundary And Next Stage

C84FL2 establishes implementation and lock readiness only. It provides no new
EEG evidence, target prediction, selector result, external-validity result or
scientific taxonomy. A future fresh direct C84F authorization may execute only
the locked field. Successful C84F must stop at
`{protocol.FIELD_GATE}`. C84S requires a separate implementation, lock, PM
review and authorization after the complete field freezes.
"""


def _update_handoff(
    path: Path, head: str, regressions: Sequence[Mapping[str, Any]], red_team_count: int,
) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "### Accepted C84C engineering base"
    if marker not in text:
        raise RuntimeError("C84FL2 handoff insertion marker is absent")
    suffix = text[text.index(marker):]
    intro = f"""# OACI (Direction 1) - Codex Handoff

**Purpose.** Everything a fresh agent needs to continue the OACI mechanism-audit
and external-validity line without losing scientific context, chronology,
authorization boundaries or immutable evidence identities. Read the scientific
and collaboration sections before acting.

> **One-sentence orientation.** OACI is a strict-DG (source-only, no target data)
> EEG mechanism study. The method line is closed/negative; the surviving
> diagnostic chain explains why source-only competence selection fails to
> transport. C84 is a separate prospective external-validity branch and does
> not turn historical diagnostics into a deployable method or selector.

---

## 0. Current continuation state (2026-07-14)

C84FL2 has completed the no-real-data dual-level full-field implementation and
C84F execution-lock readiness milestone:

```text
accepted base:              {protocol.BASE_HEAD}
C84FL2 protocol commit:     {lock_builder.PROTOCOL_COMMIT}
C84FL2 implementation:      {lock_builder.IMPLEMENTATION_COMMIT}
C84F lock commit:           {LOCK_COMMIT}
verification/report base:   {head}
reconciliation SHA-256:     {RECONCILIATION_SHA256}
field V7 SHA-256:           {FIELD_V7_SHA256}
full-field V2 SHA-256:      {FULL_FIELD_V2_SHA256}
C84F lock SHA-256:          {LOCK_SHA256}
gate:                       {protocol.SUCCESS_GATE}
```

The lock reuses 243 accepted C84C level-0 and 243 accepted C84L1C level-1
model/state/source-audit units. It rejects failed jobs 895366 and 895928. The
remaining scope is 1,458 units / 54 phases in three paired waves of 486 units /
18 phases. Levels share model initialization; level 0 is the full source panel
and level 1 is the fixed registered subject x `left_hand` deletion.

C84F enforces a no-new-target-access barrier until the atomic 1,944-unit model
field freezes. It then permits label-free instrumentation of 118 subjects, 944
contexts and 76,464 candidate-context slices. The six canary contexts are replay
witnesses only. Linear tolerance is fixed at 2e-5 and strict replay tolerances at
1e-6; target failure cannot retrain.

Runtime replay covers 79 repository objects, 38 implementation files, seven
protocols, all 1,944 IDs and 2,430 dual-canary external artifacts. Synthetic
validation is 25/25 and red team {red_team_count}/{red_team_count} PASS.

Regression: focused {regressions[0]['passed']}, C65 {regressions[1]['passed']},
C23 {regressions[2]['passed']} and full {regressions[3]['passed']} passed. The
cumulative suites have one explained C78F skip and three established C79
deselections; all stderr is empty. Scheduler monitoring used `squeue`, not
`sacct`.

No real EEG, label, remaining training, forward, GPU or scientific outcome was
accessed in C84FL2. C84F is not authorized or executed; C84S has no lock. The
complete report is `reports/C84FL2_OVERALL_REPORT.md`. The next permissible
action is PM review followed by a fresh direct `\u6388\u6743 C84F` statement
bound to the unique current lock. This handoff itself authorizes nothing.

"""
    path.write_text(intro + suffix, encoding="utf-8")


def generate() -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != _git("rev-parse", "origin/oaci") or _git("status", "--porcelain"):
        raise RuntimeError("C84FL2 finalization requires clean HEAD == origin/oaci")
    regressions = _regression_rows()
    attempts = _attempt_rows()
    red_team, context = _red_team_rows(regressions, attempts)
    suite_rows = context["suite_rows"]
    hygiene = context["hygiene"]

    _write_csv(TABLE_DIR / "regression_verification.csv", regressions)
    _write_csv(TABLE_DIR / "regression_attempt_ledger.csv", attempts)
    _write_csv(TABLE_DIR / "suite_file_registry.csv", suite_rows)
    _write_csv(TABLE_DIR / "final_report_red_team.csv", red_team)
    _write_csv(TABLE_DIR / "artifact_hygiene.csv", [{
        "tracked_files": hygiene["tracked_files"],
        "maximum_file": hygiene["maximum_file"],
        "maximum_file_bytes": hygiene["maximum_file_bytes"],
        "forbidden_payload_count": len(hygiene["forbidden_payloads"]),
        "oversized_payload_count": len(hygiene["oversized_payloads"]),
        "status": "PASS",
    }])

    regression_md = "# C84FL2 Regression Verification\n\n" + "\n".join(
        f"- `{row['suite']}`: replacement job `{row['job_id']}`, {row['passed']} passed, "
        f"{row['skipped']} skipped, {row['deselected']} deselected, stderr {row['stderr_bytes']} bytes."
        for row in regressions
    ) + (
        "\n\nInitial jobs `896157`-`896160` are preserved: each exposed six stale historical "
        "no-C84F-lock assertions. The replacement commit changed only those tests' lifecycle "
        "semantics. All jobs used `cpu-high`, 48 CPUs, 96 GiB and GPU 0. The sole cumulative "
        "skip is finalized C78F; the three cumulative deselections are historical C79 "
        "authorization-state checks. Every stderr file is empty.\n"
    )
    (REPORT_DIR / "C84FL2_REGRESSION_VERIFICATION.md").write_text(regression_md, encoding="utf-8")

    red_md = f"""# C84FL2 Final Report Red Team

All {len(red_team)}/{len(red_team)} blocking checks passed. The audit replayed both canary
identities and 2,430 external artifact files, all protocols and 79 lock-bound
objects, 1,944 operative IDs, paired waves, the model-freeze target barrier,
944-context/76,464-slice schemas, numerical gates, retry isolation, resources,
regression lifecycle, scheduler state and Git hygiene.

C84FL2 performed zero real-data access, label access, training, forward, GPU or
scientific computation. C84F has no authorization record and was not executed.
C84S has no execution lock.
"""
    (REPORT_DIR / "C84FL2_FINAL_REPORT_RED_TEAM.md").write_text(red_md, encoding="utf-8")

    readiness = {
        "schema_version": "c84fl2_protocol_readiness_v1",
        "milestone": "C84FL2", "gate": protocol.SUCCESS_GATE,
        "verification_HEAD": context["head"], "protocol_commit": lock_builder.PROTOCOL_COMMIT,
        "implementation_commit": lock_builder.IMPLEMENTATION_COMMIT,
        "execution_lock_commit": LOCK_COMMIT, "execution_lock_sha256": LOCK_SHA256,
        "protocol_hashes": {
            "reconciliation": RECONCILIATION_SHA256,
            "field_v7": FIELD_V7_SHA256,
            "full_field_v2": FULL_FIELD_V2_SHA256,
        },
        "runtime_bound_objects": context["bound_objects"], "implementation_files": 38,
        "dual_canary_reuse": {"units": 486, "phases": 18, "artifact_files_replayed": 2430},
        "field": {"units": 1944, "levels": {"level0": 972, "level1": 972}, "phases": 72},
        "remaining": {"units": 1458, "phases": 54, "waves": [486, 486, 486]},
        "target_instrumentation": {"subjects": 118, "contexts": 944, "slices": 76464},
        "synthetic_calibration": "25/25 PASS", "red_team": f"{len(red_team)}/{len(red_team)} PASS",
        "regressions": list(regressions), "failed_regression_attempts": list(attempts),
        "authorization_record_present": False, "C84F_executed": False,
        "C84S_lock_present": False, "real_EEG_access": 0, "real_label_reads": 0,
        "training_forward_GPU": 0, "scientific_outcomes": 0,
        "fresh_direct_C84F_authorization_required": True,
    }
    _write_json(REPORT_DIR / "C84FL2_PROTOCOL_READINESS.json", readiness)
    (REPORT_DIR / "C84FL2_PROTOCOL_READINESS.md").write_text(
        "# C84FL2 Protocol Readiness\n\n"
        "The accepted level-0 and level-1 canaries are reconciled into one fail-closed "
        "dual-level full-field runtime and one scope-specific C84F execution lock. The "
        "lock binds 79 runtime objects, 38 implementation files, 1,944 operative units "
        "and 2,430 reusable external artifacts.\n\n"
        f"Lock SHA-256: `{LOCK_SHA256}`. Synthetic calibration: 25/25 PASS. Red team: "
        f"{len(red_team)}/{len(red_team)} PASS. Focused/C65/C23/full: "
        f"{regressions[0]['passed']}/{regressions[1]['passed']}/{regressions[2]['passed']}/"
        f"{regressions[3]['passed']} passed.\n\n"
        "No real EEG, label, training, forward, GPU or scientific outcome was accessed. "
        "C84F is not authorized or executed; C84S has no lock.\n\n"
        f"Final gate: `{protocol.SUCCESS_GATE}`.\n",
        encoding="utf-8",
    )

    overall_md = _report_markdown(context, regressions, attempts, red_team)
    overall_json = {
        **readiness,
        "accepted_canaries": {
            "C84C_result_sha256": protocol.HASHES["c84c_result"],
            "C84C_manifest_sha256": protocol.HASHES["c84c_manifest"],
            "C84L1C_result_sha256": protocol.HASHES["c84l1c_result"],
            "C84L1C_manifest_sha256": protocol.HASHES["c84l1c_manifest"],
        },
        "failed_roots_rejected": [895366, 895928],
        "model_field_barrier": "1944_units_before_any_new_target_access",
        "numerical_gates": {"linear": 2e-5, "strict": 1e-6},
        "risk_register": "30/30 CLOSED", "evidence_boundary": "no_new_real_data_readiness_only",
        "next_stage": "fresh_direct_C84F_authorization_after_PM_review",
    }
    overall_md_path = REPORT_DIR / "C84FL2_OVERALL_REPORT.md"
    overall_json_path = REPORT_DIR / "C84FL2_OVERALL_REPORT.json"
    overall_md_path.write_text(overall_md, encoding="utf-8")
    _write_json(overall_json_path, overall_json)
    (REPORT_DIR / "C84FL2_OVERALL_REPORT.sha256").write_text(
        f"{_sha256(overall_md_path)}  {overall_md_path.name}\n"
        f"{_sha256(overall_json_path)}  {overall_json_path.name}\n",
        encoding="ascii",
    )

    memory = f"""# OACI EEG-DG Project Memory Through C84FL2

C84FL2 combines accepted C84C level-0 and C84L1C level-1 engineering evidence
into a single unexecuted C84F lock. It reuses 486 model/state/source-audit units
and leaves 1,458 units / 54 phases in three paired waves. The complete model
field is 1,944 units / 72 phases, with 972 units per level.

No new target can be loaded until the complete model-field manifest freezes.
After that barrier, the locked label-free target stage covers 118 subjects, 944
contexts and 76,464 candidate-context slices. Six canary contexts are replay
witnesses only. Linear replay is fixed at 2e-5; strict replay remains 1e-6.

Protocol SHA-256: `{RECONCILIATION_SHA256}`. C84F lock SHA-256:
`{LOCK_SHA256}`. Runtime replay covers 79 repository objects and 2,430 external
canary artifacts. Verification: 25/25 synthetic, {len(red_team)}/{len(red_team)}
red-team, focused 227, C65 713, C23 1,124 and full 2,048 passed.

No real EEG, labels, training, forward, GPU or scientific outcome occurred in
C84FL2. C84F is not authorized or executed. C84S has no lock.

Gate: `{protocol.SUCCESS_GATE}`.
"""
    (REPORT_DIR / "OACI_EEG_DG_PROJECT_MEMORY_THROUGH_C84FL2.md").write_text(memory, encoding="utf-8")
    _update_handoff(REPO_ROOT / "oaci/OACI_CODEX_HANDOFF.md", context["head"], regressions, len(red_team))
    return {
        "gate": protocol.SUCCESS_GATE, "verification_HEAD": context["head"],
        "execution_lock_sha256": LOCK_SHA256, "synthetic": 25,
        "red_team": len(red_team),
        "regressions": [{"suite": row["suite"], "passed": row["passed"]} for row in regressions],
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
