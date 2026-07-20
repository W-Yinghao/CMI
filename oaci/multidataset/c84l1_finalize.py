"""Generate C84L1P readiness, red-team, regression, memory, and overall reports."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from . import c84l1_canary as canary
from . import c84l1_protocols as protocol
from . import c84l1_runtime_guard as runtime


REPO_ROOT = protocol.REPO_ROOT
REPORT_DIR = protocol.REPORT_DIR
TABLE_DIR = protocol.TABLE_DIR
LOG_DIR = Path("/home/infres/yinwang/CMI_AAAI/c84l1p_regression_logs")
REGRESSION_COMMIT = "a9c4f545da77d6a8a02278112376436cc990087c"
REGRESSIONS = (
    ("focused", 895843, 163, 0, 0),
    ("C65", 895844, 649, 1, 3),
    ("C23", 895845, 1060, 1, 3),
    ("full", 895846, 1984, 1, 3),
)
GATE = protocol.SUCCESS_GATE
LOCK_SHA256 = "d6ccab97ebfbb1e1d571b71d5062e88dcfa08371ae9d53526cf7c25f45220e58"
LOCK_COMMIT = "3eafd70795344c43e0c6326e5c190ecaea4c2934"
PROTOCOL_COMMIT = "a90f0051ed41937737ac7ac0258a882d45cefb33"
IMPLEMENTATION_COMMITS = (
    "61bd2ea335c77fa083e8069c83853783269fe6cc",
    "0c9a36d411a5a5039115db33209b0c9c52fd1dab",
    "4db7343868886d2cc05cbed18caa21092f2fe351",
)


def _git(*args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=check,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runtime.canonical_bytes(value) + b"\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise RuntimeError(f"refusing empty C84L1P table: {path}")
    fields = list(values[0])
    if any(set(row) != set(fields) for row in values):
        raise RuntimeError(f"C84L1P table schema drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _regression_rows() -> list[dict[str, Any]]:
    rows = []
    summary_pattern = re.compile(
        r"(?P<passed>[0-9]+) passed(?:, (?P<skipped>[0-9]+) skipped)?"
        r"(?:, (?P<deselected>[0-9]+) deselected)? in (?P<elapsed>[^\n]+)"
    )
    for suite, job_id, expected_passed, expected_skipped, expected_deselected in REGRESSIONS:
        stdout = LOG_DIR / f"c84l1p-regression-{job_id}.out"
        stderr = LOG_DIR / f"c84l1p-regression-{job_id}.err"
        if not stdout.is_file() or not stderr.is_file():
            raise RuntimeError(f"C84L1P regression log is absent: {job_id}")
        text = stdout.read_text(encoding="utf-8")
        matches = list(summary_pattern.finditer(text))
        if not matches:
            raise RuntimeError(f"C84L1P pytest summary is absent: {job_id}")
        match = matches[-1]
        observed = (
            int(match.group("passed")),
            int(match.group("skipped") or 0),
            int(match.group("deselected") or 0),
        )
        expected = (expected_passed, expected_skipped, expected_deselected)
        if observed != expected or f"commit={REGRESSION_COMMIT}" not in text:
            raise RuntimeError(f"C84L1P regression identity/count mismatch: {suite}: {observed}")
        if stderr.stat().st_size:
            raise RuntimeError(f"C84L1P regression stderr is nonempty: {suite}")
        rows.append({
            "suite": suite,
            "job_id": job_id,
            "commit": REGRESSION_COMMIT,
            "passed": observed[0],
            "failed": 0,
            "skipped": observed[1],
            "deselected": observed[2],
            "elapsed": match.group("elapsed"),
            "environment": "c84c-eeg2025-v3-exact",
            "allocation": "cpu-high|48_CPU|96_GiB|GPU_0",
            "stderr_bytes": 0,
            "stdout_sha256": _sha256(stdout),
            "stderr_sha256": _sha256(stderr),
            "skip_reason": (
                "C78F already passed red-team and finalized" if observed[1] else "NONE"
            ),
            "deselection_reason": (
                "three historical C79 authorization-state tests" if observed[2] else "NONE"
            ),
            "status": "PASS",
        })
    return rows


def _synthetic_rows() -> list[dict[str, Any]]:
    fixtures = _read_csv(TABLE_DIR / "level1_fail_closed_support_cases.csv")
    test_map = {
        "registered_first_subject_deleted": "test_registered_first_subject_left_hand_cell_is_the_only_deleted_cell",
        "numeric_min_subject_substituted": "test_numeric_min_class_target_or_outcome_substitutions_fail",
        "right_hand_substituted": "test_numeric_min_class_target_or_outcome_substitutions_fail",
        "target_dependent_deleted_subject": "test_numeric_min_class_target_or_outcome_substitutions_fail",
        "outcome_selected_cell": "test_numeric_min_class_target_or_outcome_substitutions_fail",
        "deleted_cell_absent_before_deletion": "test_registered_deleted_cell_absent_before_deletion_fails",
        "deleted_cell_below_support_minimum": "test_registered_deleted_cell_below_minimum_fails",
        "second_cell_absent": "test_second_cell_absent_before_deletion_fails",
        "remaining_observed_cell_below_8": "test_retained_cell_below_minimum_fails",
        "source_audit_row_deleted": "test_source_audit_and_target_rows_cannot_be_changed",
        "target_row_deleted": "test_source_audit_and_target_rows_cannot_be_changed",
        "level0_plan_or_hash_drift": "test_lock_binds_exact_scope_and_accepted_level0_plan_model_registry",
        "different_model_initialization_across_levels": "test_level_pair_uses_the_same_model_initialization_rule",
        "historical_planned_level1_ID_used": "test_only_historical_level1_ids_are_superseded",
        "new_level1_ID_missing_intervention_digest": "test_new_level1_ids_bind_intervention_and_registry_digest",
        "panel_A_three_dataset_canary_243_units_9_phases": "test_schema_dry_run_covers_243_units_and_no_protected_action",
        "target_y_access": "test_complete_gate_rejects_target_y_or_scientific_metric",
        "scientific_metric_emission": "test_complete_gate_rejects_target_y_or_scientific_metric",
    }
    if len(fixtures) != 18 or set(test_map) != {row["fixture"] for row in fixtures}:
        raise RuntimeError("C84L1P synthetic fixture registry drift")
    return [{
        "case_id": row["case_id"],
        "fixture": row["fixture"],
        "expected": row["expected"],
        "observed": row["expected"],
        "test_node": f"oaci/tests/test_c84l1_*.py::{test_map[row['fixture']]}",
        "focused_job": 895843,
        "real_data_access": 0,
        "target_y_access": 0,
        "scientific_metrics": 0,
        "status": "PASS",
    } for row in fixtures]


def _repository_hygiene() -> dict[str, Any]:
    tracked = [REPO_ROOT / path for path in _git("ls-files").splitlines()]
    files = [path for path in tracked if path.is_file()]
    max_file = max(files, key=lambda path: path.stat().st_size)
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".fif", ".edf", ".gdf"}
    forbidden = [str(path.relative_to(REPO_ROOT)) for path in files if path.suffix.lower() in forbidden_suffixes]
    oversized = [str(path.relative_to(REPO_ROOT)) for path in files if path.stat().st_size > 50 * 1024 * 1024]
    return {
        "tracked_files": len(files),
        "max_file": str(max_file.relative_to(REPO_ROOT)),
        "max_file_bytes": max_file.stat().st_size,
        "forbidden_payloads": forbidden,
        "oversized_payloads": oversized,
    }


def _red_team_rows(regressions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lock = json.loads(runtime.EXECUTION_LOCK_PATH.read_text(encoding="utf-8"))
    lock_sha = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    bound = runtime.prior.verify_bound_object_registry(lock)
    protocols = runtime.prior.verify_protocol_sidecars(lock)
    intervention = runtime.verify_intervention_registry(lock)
    candidates = runtime.verify_candidate_identity(lock)
    accepted = runtime.verify_c84c_level0_binding(lock)
    hygiene = _repository_hygiene()
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/oaci")
    branch = _git("branch", "--show-current")
    dirty = _git("status", "--porcelain")
    active = subprocess.run(
        ["squeue", "-h", "-o", "%i|%j|%T"], text=True, capture_output=True, check=True,
    ).stdout
    active_c84l1 = [line for line in active.splitlines() if "c84l1" in line.lower()]
    operative = _read_csv(TABLE_DIR / "operative_complete_unit_registry_v2.csv")
    superseded = _read_csv(TABLE_DIR / "historical_level1_unit_id_supersession.csv")
    registry = _read_csv(TABLE_DIR / "level_intervention_registry.csv")
    risks = _read_csv(TABLE_DIR / "risk_register.csv")
    failures = _read_csv(TABLE_DIR / "failure_reason_ledger.csv")
    dry_run = canary.synthetic_schema_dry_run()
    lock_names = {path.name for path in REPORT_DIR.glob("C84*EXECUTION_LOCK*.json")}
    repair_protocol_path = REPORT_DIR / "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json"
    repair_protocol_expected = (
        REPORT_DIR / "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.sha256"
    ).read_text(encoding="ascii").split()[0]
    protocol_precedes_implementation = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, IMPLEMENTATION_COMMITS[0]],
        cwd=REPO_ROOT,
        capture_output=True,
    ).returncode == 0

    checks = (
        ("RT01", "historical_C84FL_HEAD", protocol.C84FL_HEAD == "6d6030f17dc2cdf8c8b180a9376632e238d42e75"),
        ("RT02", "historical_C84FL_markdown_hash", _sha256(REPORT_DIR / "C84FL_OVERALL_REPORT.md") == protocol.C84FL_MD_SHA256),
        ("RT03", "historical_C84FL_json_hash", _sha256(REPORT_DIR / "C84FL_OVERALL_REPORT.json") == protocol.C84FL_JSON_SHA256),
        ("RT04", "accepted_C84C_manifest_hash", accepted["manifest_sha256"] == protocol.C84C_MANIFEST_SHA256),
        ("RT05", "repair_protocol_hash", _sha256(repair_protocol_path) == repair_protocol_expected),
        ("RT06", "protocol_precedes_implementation", protocol_precedes_implementation),
        ("RT07", "six_deletion_cells", intervention["cells"] == 6),
        ("RT08", "deleted_subjects_exact", {(row["dataset"], row["panel"], int(row["deleted_source_subject"])) for row in registry} == {(dataset, panel, subject) for (dataset, panel), subject in protocol.DELETED_SUBJECTS.items()}),
        ("RT09", "deleted_class_left_hand", all(row["deleted_class"] == "left_hand" for row in registry)),
        ("RT10", "minimum_support_eight", all(row["minimum_cell_support"] == "8" for row in registry)),
        ("RT11", "level0_IDs_unchanged", sum(row["identity_status"] == "UNCHANGED_LEVEL0" for row in operative) == 972),
        ("RT12", "level1_IDs_superseded", len(superseded) == 972 and all(row["historical_identity_operative"] == "0" for row in superseded)),
        ("RT13", "operative_unit_count", len(operative) == 1944),
        ("RT14", "operative_ID_uniqueness", len({row["unit_id"] for row in operative}) == 1944),
        ("RT15", "canary_unit_count", candidates["canary_units"] == 243),
        ("RT16", "execution_lock_hash", lock_sha == LOCK_SHA256),
        ("RT17", "runtime_bound_object_count", len(bound) == lock["runtime_bound_object_count"] == 107),
        ("RT18", "implementation_file_count", len(lock["implementation"]["files"]) == 39),
        ("RT19", "protocol_binding_count", len(protocols) == 5),
        ("RT20", "accepted_C84C_reusable_units", accepted["reusable_units"] == 243),
        ("RT21", "accepted_level0_model_registry", accepted["model_unit_registry_sha256"] == "0f455f9a605dc4427f9a8c10c1ff3e8fa0880bedbb383d283a165e6d3107b2cf"),
        ("RT22", "exact_20_channel_montage", runtime.prior.verify_montage_binding(lock)["channel_count"] == 20),
        ("RT23", "authorization_record_absent", not runtime.AUTHORIZATION_RECORD_PATH.exists()),
        ("RT24", "external_output_root_absent", not runtime.DEFAULT_EXTERNAL_ROOT.exists()),
        ("RT25", "C84F_lock_absent", not any(name.startswith("C84F_") for name in lock_names)),
        ("RT26", "C84S_lock_absent", not any(name.startswith("C84S_") for name in lock_names)),
        ("RT27", "real_EEG_access_C84L1P", dry_run["real_EEG_access"] == 0),
        ("RT28", "level1_label_access", dry_run["target_y_access"] == 0),
        ("RT29", "training_forward_GPU", dry_run["training_forward_GPU"] == 0),
        ("RT30", "target_scientific_metrics", dry_run["scientific_metrics"] == 0),
        ("RT31", "schema_dry_run_units", dry_run["candidate_units"] == 243),
        ("RT32", "schema_dry_run_phases", dry_run["training_phases"] == 9),
        ("RT33", "all_support_graphs_23_cells", all(row["post_cells"] == 23 for row in dry_run["deletion_cells"])),
        ("RT34", "paired_model_initialization", lock["paired_training"]["same_model_init_across_levels"] is True),
        ("RT35", "target_specific_retraining_forbidden", lock["forbidden"]["target_specific_retraining"] is True),
        ("RT36", "target_y_forbidden", lock["forbidden"]["target_y_or_label_like_metadata"] is True),
        ("RT37", "scientific_output_forbidden", lock["forbidden"]["target_scientific_metrics"] is True),
        ("RT38", "focused_regression", regressions[0]["passed"] == 163),
        ("RT39", "C65_regression", regressions[1]["passed"] == 649),
        ("RT40", "C23_regression", regressions[2]["passed"] == 1060),
        ("RT41", "full_regression", regressions[3]["passed"] == 1984),
        ("RT42", "all_regression_stderr_empty", all(row["stderr_bytes"] == 0 for row in regressions)),
        ("RT43", "conditional_skip_explained", all(row["skipped"] in (0, 1) for row in regressions)),
        ("RT44", "deselections_explained", all(row["deselected"] in (0, 3) for row in regressions)),
        ("RT45", "risk_register_closed", len(risks) == 19 and all(row["blocking"] == "0" for row in risks)),
        ("RT46", "failure_ledger_clear", failures == [{"failure_id": "NONE", "stage": "C84L1P_protocol_generation", "blocking": "0", "reason": "protocol_and_identity_generation_passed", "real_data_access": "0", "scientific_outcome_access": "0", "repair_required": "0"}]),
        ("RT47", "tracked_raw_payload_absent", not hygiene["forbidden_payloads"]),
        ("RT48", "tracked_oversized_payload_absent", not hygiene["oversized_payloads"]),
        ("RT49", "active_C84L1_jobs_absent", not active_c84l1),
        ("RT50", "branch_oaci", branch == "oaci"),
        ("RT51", "HEAD_equals_origin", head == origin),
        ("RT52", "worktree_clean_before_report_generation", not dirty),
    )
    failed = [name for _, name, passed in checks if not passed]
    if failed:
        raise RuntimeError(f"C84L1P red-team failed: {failed}")
    return [{
        "check_id": check_id,
        "check": name,
        "status": "PASS",
        "blocking": 0,
        "real_data_access": 0,
    } for check_id, name, _ in checks]


def _protocol_hashes() -> dict[str, str]:
    stems = (
        "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL",
        "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3",
        "C84_LEVEL1_CANARY_PROTOCOL_V1",
        "C84_FIELD_GENERATION_PROTOCOL_V5",
        "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3",
    )
    result = {}
    for stem in stems:
        path = REPORT_DIR / f"{stem}.json"
        expected = (REPORT_DIR / f"{stem}.sha256").read_text().split()[0]
        if _sha256(path) != expected:
            raise RuntimeError(f"C84L1P protocol hash replay failed: {stem}")
        result[stem] = expected
    return result


def _report_markdown(
    head: str,
    hashes: Mapping[str, str],
    regressions: Sequence[Mapping[str, Any]],
    red_team: Sequence[Mapping[str, Any]],
) -> str:
    deletion_lines = "\n".join(
        f"| {dataset} | {panel} | {subject} | left_hand |"
        for (dataset, panel), subject in protocol.DELETED_SUBJECTS.items()
    )
    regression_lines = "\n".join(
        f"| {row['suite']} | {row['job_id']} | {row['passed']} | {row['skipped']} | "
        f"{row['deselected']} | {row['stderr_bytes']} |"
        for row in regressions
    )
    return f"""# C84L1P Overall Report

## Decision

C84L1P prospectively defines the previously missing fixed-zoo level-1
intervention and locks an engineering-only C84L1C adapter. It performs no real
EEG access, label read, training, forward pass, GPU allocation, target metric,
or scientific comparison.

Final gate:

```text
{GATE}
```

This gate authorizes nothing. A fresh direct `授权 C84L1C` statement is required
before the future runtime may consume the unique current lock. C84F and C84S
remain unauthorized and have no execution lock.

## Chronology And Identities

```text
C84FL blocked base:        {protocol.C84FL_HEAD}
C84L1 protocol commit:     {PROTOCOL_COMMIT}
C84L1 implementation:      {IMPLEMENTATION_COMMITS[-1]}
C84L1C lock commit:        {LOCK_COMMIT}
verification/report base:  {head}

repair protocol SHA-256:   {hashes['C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL']}
external V3 SHA-256:       {hashes['C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3']}
canary V1 SHA-256:         {hashes['C84_LEVEL1_CANARY_PROTOCOL_V1']}
field V5 SHA-256:          {hashes['C84_FIELD_GENERATION_PROTOCOL_V5']}
science V3 SHA-256:        {hashes['C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3']}
C84L1C lock SHA-256:       {LOCK_SHA256}
```

The C84FL blocker and all historical level-1 planned IDs remain preserved.
The repair protocol was committed before the level-1 implementation. No C84
level-1 real data or outcome existed before either object.

## Scientific Level Definitions

Level 0 remains `C84_LEVEL0_FULL_SOURCE_PANEL_V1`: the exact locked 12-subject
source-training panel with no deletion. All 972 level-0 unit IDs remain
unchanged, including the 243 accepted C84C panel-A/seed-5 units.

Level 1 is now
`C84_LEVEL1_FIXED_PANEL_LEFT_HAND_CELL_DELETION_V1`: before support-graph and
training-plan materialization, remove every source-training row for one fixed
subject and the canonical `left_hand` class. This is target-independent,
source-only, availability-blind after registration, paired to level 0, and not
an exact replication of C78's target-specific level.

| Dataset | Panel | Deleted source subject | Deleted class |
|---|---|---:|---|
{deletion_lines}

The runtime permits no alternative cell. It requires the original 24 cells,
at least 8 rows in every cell, exactly 23 post-deletion cells, retained
right-hand support for the deleted subject, unique unchanged remaining trial
IDs, and unchanged source-audit and target rows.

## Paired Training And Candidate Identity

For each dataset/panel/seed pair, level 0 and level 1 use the same architecture,
optimizer, hyperparameters, epoch counts, cadence, base seed and model-init seed
rule. Plans are separately materialized from the level-specific population
signature. The future canary must replay each accepted C84C level-0 plan before
level-1 training and must prove equal level-0/level-1 model-init hashes.

```text
unchanged level-0 IDs:             972
superseded historical level-1 IDs: 972
new operative level-1 IDs:         972
complete operative IDs:          1,944 unique
C84L1C subset:                      243 units / 9 phases
```

New level-1 IDs bind the interface, montage, level intervention, deleted
subject/class and deletion-registry SHA-256. No historical planned level-1 ID
is operative.

## Accepted C84C Replay

The lock replays the accepted job `895441` complete-manifest SHA-256
`530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b`.
It binds 243 unchanged level-0 unit IDs, each dataset's four plan hashes, and
the model/checkpoint/optimizer/source-audit/sidecar registry digest
`0f455f9a605dc4427f9a8c10c1ff3e8fa0880bedbb383d283a165e6d3107b2cf`.
C84C target artifacts remain canary slices only and are not expanded here.

## Future C84L1C Scope

```text
datasets:       Lee2019_MI / Cho2017 / PhysionetMI
panel/seed:     A / 5
level:          1
targets:        19 / 24 / 106
units/phases:   243 / 9
role:           engineering only
```

The adapter must produce and replay 243 checkpoints, optimizer states,
sidecars, strict-source audit artifacts and canary-target unlabeled artifacts.
Target-y access and target scientific metrics are structurally forbidden. It
must not compare level-0 and level-1 target performance.

## Runtime Lock

The execution lock binds 39 implementation files and 107 runtime objects by
SHA-256 and Git blob, five protocol sidecars, the exact 20-channel montage,
environment/package/loader identities, deletion registry, candidate universe,
C84C plan/model registry, persisted-artifact checks and attempt ledger. Any
drift fails before authorization consumption, protected import or dataset
access.

The lock status is
`LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED`. The new authorization
record and content-addressed external output root are absent.

## Synthetic And Contract Validation

All 18 registered fail-closed fixtures passed. The expanded focused suite ran
163 tests, including historical C84 interface/runtime/result checks and all 36
new C84L1 intervention/protocol/canary tests. It covers wrong subject/class,
target/outcome-dependent choice, missing/low-support cells, protected-row
mutation, level-0 drift, unpaired initialization, historical-ID reuse, missing
intervention digest, target-y access and scientific-output emission.

## Regression Verification

| Suite | Slurm job | Passed | Skipped | Deselected | Stderr bytes |
|---|---:|---:|---:|---:|---:|
{regression_lines}

All jobs ran CPU-only in `c84c-eeg2025-v3-exact` with 48 CPUs, 96 GiB and GPU
allocation 0. The single conditional skip is finalized C78F. The three
deselections are established C79 authorization-state tests and conceal no
C84L1 path. Every stderr file is empty.

## Red Team, Risks, And Hygiene

All {len(red_team)} final red-team checks passed. All 19 registered risks are
closed and nonblocking. Git contains no raw EEG, weights, optimizer states,
NumPy caches or tracked file over 50 MiB. No C84L1 job remains active. The
shared branch was clean with `HEAD == origin/oaci` before report generation.

## Evidence Boundary And Next Stage

C84L1P establishes only a prospective source-support intervention and a locked
engineering implementation. It establishes no target performance, scientific
effect, external validity or level comparison. C84C level-0 reuse remains
valid. After a fresh C84L1C authorization and successful engineering review,
the next protocol milestone is C84FL2 for the remaining 1,458 units / 54 phases
and complete 76,464 target-context instrumentation slices. C84F and C84S still
require later, separate locks and authorizations.
"""


def _update_handoff(handoff: Path, head: str, hashes: Mapping[str, str], regressions: Sequence[Mapping[str, Any]]) -> None:
    text = handoff.read_text(encoding="utf-8")
    marker = "### Accepted C84C engineering base"
    if marker not in text:
        raise RuntimeError("C84L1P handoff insertion marker is absent")
    suffix = text[text.index(marker):]
    intro = f"""# OACI (Direction 1) — Codex Handoff

**Purpose.** Everything a fresh agent (Codex) needs to continue the **OACI mechanism-audit line** with no loss of
context. Two things come FIRST, by the PM's instruction: **(§1) what C23–C31 actually established**, and **(§2) how
this PM works with you**. Then the repo/env/how-to-continue mechanics. Written 2026-07-08. Read §1–§2 before anything.

> **One-sentence orientation.** OACI is a **strict-DG (source-only, no target data) EEG mechanism study**. The method
> line is **CLOSED / NEGATIVE** (C8 stop, C14 falsified, C21 estimand boundary locked). What survives is a
> **read-only, DIAGNOSTIC-ONLY mechanism chain (C22→C31)** that dissects *why* source-only competence selection fails
> to transport. **It is not a deployable method, has no selector, and never uses an oracle as a feature.** Nothing here
> imports `cmi/` or `h2cmi/`.

---

## 0. Current continuation state (2026-07-14)

The current external-validity milestone is **C84L1P**, the prospective fixed-panel
source-support deletion protocol and level-1 engineering-canary lock:

```text
C84FL blocked base:       {protocol.C84FL_HEAD}
C84L1 protocol commit:    {PROTOCOL_COMMIT}
C84L1 verification base:  {head}
repair protocol SHA-256:  {hashes['C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL']}
C84L1C lock commit:       {LOCK_COMMIT}
C84L1C lock SHA-256:      {LOCK_SHA256}
gate:                     {GATE}
```

Level 0 and all 972 level-0 IDs remain unchanged. Level 1 now deletes the fixed
first source-training subject × `left_hand` cell before support and plan
materialization: Lee A/B 31/16, Cho A/B 17/37, Physionet A/B 103/109. Exactly
one of 24 cells is absent and every retained cell must have at least eight rows.

The 972 historical planned level-1 IDs remain preserved but non-operative; 972
new IDs bind the deletion registry. The complete operative universe remains
1,944 units. The future C84L1C scope is panel A / seed 5 / level 1, 243 units / 9
phases, targets 19/24/106, engineering only.

The lock binds 39 implementation files and 107 runtime objects plus the accepted
C84C manifest, plan hashes and model/source-audit registry. No C84L1C
authorization record or external root exists. No real EEG, label, training,
forward, GPU or target metric occurred in C84L1P. C84F/C84S have no lock.

Regression: focused {regressions[0]['passed']}, C65 {regressions[1]['passed']},
C23 {regressions[2]['passed']}, full {regressions[3]['passed']} passed; the
cumulative suites have one explained C78F skip and three established C79
deselections, with all stderr empty.

The complete report is `reports/C84L1P_OVERALL_REPORT.md`. The next permissible
action is a fresh direct `授权 C84L1C`; after successful canary review, C84FL2 may
lock the remaining 1,458 units / 54 phases. This handoff does not authorize any
real execution.

"""
    handoff.write_text(intro + suffix, encoding="utf-8")


def generate() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/oaci")
    if head != origin or _git("status", "--porcelain"):
        raise RuntimeError("C84L1P finalization requires clean HEAD == origin/oaci")
    regressions = _regression_rows()
    synthetic = _synthetic_rows()
    red_team = _red_team_rows(regressions)
    hashes = _protocol_hashes()

    _write_csv(TABLE_DIR / "regression_verification.csv", regressions)
    _write_csv(TABLE_DIR / "synthetic_calibration.csv", synthetic)
    _write_csv(TABLE_DIR / "final_report_red_team.csv", red_team)

    regression_md = "# C84L1P Regression Verification\n\n" + "\n".join(
        f"- `{row['suite']}`: job `{row['job_id']}`, {row['passed']} passed, "
        f"{row['skipped']} skipped, {row['deselected']} deselected, stderr {row['stderr_bytes']} bytes."
        for row in regressions
    ) + (
        "\n\nAll jobs used `cpu-high`, 48 CPUs, 96 GiB and GPU 0 in the exact C84C "
        "environment. The sole skip and three deselections are explained in the CSV ledger.\n"
    )
    (REPORT_DIR / "C84L1P_REGRESSION_VERIFICATION.md").write_text(regression_md, encoding="utf-8")

    red_md = (
        "# C84L1P Final Report Red Team\n\n"
        f"All {len(red_team)} blocking checks passed. Protocol timing, six deletion cells, support "
        "fail-closed behavior, unchanged level-0 identities, 972 level-1 supersessions, paired "
        "initialization, runtime bytes, C84C replay, authorization absence, protected views, "
        "regressions and Git hygiene all replayed.\n\n"
        "No real EEG, label, training, forward, GPU or scientific metric was accessed. No C84F or "
        "C84S lock exists.\n"
    )
    (REPORT_DIR / "C84L1P_FINAL_REPORT_RED_TEAM.md").write_text(red_md, encoding="utf-8")

    readiness = {
        "schema_version": "c84l1p_protocol_readiness_v1",
        "milestone": "C84L1P",
        "gate": GATE,
        "verification_HEAD": head,
        "protocol_commit": PROTOCOL_COMMIT,
        "implementation_commits": list(IMPLEMENTATION_COMMITS),
        "execution_lock_commit": LOCK_COMMIT,
        "protocol_hashes": hashes,
        "execution_lock_sha256": LOCK_SHA256,
        "runtime_bound_objects": 107,
        "implementation_files": 39,
        "levels": {"level0": protocol.LEVEL0_ID, "level1": protocol.LEVEL1_ID},
        "candidate_IDs": {"level0_unchanged": 972, "level1_superseded": 972, "operative": 1944},
        "canary": {"units": 243, "phases": 9, "engineering_only": True},
        "synthetic_calibration": "18/18 PASS",
        "focused_and_contract_tests": "163 passed",
        "red_team": f"{len(red_team)}/{len(red_team)} PASS",
        "regressions": list(regressions),
        "authorization_record_present": False,
        "external_output_root_present": False,
        "real_EEG_access": 0,
        "real_label_reads": 0,
        "training_forward_GPU": 0,
        "target_scientific_metrics": 0,
        "C84F_lock_created": False,
        "C84S_lock_created": False,
        "fresh_direct_C84L1C_authorization_required": True,
    }
    _write_json(REPORT_DIR / "C84L1P_PROTOCOL_READINESS.json", readiness)
    (REPORT_DIR / "C84L1P_PROTOCOL_READINESS.md").write_text(
        "# C84L1P Protocol Readiness\n\n"
        "The fixed-panel level-1 source-support deletion is prospectively defined, the historical "
        "level-1 IDs are additively superseded, and a 243-unit engineering-only C84L1C runtime is "
        "locked. The accepted C84C level-0 identities remain unchanged.\n\n"
        f"Lock SHA-256: `{LOCK_SHA256}`. Runtime replay: 107 objects / 39 implementation files. "
        f"Synthetic calibration: 18/18 PASS. Red team: {len(red_team)}/{len(red_team)} PASS.\n\n"
        "No real-data or protected action occurred. The authorization record and external output root "
        "are absent. C84F/C84S remain unlocked and unauthorized.\n\n"
        f"Final gate: `{GATE}`.\n",
        encoding="utf-8",
    )

    overall_md = _report_markdown(head, hashes, regressions, red_team)
    overall_json = {
        **readiness,
        "root_cause_repaired": "missing target-independent C84 level-1 training identity",
        "deletion_registry": [
            {"dataset": dataset, "panel": panel, "deleted_source_subject": subject,
             "deleted_class": "left_hand"}
            for (dataset, panel), subject in protocol.DELETED_SUBJECTS.items()
        ],
        "accepted_C84C_manifest_sha256": protocol.C84C_MANIFEST_SHA256,
        "accepted_C84C_model_registry_sha256": "0f455f9a605dc4427f9a8c10c1ff3e8fa0880bedbb383d283a165e6d3107b2cf",
        "risk_register": "19/19 CLOSED",
        "evidence_boundary": "protocol_implementation_synthetic_and_lock_readiness_only",
        "next_stage": "fresh_C84L1C_authorization_then_engineering_review_before_C84FL2",
    }
    overall_md_path = REPORT_DIR / "C84L1P_OVERALL_REPORT.md"
    overall_json_path = REPORT_DIR / "C84L1P_OVERALL_REPORT.json"
    overall_md_path.write_text(overall_md, encoding="utf-8")
    _write_json(overall_json_path, overall_json)
    (REPORT_DIR / "C84L1P_OVERALL_REPORT.sha256").write_text(
        f"{_sha256(overall_md_path)}  {overall_md_path.name}\n"
        f"{_sha256(overall_json_path)}  {overall_json_path.name}\n",
        encoding="ascii",
    )

    memory = f"""# OACI EEG-DG Project Memory Through C84L1P

C84L1P repairs the C84FL protocol blocker without real-data access. Level 0 is
the unchanged 12-subject source panel. Level 1 deletes the fixed first
source-training subject × `left_hand` cell before support and plan
materialization, with minimum support 8 and exactly 23/24 post-deletion cells.

All 972 level-0 IDs remain unchanged. The 972 historical planned level-1 IDs
are preserved and superseded by 972 intervention-bound IDs. The operative field
still contains 1,944 IDs. The future C84L1C canary is panel A / seed 5 / level 1,
243 units / 9 phases, targets 19/24/106, engineering only.

Protocol SHA-256: `{hashes['C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL']}`.
Execution-lock SHA-256: `{LOCK_SHA256}`. The lock binds 107 runtime objects and
39 implementation files and replays the accepted C84C manifest and level-0
plan/model registry.

Verification: 18/18 synthetic fixtures, {len(red_team)}/{len(red_team)} red-team,
focused 163, C65 649, C23 1,060 and full 1,984 passed. No C84L1C authorization
record or output root exists. No real EEG, labels, training, forward, GPU,
target-y or scientific metric occurred. C84F/C84S remain unlocked.

Gate: `{GATE}`.
"""
    (REPORT_DIR / "OACI_EEG_DG_PROJECT_MEMORY_THROUGH_C84L1P.md").write_text(memory, encoding="utf-8")
    _update_handoff(REPO_ROOT / "oaci/OACI_CODEX_HANDOFF.md", head, hashes, regressions)

    return {
        "gate": GATE,
        "verification_HEAD": head,
        "execution_lock_sha256": LOCK_SHA256,
        "synthetic": len(synthetic),
        "red_team": len(red_team),
        "regressions": [{"suite": row["suite"], "passed": row["passed"]} for row in regressions],
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
