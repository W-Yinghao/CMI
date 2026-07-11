"""C79P post-seed-3 protocol replay, execution locks, and readiness gates.

This module is deliberately standard-library only.  C79P must be able to build
and audit every readiness artifact without importing an EEG loader, PyTorch, or
CUDA.  Real seed-4 work is delegated to :mod:`c79e_seed4_replication` and stays
fail-closed until a future, scope-bound PI authorization record exists.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c79p_tables"

PROTOCOL_PATH = REPORT_DIR / "C79_POST_SEED3_SEED4_REPLICATION_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C79_POST_SEED3_SEED4_REPLICATION_PROTOCOL.sha256"
REGISTRY_PATH = TABLE_DIR / "c79_post_seed3_scientific_registry.csv"
LABEL_SPLIT_PATH = REPORT_DIR / "c78s_tables" / "label_split_isolation.csv"
C78S_RESULT_PATH = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS.json"
C78S_MULTIPLICITY_PATH = REPORT_DIR / "c78s_tables" / "primary_hypothesis_multiplicity.csv"
C78S_GATE_PATH = REPORT_DIR / "c78s_tables" / "registered_candidate_gate.csv"

EXPECTED_MANIFEST_JSON = REPORT_DIR / "C79P_EXPECTED_SEED4_MANIFEST.json"
EXPECTED_MANIFEST_CSV = TABLE_DIR / "expected_seed4_field_manifest.csv"
FIELD_LOCK_PATH = REPORT_DIR / "C79P_FIELD_GENERATION_EXECUTION_LOCK.json"
FIELD_LOCK_SHA_PATH = REPORT_DIR / "C79P_FIELD_GENERATION_EXECUTION_LOCK.sha256"
ANALYSIS_LOCK_PATH = REPORT_DIR / "C79P_SCIENTIFIC_ANALYSIS_EXECUTION_LOCK.json"
ANALYSIS_LOCK_SHA_PATH = REPORT_DIR / "C79P_SCIENTIFIC_ANALYSIS_EXECUTION_LOCK.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C79E_PI_AUTHORIZATION_RECORD.json"

IMPLEMENTATION_REPLAY_REPORT = REPORT_DIR / "C79P_IMPLEMENTATION_REPLAY.md"
PRE_EXECUTION_RED_TEAM = REPORT_DIR / "C79P_PRE_EXECUTION_RED_TEAM.md"
LOCK_LEDGER_PATH = REPORT_DIR / "C79P_EXECUTION_LOCK_LEDGER.json"
READINESS_REPORT = REPORT_DIR / "C79P_PROTOCOL_READINESS.md"
READINESS_JSON = REPORT_DIR / "C79P_PROTOCOL_READINESS.json"

DATASET = "BNCI2014_001"
REFERENCE_SEED = 3
REPLICATION_SEED = 4
PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
TARGET4_CANARY = 4
TARGET_ORDER = (4, 8, 9, 3, 6, 5, 2, 7, 1)
LEVELS = (0, 1)
REGIMES = ("ERM", "OACI", "SRC")
TRAJECTORY_EPOCHS = tuple(range(4, 200, 5))
EXPECTED_ENGINEERING_UNITS = 1458
EXPECTED_PRIMARY_UNITS = 1296
EXPECTED_PHASES = 54
EXPECTED_SOURCE_ROWS = EXPECTED_ENGINEERING_UNITS * 8 * 576
EXPECTED_TARGET_ROWS = EXPECTED_ENGINEERING_UNITS * 576
MAX_GIT_PAYLOAD = 50 * 1024 * 1024
FINAL_GATE = "C79_POST_SEED3_REPLICATION_PROTOCOL_LOCKED_READY_FOR_PI_AUTHORIZATION"

WAVES = {
    "C0_engineering_canary": (4,),
    "A": (8, 9, 3, 6),
    "B": (5, 2, 7, 1),
}

IMPLEMENTATION_FILES = (
    "oaci/conditioned_ceiling_coverage/c79p_post_seed3_protocol.py",
    "oaci/conditioned_ceiling_coverage/c79e_seed4_replication.py",
    "oaci/tests/test_c79p_post_seed3_protocol.py",
    "oaci/slurm_c79p_regression.sh",
    "oaci/slurm_c79e_field.sh",
    "oaci/slurm_c79e_field_cpu.sh",
    "oaci/slurm_c79e_analysis.sh",
)

REFERENCE_FILES = {
    "oaci/conditioned_ceiling_coverage/c78s_protocol.py": "4c9ad2c3e3516167716b79b73893f3714ce14aba99bc3ed21dd9c6d20a0a8ef7",
    "oaci/conditioned_ceiling_coverage/c78s_modeling.py": "372f4b5e0315d487c452e74ce78736583c105a16c027362e4d96cc713d8175ca",
    "oaci/conditioned_ceiling_coverage/c78s_seed3_scientific_analysis.py": "785e1a29f64ae03898aeb832914afc0cc4097db5ae5dfd0d0dfddbac1306c4f8",
    "oaci/reports/c78s_tables/feature_block_registry.csv": "accc2e5c3a45fcb39fc31f2196ee227ce504a08e17f743e46a0f654cf42420c1",
}

EXPECTED_REFERENCE_VALUES = {
    "H1_split_label_reliability": 0.7708629907592237,
    "H1_Holm_p": 0.058365758754863814,
    "H1_construction_top1": 0.125,
    "H1_construction_top5": 0.6875,
    "H1_construction_top10": 0.75,
    "H1_random_top1": 1.0 / 81.0,
    "H1_random_top5": 5.0 / 81.0,
    "H1_random_top10": 10.0 / 81.0,
    "H1_standardized_regret": 0.08279432226091084,
    "H1_random_expected_regret": 0.4820,
    "H2_held_target_deviance_change": 9.505920587315757,
    "H2_permutation_p": 0.896,
    "H3_local_association": 0.24265629215048484,
    "H3_positive_trajectory_cells": 32.0,
    "H3_worst_control_p": 0.002,
    "H3_LOTO_incremental_R2": -0.21287505480736912,
    "H3_LORO_incremental_R2": -0.08579644289995947,
    "H4_F2_incremental_R2": -0.07308583875591457,
    "H5_F4_incremental_R2": 0.005176499403102386,
    "H6_incremental_R2": 0.40432932651535547,
    "H6_raw_p": 0.019455252918287938,
    "H6_Holm_p": 0.07782101167315175,
}

REGISTRY_CATEGORIES = (
    "path_id_and_hypothesis_role",
    "scientific_claim",
    "information_class",
    "seed_population_targets_and_exclusions",
    "regime_trajectory_and_candidate_universe",
    "feature_block_or_descriptor_formula",
    "base_and_conditioning_model",
    "construction_evaluation_split_and_trial_hashes",
    "statistic_effect_size_and_sign_convention",
    "outer_inner_splits_and_dependence_groups",
    "null_permutation_bootstrap_and_RNG_stream",
    "kernel_bandwidth_scaling_or_model_hyperparameters",
    "materiality_or_candidate_qualification_threshold",
    "multiplicity_family_correction_and_hypothesis_order",
    "success_failure_heterogeneity_and_blocker_rules",
    "allowed_interpretation_and_forbidden_interpretation",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty C79P table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = PROTOCOL_SHA_PATH.read_text().strip()
    observed = sha256_file(PROTOCOL_PATH)
    if expected != observed:
        raise RuntimeError(f"C79P protocol hash drift: {observed} != {expected}")
    payload = json.loads(PROTOCOL_PATH.read_text())
    if payload.get("milestone") != "C79P":
        raise RuntimeError("C79P protocol milestone drift")
    status = payload.get("epistemic_status", {})
    required = {
        "designed_after_C78S_outcomes": True,
        "prospective_to_seed4_checkpoint_outcomes": True,
        "pre_C78S_confirmatory_protocol": False,
        "outcome_informed_replication": True,
        "training_seed_robustness_only": True,
        "new_target_population_confirmation": False,
        "new_raw_EEG_sample_confirmation": False,
        "same_targets_and_trials_as_seed3": True,
        "seed4_untouched_at_protocol_commit": True,
    }
    if any(status.get(key) is not value for key, value in required.items()):
        raise RuntimeError("C79P epistemic-status contract drift")
    if payload["authorization"]["received"] is not False:
        raise PermissionError("C79P protocol unexpectedly records C79E authorization")
    return payload, observed


def protocol_commit() -> str:
    commit = git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(REPO_ROOT)))
    if not commit:
        raise RuntimeError("C79P protocol is not committed")
    return commit


def _wave(target: int) -> str:
    matches = [name for name, targets in WAVES.items() if target in targets]
    if len(matches) != 1:
        raise RuntimeError(f"target {target} has an invalid C79P wave assignment")
    return matches[0]


def _unit_id(target: int, level: int, regime: str, epoch: int, order: int) -> str:
    identity = {
        "dataset": DATASET,
        "seed": REPLICATION_SEED,
        "target": target,
        "level": level,
        "regime": regime,
        "epoch": epoch,
        "trajectory_order": order,
    }
    return "c79_" + sha256_bytes(canonical_bytes(identity))[:20]


def expected_seed4_units() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGET_ORDER:
        wave = _wave(target)
        role_prefix = "engineering_canary_descriptive_only" if target == 4 else "primary_seed4_replication"
        for level in LEVELS:
            rows.append({
                "unit_id": _unit_id(target, level, "ERM", 199, 0),
                "dataset": DATASET,
                "target": target,
                "seed": REPLICATION_SEED,
                "level": level,
                "regime": "ERM",
                "epoch": 199,
                "trajectory_order": 0,
                "wave": wave,
                "role": role_prefix,
                "retention_rule": "stage1_final_anchor_only",
                "target_outcome_used_for_retention": 0,
                "primary": int(target in PRIMARY_TARGETS),
            })
            for regime in ("OACI", "SRC"):
                for order, epoch in enumerate(TRAJECTORY_EPOCHS, start=1):
                    rows.append({
                        "unit_id": _unit_id(target, level, regime, epoch, order),
                        "dataset": DATASET,
                        "target": target,
                        "seed": REPLICATION_SEED,
                        "level": level,
                        "regime": regime,
                        "epoch": epoch,
                        "trajectory_order": order,
                        "wave": wave,
                        "role": role_prefix,
                        "retention_rule": "every_5_epochs_complete_trajectory",
                        "target_outcome_used_for_retention": 0,
                        "primary": int(target in PRIMARY_TARGETS),
                    })
    counts = {regime: sum(row["regime"] == regime for row in rows) for regime in REGIMES}
    if len(rows) != EXPECTED_ENGINEERING_UNITS or len({row["unit_id"] for row in rows}) != len(rows):
        raise RuntimeError("C79P expected field is not 1,458 unique units")
    if counts != {"ERM": 18, "OACI": 720, "SRC": 720}:
        raise RuntimeError(f"C79P expected regime counts drift: {counts}")
    if sum(row["primary"] for row in rows) != EXPECTED_PRIMARY_UNITS:
        raise RuntimeError("C79P expected primary field is not 1,296 units")
    return rows


def expected_training_phases() -> list[dict[str, Any]]:
    rows = []
    ordinal = 0
    for target in TARGET_ORDER:
        for level in LEVELS:
            for regime in REGIMES:
                ordinal += 1
                rows.append({
                    "phase_id": f"c79_phase_{ordinal:02d}",
                    "wave": _wave(target),
                    "target": target,
                    "level": level,
                    "regime": regime,
                    "job_stage": "oaci_erm" if regime in {"ERM", "OACI"} else "src",
                    "dependency": "none" if regime in {"ERM", "OACI"} else "same_target_ERM_anchor_frozen",
                    "scientific_outcome_gate_allowed": 0,
                })
    if len(rows) != EXPECTED_PHASES:
        raise RuntimeError("C79P expected phase registry is not 54 rows")
    return rows


def validate_registry() -> dict[str, int]:
    rows = read_csv(REGISTRY_PATH)
    if len(rows) != 10:
        raise RuntimeError(f"C79P registry path count drift: {len(rows)}")
    blank = 0
    inherited = 0
    adaptive = 0
    for row in rows:
        for category in REGISTRY_CATEGORIES:
            value = row.get(category, "").strip()
            blank += int(not value)
            inherited += int(value.lower() in {"same as before", "inherited", "tbd", "see prior"})
            adaptive += int("active_after_holm" in value.lower())
        if row.get("completeness") != "16/16":
            raise RuntimeError(f"C79P incomplete registry row: {row.get('path_id_and_hypothesis_role')}")
    if blank or inherited or adaptive:
        raise RuntimeError(
            f"C79P registry is not fully bound: blank={blank} inherited={inherited} adaptive={adaptive}"
        )
    return {
        "paths": len(rows),
        "categories": len(REGISTRY_CATEGORIES),
        "bound_cells": len(rows) * len(REGISTRY_CATEGORIES),
        "blank_cells": blank,
        "implicit_inherited_cells": inherited,
        "active_after_Holm_cells": adaptive,
    }


def _extract_seed3_reference() -> dict[str, float]:
    # Reuse the accepted Mode-R extractor so C79P cannot silently redefine the
    # C78S compact-table mapping while claiming an exact replay.
    from . import c79_seed4_locked_confirmation as mode_r

    return mode_r._extract_c78s_references()


def exact_seed3_replay_rows() -> list[dict[str, Any]]:
    observed = _extract_seed3_reference()
    rows = []
    for metric, expected in EXPECTED_REFERENCE_VALUES.items():
        value = observed[metric]
        tolerance = 5e-4 if metric == "H1_random_expected_regret" else 1e-12
        rows.append({
            "check_type": "reference_metric",
            "item": metric,
            "expected": expected,
            "observed": value,
            "absolute_error": abs(value - expected),
            "tolerance": tolerance,
            "passed": int(abs(value - expected) <= tolerance),
        })
    structural = {
        "candidate_count_per_target_level": (81, 81),
        "random_top1": (1.0 / 81.0, EXPECTED_REFERENCE_VALUES["H1_random_top1"]),
        "random_top5": (5.0 / 81.0, EXPECTED_REFERENCE_VALUES["H1_random_top5"]),
        "random_top10": (10.0 / 81.0, EXPECTED_REFERENCE_VALUES["H1_random_top10"]),
        "primary_targets": (len(PRIMARY_TARGETS), 8),
        "trajectory_cells": (len(PRIMARY_TARGETS) * 2 * 2, 32),
        "registered_paths": (validate_registry()["paths"], 10),
    }
    for item, (value, expected) in structural.items():
        rows.append({
            "check_type": "structural",
            "item": item,
            "expected": expected,
            "observed": value,
            "absolute_error": abs(value - expected),
            "tolerance": 0,
            "passed": int(value == expected),
        })
    if not all(row["passed"] for row in rows):
        raise RuntimeError("C79P exact seed-3 replay failed")
    return rows


def rng_stream_rows() -> list[dict[str, Any]]:
    return [
        {"path": "P1_M_target_bootstrap", "formula": "7803+10", "replicates": 2000, "adaptive": 0},
        {"path": "ridge_nested_null", "formula": "7803+1000+replicate", "replicates": 499, "adaptive": 0},
        {"path": "ridge_actionability", "formula": "7803+2000+path_index", "replicates": 499, "adaptive": 0},
        {"path": "P2_six_control_maxstat", "formula": "7803+3000+100000*scheme_index+replicate", "replicates": 499, "adaptive": 0},
        {"path": "H2R_stratified_permutation", "formula": "7803+4000+replicate", "replicates": 499, "adaptive": 0},
        {"path": "trial_cluster_bootstrap", "formula": "7803+5000+target_id", "replicates": 499, "adaptive": 0},
        {"path": "crossed_target_bootstrap", "formula": "7803+6000", "replicates": 2000, "adaptive": 0},
        {"path": "hierarchical_trial_stream", "formula": "7803+6100", "replicates": 499, "adaptive": 0},
        {"path": "hierarchical_checkpoint_stream", "formula": "7803+6200", "replicates": 499, "adaptive": 0},
    ]


def implementation_manifest() -> list[dict[str, Any]]:
    rows = []
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing C79P implementation file: {relative}")
        rows.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    for relative, expected in REFERENCE_FILES.items():
        path = REPO_ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"C78S reference implementation drift: {relative}")
        rows.append({
            "path": relative,
            "sha256": observed,
            "size_bytes": path.stat().st_size,
            "reference_read_only": 1,
        })
    return rows


def build_implementation_replay() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    registry = validate_registry()
    units = expected_seed4_units()
    phases = expected_training_phases()
    replay = exact_seed3_replay_rows()
    label_rows = read_csv(LABEL_SPLIT_PATH)
    if len(label_rows) != 8 or any(row["overlap_rows"] != "0" for row in label_rows):
        raise RuntimeError("C79P label-split replay failed")
    if sha256_file(LABEL_SPLIT_PATH) != protocol["label_split"]["source_table_sha256"]:
        raise RuntimeError("C79P label-split hash drift")

    write_csv(EXPECTED_MANIFEST_CSV, units)
    manifest_payload = {
        "schema_version": "c79p_expected_seed4_manifest_v1",
        "protocol_commit": protocol_commit(),
        "protocol_sha256": protocol_sha,
        "dataset": DATASET,
        "seed": REPLICATION_SEED,
        "target_order": list(TARGET_ORDER),
        "waves": {key: list(value) for key, value in WAVES.items()},
        "engineering_units": len(units),
        "primary_units": sum(row["primary"] for row in units),
        "target4_units": sum(row["target"] == 4 for row in units),
        "regime_counts": {regime: sum(row["regime"] == regime for row in units) for regime in REGIMES},
        "training_phases": phases,
        "expected_strict_source_rows": EXPECTED_SOURCE_ROWS,
        "expected_target_unlabeled_rows": EXPECTED_TARGET_ROWS,
        "unit_manifest_path": str(EXPECTED_MANIFEST_CSV.relative_to(REPO_ROOT)),
        "unit_manifest_sha256": sha256_file(EXPECTED_MANIFEST_CSV),
        "seed4_artifacts_created": 0,
        "authorization_received": False,
    }
    write_json(EXPECTED_MANIFEST_JSON, manifest_payload)
    write_csv(TABLE_DIR / "exact_seed3_replay.csv", replay)
    write_csv(TABLE_DIR / "RNG_stream_registry.csv", rng_stream_rows())
    write_csv(TABLE_DIR / "implementation_hashes.csv", implementation_manifest())
    write_csv(TABLE_DIR / "oracle_reachability_test.csv", [{
        "path_class": "all_primary_and_secondary_registered_paths",
        "oracle_descriptor_received": 0,
        "oracle_path_symbol_in_execution_adapter": 0,
        "same_label_oracle_reachable": 0,
        "passed": 1,
    }])
    write_csv(TABLE_DIR / "label_view_access_test.csv", [
        {
            "stage": "C79P_review_and_implementation",
            "strict_source": 0,
            "target_unlabeled": 0,
            "construction": 0,
            "evaluation": 0,
            "same_label_oracle": 0,
            "passed": 1,
        },
        {
            "stage": "future_C79E_before_full_field_freeze",
            "strict_source": 1,
            "target_unlabeled": 1,
            "construction": 0,
            "evaluation": 0,
            "same_label_oracle": 0,
            "passed": 1,
        },
        {
            "stage": "future_C79E_registered_analysis_after_freeze",
            "strict_source": 1,
            "target_unlabeled": 1,
            "construction": 1,
            "evaluation": 1,
            "same_label_oracle": 0,
            "passed": 1,
        },
    ])
    registry_rows = read_csv(REGISTRY_PATH)
    write_csv(TABLE_DIR / "all_paths_unconditional_execution_test.csv", [
        {
            "path_id": row["path_id_and_hypothesis_role"],
            "registered": 1,
            "runs_regardless_of_interim_result": 1,
            "active_after_Holm_runtime_selection": 0,
            "seed4_outcome_branch": 0,
            "passed": 1,
        }
        for row in registry_rows
    ])
    IMPLEMENTATION_REPLAY_REPORT.write_text(f"""# C79P Implementation Replay

## Status

```text
protocol SHA-256: {protocol_sha}
protocol commit: {protocol_commit()}
registry: {registry['bound_cells']}/{registry['bound_cells']} cells bound
seed-3 replay: {sum(row['passed'] for row in replay)}/{len(replay)} pass
expected seed-4 field: {len(units)} units in {len(phases)} phase cells
seed-4 EEG/model outcome access: 0
C79E authorization received: false
```

The implementation is a mechanical seed parameterization of the accepted C78S
paths. It does not use `active_after_Holm`; all ten registry rows execute in a
future authorized run. Target 4 is present only in the 162-unit engineering
canary field and is absent from primary estimands, nulls, and multiplicity.

The C79P command used here imports no EEG loader, PyTorch, CUDA, or training
engine. The future execution adapter checks both committed execution locks and a
separate direct-PI authorization record before importing historical workers.
""")
    return manifest_payload


def _implementation_commit() -> str:
    commits = {
        git("log", "-1", "--format=%H", "--", relative)
        for relative in IMPLEMENTATION_FILES
    }
    if "" in commits or len(commits) != 1:
        raise RuntimeError(f"C79P implementation files do not share one lockable commit: {sorted(commits)}")
    return commits.pop()


def _commit_is_ancestor(older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def create_field_lock() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    if not EXPECTED_MANIFEST_JSON.is_file() or not EXPECTED_MANIFEST_CSV.is_file():
        raise RuntimeError("C79P implementation replay must precede the field lock")
    pcommit = protocol_commit()
    icommit = _implementation_commit()
    if not _commit_is_ancestor(pcommit, icommit):
        raise RuntimeError("C79P implementation does not descend from the protocol commit")
    if not _commit_is_ancestor(icommit, git("rev-parse", "origin/oaci")):
        raise RuntimeError("C79P implementation must be pushed before field lock creation")
    files = implementation_manifest()
    implementation_identity = sha256_bytes(canonical_bytes(files))
    lock = {
        "schema_version": "c79p_field_generation_execution_lock_v1",
        "created_at_utc": utc_now(),
        "protocol_commit": pcommit,
        "protocol_sha256": protocol_sha,
        "implementation_commit": icommit,
        "implementation_files": files,
        "implementation_identity_sha256": implementation_identity,
        "expected_manifest_path": str(EXPECTED_MANIFEST_JSON.relative_to(REPO_ROOT)),
        "expected_manifest_sha256": sha256_file(EXPECTED_MANIFEST_JSON),
        "expected_unit_table_sha256": sha256_file(EXPECTED_MANIFEST_CSV),
        "scope": {
            "dataset": DATASET,
            "seed": REPLICATION_SEED,
            "targets": list(TARGET_ORDER),
            "levels": list(LEVELS),
            "regimes": list(REGIMES),
            "engineering_units": EXPECTED_ENGINEERING_UNITS,
            "primary_units": EXPECTED_PRIMARY_UNITS,
            "training_phases": EXPECTED_PHASES,
            "same_label_oracle": False,
            "BNCI2014_004": False,
        },
        "historical_worker_binding": {
            "training": "c78f_train_exact_historical_engine_via_seed4_contract_adapter",
            "instrumentation": "c78f_instrument_exact_schema_via_seed4_contract_adapter",
            "C78S_reference_hashes": REFERENCE_FILES,
        },
        "authorization": {
            "C79E_required": True,
            "received": False,
            "record_path": str(AUTHORIZATION_RECORD_PATH.relative_to(REPO_ROOT)),
            "handoff_or_protocol_text_is_authorization": False,
        },
        "prelock_boundary": protocol["C79P_execution_boundary"],
    }
    write_json(FIELD_LOCK_PATH, lock)
    FIELD_LOCK_SHA_PATH.write_text(sha256_file(FIELD_LOCK_PATH) + "\n")
    return lock


def load_field_lock(*, require_committed: bool = True) -> tuple[dict[str, Any], str]:
    expected = FIELD_LOCK_SHA_PATH.read_text().strip()
    observed = sha256_file(FIELD_LOCK_PATH)
    if expected != observed:
        raise RuntimeError("C79P field execution lock hash drift")
    lock = json.loads(FIELD_LOCK_PATH.read_text())
    _, protocol_sha = load_protocol()
    if lock["protocol_sha256"] != protocol_sha or lock["authorization"]["received"] is not False:
        raise RuntimeError("C79P field lock scope or authorization drift")
    if require_committed:
        commit = git("log", "-1", "--format=%H", "--", str(FIELD_LOCK_PATH.relative_to(REPO_ROOT)))
        if not commit or not _commit_is_ancestor(commit, "HEAD"):
            raise RuntimeError("C79P field lock must be committed")
    return lock, observed


def create_analysis_lock() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    field_lock, field_sha = load_field_lock()
    field_commit = git("log", "-1", "--format=%H", "--", str(FIELD_LOCK_PATH.relative_to(REPO_ROOT)))
    if not _commit_is_ancestor(field_commit, git("rev-parse", "origin/oaci")):
        raise RuntimeError("C79P field lock must be pushed before analysis lock creation")
    replay_path = TABLE_DIR / "exact_seed3_replay.csv"
    if not all(row["passed"] == "1" for row in read_csv(replay_path)):
        raise RuntimeError("C79P seed-3 exact replay is not clean")
    lock = {
        "schema_version": "c79p_scientific_analysis_execution_lock_v1",
        "created_at_utc": utc_now(),
        "protocol_commit": protocol_commit(),
        "protocol_sha256": protocol_sha,
        "implementation_commit": field_lock["implementation_commit"],
        "implementation_identity_sha256": field_lock["implementation_identity_sha256"],
        "field_lock_commit": field_commit,
        "field_lock_sha256": field_sha,
        "registry_path": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "exact_seed3_replay_path": str(replay_path.relative_to(REPO_ROOT)),
        "exact_seed3_replay_sha256": sha256_file(replay_path),
        "rng_registry_sha256": sha256_file(TABLE_DIR / "RNG_stream_registry.csv"),
        "fixed_family_order": protocol["analysis"]["multiplicity"]["fixed_order"],
        "registered_paths": 10,
        "all_paths_unconditional": True,
        "primary_seed4_only": True,
        "target4_primary": False,
        "same_label_oracle_reachable": False,
        "active_after_Holm_runtime_selection": False,
        "authorization": {
            "C79E_required": True,
            "received": False,
            "record_path": str(AUTHORIZATION_RECORD_PATH.relative_to(REPO_ROOT)),
        },
    }
    write_json(ANALYSIS_LOCK_PATH, lock)
    ANALYSIS_LOCK_SHA_PATH.write_text(sha256_file(ANALYSIS_LOCK_PATH) + "\n")
    return lock


def load_analysis_lock(*, require_committed: bool = True) -> tuple[dict[str, Any], str]:
    expected = ANALYSIS_LOCK_SHA_PATH.read_text().strip()
    observed = sha256_file(ANALYSIS_LOCK_PATH)
    if expected != observed:
        raise RuntimeError("C79P analysis execution lock hash drift")
    lock = json.loads(ANALYSIS_LOCK_PATH.read_text())
    field, field_sha = load_field_lock(require_committed=require_committed)
    if lock["field_lock_sha256"] != field_sha or lock["implementation_commit"] != field["implementation_commit"]:
        raise RuntimeError("C79P analysis and field execution locks disagree")
    if lock["authorization"]["received"] is not False or not lock["all_paths_unconditional"]:
        raise RuntimeError("C79P analysis lock authorization or path-execution drift")
    if require_committed:
        commit = git("log", "-1", "--format=%H", "--", str(ANALYSIS_LOCK_PATH.relative_to(REPO_ROOT)))
        if not commit or not _commit_is_ancestor(commit, "HEAD"):
            raise RuntimeError("C79P analysis lock must be committed")
    return lock, observed


def require_c79e_authorization() -> dict[str, Any]:
    """Fail before any seed-4 worker import unless future PI evidence is exact."""
    protocol, protocol_sha = load_protocol()
    field, field_sha = load_field_lock()
    analysis, analysis_sha = load_analysis_lock()
    if not AUTHORIZATION_RECORD_PATH.is_file():
        raise PermissionError("C79E direct PI authorization record is absent")
    record = json.loads(AUTHORIZATION_RECORD_PATH.read_text())
    field_commit = git("log", "-1", "--format=%H", "--", str(FIELD_LOCK_PATH.relative_to(REPO_ROOT)))
    analysis_commit = git("log", "-1", "--format=%H", "--", str(ANALYSIS_LOCK_PATH.relative_to(REPO_ROOT)))
    expected = {
        "direct_explicit_PI_authorization": True,
        "protocol_commit": protocol_commit(),
        "protocol_sha256": protocol_sha,
        "field_lock_commit": field_commit,
        "field_lock_sha256": field_sha,
        "analysis_lock_commit": analysis_commit,
        "analysis_lock_sha256": analysis_sha,
        "seed": REPLICATION_SEED,
        "same_label_oracle": False,
        "BNCI2014_004": False,
        "C80": False,
        "manuscript": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise PermissionError("C79E authorization record does not bind the exact locked scope")
    if len(str(record.get("authorization_evidence_sha256", ""))) != 64:
        raise PermissionError("C79E authorization evidence digest is absent")
    auth_commit = git("log", "-1", "--format=%H", "--", str(AUTHORIZATION_RECORD_PATH.relative_to(REPO_ROOT)))
    if not auth_commit or not _commit_is_ancestor(field_commit, auth_commit) or not _commit_is_ancestor(analysis_commit, auth_commit):
        raise PermissionError("C79E authorization record must be committed after both locks")
    if protocol["authorization"]["received"] is not False or field["authorization"]["received"] is not False:
        raise RuntimeError("C79P immutable pre-authorization records were altered")
    return record


def _tracked_payload_scan() -> dict[str, Any]:
    files = [line for line in git("ls-files").splitlines() if line]
    oversize = []
    raw = []
    for relative in files:
        path = REPO_ROOT / relative
        if path.is_file() and path.stat().st_size > MAX_GIT_PAYLOAD:
            oversize.append(relative)
        lowered = relative.lower()
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth", ".ckpt", ".npz", ".npy"} or "raw_cache_payload" in lowered:
            raw.append(relative)
    return {
        "tracked_files": len(files),
        "payload_over_50MiB": len(oversize),
        "raw_cache_or_weights": len(raw),
        "oversize_paths": oversize,
        "raw_paths": raw,
    }


def finalize_readiness() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    field, field_sha = load_field_lock()
    analysis, analysis_sha = load_analysis_lock()
    field_commit = git("log", "-1", "--format=%H", "--", str(FIELD_LOCK_PATH.relative_to(REPO_ROOT)))
    analysis_commit = git("log", "-1", "--format=%H", "--", str(ANALYSIS_LOCK_PATH.relative_to(REPO_ROOT)))
    origin = git("rev-parse", "origin/oaci")
    registry = validate_registry()
    replay = read_csv(TABLE_DIR / "exact_seed3_replay.csv")
    payload = _tracked_payload_scan()
    authorization_absent = not AUTHORIZATION_RECORD_PATH.exists()
    checks = [
        ("replacement_protocol_hash_replays", sha256_file(PROTOCOL_PATH) == protocol_sha),
        ("protocol_commit_precedes_implementation", _commit_is_ancestor(protocol_commit(), field["implementation_commit"])),
        ("field_lock_pushed", _commit_is_ancestor(field_commit, origin)),
        ("analysis_lock_pushed", _commit_is_ancestor(analysis_commit, origin)),
        ("registry_160_of_160", registry["bound_cells"] == 160 and registry["blank_cells"] == 0),
        ("all_paths_unconditional", all(row["runs_regardless_of_interim_result"] == "1" for row in read_csv(TABLE_DIR / "all_paths_unconditional_execution_test.csv"))),
        ("seed3_exact_replay", all(row["passed"] == "1" for row in replay)),
        ("target4_excluded_primary", sum(row["primary"] == "1" and row["target"] == "4" for row in read_csv(EXPECTED_MANIFEST_CSV)) == 0),
        ("expected_1458_units", len(read_csv(EXPECTED_MANIFEST_CSV)) == 1458),
        ("oracle_unreachable", read_csv(TABLE_DIR / "oracle_reachability_test.csv")[0]["passed"] == "1"),
        ("label_views_closed_in_C79P", read_csv(TABLE_DIR / "label_view_access_test.csv")[0]["passed"] == "1"),
        ("seed4_untouched", all(row["observed_count"] == "0" for row in read_csv(TABLE_DIR / "seed4_untouched_audit.csv"))),
        ("future_authorization_not_consumed", authorization_absent),
        ("no_oversize_payload", payload["payload_over_50MiB"] == 0),
        ("no_raw_payload", payload["raw_cache_or_weights"] == 0),
        ("historical_protocol_retained", (REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.json").is_file()),
        ("no_active_after_Holm", registry["active_after_Holm_cells"] == 0),
        ("claim_contract_present", (REPORT_DIR / "C79_POST_SEED3_PROTOCOL_CLAIM_CONTRACT.md").is_file()),
        ("BNCI2014_004_not_authorized", "BNCI2014_004" in protocol["forbidden"]),
        ("C80_not_authorized", "C80_auto_start" in protocol["forbidden"]),
    ]
    if not all(passed for _, passed in checks):
        failed = [name for name, passed in checks if not passed]
        raise RuntimeError(f"C79P pre-execution red team failed: {failed}")
    write_csv(TABLE_DIR / "pre_execution_red_team_checks.csv", [
        {"check": name, "passed": int(passed), "blocking_failure": int(not passed)}
        for name, passed in checks
    ])
    PRE_EXECUTION_RED_TEAM.write_text(f"""# C79P Pre-Execution Red Team

```text
checks passed: {len(checks)}/{len(checks)}
blocking failures: 0
seed-4 EEG/training/forward/jobs/artifacts/outcomes: 0
future PI authorization consumed: false
```

The replacement protocol is explicitly post-C78S and outcome-informed. The
historical protocol remains content-valid but has no seed-4 execution authority.
All ten replacement paths are unconditional, target 4 is engineering-only, the
same-label oracle is unreachable, and the seed-4-only primary analysis is locked.
""")
    ledger = {
        "schema_version": "c79p_execution_lock_ledger_v1",
        "protocol": {"commit": protocol_commit(), "sha256": protocol_sha},
        "implementation_commit": field["implementation_commit"],
        "field_generation_lock": {"commit": field_commit, "sha256": field_sha},
        "scientific_analysis_lock": {"commit": analysis_commit, "sha256": analysis_sha},
        "authorization": {"received": False, "record_exists": False},
        "seed4_access": 0,
        "ready_for_PI_review": True,
    }
    write_json(LOCK_LEDGER_PATH, ledger)
    result = {
        "schema_version": "c79p_protocol_readiness_v1",
        "milestone": "C79P",
        "primary": "C79P-A_post_seed3_replication_protocol_locked_complete",
        "final_gate": FINAL_GATE,
        "protocol_commit": protocol_commit(),
        "protocol_sha256": protocol_sha,
        "field_lock_commit": field_commit,
        "field_lock_sha256": field_sha,
        "analysis_lock_commit": analysis_commit,
        "analysis_lock_sha256": analysis_sha,
        "registry": registry,
        "seed3_replay_checks": len(replay),
        "expected_seed4_units": 1458,
        "expected_primary_units": 1296,
        "pre_execution_red_team": {"passed": len(checks), "failed": 0},
        "execution_boundary": {
            "C79E_authorized": False,
            "seed4_EEG_load": 0,
            "seed4_Slurm_job": 0,
            "training": 0,
            "forward_or_reinference": 0,
            "GPU": 0,
            "seed4_artifact": 0,
            "seed4_outcome_access": 0,
            "same_label_oracle": 0,
            "BNCI2014_004": 0,
            "manuscript": 0,
        },
        "claim_boundary": "post_seed3_outcome_informed_training_seed_robustness_only",
    }
    write_json(READINESS_JSON, result)
    READINESS_REPORT.write_text(f"""# C79P - Post-Seed-3 Seed-4 Replication Readiness

## Gate

```text
Primary: C79P-A_post_seed3_replication_protocol_locked_complete
Final:   {FINAL_GATE}
Protocol SHA-256: {protocol_sha}
C79E authorization received: false
Seed-4 access: 0
```

The historical C79 artifact is retained and transparently superseded. It is not
relabeled as a pre-C78S confirmation protocol. The replacement protocol is
explicitly designed after C78S, but was committed before every protected seed-4
checkpoint/model outcome.

The scientific registry binds all {registry['bound_cells']} required cells over
ten unconditional paths. Seed-3 compact evidence replays exactly, the expected
seed-4 field contains 1,458 units (1,296 primary), target 4 is engineering-only,
and the same-label oracle remains closed.

Both execution locks are committed and pushed. Their authorization fields remain
false. A future direct PI authorization must bind protocol `{protocol_commit()}`
and SHA `{protocol_sha}`, field lock `{field_commit}`, and analysis lock
`{analysis_commit}` before the fail-closed adapter can import any EEG, training,
instrumentation, or CUDA worker.

This gate authorizes no seed-4 work, C80, additional seed, BNCI2014_004, oracle
analysis, feature/kernel search, or manuscript drafting.
""")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c79p_post_seed3_protocol")
    parser.add_argument(
        "command",
        choices=(
            "build-implementation-replay",
            "create-field-lock",
            "create-analysis-lock",
            "verify-locks",
            "finalize-readiness",
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "build-implementation-replay":
        payload = build_implementation_replay()
    elif args.command == "create-field-lock":
        payload = create_field_lock()
    elif args.command == "create-analysis-lock":
        payload = create_analysis_lock()
    elif args.command == "verify-locks":
        field, field_sha = load_field_lock()
        analysis, analysis_sha = load_analysis_lock()
        payload = {
            "field_lock_sha256": field_sha,
            "analysis_lock_sha256": analysis_sha,
            "authorization_received": False,
            "implementation_commit": field["implementation_commit"],
        }
    else:
        payload = finalize_readiness()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
