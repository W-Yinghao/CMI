"""C78S prospective analysis lock, view routing, and execution governance."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


MILESTONE = "C78S"
PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
TARGET4_CANARY = 4
SEED = 3
LEVELS = (0, 1)
REGIMES = ("ERM", "OACI", "SRC")
PRIMARY_UNITS = 1296
FULL_FIELD_UNITS = 1458
NULL_REPLICATES = 499
BOOTSTRAP_REPLICATES = 2000
TRIAL_BOOTSTRAP_REPLICATES = 499
RNG_SEED = 7803
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
KERNEL_FAMILIES = ("rbf", "laplacian")
BANDWIDTH_FACTORS = (0.5, 1.0, 2.0)
ASSOCIATION_STATISTICS = ("normalized_alignment", "centered_hsic")
PRIMARY_GEOMETRY_EPSILON = 0.05
PREFIX_SIZES = (5, 10, 20, 40)

REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c78s_tables"
PROTOCOL_PATH = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS_PROTOCOL.sha256"
ROUTE_PATH = REPORT_DIR / "C78S_PRIMARY_VIEW_ROUTE.json"
ROUTE_SHA_PATH = REPORT_DIR / "C78S_PRIMARY_VIEW_ROUTE.sha256"
LOCK_PATH = REPORT_DIR / "C78S_AUTHORIZED_ANALYSIS_EXECUTION_LOCK.json"
LOCK_SHA_PATH = REPORT_DIR / "C78S_AUTHORIZED_ANALYSIS_EXECUTION_LOCK.sha256"
TIMING_PATH = REPORT_DIR / "C78S_PROTOCOL_TIMING_AUDIT.md"

C78F_RESULT = REPORT_DIR / "C78F_FULL_SEED3_FIELD.json"
C78F_PROTOCOL = REPORT_DIR / "C78F_FULL_SEED3_FIELD_PROTOCOL.json"
C78F_PROTOCOL_SHA = REPORT_DIR / "C78F_FULL_SEED3_FIELD_PROTOCOL.sha256"
C78F_FULL_UNITS = REPORT_DIR / "c78f_tables/full_unit_manifest.csv"
C78F_PHYSICAL_VIEWS = REPORT_DIR / "c78f_tables/physical_view_manifest.csv"
C78F_RISK = REPORT_DIR / "c78f_tables/risk_register.csv"
C78F_SEED4 = REPORT_DIR / "c78f_tables/seed4_protection_audit.csv"
C78F_EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c78f-full-seed3")

AUTHORIZATION_MODE = "direct_explicit_user_authorization"
AUTHORIZATION_EVIDENCE = "我明确授权C78S"
AUTHORIZATION_EVIDENCE_SHA256 = hashlib.sha256(AUTHORIZATION_EVIDENCE.encode()).hexdigest()

IMPLEMENTATION_FILES = (
    "oaci/conditioned_ceiling_coverage/c78s_protocol.py",
    "oaci/conditioned_ceiling_coverage/c78s_data.py",
    "oaci/conditioned_ceiling_coverage/c78s_modeling.py",
    "oaci/conditioned_ceiling_coverage/c78s_seed3_scientific_analysis.py",
    "oaci/conditioned_ceiling_coverage/c78s_red_team.py",
    "oaci/tests/test_c78s_seed3_scientific_analysis.py",
    "oaci/slurm_c78s_analysis.sh",
    "oaci/slurm_c78s_red_team.sh",
    "oaci/slurm_c78s_regression.sh",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty C78S table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = PROTOCOL_SHA_PATH.read_text().strip()
    observed = sha256_file(PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError(f"C78S protocol hash drift: {observed} != {expected}")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get("milestone") != MILESTONE:
        raise RuntimeError("C78S protocol milestone drift")
    if tuple(protocol["data_roles"]["primary_targets"]) != PRIMARY_TARGETS:
        raise RuntimeError("C78S primary target registry drift")
    if protocol["data_roles"]["target4_canary"] != "descriptive_only_excluded_from_all_primary_tests":
        raise RuntimeError("C78S target-4 exclusion drift")
    return protocol, observed


def _expected_external_root() -> Path:
    protocol_sha = json.loads(C78F_RESULT.read_text())["protocol_sha256"]
    implementation = json.loads(C78F_RESULT.read_text())["execution_lock_commit"]
    # The authoritative implementation identity is embedded in the external
    # path and replayed from the committed physical-view table below.
    candidates = sorted(C78F_EXTERNAL_ROOT.glob(f"protocol_{protocol_sha[:16]}/implementation_*"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"C78S expected one C78F external implementation root for {implementation}: {candidates}"
        )
    return candidates[0]


def build_primary_route() -> dict[str, Any]:
    """Create the only descriptor presented to the primary C78S process.

    This metadata-only step may inspect the committed C78F view ledger.  It
    deliberately strips the same-label oracle descriptor before the primary
    process is locked or launched.
    """

    protocol, protocol_sha = load_protocol()
    rows = read_csv(C78F_PHYSICAL_VIEWS)
    allowed_names = {
        "strict_source_input",
        "target_unlabeled_input",
        "target_construction_view",
        "target_evaluation_view",
    }
    selected = [
        row for row in rows
        if int(row["target"]) in PRIMARY_TARGETS and row["view_name"] in allowed_names
    ]
    if len(selected) != len(PRIMARY_TARGETS) * len(allowed_names):
        raise RuntimeError("C78S primary route does not contain exactly four views per primary target")
    if any(int(row["target"]) == TARGET4_CANARY for row in selected):
        raise RuntimeError("C78S primary route contains target 4")
    by_target: dict[str, dict[str, Any]] = {}
    external_root = _expected_external_root()
    for target in PRIMARY_TARGETS:
        target_rows = {row["view_name"]: row for row in selected if int(row["target"]) == target}
        if set(target_rows) != allowed_names:
            raise RuntimeError(f"C78S route view mismatch for target {target}")
        construction = target_rows["target_construction_view"]
        evaluation = target_rows["target_evaluation_view"]
        if construction["uses_evaluation_labels"] != "0" or evaluation["uses_evaluation_labels"] != "1":
            raise RuntimeError(f"C78S label-role drift for target {target}")
        by_target[str(target)] = {
            "instrumentation_manifest": str(
                external_root / "targets" / f"target-{target:03d}" / "instrumentation" / "INSTRUMENTATION_COMPLETE.json"
            ),
            "strict_source_input": {
                key: target_rows["strict_source_input"][key]
                for key in ("path", "sha256", "rows", "allowed_columns")
            },
            "target_unlabeled_input": {
                key: target_rows["target_unlabeled_input"][key]
                for key in ("path", "sha256", "rows", "allowed_columns", "forbidden_columns")
            },
            "target_construction_view": {
                key: construction[key] for key in ("path", "sha256", "rows", "allowed_columns")
            },
            "target_evaluation_view": {
                key: evaluation[key] for key in ("path", "sha256", "rows", "allowed_columns")
            },
        }
    route = {
        "schema_version": "c78s_primary_view_route_v1",
        "created_at_utc": utc_now(),
        "C78S_protocol_sha256": protocol_sha,
        "C78F_result_commit": "51022f4",
        "primary_targets": list(PRIMARY_TARGETS),
        "target4_included": False,
        "same_label_oracle_descriptor_included": False,
        "trial_id_role": "join_split_and_dependence_cluster_only_never_predictor",
        "row_order_role": "alignment_only_never_predictor",
        "views": by_target,
        "scope": {
            "training": False,
            "forward": False,
            "reinference": False,
            "GPU": False,
            "seed4": False,
            "BNCI2014_004": False,
            "manuscript": False,
        },
    }
    serialized = canonical_bytes(route).decode()
    if "same_label_oracle_view" in serialized or "/oracle/" in serialized:
        raise RuntimeError("C78S primary route leaked an oracle descriptor")
    write_json(ROUTE_PATH, route)
    ROUTE_SHA_PATH.write_text(sha256_file(ROUTE_PATH) + "\n")
    return route


def load_primary_route() -> tuple[dict[str, Any], str]:
    expected = ROUTE_SHA_PATH.read_text().strip()
    observed = sha256_file(ROUTE_PATH)
    if observed != expected:
        raise RuntimeError("C78S primary route hash drift")
    raw = ROUTE_PATH.read_text()
    if "same_label_oracle_view" in raw or "/oracle/" in raw:
        raise RuntimeError("C78S primary route contains an oracle descriptor")
    route = json.loads(raw)
    if tuple(route["primary_targets"]) != PRIMARY_TARGETS or route["target4_included"]:
        raise RuntimeError("C78S primary route target registry drift")
    return route, observed


def implementation_registry() -> list[dict[str, str]]:
    rows = []
    for path in IMPLEMENTATION_FILES:
        file_path = Path(path)
        if not file_path.is_file():
            raise RuntimeError(f"C78S implementation file missing: {path}")
        rows.append({"path": path, "sha256": sha256_file(file_path)})
    return rows


def create_execution_lock() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    route, route_sha = load_primary_route()
    implementation = implementation_registry()
    head = git("rev-parse", "HEAD")
    for item in implementation:
        committed = git("log", "-1", "--format=%H", "--", item["path"])
        if not committed or git("merge-base", "--is-ancestor", committed, head) != "":
            raise RuntimeError(f"C78S implementation is not committed at HEAD: {item['path']}")
    route_commit = git("log", "-1", "--format=%H", "--", str(ROUTE_PATH))
    if not route_commit or git("merge-base", "--is-ancestor", route_commit, head) != "":
        raise RuntimeError("C78S primary route must be committed before execution lock")
    payload = {
        "schema_version": "c78s_authorized_analysis_execution_lock_v1",
        "created_at_utc": utc_now(),
        "protocol_sha256": protocol_sha,
        "protocol_commit": git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH)),
        "primary_route_sha256": route_sha,
        "primary_route_commit": route_commit,
        "implementation_commit": head,
        "implementation_files": implementation,
        "implementation_identity_sha256": sha256_bytes(canonical_bytes(implementation)),
        "authorization": {
            "received": True,
            "mode": AUTHORIZATION_MODE,
            "evidence_sha256": AUTHORIZATION_EVIDENCE_SHA256,
            "scope_bound": True,
        },
        "scope": {
            "primary_targets": list(PRIMARY_TARGETS),
            "target4_primary": False,
            "seed": SEED,
            "analysis_only": True,
            "training": False,
            "forward": False,
            "reinference": False,
            "GPU": False,
            "seed4": False,
            "C79": False,
            "BNCI2014_004": False,
            "manuscript": False,
            "same_label_oracle": False,
        },
        "before_lock": {
            "quarantined_label_payload_reads_by_C78S": 0,
            "scientific_outcomes_computed_by_C78S": 0,
            "seed4_access": 0,
        },
        "locked_analysis": {
            "hypotheses": [item["id"] for item in protocol["primary_hypotheses"]],
            "null_replicates": NULL_REPLICATES,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "trial_bootstrap_replicates": TRIAL_BOOTSTRAP_REPLICATES,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "kernel_families": list(KERNEL_FAMILIES),
            "bandwidth_factors": list(BANDWIDTH_FACTORS),
            "association_statistics": list(ASSOCIATION_STATISTICS),
            "geometry_epsilon": PRIMARY_GEOMETRY_EPSILON,
            "candidate_prefix_sizes": list(PREFIX_SIZES),
            "primary_cell": "target_x_level_all_81_candidates",
            "trajectory_cell": "target_x_level_x_regime_ERM_anchor_not_symmetric",
            "continuous_utility": "mean_within_target_x_level_midrank(target_bAcc,-target_NLL,-target_ECE)",
            "primary_joint_good": "all_three_oriented_within_cell_midranks_at_least_0.75",
            "primary_feature_paths": [
                "strict_source_F2_given_F0_F1",
                "target_unlabeled_F4_geometry_given_F0_F1_F3",
                "target_unlabeled_F4_full_mixed_secondary",
                "construction_F5_given_F0_F1_F3_F4_geometry",
            ],
            "nonlinear_prediction_paths": {
                "strict_source": {"kernel": "rbf", "bandwidth_factor": 1.0, "alpha": 1.0},
                "target_unlabeled": {"kernel": "laplacian", "bandwidth_factor": 1.0, "alpha": 1.0},
            },
            "same_label_oracle_stage": "not_run_in_C78S",
        },
        "route_contract": {
            "same_label_oracle_descriptor_present": route["same_label_oracle_descriptor_included"],
            "trial_id_predictor": False,
            "row_order_predictor": False,
        },
    }
    write_json(LOCK_PATH, payload)
    LOCK_SHA_PATH.write_text(sha256_file(LOCK_PATH) + "\n")
    return payload


def load_execution_lock() -> tuple[dict[str, Any], str]:
    expected = LOCK_SHA_PATH.read_text().strip()
    observed = sha256_file(LOCK_PATH)
    if observed != expected:
        raise RuntimeError("C78S execution lock hash drift")
    lock = json.loads(LOCK_PATH.read_text())
    _, protocol_sha = load_protocol()
    _, route_sha = load_primary_route()
    if lock["protocol_sha256"] != protocol_sha or lock["primary_route_sha256"] != route_sha:
        raise RuntimeError("C78S lock protocol/route mismatch")
    if lock["authorization"] != {
        "evidence_sha256": AUTHORIZATION_EVIDENCE_SHA256,
        "mode": AUTHORIZATION_MODE,
        "received": True,
        "scope_bound": True,
    }:
        raise PermissionError("C78S direct authorization evidence drift")
    lock_commit = git("log", "-1", "--format=%H", "--", str(LOCK_PATH))
    if not lock_commit or git("merge-base", "--is-ancestor", lock_commit, "HEAD") != "":
        raise RuntimeError("C78S execution lock must be committed before analysis")
    if lock["scope"]["target4_primary"] or lock["scope"]["seed4"] or lock["scope"]["same_label_oracle"]:
        raise PermissionError("C78S execution lock expanded beyond authorization")
    for item in lock["implementation_files"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C78S locked implementation drift: {item['path']}")
    return lock, observed


def feature_registry() -> list[dict[str, Any]]:
    return [
        {
            "block": "F0",
            "information_class": "metadata_plus_strict_source_endpoints",
            "dimension": 9,
            "formula": "regime_onehot[3]+level_onehot[2]+order_scaled+source_bAcc+source_NLL+source_ECE",
            "target_labels": 0,
            "predictor_trial_id": 0,
            "factorization_status": "functional_metadata_baseline",
        },
        {
            "block": "F1",
            "information_class": "strict_source_functional",
            "dimension": 25,
            "formula": "exact_C75_registered_source_logits_probabilities_calibration_block",
            "target_labels": 0,
            "predictor_trial_id": 0,
            "factorization_status": "function_level",
        },
        {
            "block": "F2",
            "information_class": "strict_source_architecture_tied",
            "dimension": 25,
            "formula": "exact_C75_registered_z_spectrum_W_geometry_alignment_block",
            "target_labels": 0,
            "predictor_trial_id": 0,
            "factorization_status": "architecture_tied_nonidentified",
        },
        {
            "block": "F3",
            "information_class": "target_unlabeled_functional",
            "dimension": 18,
            "formula": "exact_C75_registered_target_logits_probabilities_common_shift_block",
            "target_labels": 0,
            "predictor_trial_id": 0,
            "factorization_status": "function_level_target_unlabeled",
        },
        {
            "block": "F4",
            "information_class": "target_unlabeled_architecture_tied_mixed",
            "dimension": 35,
            "formula": "exact_C75_registered_target_z_W_geometry_Wz_projection_block",
            "target_labels": 0,
            "predictor_trial_id": 0,
            "factorization_status": "first20_architecture_tied_last15_functional_projection",
        },
        {
            "block": "F5",
            "information_class": "target_construction_labels_diagnostic_positive_control",
            "dimension": 15,
            "formula": "exact_C75_registered_split_label_endpoint_block_ranked_within_target_x_level",
            "target_labels": 1,
            "predictor_trial_id": 0,
            "factorization_status": "diagnostic_label_derived",
        },
    ]


def risk_registry() -> list[dict[str, Any]]:
    risks = (
        "protocol_hash_or_execution_lock_drift",
        "target4_in_primary_estimand",
        "target4_in_primary_null_pool",
        "target4_in_primary_multiplicity_family",
        "quarantined_label_read_before_execution_lock",
        "construction_evaluation_trial_overlap",
        "same_label_oracle_early_access",
        "trial_id_or_row_order_as_predictor",
        "target_label_in_unlabeled_feature",
        "outcome_adaptive_retry_or_report_scope",
        "bandwidth_selected_outside_nested_contract",
        "row_iid_inference",
        "checkpoint_units_treated_as_population_replication",
        "ERM_OACI_SRC_false_symmetry",
        "association_called_prediction",
        "prediction_called_actionability",
        "seed3_called_seed_confirmation",
        "source_or_target_unlabeled_null_called_universal_failure",
        "target_population_overclaim",
        "seed4_or_C79_access",
        "BNCI2014_004_access",
        "training_forward_reinference_or_GPU",
        "raw_cache_or_weights_in_git",
        "selector_or_checkpoint_recommendation",
        "manuscript_drafting",
    )
    return [
        {
            "risk": risk,
            "status": "LOCKED_MITIGATION_PENDING_EXECUTION_REPLAY",
            "blocking": 1,
            "mitigation": "verified_by_C78S_preflight_and_independent_result_red_team",
        }
        for risk in risks
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78s_protocol")
    parser.add_argument("command", choices=("build-route", "create-execution-lock", "verify-lock"))
    args = parser.parse_args(argv)
    if args.command == "build-route":
        print(json.dumps(build_primary_route(), indent=2, sort_keys=True))
    elif args.command == "create-execution-lock":
        print(json.dumps(create_execution_lock(), indent=2, sort_keys=True))
    else:
        lock, digest = load_execution_lock()
        print(json.dumps({"lock_sha256": digest, "scope": lock["scope"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
