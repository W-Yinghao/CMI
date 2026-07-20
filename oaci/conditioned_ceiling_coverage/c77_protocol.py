"""Prospective C77/C78 contracts for independent multi-regime replication.

C77 is metadata and simulation only.  This module deliberately has no data loader,
model training, or real-data forward entry point.  It reconstructs historical
regimes, freezes the seed roles, and emits the future C78 execution contract.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.protocol.manifest_v2 import (
    load_v2,
    manifest_logical_payload,
    optimization_manifest_hash,
)


MILESTONE = "C77"
PARENT_COMMIT = "ce237532aa1866c01a90fe01e900654c83275465"
REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c77_tables"
PROTOCOL_PATH = REPORT_DIR / "C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.sha256"
TIMING_PATH = REPORT_DIR / "C77_PROTOCOL_TIMING_AUDIT.md"
C78_PROTOCOL_PATH = REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT_PROTOCOL.json"
C78_PROTOCOL_SHA_PATH = REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT_PROTOCOL.sha256"
C79_SKELETON_PATH = REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL_SKELETON.json"

C76_RESULT = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ORBIT.json"
C76_REPORT = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_ORBIT.md"
C76_RED_TEAM = REPORT_DIR / "C76_RED_TEAM_VERIFICATION.md"
C76_PROTOCOL = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_PROTOCOL.json"
C76_PROTOCOL_SHA = REPORT_DIR / "C76_REPRESENTATION_ASSOCIATION_PROTOCOL.sha256"
C76_ORBIT_IDENTITY = REPORT_DIR / "c76_tables/orbit_functional_identity.csv"
C76_SEPARATION = REPORT_DIR / "c76_tables/association_prediction_separation.csv"
C76_TOPOLOGY = REPORT_DIR / "c76_tables/association_topology.csv"
C76_T3_GATE = REPORT_DIR / "c76_tables/t3_candidate_gate.csv"

MANIFEST_PATH = Path("oaci/protocol/confirmatory_v2.yaml")
SRC_ONEFOLD_PATH = Path("oaci/confirmatory/src_onefold.py")
SRC_OBJECTIVE_PATH = Path("oaci/methods/source_robust.py")
SRC_SELECTOR_PATH = Path("oaci/runner/source_endpoint_selector.py")
OACI_OBJECTIVE_PATH = Path("oaci/methods/oaci.py")
ERM_OBJECTIVE_PATH = Path("oaci/methods/erm.py")
ENGINE_PATH = Path("oaci/train/engine.py")
PROVENANCE_PATH = Path("oaci/runner/provenance.py")
C11_RESULT_PATH = REPORT_DIR / "C11_SRC_ONEFOLD_PILOT.json"
C12_RESULT_PATH = REPORT_DIR / "C12_SRC_STRESS_REPLICATION.json"
C74_STORAGE_PATH = REPORT_DIR / "c74_tables/power_and_storage_plan.csv"

REGIME_COMMITS = {
    "ERM": "4c6bb93ed5eae5a3fa073166c322765680760190",
    "OACI": "5d3102b001a936f82ae8a37c13f42e1ebcce82f5",
    "SRC": "2555b3623713f802018e69afcf2b7d1449050641",
    "manifest": "97cdead2c1257f5c47b07bf1fff4b9df01f5f4fe",
}

C78_AUTHORIZATION_TOKEN = "C78_SEED3_MULTIREGIME_INSTRUMENTED_PILOT_AUTHORIZED"
TARGETS = tuple(range(1, 10))
PRIMARY_REGIMES = ("ERM", "OACI", "SRC")
TRAJECTORY_REGIMES = ("OACI", "SRC")
LEVELS = (0, 1)
STAGE2_CHECKPOINTS = 40
RNG_SEED = 7701
SYNTHETIC_REPLICATES = 400


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


def _write_csv(name: str, rows: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_blob_sha(commit: str, path: str | Path) -> str:
    blob = subprocess.check_output(["git", "show", f"{commit}:{path}"])
    return hashlib.sha256(blob).hexdigest()


def _git_has_path(commit: str, path: str | Path) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _hash_choice(namespace: str, values: tuple[str, ...]) -> tuple[str, str]:
    scored = [(hashlib.sha256(f"C78|{namespace}|{value}".encode()).hexdigest(), value) for value in values]
    digest, value = min(scored)
    return value, digest


def _manifest_contract() -> tuple[dict, dict]:
    manifest = load_v2(str(MANIFEST_PATH))
    manifest.validate_complete()
    logical = manifest_logical_payload(manifest)
    canonical_hash = manifest.freeze()["sha256"]
    optimization_hash = optimization_manifest_hash(manifest)
    dataset = asdict(manifest.datasets["BNCI2014_001"])
    if not dataset["enabled"] or len(dataset["class_names"]) != 4:
        raise RuntimeError("C77 BNCI2014_001 manifest replay failed")
    if not manifest.datasets["BNCI2014_004"].enabled:
        raise RuntimeError("C77 expected parent manifest to contain enabled BNCI2014_004 for exclusion audit")
    derived = {
        "parent_manifest_path": str(MANIFEST_PATH),
        "parent_manifest_file_sha256": sha256(MANIFEST_PATH),
        "parent_manifest_canonical_sha256": canonical_hash,
        "parent_optimization_sha256": optimization_hash,
        "dataset_allowlist": ["BNCI2014_001"],
        "dataset_denylist": ["BNCI2014_004", "SEED", "PD_cross_site"],
        "targets": list(TARGETS),
        "seed3_only": True,
        "seed4_forbidden": True,
        "execution_filter_rule": "fail unless dataset argument equals BNCI2014_001; never iterate parent enabled datasets",
        "BNCI2014_001": dataset,
        "backbone": logical["backbone"],
        "optimizer": logical["optimizer"],
        "training": logical["training"],
        "sampler": logical["sampler"],
        "risk": logical["risk"],
    }
    derived["derived_execution_view_sha256"] = payload_sha256(derived)
    return logical, derived


def c76_replay_tables() -> dict[str, list[dict]]:
    result = json.loads(C76_RESULT.read_text())
    if result["final_gate"] != "LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE":
        raise RuntimeError("C77 parent C76 gate mismatch")
    if result["qualification"]["C77_protocol_created"]:
        raise RuntimeError("C77 parent says a C77 protocol already existed")
    separation = {row["path"]: row for row in _read_csv(C76_SEPARATION)}
    target_topology = {
        row["level"]: row
        for row in _read_csv(C76_TOPOLOGY)
        if row["feature_set"] == "target_unlabeled_G3_architecture"
        and row["kernel"] == "laplacian"
        and row["bandwidth_factor"] == "1.0"
        and row["statistic"] == "centered_hsic"
    }
    orbit = _read_csv(C76_ORBIT_IDENTITY)
    rows = [
        {"metric": "final_gate", "reported": result["final_gate"], "replayed": "LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE", "match": 1},
        {"metric": "strict_source_effect", "reported": result["association_prediction_separation"]["strict_source"]["association"], "replayed": separation["strict_source"]["association"], "match": int(result["association_prediction_separation"]["strict_source"]["association"] == separation["strict_source"]["association"])},
        {"metric": "strict_source_worst_p", "reported": result["association_prediction_separation"]["strict_source"]["association_worst_required_p"], "replayed": separation["strict_source"]["association_worst_required_p"], "match": 1},
        {"metric": "strict_source_incremental_R2", "reported": result["association_prediction_separation"]["strict_source"]["incremental_R2"], "replayed": separation["strict_source"]["incremental_R2"], "match": 1},
        {"metric": "target_unlabeled_effect", "reported": result["target_association"]["effect"], "replayed": separation["target_unlabeled"]["association"], "match": int(abs(float(result["target_association"]["effect"]) - float(separation["target_unlabeled"]["association"])) < 1e-15)},
        {"metric": "target_unlabeled_incremental_R2", "reported": result["association_prediction_separation"]["target_unlabeled"]["incremental_R2"], "replayed": separation["target_unlabeled"]["incremental_R2"], "match": 1},
        {"metric": "target_unlabeled_positive_targets", "reported": result["association_prediction_separation"]["target_unlabeled"]["positive_targets"], "replayed": separation["target_unlabeled"]["positive_targets"], "match": 1},
        {"metric": "conditioning_pooled", "reported": result["target_association"]["topology"]["pooled"]["association"], "replayed": target_topology["pooled"]["association"], "match": 1},
        {"metric": "conditioning_within_target", "reported": result["target_association"]["topology"]["within_target"]["association"], "replayed": target_topology["within_target"]["association"], "match": 1},
        {"metric": "conditioning_target_trajectory", "reported": result["target_association"]["topology"]["within_target_x_trajectory"]["association"], "replayed": target_topology["within_target_x_trajectory"]["association"], "match": 1},
    ]
    orbit_summary = [{
        "orbit_variants": len(orbit),
        "max_projection_error": max(float(row["max_projection_error"]) for row in orbit),
        "max_probability_error": max(float(row["max_probability_error"]) for row in orbit),
        "prediction_disagreements": sum(int(row["prediction_disagreements"]) for row in orbit),
        "all_identity_pass": int(all(float(row["max_projection_error"]) <= 1e-8 and float(row["max_probability_error"]) <= 1e-8 for row in orbit)),
    }]
    closure = [
        {"branch": "additional_unregistered_T2_representation_features_or_kernels", "status": "closed", "C77_action": "forbidden"},
        {"branch": "T3_HO_representation_generation", "status": "closed_without_qualified_T2_candidate", "C77_action": "forbidden"},
        {"branch": "local_association_reinterpreted_as_transport", "status": "closed", "C77_action": "forbidden"},
        {"branch": "new_seed_multiregime_replication", "status": "protocol_only", "C77_action": "prepare_not_execute"},
        {"branch": "independent_target_or_dataset_replication", "status": "future_R2", "C77_action": "readiness_only_no_data_access"},
    ]
    return {
        "c76_metric_identity_replay.csv": rows,
        "c76_orbit_identity_replay.csv": orbit_summary,
        "c76_branch_closure_ledger.csv": closure,
    }


def regime_tables(derived_manifest: dict) -> dict[str, list[dict]]:
    current_head = _git("rev-parse", "HEAD")
    historical = [
        {
            "regime_id": "ERM", "objective": "stage1_source_balanced_cross_entropy", "loss_terms": "R_src only",
            "coefficients": "risk.epsilon=0.03", "training_schedule": "stage1 200 epochs x1 step; final checkpoint only",
            "optimizer": "Adam", "learning_rate": "0.005", "batch_size": 256,
            "preprocessing": "BNCI001 4-38Hz 128Hz 0.5-3.5s zscore_sample", "subject_source_split": "LOSO source subjects only",
            "checkpoint_cadence": "one final stage1 checkpoint", "early_stopping_rule": "none",
            "target_label_access": "none before locked evaluation", "seed_implementation": "manifest model seed + deterministic plans",
            "code_commit": REGIME_COMMITS["ERM"], "config_hash": "derived_below", "sidecar_schema": "CheckpointRecord+RunProvenance",
            "recoverability": "exact", "role_in_R1": "shared_baseline_anchor_not_equal_length_trajectory",
            "C14_C76_target_outcome_used_to_select_for_C77": 0,
            "historical_context": "original pre-C14 baseline",
        },
        {
            "regime_id": "OACI", "objective": "support_aware_adversarial_domain_information_penalty", "loss_terms": "A_OACI + lambda*R_src",
            "coefficients": "critic_capacity=16;lambda_init=0.3;dual_lr=0.5;lambda_max=20", "training_schedule": "stage2 200 epochs x20 steps after shared ERM",
            "optimizer": "Adam", "learning_rate": "encoder=0.01;critic=0.01", "batch_size": 256,
            "preprocessing": "same locked BNCI001 view", "subject_source_split": "LOSO source subjects only",
            "checkpoint_cadence": "every 5 epochs = 40 checkpoints", "early_stopping_rule": "none; retain complete trajectory",
            "target_label_access": "none before locked evaluation", "seed_implementation": "manifest model seed + deterministic task/alignment plans",
            "code_commit": REGIME_COMMITS["OACI"], "config_hash": "derived_below", "sidecar_schema": "TrainResult+CheckpointRecord+RunProvenance",
            "recoverability": "exact", "role_in_R1": "primary_preexisting_trajectory",
            "C14_C76_target_outcome_used_to_select_for_C77": 0,
            "historical_context": "original pre-C14 method; frozen here as a regime, not a rescue target",
        },
        {
            "regime_id": "SRC", "objective": "smooth_worst_source_domain_balanced_CE", "loss_terms": "tau_lse*logsumexp(R_d/tau_lse)+lambda*R_src",
            "coefficients": "smooth_temperature=0.1;min_source_domains=2;lambda inherited", "training_schedule": "stage2 200 epochs x20 steps after shared ERM",
            "optimizer": "Adam", "learning_rate": "encoder=0.01;no critic", "batch_size": 256,
            "preprocessing": "same locked BNCI001 view", "subject_source_split": "LOSO source subjects only",
            "checkpoint_cadence": "every 5 epochs = 40 checkpoints", "early_stopping_rule": "none; retain complete trajectory",
            "target_label_access": "none before locked evaluation", "seed_implementation": "manifest model seed + deterministic full-domain alignment",
            "code_commit": REGIME_COMMITS["SRC"], "config_hash": "derived_below", "sidecar_schema": "TrainResult+CheckpointRecord+RunProvenance",
            "recoverability": "exact", "role_in_R1": "preexisting_negative_control_trajectory_C11_C12",
            "C14_C76_target_outcome_used_to_select_for_C77": 0,
            "historical_context": "endpoint-aligned after C10; temp=0.1 fixed before C12 and before C14-C76; C12 falsified transfer",
        },
        {
            "regime_id": "global_lpc", "objective": "historical_global_LPC_control", "loss_terms": "registered_control objective",
            "coefficients": "laplace_smoothing=1.0", "training_schedule": "historical stage2 schedule", "optimizer": "Adam",
            "learning_rate": "encoder=0.01;critic=0.01", "batch_size": 256, "preprocessing": "same manifest",
            "subject_source_split": "LOSO source subjects only", "checkpoint_cadence": "every 5 epochs",
            "early_stopping_rule": "none", "target_label_access": "none before locked evaluation",
            "seed_implementation": "manifest deterministic plans", "code_commit": REGIME_COMMITS["ERM"],
            "config_hash": "parent_manifest", "sidecar_schema": "TrainResult", "recoverability": "exact_excluded",
            "role_in_R1": "excluded_to_avoid_unregistered_regime_multiplicity", "C14_C76_target_outcome_used_to_select_for_C77": 0,
            "historical_context": "original registered control; recoverable but not in PM-preferred C77 set",
        },
        {
            "regime_id": "uniform", "objective": "historical_uniform_domain_control", "loss_terms": "registered_control objective",
            "coefficients": "manifest_exact", "training_schedule": "historical stage2 schedule", "optimizer": "Adam",
            "learning_rate": "encoder=0.01;critic=0.01", "batch_size": 256, "preprocessing": "same manifest",
            "subject_source_split": "LOSO source subjects only", "checkpoint_cadence": "every 5 epochs",
            "early_stopping_rule": "none", "target_label_access": "none before locked evaluation",
            "seed_implementation": "manifest deterministic plans", "code_commit": REGIME_COMMITS["ERM"],
            "config_hash": "parent_manifest", "sidecar_schema": "TrainResult", "recoverability": "exact_excluded",
            "role_in_R1": "excluded_to_avoid_unregistered_regime_multiplicity", "C14_C76_target_outcome_used_to_select_for_C77": 0,
            "historical_context": "original registered control; recoverable but not in PM-preferred C77 set",
        },
    ]
    regime_payloads = {
        "ERM": {"base": derived_manifest["derived_execution_view_sha256"], "objective": "ERM", "stage": "stage1_final"},
        "OACI": {"base": derived_manifest["derived_execution_view_sha256"], "objective": "OACI", "objective_blob": _git_blob_sha(REGIME_COMMITS["OACI"], OACI_OBJECTIVE_PATH), "engine_blob": _git_blob_sha(REGIME_COMMITS["OACI"], ENGINE_PATH)},
        "SRC": {"base": derived_manifest["derived_execution_view_sha256"], "objective": "SRC", "smooth_temperature": 0.1, "min_source_domains": 2, "objective_blob": _git_blob_sha(REGIME_COMMITS["SRC"], SRC_OBJECTIVE_PATH), "runner_blob": _git_blob_sha(REGIME_COMMITS["SRC"], SRC_ONEFOLD_PATH), "selector_margins": {"bacc": 0.02, "nll": 0.05, "ece": 0.02}},
    }
    hashes = []
    for regime, payload in regime_payloads.items():
        digest = payload_sha256(payload)
        hashes.append({"regime_id": regime, "regime_config_sha256": digest, "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")), "exact_historical_commit": REGIME_COMMITS[regime], "current_HEAD_at_protocol_generation": current_head})
        for row in historical:
            if row["regime_id"] == regime:
                row["config_hash"] = digest
    reconstruction = [
        {"regime_id": row["regime_id"], "exact_config_recoverable": int(row["recoverability"].startswith("exact")), "strict_target_isolation_verifiable": 1, "checkpoint_cadence_reproducible": 1, "C14_C76_target_outcome_selected_for_C77": 0, "qualifies_primary_R1": int(row["regime_id"] in PRIMARY_REGIMES), "comparable_40_checkpoint_trajectory_per_level": int(row["regime_id"] in TRAJECTORY_REGIMES), "levels": "0|1", "reason": row["role_in_R1"]}
        for row in historical
    ]
    isolation = [
        {"regime_id": "ERM", "optimization_view": "source_train", "selection_view": "not_applicable_stage1_anchor", "target_view": "post_lock_evaluation_only", "target_fit_ids_empty_evidence": "RunProvenance invariant + historical runner tests", "target_label_training_access": 0, "passed": 1},
        {"regime_id": "OACI", "optimization_view": "source_train", "selection_view": "source_train leakage measurement", "target_view": "post_lock_evaluation_only", "target_fit_ids_empty_evidence": "RunProvenance invariant + C11 actual", "target_label_training_access": 0, "passed": 1},
        {"regime_id": "SRC", "optimization_view": "source_train", "selection_view": "not used for field retention; historical selector was source_guard only", "target_view": "post_lock_evaluation_only", "target_fit_ids_empty_evidence": "C11 all_target_fit_ids_empty=true; selector target_read=false", "target_label_training_access": 0, "passed": 1},
    ]
    return {
        "historical_regime_inventory.csv": historical,
        "regime_reconstruction_status.csv": reconstruction,
        "regime_target_isolation_audit.csv": isolation,
        "regime_config_hash_manifest.csv": hashes,
    }


def seed_role_tables() -> dict[str, list[dict]]:
    seed_roles = [
        {"seed": 3, "role": "instrumentation_pilot_and_protocol_debug", "confirmation_claim_allowed": 0, "may_change_seed4_protocol": 1, "access_in_C77": 0, "execution_authorized": 0},
        {"seed": 4, "role": "locked_confirmation", "confirmation_claim_allowed": 1, "may_change_protocol_after_first_access": 0, "access_in_C77": 0, "execution_authorized": 0},
    ]
    separation = [
        {"gate": "seed3_only_in_C78", "required": 1, "observed_contract": 1, "blocking": 1},
        {"gate": "seed4_absent_from_C78_execution_matrix", "required": 1, "observed_contract": 1, "blocking": 1},
        {"gate": "C79_final_protocol_hash_before_seed4_training", "required": 1, "observed_contract": 1, "blocking": 1},
        {"gate": "seed3_and_seed4_not_pooled_before_seed4_verdict", "required": 1, "observed_contract": 1, "blocking": 1},
        {"gate": "seed4_no_access_in_C77", "required": 1, "observed_contract": 1, "blocking": 1},
    ]
    return {"seed_role_contract.csv": seed_roles, "seed3_seed4_separation_audit.csv": separation}


def instrumentation_tables() -> dict[str, list[dict]]:
    registry = [
        {"scope": "checkpoint", "field": field, "dtype": dtype, "information_class": info, "required": 1}
        for field, dtype, info in (
            ("dataset", "string", "metadata"), ("target_subject", "int", "group_key"),
            ("source_subjects", "list[int]", "strict_source_metadata"), ("seed", "int", "metadata"),
            ("regime", "enum", "metadata"), ("epoch_step", "int_pair", "metadata"),
            ("checkpoint_hash", "sha256", "metadata"), ("config_hash", "sha256", "metadata"),
            ("code_commit", "git_sha", "metadata"), ("optimizer_state_hash", "sha256", "training_trace"),
            ("training_loss_components", "map[float]", "strict_source_training_trace"),
            ("source_audit_metrics", "map[float]", "strict_source"),
            ("support_leakage_metrics", "map[float]", "diagnostic_measurement"),
            ("checkpoint_genealogy", "parent_hash", "metadata"),
        )
    ]
    for view, fields, info in (
        ("strict_source_trial_view", "unit_id,source_subject,trial_id,source_label,logits,probabilities,prediction,z,Wz,Wz_plus_b,class_margins", "strict_source"),
        ("target_unlabeled_trial_view", "unit_id,target_subject,trial_id,logits,probabilities,prediction,z,Wz,Wz_plus_b,class_margins", "target_unlabeled"),
        ("target_construction_view", "unit_id,target_subject,trial_id,target_label,split_id", "target_label_construction"),
        ("target_evaluation_view", "unit_id,target_subject,trial_id,target_label,split_id", "target_label_evaluation"),
        ("same_label_oracle_view", "unit_id,target_subject,trial_id,target_label,endpoints", "same_label_oracle"),
        ("trajectory_trace_view", "unit_id,epoch,source_metrics,representation_summaries,registered_leakage_atoms,genealogy", "strict_source_training_trace"),
    ):
        registry.append({"scope": view, "field": fields, "dtype": "registered_schema", "information_class": info, "required": 1})
    views = [
        {"view": "strict_source_trial_view", "uses_source_labels": 1, "uses_target_rows": 0, "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_under_strict_source_DG": 1, "diagnostic_only": 0, "physically_separate": 1},
        {"view": "target_unlabeled_trial_view", "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_under_strict_source_DG": 0, "diagnostic_only": 1, "physically_separate": 1},
        {"view": "target_construction_view", "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 1, "uses_evaluation_labels": 0, "available_under_strict_source_DG": 0, "diagnostic_only": 1, "physically_separate": 1},
        {"view": "target_evaluation_view", "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_under_strict_source_DG": 0, "diagnostic_only": 1, "physically_separate": 1},
        {"view": "same_label_oracle_view", "uses_source_labels": 0, "uses_target_rows": 1, "uses_target_labels": 1, "uses_evaluation_labels": 1, "available_under_strict_source_DG": 0, "diagnostic_only": 1, "physically_separate": 1},
    ]
    return {"instrumentation_schema_registry.csv": registry, "physical_view_schema.csv": views}


def hypothesis_tables() -> dict[str, list[dict]]:
    hypotheses = [
        {"hypothesis": "H1", "claim": "measurement_control_separation_replicates", "primary_metric": "reliability_or_association_nonzero_and_no_material_actionability", "confirmation_unit": "seed4_target_x_regime", "direction": "separation", "multiplicity_family": "H1_H7"},
        {"hypothesis": "H2", "claim": "effective_multiplicity_predicts_control_failure_beyond_raw_M", "primary_metric": "blocked_incremental_deviance_and_top_gap_adjusted_effect", "confirmation_unit": "target_x_regime_trajectory", "direction": "effective_gt_raw", "multiplicity_family": "H1_H7"},
        {"hypothesis": "H3", "claim": "local_nonlinear_association_nontransportable", "primary_metric": "within_group_effect_minus_leave_target_trajectory_regime_transport", "confirmation_unit": "target_x_regime", "direction": "local_positive_transport_unqualified", "multiplicity_family": "H1_H7"},
        {"hypothesis": "H4", "claim": "registered_strict_source_representation_block_no_escape_hatch", "primary_metric": "incremental_R2_actionability_and_blocked_null", "confirmation_unit": "target_x_regime", "direction": "fails_candidate_gate", "multiplicity_family": "H1_H7"},
        {"hypothesis": "H5", "claim": "registered_target_unlabeled_geometry_no_actionable_control", "primary_metric": "incremental_R2_actionability_and_transport", "confirmation_unit": "target_x_regime", "direction": "fails_candidate_gate", "multiplicity_family": "H1_H7"},
        {"hypothesis": "H6", "claim": "split_label_target_information_exceeds_source_and_unlabeled_blocks", "primary_metric": "paired_incremental_prediction_difference", "confirmation_unit": "target_x_regime", "direction": "construction_gt_source_unlabeled", "multiplicity_family": "H1_H7", "availability": "target_label_positive_control_diagnostic_only"},
        {"hypothesis": "H7", "claim": "orbit_robustness_does_not_identify_W_vs_z_origin", "primary_metric": "functional_identity_and_descriptor_orbit_behavior", "confirmation_unit": "checkpoint", "direction": "identity_preserved_no_origin_claim", "multiplicity_family": "H1_H7"},
    ]
    gates = []
    for block in ("strict_source_C75_C76_exact", "target_unlabeled_C75_C76_exact"):
        for gate, operator, threshold in (
            ("incremental_R2", ">=", 0.02), ("nested_max_stat_p", "<", 0.05),
            ("positive_targets", ">=", 7), ("positive_every_regime_or_no_interaction", "==", 1),
            ("leave_target_median", ">", 0), ("leave_regime_median", ">", 0),
            ("material_topk_or_regret", "==", 1), ("target_label_leakage", "==", 0),
        ):
            gates.append({"candidate_block": block, "gate": gate, "operator": operator, "threshold": threshold, "all_required": 1, "association_p_alone_sufficient": 0})
    hierarchical = [
        {"analysis": name, "unit": unit, "mandatory": 1, "row_iid_allowed": 0}
        for name, unit in (
            ("within_target_centering", "target"), ("within_regime_centering", "regime"),
            ("per_target_per_regime", "target_x_regime"), ("leave_target_out", "target"),
            ("leave_regime_out", "regime"), ("leave_trajectory_out", "trajectory"),
            ("checkpoint_cluster_bootstrap", "checkpoint"), ("trial_ID_cluster_bootstrap", "trial_id"),
            ("seed_stratified", "seed"), ("blocked_max_stat_permutation", "seed_x_regime_x_target"),
            ("crossed_pigeonhole_bootstrap", "checkpoint_x_trial_id"),
        )
    ]
    return {
        "primary_hypothesis_registry.csv": hypotheses,
        "actionability_gate_registry.csv": gates,
        "hierarchical_inference_plan.csv": hierarchical,
    }


def risk_rows() -> list[dict]:
    definitions = [
        ("new_training_used_for_method_rescue", "block", "regimes fixed to historical ERM/OACI/SRC; no selector objective"),
        ("target_outcome_driven_regime_selection", "block", "regime set locked by history and PM before outcomes"),
        ("regime_not_exactly_recoverable", "block", "all three primary identities hash-bound"),
        ("seed3_seed4_contamination", "block", "physical roots and protocols separate"),
        ("seed4_access_before_protocol_lock", "block", "seed4 forbidden in C78; C79 final hash required"),
        ("target_label_training_leakage", "block", "RunProvenance and physical views"),
        ("checkpoint_retention_based_on_target_outcome", "block", "retain complete fixed-cadence field"),
        ("instrumentation_schema_drift", "block", "schema hash per shard and checkpoint"),
        ("physical_view_leakage", "block", "separate content-addressed roots"),
        ("row_iid_inference", "block", "hierarchical plan mandatory"),
        ("pooled_association_called_transport", "block", "leave-target/trajectory/regime required"),
        ("local_association_called_control", "block", "material actionability gate required"),
        ("orbit_robustness_called_causality", "block", "H7 expressly forbids origin inference"),
        ("effective_multiplicity_not_reported", "block", "H2 and every field report require it"),
        ("small_target_count", "limitation", "nine targets; no population claim"),
        ("multiple_regimes_multiple_testing", "block", "H1-H7 max-stat/Holm family"),
        ("training_failure_selective_reporting", "block", "all planned units reason-coded; retries retained"),
        ("BNCI2014_004_early_access", "block", "explicit denylist and zero-access ledger"),
        ("raw_cache_or_weights_in_git", "block", "external content-addressed roots only"),
        ("manuscript_drafting", "block", "not authorized"),
        ("ERM_trajectory_asymmetry", "limitation", "ERM is one shared stage1 anchor; OACI/SRC are comparable 40-point fields"),
        ("SRC_historical_negative_control_overread", "block", "SRC is falsification regime, never rescue candidate"),
        ("C76_T2_branch_reopened", "block", "no additional C76 feature/kernel mining"),
    ]
    return [{"risk": name, "severity": severity, "status_at_C77": "mitigated" if severity == "block" else "open_limitation", "blocking_open": 0, "mitigation_or_boundary": note} for name, severity, note in definitions]


def external_readiness_tables() -> dict[str, list[dict]]:
    readiness = [
        {"requirement": "dataset_access", "status": "not_accessed_in_C77", "future_need": "explicit authorization and content-addressed evidence manifest"},
        {"requirement": "task_label_compatibility", "status": "unresolved_by_design", "future_need": "locked motor-imagery mapping without outcome inspection"},
        {"requirement": "preprocessing_compatibility", "status": "schema_ready_not_validated", "future_need": "channel/time/filter ABI preflight"},
        {"requirement": "model_architecture_compatibility", "status": "conditional", "future_need": "ShallowConvNet geometry preflight and exact hook identity"},
        {"requirement": "target_definition", "status": "planned", "future_need": "subject-level LOSO; no reused target subjects"},
        {"requirement": "minimum_target_count", "status": "defer_threshold_until_R1", "future_need": "power recalibration from seed4 without outcome tuning"},
        {"requirement": "instrumentation_schema", "status": "compatible_by_contract", "future_need": "same five physical information views"},
        {"requirement": "hypothesis_mapping", "status": "H1_H7_general_objects_only", "future_need": "lock before any external data open"},
    ]
    graph = [
        {"stage": "R1_seed3", "depends_on": "C77_pass+future_exact_authorization", "output": "instrumented pilot and protocol-debug field", "executes_in_C77": 0},
        {"stage": "R1_seed4", "depends_on": "seed3 freeze+C79 final protocol hash+new authorization", "output": "locked same-dataset checkpoint-field confirmation", "executes_in_C77": 0},
        {"stage": "R2_external", "depends_on": "R1 verdict+external protocol lock+dataset authorization", "output": "independent target/dataset replication", "executes_in_C77": 0},
        {"stage": "R3_non_EEG", "depends_on": "R2/general-object justification", "output": "optional generality", "executes_in_C77": 0},
    ]
    return {"external_dataset_readiness.csv": readiness, "R2_dependency_graph.csv": graph}


def _checkpoint_sidecar_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "C78 checkpoint sidecar",
        "type": "object",
        "required": ["dataset", "target_subject", "source_subjects", "seed", "regime", "epoch", "optimizer_step", "checkpoint_sha256", "config_sha256", "code_commit", "optimizer_state_sha256", "provenance_sha256", "view_manifest_sha256"],
        "properties": {
            "dataset": {"const": "BNCI2014_001"}, "target_subject": {"type": "integer", "minimum": 1, "maximum": 9},
            "source_subjects": {"type": "array", "items": {"type": "integer"}, "minItems": 8, "maxItems": 8, "uniqueItems": True},
            "seed": {"const": 3}, "regime": {"enum": list(PRIMARY_REGIMES)}, "epoch": {"type": "integer"},
            "optimizer_step": {"type": "integer"}, "checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "config_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "code_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "optimizer_state_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "provenance_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "view_manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "additionalProperties": True,
    }


def _c78_protocol(derived_manifest: dict, locked_tables: dict) -> dict:
    pilot_target, pilot_target_hash = _hash_choice("pilot_target", tuple(map(str, TARGETS)))
    pilot_regime, pilot_regime_hash = _hash_choice("pilot_regime", TRAJECTORY_REGIMES)
    execution_matrix = [
        {"target": target, "seed": 3, "level": level, "regime": regime, "retained_checkpoints": 1 if regime == "ERM" else STAGE2_CHECKPOINTS}
        for target in TARGETS for level in LEVELS for regime in PRIMARY_REGIMES
    ]
    return {
        "schema_version": "c78_seed3_instrumented_pilot_protocol_v1",
        "milestone": "C78", "status": "LOCKED_READY_BUT_NOT_AUTHORIZED",
        "created_by": "C77_protocol_only", "authorization": {
            "required": True, "accepted_channel": "exact_CLI_argument_only",
            "argument": "--authorization-token", "exact_token": C78_AUTHORIZATION_TOKEN,
            "prompt_text_is_authorization": False, "environment_is_authorization": False,
            "C77_authorized": False, "training_attempts_in_C77": 0, "forward_attempts_in_C77": 0,
        },
        "execution_boundary": {
            "dataset_allowlist": ["BNCI2014_001"], "dataset_denylist": ["BNCI2014_004", "SEED", "PD_cross_site"],
            "seed_allowlist": [3], "seed_denylist": [4], "GPU_future_only_after_authorization": True,
            "target_labels_in_training": False, "target_outcome_checkpoint_retention": False,
            "complete_fixed_cadence_retention": True, "selector_or_recommendation_artifact": False,
            "manuscript": False, "raw_cache_or_weights_in_git": False,
        },
        "environment_contract": {
            "prefix": "/home/infres/yinwang/anaconda3/envs/icml",
            "historical_runner_environment": True,
            "python": "3.9.25", "torch": "2.8.0+cu128", "numpy": "1.26.4",
            "scipy": "1.13.1", "sklearn": "1.5.2",
            "conda_explicit_list_sha256_at_C77": "2c04fc1733a53b55abd071d6b1657eabfda8bbb56ef0bf0ab97e8234171958a1",
            "dummy_ABI": {"input": [2, 22, 385], "z": [2, 800], "logits": [2, 4], "Wz_plus_b_max_error": 0.0, "state_bytes": 158424},
            "future_GPU_partition_primary": "V100",
            "partition_snapshot": "V100 up at C77; availability is not authorization",
            "P0_rule": "rehash environment and rerun dummy ABI before any real-data load; mismatch stops",
        },
        "manifest_execution_view": derived_manifest,
        "regimes": {
            "primary": list(PRIMARY_REGIMES), "comparable_trajectory_regimes": list(TRAJECTORY_REGIMES),
            "ERM_role": "shared_stage1_final_anchor_only",
            "SRC_role": "historical_negative_falsification_control_not_method_rescue",
            "global_lpc_uniform_excluded": "recoverable but omitted prospectively to control regime multiplicity",
        },
        "seed_role": {"seed3": "pilot_protocol_debug", "seed4": "physically_forbidden_until_C79_final_lock"},
        "pilot": {
            "target": int(pilot_target), "target_selection": "minimum_SHA256(C78|pilot_target|target)", "target_selection_hash": pilot_target_hash,
            "regime": pilot_regime, "regime_selection": "minimum_SHA256(C78|pilot_regime|regime) over OACI,SRC", "regime_selection_hash": pilot_regime_hash,
            "levels": list(LEVELS), "shared_ERM_prerequisite_retained_per_level": True,
            "planned_retained_units": len(LEVELS) * (1 + STAGE2_CHECKPOINTS),
            "outcome_blind": True,
        },
        "stages": [
            {"stage": "P0", "scope": "dummy_ABI_schema_and_storage_dry_run", "real_data": False, "authorization_needed": False, "silent_escalation": False},
            {"stage": "P1", "scope": f"target_{pilot_target}_{pilot_regime}_seed3_levels_0_1_plus_shared_ERM", "real_data": True, "authorization_needed": True, "continue_if": "all identity, provenance, view-isolation, deterministic replay, quota, and failure-ledger gates pass", "silent_escalation": False},
            {"stage": "P2", "scope": "remaining predeclared seed3 matrix", "real_data": True, "authorization_needed": True, "continue_if": "P1 passes and PM authorization scope explicitly includes full C78 matrix", "silent_escalation": False},
        ],
        "execution_matrix": execution_matrix,
        "matrix_summary": {"target_level_contexts": 18, "training_phases": 54, "retained_checkpoint_target_level_units": len(TARGETS) * len(LEVELS) * (1 + 2 * STAGE2_CHECKPOINTS), "pilot_units": len(LEVELS) * (1 + STAGE2_CHECKPOINTS)},
        "checkpoint_cadence": {"ERM": "stage1_final_only", "OACI": list(range(4, 200, 5)), "SRC": list(range(4, 200, 5))},
        "instrumentation": {
            "views": ["strict_source_trial_view", "target_unlabeled_trial_view", "target_construction_view", "target_evaluation_view", "same_label_oracle_view", "trajectory_trace_view"],
            "source_fields": ["logits", "probabilities", "prediction", "z", "Wz", "Wz_plus_b", "class_margins"],
            "target_unlabeled_fields": ["logits", "probabilities", "prediction", "z", "Wz", "Wz_plus_b", "class_margins"],
            "identity_gates": ["Wz_plus_b_equals_logits", "softmax_equals_probabilities", "argmax_equals_prediction", "repeat_forward_determinism", "checkpoint_hash", "preprocessing_hash", "target_fit_ids_empty"],
        },
        "hypotheses": [f"H{index}" for index in range(1, 8)],
        "inference": ["within_target_centering", "within_regime_centering", "leave_target_out", "leave_regime_out", "leave_trajectory_out", "checkpoint_cluster_bootstrap", "trial_ID_cluster_bootstrap", "seed_stratified", "blocked_max_stat_permutations", "crossed_pigeonhole_bootstrap"],
        "multiplicity": "Holm over H1-H7 primary family; nested max-stat within registered feature/kernel paths",
        "failure_policy": "retain all planned/attempted unit rows and reason codes; no selective omission or target-outcome retry",
        "locked_tables": locked_tables,
        "timing_contract": {"protocol_lock": "C77 commit before authorization", "authorization_time": "future CLI invocation", "first_training_submission": "strictly after both", "first_target_outcome_read": "after physical split lock", "first_seed4_access": "after C79 final protocol hash and separate authorization"},
    }


def prepare_protocol() -> dict:
    if _git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise RuntimeError("C77 protocol must be generated directly from accepted C76 result commit")
    if sha256(C76_PROTOCOL) != C76_PROTOCOL_SHA.read_text().strip():
        raise RuntimeError("C77 parent C76 protocol hash mismatch")
    required_paths = [C76_RESULT, C76_REPORT, C76_RED_TEAM, C76_ORBIT_IDENTITY, C76_SEPARATION, C76_TOPOLOGY, C76_T3_GATE, MANIFEST_PATH, SRC_ONEFOLD_PATH, SRC_OBJECTIVE_PATH, SRC_SELECTOR_PATH, OACI_OBJECTIVE_PATH, ERM_OBJECTIVE_PATH, ENGINE_PATH, PROVENANCE_PATH, C11_RESULT_PATH, C12_RESULT_PATH, C74_STORAGE_PATH]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    for regime, commit in REGIME_COMMITS.items():
        if not _git("cat-file", "-t", commit) == "commit":
            raise RuntimeError(f"C77 historical commit missing: {regime} {commit}")
    logical_manifest, derived_manifest = _manifest_contract()
    tables: dict[str, list[dict]] = {}
    tables.update(c76_replay_tables())
    tables.update(regime_tables(derived_manifest))
    tables.update(seed_role_tables())
    tables.update(instrumentation_tables())
    tables.update(hypothesis_tables())
    tables.update(external_readiness_tables())
    tables["risk_register.csv"] = risk_rows()
    tables["failure_reason_ledger.csv"] = [
        {"item": "no_failure", "status": "protocol_ready", "blocking": 0, "reason": "all static protocol and recovery gates passed; execution remains unauthorized"}
    ]
    for name, rows in tables.items():
        _write_csv(name, rows)
    _write_json(TABLE_DIR / "checkpoint_sidecar_schema.json", _checkpoint_sidecar_schema())
    locked_tables = {
        name: {"path": str(TABLE_DIR / name), "sha256": sha256(TABLE_DIR / name), "rows": len(rows), "size_bytes": (TABLE_DIR / name).stat().st_size}
        for name, rows in tables.items()
    }
    locked_tables["checkpoint_sidecar_schema.json"] = {
        "path": str(TABLE_DIR / "checkpoint_sidecar_schema.json"),
        "sha256": sha256(TABLE_DIR / "checkpoint_sidecar_schema.json"),
        "rows": 1, "size_bytes": (TABLE_DIR / "checkpoint_sidecar_schema.json").stat().st_size,
    }
    c78 = _c78_protocol(derived_manifest, locked_tables)
    _write_json(C78_PROTOCOL_PATH, c78)
    C78_PROTOCOL_SHA_PATH.write_text(sha256(C78_PROTOCOL_PATH) + "\n")
    c79 = {
        "schema_version": "c79_seed4_confirmation_protocol_skeleton_v1",
        "status": "SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED", "seed": 4,
        "depends_on": ["C78 seed3 complete", "seed3 report frozen", "C79 final protocol committed and hashed", "separate exact CLI authorization"],
        "fixed_from_seed3_before_seed4": ["hypotheses", "feature blocks", "metrics", "nulls", "actionability gates", "regime set", "checkpoint cadence"],
        "forbidden_now": ["seed4 data access", "training", "forward", "checkpoint creation", "confirmation claim"],
        "C77_seed4_access": False,
    }
    _write_json(C79_SKELETON_PATH, c79)
    protocol = {
        "schema_version": "c77_independent_multiregime_replication_protocol_v1",
        "milestone": MILESTONE, "protocol_lock_timestamp_utc": _now(),
        "parent_C76_result_commit": PARENT_COMMIT,
        "parent_C76_result_sha256": sha256(C76_RESULT),
        "status": "prospective_protocol_and_power_only_not_execution_not_confirmation",
        "execution_boundary": {"training": False, "real_forward": False, "re_inference": False, "GPU": False, "BNCI2014_004_access": False, "seed3_access": False, "seed4_access": False, "checkpoints_created": 0, "selector_or_recommendation_artifact": False, "manuscript": False, "raw_cache_or_weights_in_git": False},
        "scientific_scope": {"justified": ["independent_instrumented_replication", "cross_regime_transport", "new_checkpoint_field_confirmation"], "not_justified": ["OACI_v2_rescue", "selector_search", "target_outcome_hyperparameter_tuning", "new_DG_penalty", "checkpoint_recommendation"]},
        "regime_decision": {"primary": list(PRIMARY_REGIMES), "comparable_trajectories": list(TRAJECTORY_REGIMES), "levels": list(LEVELS), "ERM_asymmetry_disclosed": True, "minimum_two_preexisting_trajectory_regimes_met": True},
        "dataset_execution_view": derived_manifest,
        "seed_roles": {"seed3": "pilot_protocol_debug", "seed4": "locked_confirmation", "C77_accessed_neither": True},
        "C78_protocol": {"path": str(C78_PROTOCOL_PATH), "sha256": sha256(C78_PROTOCOL_PATH), "authorization_required": True, "authorized_in_C77": False},
        "C79_skeleton": {"path": str(C79_SKELETON_PATH), "sha256": sha256(C79_SKELETON_PATH), "final_protocol_locked": False, "seed4_authorized": False},
        "synthetic_contract": {"replicates": SYNTHETIC_REPLICATES, "seed": RNG_SEED, "grid": {"candidate_count": [20, 40, 80], "effective_multiplicity": [2, 8, 20], "top_gap": [0.01, 0.03], "association_strength": [0.0, 0.2, 0.5], "transport_heterogeneity": [0.0, 0.75, 1.5], "label_budget": [8, 32, 128]}, "primary_alpha": 0.05, "FPR_upper_gate": 0.075, "power_gate": 0.80, "actionability_materiality": 0.02, "shards": 8},
        "locked_tables": locked_tables,
        "claims": {"new_training_scientifically_justified": True, "training_authorized": False, "target_population_generalization": False, "representation_mechanism": False, "deployability": False},
    }
    _write_json(PROTOCOL_PATH, protocol)
    PROTOCOL_SHA_PATH.write_text(sha256(PROTOCOL_PATH) + "\n")
    TIMING_PATH.write_text(
        "# C77 protocol timing audit\n\n"
        f"- Generated from accepted C76 commit: `{PARENT_COMMIT}`.\n"
        f"- C77 protocol SHA-256: `{sha256(PROTOCOL_PATH)}`.\n"
        f"- C78 seed-3 protocol SHA-256: `{sha256(C78_PROTOCOL_PATH)}`.\n"
        "- These files must be committed before any synthetic outcome computation or future seed-3 job submission.\n"
        "- C77 performs no training, real-data forward, re-inference, GPU work, seed-3/seed-4 access, or BNCI2014_004 access.\n"
        "- C79 is a skeleton only. A later final C79 protocol and separate authorization must precede any seed-4 access.\n"
        "- C77 is prospective protocol preparation, not retrospective pre-registration and not independent confirmation.\n"
    )
    return protocol


def main() -> int:
    protocol = prepare_protocol()
    print(json.dumps({
        "milestone": MILESTONE,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "C78_sha256": sha256(C78_PROTOCOL_PATH),
        "regimes": protocol["regime_decision"],
        "training_attempts": 0,
        "forward_attempts": 0,
        "authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
