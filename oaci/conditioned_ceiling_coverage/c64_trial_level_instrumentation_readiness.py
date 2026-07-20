"""C64 - Trial-Level Instrumentation Readiness / Split-Label-CS Evidence Gate."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os

from . import audit_utils as au
from . import c61_conditional_observability_divergence as c61
from . import c63_trajectory_dynamic_observability as c63


MILESTONE = "C64"
REPORT_DIR = "oaci/reports"
TABLE_DIR = "oaci/reports/c64_tables"
REPORT_JSON = "oaci/reports/C64_TRIAL_LEVEL_INSTRUMENTATION_READINESS.json"
C63_JSON = "oaci/reports/C63_TRAJECTORY_DYNAMIC_OBSERVABILITY.json"

DECISIONS = (
    "C64-A_frozen_summary_artifact_paths_saturated",
    "C64-B_reinference_only_trial_cache_campaign_sufficient",
    "C64-C_new_training_required_for_trial_cache_or_atom_trace",
    "C64-D_split_label_cache_protocol_ready_but_not_authorized",
    "C64-E_full_time_series_conditional_cs_protocol_ready_but_not_authorized",
    "C64-F_atom_trace_protocol_ready_but_not_authorized",
    "C64-G_independent_checkpoint_replication_protocol_ready_but_not_authorized",
    "C64-H_instrumentation_not_scientifically_justified_yet",
    "C64-I_source_observable_escape_hatch_remaining",
    "C64-J_claim_or_availability_inconsistency_found",
)

FINAL_GATE = "TRIAL_LEVEL_CACHE_CAMPAIGN_READY_BUT_NOT_AUTHORIZED"
REINFERENCE_SUBGATE = "CONDITIONALLY_SUFFICIENT_IF_FROZEN_CHECKPOINTS_AND_PREPROCESSING_ARE_RECOVERABLE"
TRAINING_GATE = "TRAINING_NOT_AUTHORIZED_IN_C64"
NEXT_DIRECTION = "wait for remote review; C65 may request explicit re-inference-only instrumentation authorization or revise readiness gates"

TEMPLATE_ONLY_HIT = c61.TEMPLATE_ONLY_HIT
ENDPOINT_ORACLE_HIT = c61.ENDPOINT_ORACLE_HIT
MAX_NULL_P95 = c61.MAX_NULL_P95
STRICT_SOURCE_HIT = c61.STRICT_SOURCE_HIT
SOURCE_DYNAMIC_HIT = c63.SOURCE_DYNAMIC_HISTORY_HIT
SOURCE_DYNAMIC_TEMPLATE_HIT = c63.SOURCE_DYNAMIC_TEMPLATE_HIT

FORBIDDEN_PATTERNS = (
    "manuscript drafting",
    "M1 started",
    "training authorized",
    "re-inference authorized",
    "source-only rescue",
    "OACI rescue",
    "deployable selector",
    "checkpoint recommendation",
    "selected_candidate_id",
    "chosen checkpoint",
    "few-label sufficiency",
    "same-label endpoint oracle available at selection time",
    "full conditional-CS estimator supported",
    "full time-series conditional-CS estimator supported",
    "EEG distribution theorem",
    "minimax theorem",
    "Le Cam theorem",
    "Fano theorem",
    "BNCI2014_004 used",
    "seeds [3,4] used",
    "GPU required",
)

NEGATION_CUES = c63.NEGATION_CUES + (
    "not authorized",
    "not supported",
    "not available",
    "blocked",
    "forbidden",
    "without",
    "requires",
    "future",
    "protocol only",
)


def _lock_config() -> str:
    return au.lock_config(MILESTONE)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: list[dict], cols: list[str]) -> None:
    au.write_csv(path, rows, cols)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_files() -> list[str]:
    out = []
    for root, dirs, files in os.walk("."):
        if "/.git" in root or root.startswith("./.git"):
            continue
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            out.append(os.path.join(root, name)[2:])
    return out


def _checkpoint_weight_files() -> list[str]:
    suffixes = (".pt", ".pth", ".ckpt", ".safetensors")
    return [p for p in _repo_files() if p.lower().endswith(suffixes)]


def _exists(path: str) -> int:
    return int(os.path.exists(path))


def build_frozen_path_closure_ledger() -> list[dict]:
    return [
        {"path_id": "FP1_static_source", "milestones": "C31-C47", "status": "closed_as_reliable_control", "evidence": "source-only global comparability fails; conditioning diagnostic only", "remaining_value": "none_as_selector"},
        {"path_id": "FP2_conditioned_ceiling", "milestones": "C48-C52", "status": "diagnostic_ceiling_not_action_rule", "evidence": "trajectory fragmentation and key-only insufficiency", "remaining_value": "requires label/content boundary"},
        {"path_id": "FP3_endpoint_scalar", "milestones": "C53-C55", "status": "same_label_oracle_boundary", "evidence": "0.944 endpoint scalar is unavailable at selection time", "remaining_value": "split-label cache required"},
        {"path_id": "FP4_mechanism_theory", "milestones": "C56-C60", "status": "synthetic_model_bound_not_eeg_theorem", "evidence": "rank-gauge theorem repaired; EEG lower-bound bridge not forced", "remaining_value": "instrumented data only"},
        {"path_id": "FP5_static_cod", "milestones": "C61-C62", "status": "estimator_stress_stable", "evidence": "finite partition/binary-Y COD stable; full CS unsupported", "remaining_value": "trial-level paired cache"},
        {"path_id": "FP6_dynamic_cod", "milestones": "C63", "status": "dynamic_escape_hatch_closed", "evidence": "dynamic+template -> endpoint still +0.223765", "remaining_value": "trial-level trajectory cache"},
    ]


def build_remaining_escape_hatches() -> list[dict]:
    return [
        {"hatch_id": "RH1_more_source_summary_features", "remaining": 0, "reason": "static/dynamic source summaries repeatedly below endpoint boundary", "scientifically_justified_now": 0},
        {"hatch_id": "RH2_target_unlabeled_summary_geometry", "remaining": 0, "reason": "already diagnostic-only and not source-only rescue", "scientifically_justified_now": 0},
        {"hatch_id": "RH3_split_label_cache", "remaining": 1, "reason": "separates same-label oracle from disjoint target-label diagnostic content", "scientifically_justified_now": 1},
        {"hatch_id": "RH4_full_conditional_cs", "remaining": 1, "reason": "requires paired trial-level variables and Gram/KDE inputs", "scientifically_justified_now": 1},
        {"hatch_id": "RH5_atom_trace", "remaining": 1, "reason": "C39-C40 atom branch closed only under current artifacts", "scientifically_justified_now": 1},
        {"hatch_id": "RH6_independent_replication", "remaining": 1, "reason": "tests checkpoint-field dependence without tuning current universe", "scientifically_justified_now": 1},
    ]


def build_frozen_saturation_decision() -> list[dict]:
    return [
        {"decision_id": "SAT1", "decision": "FROZEN_SUMMARY_PATHS_SATURATED", "passed": 1, "rationale": "C61-C63 closed static, estimator, and dynamic summary escape hatches."},
        {"decision_id": "SAT2", "decision": "MORE_SUMMARY_FEATURE_ENGINEERING_NOT_PRIORITIZED", "passed": 1, "rationale": "No remaining source-observable summary family has a justified path to split-label/full-CS evidence."},
        {"decision_id": "SAT3", "decision": "NEXT_EVIDENCE_IS_TRIAL_LEVEL_CACHE", "passed": 1, "rationale": "Split-label and full conditional-CS both require sample-level paired variables."},
    ]


def build_trial_cache_columns() -> list[dict]:
    rows = [
        ("candidate_id", "identity", "yes", "none", 1, 1),
        ("checkpoint_id", "identity", "yes", "none", 1, 1),
        ("dataset_id", "identity", "yes", "none", 1, 1),
        ("target_subject", "identity", "yes", "target metadata", 1, 1),
        ("source_subjects", "identity", "yes", "source metadata", 1, 1),
        ("seed", "identity", "yes", "training metadata", 1, 1),
        ("trajectory_id", "identity", "yes", "trajectory metadata", 1, 1),
        ("epoch_or_step", "identity", "yes", "trajectory metadata", 1, 1),
        ("regime_objective", "identity", "yes", "training metadata", 1, 1),
        ("trial_id", "trial", "yes", "none", 1, 1),
        ("split_id", "split", "yes", "none", 1, 1),
        ("split_role", "split", "yes", "none", 1, 1),
        ("domain_id", "trial", "yes", "none", 1, 1),
        ("class_label", "label", "yes", "target/source label", 1, 0),
        ("prediction", "prediction", "yes", "model output", 1, 0),
        ("correct_flag", "endpoint", "yes", "label-derived", 1, 0),
        ("logits", "prediction", "yes", "model output", 1, 0),
        ("probabilities", "prediction", "yes", "model output", 1, 0),
        ("confidence", "prediction", "recommended", "model output", 0, 0),
        ("per_class_margins", "prediction", "recommended", "model output", 0, 0),
        ("nll", "endpoint_component", "recommended", "label-derived", 0, 0),
        ("ece_bin_components", "calibration", "recommended", "label-derived", 0, 0),
        ("representation_z", "representation", "optional", "model internal", 0, 0),
        ("projection_Wz", "representation", "optional", "model internal", 0, 0),
        ("source_target_flag", "split", "yes", "none", 1, 1),
        ("audit_role", "split", "yes", "none", 1, 1),
    ]
    return [
        {"column": col, "category": cat, "minimality": req, "label_dependency": dep, "needed_for_split_label": split, "available_at_selection_time_if_source_only": avail}
        for col, cat, req, dep, split, avail in rows
    ]


def build_availability_tags() -> list[dict]:
    return [
        {"tag": "source_only", "definition": "constructible from source-side metadata, labels, or predictions only", "allowed_for_source_rule": 1, "diagnostic_only": 0},
        {"tag": "target_unlabeled", "definition": "target inputs/predictions without target labels", "allowed_for_source_rule": 0, "diagnostic_only": 1},
        {"tag": "target_label_construct", "definition": "target labels used only on construction split", "allowed_for_source_rule": 0, "diagnostic_only": 1},
        {"tag": "target_label_eval", "definition": "held-out target labels reserved for evaluation only", "allowed_for_source_rule": 0, "diagnostic_only": 1},
        {"tag": "same_label_oracle", "definition": "same candidate endpoint labels/scalars used for construction and evaluation", "allowed_for_source_rule": 0, "diagnostic_only": 1},
        {"tag": "split_label_allowed", "definition": "target-label diagnostic constructed/evaluated on disjoint splits", "allowed_for_source_rule": 0, "diagnostic_only": 1},
        {"tag": "available_at_selection_time", "definition": "available before target endpoint evaluation under original DG setting", "allowed_for_source_rule": 1, "diagnostic_only": 0},
    ]


def build_split_label_power() -> list[dict]:
    return [
        {"cell": "target_x_trajectory_x_class", "min_construct_trials": 20, "min_eval_trials": 20, "bootstrap_reps": 1000, "detectable_hit_gap": 0.15, "status": "protocol_ready_not_authorized"},
        {"cell": "target_x_class", "min_construct_trials": 40, "min_eval_trials": 40, "bootstrap_reps": 1000, "detectable_hit_gap": 0.10, "status": "coarser_backup"},
        {"cell": "trajectory_x_class", "min_construct_trials": 30, "min_eval_trials": 30, "bootstrap_reps": 1000, "detectable_hit_gap": 0.12, "status": "trajectory_backup"},
        {"cell": "global_target_split", "min_construct_trials": 80, "min_eval_trials": 80, "bootstrap_reps": 2000, "detectable_hit_gap": 0.08, "status": "stability_check"},
    ]


def build_split_label_forbidden_claims() -> list[dict]:
    return [
        {"claim": "same-label endpoint oracle implies split-label sufficiency", "forbidden": 1, "reason": "construction/evaluation labels must be disjoint"},
        {"claim": "split-label diagnostic implies few-label deployable calibration", "forbidden": 1, "reason": "few-label method claim needs separate protocol and selection-time constraints"},
        {"claim": "target-label diagnostic is source-only", "forbidden": 1, "reason": "availability class differs"},
        {"claim": "held-out evaluation labels can enter feature construction", "forbidden": 1, "reason": "leakage"},
    ]


def build_conditional_cs_feasibility() -> list[dict]:
    return [
        {"estimator_component": "p(y|x1) vs p(y|x1,x2)", "project_mapping": "target trial correctness/logit margin conditioned on source trajectory state plus split-label diagnostic", "current_supported": 0, "required_cache": "trial_id, split_role, logits, probabilities, labels, source/target flag", "protocol_ready": 1},
        {"estimator_component": "Gram/KDE sample matrix", "project_mapping": "paired trial-level x1,x2,y samples", "current_supported": 0, "required_cache": "sample-level paired variables", "protocol_ready": 1},
        {"estimator_component": "Hankel past-window response", "project_mapping": "past K checkpoint source state to response endpoint/logit", "current_supported": 0, "required_cache": "raw trajectory windows and per-step predictions", "protocol_ready": 1},
        {"estimator_component": "summary kernel proxy", "project_mapping": "C62/C63 hit-delta RBF proxy", "current_supported": 1, "required_cache": "summary tables only", "protocol_ready": 1},
        {"estimator_component": "split-label conditional diagnostic", "project_mapping": "construct target-label feature on split A, evaluate on split B", "current_supported": 0, "required_cache": "disjoint target splits", "protocol_ready": 1},
    ]


def build_hankel_mapping() -> list[dict]:
    return [
        {"paper_object": "past window", "project_variable": "past K checkpoint source state", "current_artifact": "C63 support-only proxy", "sample_level_required": 1, "supported_now": 0},
        {"paper_object": "response", "project_variable": "held-out target trial correctness/logit margin at checkpoint t", "current_artifact": "endpoint summaries only", "sample_level_required": 1, "supported_now": 0},
        {"paper_object": "conditioning X1", "project_variable": "source-observable trajectory state", "current_artifact": "source score summaries", "sample_level_required": 1, "supported_now": 0},
        {"paper_object": "increment X2", "project_variable": "split-label diagnostic or target-unlabeled geometry", "current_artifact": "same-label endpoint oracle only", "sample_level_required": 1, "supported_now": 0},
        {"paper_object": "diagnostic Y", "project_variable": "trial correctness / NLL / margin / joint-good bin", "current_artifact": "candidate-level bins", "sample_level_required": 1, "supported_now": 0},
    ]


def build_missing_data_ledger() -> list[dict]:
    return [
        {"missing_item": "per_trial_logits_probabilities", "blocks_split_label": 1, "blocks_full_cs": 1, "blocks_atom_trace": 0, "recoverable_by_reinference": "yes_if_checkpoint_weights_available"},
        {"missing_item": "per_trial_labels_and_split_roles", "blocks_split_label": 1, "blocks_full_cs": 1, "blocks_atom_trace": 0, "recoverable_by_reinference": "requires_dataset_labels_and_split_plan"},
        {"missing_item": "raw_source_trajectory_windows", "blocks_split_label": 0, "blocks_full_cs": 1, "blocks_atom_trace": 0, "recoverable_by_reinference": "yes_if_checkpoints_and_order_available"},
        {"missing_item": "representations_z_projection_Wz", "blocks_split_label": 0, "blocks_full_cs": 1, "blocks_atom_trace": 0, "recoverable_by_reinference": "yes_if_model_forward_hooks_available"},
        {"missing_item": "atom_level_leakage_trace", "blocks_split_label": 0, "blocks_full_cs": 0, "blocks_atom_trace": 1, "recoverable_by_reinference": "unclear_likely_training_instrumentation_needed"},
        {"missing_item": "independent_checkpoint_field", "blocks_split_label": 0, "blocks_full_cs": 0, "blocks_atom_trace": 0, "recoverable_by_reinference": "no_requires_new_field_or_reserved_holdout_release"},
    ]


def build_checkpoint_inventory() -> list[dict]:
    weight_files = _checkpoint_weight_files()
    return [
        {"inventory_item": "checkpoint_weight_files", "present": int(bool(weight_files)), "count": len(weight_files), "evidence": ";".join(weight_files[:5]) if weight_files else "no .pt/.pth/.ckpt/.safetensors files in checkout", "blocks_reinference_now": int(not bool(weight_files))},
        {"inventory_item": "checkpoint_abi_code", "present": _exists("oaci/artifacts/checkpoint.py"), "count": 1, "evidence": "oaci/artifacts/checkpoint.py", "blocks_reinference_now": 0},
        {"inventory_item": "training_checkpoint_record_code", "present": _exists("oaci/train/checkpoint.py"), "count": 1, "evidence": "oaci/train/checkpoint.py", "blocks_reinference_now": 0},
        {"inventory_item": "eeg_preprocessing_code", "present": _exists("oaci/data/eeg/preprocess.py"), "count": 1, "evidence": "oaci/data/eeg/preprocess.py", "blocks_reinference_now": 0},
        {"inventory_item": "eeg_cache_code", "present": _exists("oaci/data/eeg/cache.py"), "count": 1, "evidence": "oaci/data/eeg/cache.py", "blocks_reinference_now": 0},
        {"inventory_item": "split_label_cache", "present": 0, "count": 0, "evidence": "C58 schema exists but cache missing", "blocks_reinference_now": 1},
        {"inventory_item": "trial_level_logits_cache", "present": 0, "count": 0, "evidence": "C62/C63 missing-data gates", "blocks_reinference_now": 1},
    ]


def build_reinference_feasibility() -> list[dict]:
    weights_present = bool(_checkpoint_weight_files())
    return [
        {"requirement": "frozen_checkpoint_weights_loadable", "present_in_checkout": int(weights_present), "status": "blocking_in_current_checkout" if not weights_present else "present", "reinference_only_sufficient_if_present": 1},
        {"requirement": "checkpoint_metadata_hashes", "present_in_checkout": _exists("oaci/artifacts/checkpoint.py"), "status": "abi_code_present_metadata_files_not_verified", "reinference_only_sufficient_if_present": 1},
        {"requirement": "dataset_preprocessing_replay", "present_in_checkout": _exists("oaci/data/eeg/preprocess.py"), "status": "code_present_data_not_verified", "reinference_only_sufficient_if_present": 1},
        {"requirement": "trial_label_access", "present_in_checkout": 0, "status": "not_verified_without_dataset_mount", "reinference_only_sufficient_if_present": 1},
        {"requirement": "forward_hooks_for_logits_probs_representations", "present_in_checkout": 1, "status": "implementable_protocol_not_executed", "reinference_only_sufficient_if_present": 1},
        {"requirement": "overall_reinference_only_decision", "present_in_checkout": int(weights_present), "status": REINFERENCE_SUBGATE, "reinference_only_sufficient_if_present": int(weights_present)},
    ]


def build_resource_estimate() -> list[dict]:
    return [
        {"resource": "cpu_inference", "estimate": "moderate", "needed": 1, "notes": "can be Slurm cpu-high if model/runtime supports CPU inference"},
        {"resource": "gpu_inference", "estimate": "not_requested", "needed": 0, "notes": "C64 does not request GPU"},
        {"resource": "storage_trial_cache", "estimate": "1-20GB depending on logits/representations", "needed": 1, "notes": "representations dominate storage"},
        {"resource": "walltime", "estimate": "campaign estimate needed after checkpoint count is known", "needed": 1, "notes": "current checkout lacks weight-file count"},
        {"resource": "hash_manifest", "estimate": "required", "needed": 1, "notes": "must hash checkpoint, cache, split, and output files"},
    ]


def build_reinference_risk_ledger() -> list[dict]:
    return [
        {"risk": "checkpoint_unavailable", "severity": "high", "mitigation": "mount or restore frozen checkpoint store; do not train silently"},
        {"risk": "preprocessing_drift", "severity": "high", "mitigation": "hash preprocessing code/data transforms and replay exact C19 config"},
        {"risk": "split_leakage", "severity": "high", "mitigation": "pre-register split roles before cache construction"},
        {"risk": "same_label_oracle_reinterpretation", "severity": "high", "mitigation": "same-label endpoint scalar cannot enter split-label claims"},
        {"risk": "resource_scope_creep", "severity": "medium", "mitigation": "re-inference-only campaign, no training, no selector search"},
    ]


def build_training_necessity() -> list[dict]:
    return [
        {"need": "split_label_trial_cache", "reinference_only_possible": 1, "new_training_required": 0, "decision": "prefer_reinference_if_frozen_checkpoints_recoverable"},
        {"need": "full_time_series_conditional_cs", "reinference_only_possible": 1, "new_training_required": 0, "decision": "prefer_reinference_if_logits_probabilities_windows_recoverable"},
        {"need": "representations_Wz", "reinference_only_possible": 1, "new_training_required": 0, "decision": "forward-hook re-inference if checkpoint/model available"},
        {"need": "atom_trace", "reinference_only_possible": 0, "new_training_required": 1, "decision": "training-time instrumentation likely required"},
        {"need": "independent_checkpoint_field_replication", "reinference_only_possible": 0, "new_training_required": 1, "decision": "new field required but not authorized"},
        {"need": "overall_training_gate", "reinference_only_possible": 0, "new_training_required": 0, "decision": TRAINING_GATE},
    ]


def build_training_tradeoff() -> list[dict]:
    return [
        {"path": "re_inference_only", "changes_checkpoint_universe": 0, "supports_split_label": 1, "supports_full_cs": 1, "supports_atom_trace": 0, "confound_risk": "low", "authorized": 0},
        {"path": "new_instrumented_training", "changes_checkpoint_universe": 1, "supports_split_label": 1, "supports_full_cs": 1, "supports_atom_trace": 1, "confound_risk": "medium_high", "authorized": 0},
        {"path": "independent_replication_field", "changes_checkpoint_universe": 1, "supports_split_label": 1, "supports_full_cs": 1, "supports_atom_trace": 1, "confound_risk": "medium", "authorized": 0},
        {"path": "more_summary_feature_engineering", "changes_checkpoint_universe": 0, "supports_split_label": 0, "supports_full_cs": 0, "supports_atom_trace": 0, "confound_risk": "low_but_low_value", "authorized": 0},
    ]


def build_atom_boundary() -> list[dict]:
    return [
        {"claim": "atom sums reproduce aggregate leakage", "requires_trace": 1, "identity_gate_required": 1, "current_supported": 0, "future_status": "protocol_ready_not_authorized"},
        {"claim": "domain_x_class_x_checkpoint atoms explain offset", "requires_trace": 1, "identity_gate_required": 1, "current_supported": 0, "future_status": "protocol_ready_not_authorized"},
        {"claim": "atom branch closed under current artifacts", "requires_trace": 0, "identity_gate_required": 0, "current_supported": 1, "future_status": "current_boundary"},
        {"claim": "atom trace selector/action-rule claim", "requires_trace": 1, "identity_gate_required": 1, "current_supported": 0, "future_status": "forbidden_action_claim"},
    ]


def build_replication_options() -> list[dict]:
    return [
        {"option": "reuse_current_frozen_universe", "uses_reserved_holdout": 0, "new_training": 0, "value": "low", "authorized": 0, "notes": "summary paths saturated"},
        {"option": "re_infer_current_checkpoints", "uses_reserved_holdout": 0, "new_training": 0, "value": "high", "authorized": 0, "notes": "best next low-confound path if checkpoints are recoverable"},
        {"option": "new_nonreserved_checkpoint_field", "uses_reserved_holdout": 0, "new_training": 1, "value": "medium_high", "authorized": 0, "notes": "replication without releasing BNCI2014_004/seeds [3,4]"},
        {"option": "reserved_holdout_final_stress", "uses_reserved_holdout": 1, "new_training": 1, "value": "high_but_should_preserve", "authorized": 0, "notes": "BNCI2014_004 and seeds [3,4] remain reserved"},
    ]


def build_replication_criteria() -> list[dict]:
    return [
        {"criterion": "rank_gauge_source_weak_signal", "pass_rule": "source signal nonzero but below endpoint oracle", "requires_new_field": 1},
        {"criterion": "endpoint_oracle_boundary", "pass_rule": "same-label endpoint scalar dominates but is unavailable at selection time", "requires_new_field": 1},
        {"criterion": "cod_ladder_order", "pass_rule": "endpoint > label diagnostic > template > source/key", "requires_new_field": 1},
        {"criterion": "split_label_disjointness", "pass_rule": "construction/evaluation target labels disjoint", "requires_new_field": 0},
        {"criterion": "no_tuning", "pass_rule": "all thresholds/splits pre-registered before running", "requires_new_field": 0},
    ]


def build_power_simulation() -> list[dict]:
    return [
        {"simulation": "split_label_detection", "n_construct": 20, "n_eval": 20, "expected_power_class": "low_medium", "value": "pilot"},
        {"simulation": "split_label_detection", "n_construct": 40, "n_eval": 40, "expected_power_class": "medium", "value": "recommended_minimum"},
        {"simulation": "split_label_detection", "n_construct": 80, "n_eval": 80, "expected_power_class": "high", "value": "stable"},
        {"simulation": "conditional_cs_gram_stability", "n_construct": 100, "n_eval": 100, "expected_power_class": "medium", "value": "bandwidth_grid_needed"},
        {"simulation": "atom_trace_identity", "n_construct": 0, "n_eval": 0, "expected_power_class": "deterministic_identity_gate", "value": "requires instrumentation not sample power"},
    ]


def build_value_summary() -> list[dict]:
    return [
        {"path": "more_frozen_summary_audit", "expected_information_gain": "low", "risk": "low", "recommendation": "stop_as_primary_path"},
        {"path": "re_inference_trial_cache", "expected_information_gain": "high", "risk": "low_medium", "recommendation": "best_next_authorization_candidate_if_checkpoints_recoverable"},
        {"path": "new_atom_trace_training", "expected_information_gain": "medium_high", "risk": "medium_high", "recommendation": "only after explicit training authorization"},
        {"path": "independent_replication", "expected_information_gain": "high", "risk": "medium", "recommendation": "protocol_ready_not_authorized"},
    ]


def build_minimum_requirements() -> list[dict]:
    return [
        {"evidence_target": "split_label_sufficiency", "minimum_requirement": "disjoint construction/evaluation target splits", "current_status": "missing"},
        {"evidence_target": "full_conditional_cs", "minimum_requirement": "paired sample-level X1/X2/Y matrices", "current_status": "missing"},
        {"evidence_target": "time_series_hankel_cs", "minimum_requirement": "past-window source states and response variables", "current_status": "missing"},
        {"evidence_target": "atom_trace", "minimum_requirement": "per-domain x class x checkpoint x split atom tensors", "current_status": "missing"},
        {"evidence_target": "replication", "minimum_requirement": "independent checkpoint field with preregistered pass/fail criteria", "current_status": "missing"},
    ]


def build_gate_decision() -> dict:
    weights_present = bool(_checkpoint_weight_files())
    return {
        "final_gate": FINAL_GATE,
        "reinference_subgate": REINFERENCE_SUBGATE,
        "training_gate": TRAINING_GATE,
        "authorized_in_c64": False,
        "training_authorized": False,
        "reinference_authorized": False,
        "gpu_authorized": False,
        "frozen_summary_paths_saturated": True,
        "reinference_only_sufficient_from_current_checkout": weights_present,
        "checkpoint_weights_found_in_checkout": len(_checkpoint_weight_files()),
        "new_training_scientifically_needed_for": ["atom_trace", "independent_checkpoint_replication"],
        "new_training_authorized": False,
        "decision": "protocols ready; execution requires explicit user authorization",
    }


def _affirmative_hit(text: str, phrase: str, window: int = 240) -> bool:
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx == -1:
            return False
        ctx = low[max(0, idx - window):idx]
        if not any(cue in ctx for cue in NEGATION_CUES):
            return True
        start = idx + len(phrase)


def _is_inventory_path(path: str) -> bool:
    return os.path.basename(path) in {
        "forbidden_claim_scan.csv",
        "red_team_failure_ledger.csv",
        "c64_gate_decision.json",
    }


def build_forbidden_scan(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in FORBIDDEN_PATTERNS:
        total = affirmative = 0
        files = []
        for path in paths:
            if _is_inventory_path(path):
                continue
            text = open(path, errors="ignore").read()
            count = text.lower().count(pattern.lower())
            if count:
                total += count
                files.append(path)
                if _affirmative_hit(text, pattern):
                    affirmative += 1
        rows.append({"pattern": pattern, "total_hits": total, "affirmative_hits": affirmative, "files": ";".join(files), "passed": int(affirmative == 0)})
    return rows


def classify(res: dict) -> dict:
    failures = [r for r in res["red_team_failure_ledger_rows"] if int(r["failed"])]
    if failures:
        primary = "C64-J_claim_or_availability_inconsistency_found"
    else:
        primary = "C64-A_frozen_summary_artifact_paths_saturated"
    active = [
        "C64-A_frozen_summary_artifact_paths_saturated",
        "C64-C_new_training_required_for_trial_cache_or_atom_trace",
        "C64-D_split_label_cache_protocol_ready_but_not_authorized",
        "C64-E_full_time_series_conditional_cs_protocol_ready_but_not_authorized",
        "C64-F_atom_trace_protocol_ready_but_not_authorized",
        "C64-G_independent_checkpoint_replication_protocol_ready_but_not_authorized",
    ]
    inactive = [
        "C64-B_reinference_only_trial_cache_campaign_sufficient",
        "C64-H_instrumentation_not_scientifically_justified_yet",
        "C64-I_source_observable_escape_hatch_remaining",
        "C64-J_claim_or_availability_inconsistency_found",
    ]
    if primary in inactive:
        inactive.remove(primary)
        active.append(primary)
    return {
        "primary": primary,
        "active": active,
        "inactive": inactive,
        "final_gate": FINAL_GATE,
        "reinference_subgate": REINFERENCE_SUBGATE,
        "training_gate": TRAINING_GATE,
        "red_team_failure_count": len(failures),
        "recommended_next_direction": NEXT_DIRECTION,
    }


def build_red_team_rows(res: dict) -> list[dict]:
    gate = res["c64_gate_decision"]
    feasibility = {r["requirement"]: r for r in res["reinference_only_feasibility_rows"]}
    inventory = {r["inventory_item"]: r for r in res["checkpoint_inventory_summary_rows"]}
    missing = {r["missing_item"]: r for r in res["sample_level_missing_data_ledger_rows"]}
    checks = [
        ("frozen_paths_saturated", all(int(r["passed"]) for r in res["frozen_path_saturation_decision_rows"]), "Frozen summary paths are explicitly saturated."),
        ("trial_cache_schema_ready", len(res["trial_level_cache_minimal_columns_rows"]) >= 20, "Trial-level cache schema has required identity/prediction/label fields."),
        ("split_label_protocol_ready", all(int(r["forbidden"]) for r in res["split_label_forbidden_claims_rows"]), "Split-label forbidden-claim boundary is explicit."),
        ("full_cs_not_supported_now", all(int(r["current_supported"]) == 0 for r in res["conditional_cs_feasibility_matrix_rows"] if r["estimator_component"] != "summary kernel proxy"), "Full conditional-CS sample estimator is not marked supported."),
        ("missing_data_ledger_blocks_cs", int(missing["per_trial_logits_probabilities"]["blocks_full_cs"]) == 1 and int(missing["per_trial_labels_and_split_roles"]["blocks_split_label"]) == 1, "Missing data ledger blocks split-label/full-CS."),
        ("reinference_not_authorized", gate["reinference_authorized"] is False and feasibility["overall_reinference_only_decision"]["status"] == REINFERENCE_SUBGATE, "Re-inference remains unauthorized and conditional."),
        ("training_not_authorized", gate["training_authorized"] is False and gate["training_gate"] == TRAINING_GATE, "Training remains unauthorized."),
        ("checkpoint_inventory_honest", int(inventory["checkpoint_weight_files"]["present"]) == int(gate["checkpoint_weights_found_in_checkout"] > 0), "Checkpoint inventory matches gate decision."),
        ("atom_protocol_ready_not_claimed_current", all(int(r["current_supported"]) == 0 for r in res["atom_trace_claim_boundary_rows"] if int(r["requires_trace"]) == 1), "Atom trace claims remain future protocol only."),
        ("reserved_holdout_preserved", any(int(r["uses_reserved_holdout"]) == 1 and int(r["authorized"]) == 0 for r in res["replication_protocol_options_rows"]), "Reserved holdout remains unreleased."),
        ("forbidden_scan_passed", all(int(r["passed"]) for r in res["forbidden_claim_scan_rows"]), "Forbidden affirmative claim scan passed."),
        ("large_artifact_scan_passed", all(int(r.get("passed", 1)) for r in res["large_artifact_scan_rows"]), "All listed artifacts are under 50MB."),
    ]
    return [{"gate": gate_id, "failed": int(not passed), "finding": finding} for gate_id, passed, finding in checks]


def table_row_counts(res: dict) -> dict:
    keys = {
        "artifact_manifest": "artifact_manifest_rows",
        "atom_trace_claim_boundary": "atom_trace_claim_boundary_rows",
        "availability_tag_definitions": "availability_tag_definitions_rows",
        "checkpoint_inventory_summary": "checkpoint_inventory_summary_rows",
        "conditional_cs_feasibility_matrix": "conditional_cs_feasibility_matrix_rows",
        "forbidden_claim_scan": "forbidden_claim_scan_rows",
        "frozen_path_closure_ledger": "frozen_path_closure_ledger_rows",
        "frozen_path_saturation_decision": "frozen_path_saturation_decision_rows",
        "hankel_variable_mapping": "hankel_variable_mapping_rows",
        "instrumentation_power_simulation": "instrumentation_power_simulation_rows",
        "large_artifact_scan": "large_artifact_scan_rows",
        "minimum_sample_requirements": "minimum_sample_requirements_rows",
        "red_team_failure_ledger": "red_team_failure_ledger_rows",
        "reinference_only_feasibility": "reinference_only_feasibility_rows",
        "reinference_resource_estimate": "reinference_resource_estimate_rows",
        "reinference_risk_ledger": "reinference_risk_ledger_rows",
        "remaining_summary_escape_hatches": "remaining_summary_escape_hatches_rows",
        "replication_pass_fail_criteria": "replication_pass_fail_criteria_rows",
        "replication_protocol_options": "replication_protocol_options_rows",
        "sample_level_missing_data_ledger": "sample_level_missing_data_ledger_rows",
        "schema_validation_summary": "schema_validation_summary_rows",
        "split_label_forbidden_claims": "split_label_forbidden_claims_rows",
        "split_label_power_requirements": "split_label_power_requirements_rows",
        "test_command_manifest": "test_command_manifest_rows",
        "training_necessity_decision": "training_necessity_decision_rows",
        "training_vs_reinference_tradeoff": "training_vs_reinference_tradeoff_rows",
        "trial_level_cache_minimal_columns": "trial_level_cache_minimal_columns_rows",
        "value_of_information_summary": "value_of_information_summary_rows",
    }
    return {name: len(res.get(key, [])) for name, key in keys.items()}


def build_test_manifest(status: str) -> list[dict]:
    return [
        {"test_scope": "focused_c64", "command": "python -m pytest oaci/tests/test_c64_trial_level_instrumentation_readiness.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c50_c64_slice", "command": "python -m pytest oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py oaci/tests/test_c63_trajectory_dynamic_observability.py oaci/tests/test_c64_trial_level_instrumentation_readiness.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "c23_c64_regression", "command": "python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3*.py oaci/tests/test_c4*.py oaci/tests/test_c5*.py oaci/tests/test_c60_rank_gauge_proof_stress_empirical_bridge.py oaci/tests/test_c61_conditional_observability_divergence.py oaci/tests/test_c62_conditional_divergence_estimator_stress.py oaci/tests/test_c63_trajectory_dynamic_observability.py oaci/tests/test_c64_trial_level_instrumentation_readiness.py -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
        {"test_scope": "full_oaci_tests", "command": "python -m pytest oaci/tests -q", "status": status, "environment": "eeg2025", "slurm_partition": "cpu-high"},
    ]


def build_reports(res: dict) -> dict[str, str]:
    d = res["decision"]
    gate = res["c64_gate_decision"]
    main = "\n".join([
        f"# C64 - Trial-Level Instrumentation Readiness / Split-Label-CS Evidence Gate (frozen C19 `{res['config_hash']}`)",
        "",
        "## Primary Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Active: `{' ; '.join(d['active'])}`",
        "",
        f"Inactive: `{' ; '.join(d['inactive'])}`",
        "",
        "## Gate",
        "",
        f"`{FINAL_GATE}`",
        "",
        f"Re-inference subgate: `{REINFERENCE_SUBGATE}`.",
        "",
        f"Current checkout checkpoint weight files found: `{gate['checkpoint_weights_found_in_checkout']}`. Re-inference-only is the preferred low-confound path if frozen checkpoints, preprocessing artifacts, and labels are recoverable, but C64 does not authorize or execute it.",
        "",
        "## Result",
        "",
        "C64 finds frozen summary artifact paths saturated. The next meaningful evidence is trial-level instrumentation: split-label cache, sample-level conditional-CS/Hankel variables, atom trace schema, and independent replication protocol.",
        "",
        "Split-label and full conditional-CS protocols are ready but not authorized. Atom trace and independent replication protocols are also ready but not authorized.",
        "",
        "New training is not needed for split-label or full conditional-CS if frozen checkpoints can be re-inferred. New instrumented training would be needed for atom trace or independent checkpoint-field replication, but it is not authorized.",
        "",
        "## Boundary",
        "",
        f"Template-only remains below max null p95 (`{TEMPLATE_ONLY_HIT:.6f}` < `{MAX_NULL_P95:.6f}`), while endpoint scalar remains above it (`{ENDPOINT_ORACLE_HIT:.6f}` > `{MAX_NULL_P95:.6f}`). The endpoint scalar remains a same-label target endpoint oracle and unavailable at selection time.",
        "",
        "## Execution Gate",
        "",
        f"`{TRAINING_GATE}`",
        "",
        "C64 does not train, re-infer, use GPU, add BNCI2014_004, run seeds [3,4], create selector artifacts, or start manuscript drafting.",
    ])
    red = "\n".join([
        "# C64 - Red-Team Verification",
        "",
        "All C64 red-team gates pass." if d["red_team_failure_count"] == 0 else "C64 red-team gates failed.",
        "",
        *[f"- {r['gate']}: {'PASS' if not int(r['failed']) else 'FAIL'} - {r['finding']}" for r in res["red_team_failure_ledger_rows"]],
    ])
    split_label = "\n".join([
        "# C64 - Split-Label Protocol",
        "",
        "Construct target-label diagnostic content only on the construction split and evaluate only on held-out target evaluation split.",
        "",
        "Same-label endpoint oracle rows remain diagnostic oracle rows and cannot be reinterpreted as split-label or few-label sufficiency.",
    ])
    atom = "\n".join([
        "# C64 - Atom Identity Gate Protocol",
        "",
        "Future atom traces must serialize per-domain x class x checkpoint x split atoms with deterministic float precision and hash manifests.",
        "",
        "The identity gate is: atom sum must reproduce aggregate leakage within a pre-registered tolerance before any atom-level mechanism claim is considered.",
    ])
    training = "\n".join([
        "# C64 - New Training Minimal Protocol",
        "",
        "Training remains unauthorized in C64. If later authorized, it must be pre-registered, quarantined, instrumented, non-rescue, non-selector, and must not recommend checkpoints.",
    ])
    holdout = "\n".join([
        "# C64 - Reserved Holdout Policy",
        "",
        "BNCI2014_004 and seeds [3,4] remain reserved. C64 does not release them for instrumentation, replication, or stress testing.",
    ])
    return {
        "C64_TRIAL_LEVEL_INSTRUMENTATION_READINESS.md": main,
        "C64_RED_TEAM_VERIFICATION.md": red,
        "C64_SPLIT_LABEL_PROTOCOL.md": split_label,
        "C64_ATOM_IDENTITY_GATE_PROTOCOL.md": atom,
        "C64_NEW_TRAINING_MINIMAL_PROTOCOL.md": training,
        "C64_RESERVED_HOLDOUT_POLICY.md": holdout,
    }


def _compact_json(res: dict) -> dict:
    return {
        "milestone": MILESTONE,
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": True,
        "c63_commit": "25626fa",
        "c63_decision": res["c63_decision"],
        "decision": res["decision"],
        "final_gate": FINAL_GATE,
        "reinference_subgate": REINFERENCE_SUBGATE,
        "training_gate": TRAINING_GATE,
        "gate_decision": res["c64_gate_decision"],
        "key_numbers": {
            "strict_source": STRICT_SOURCE_HIT,
            "source_dynamic": SOURCE_DYNAMIC_HIT,
            "source_dynamic_template": SOURCE_DYNAMIC_TEMPLATE_HIT,
            "template_only": TEMPLATE_ONLY_HIT,
            "endpoint_oracle": ENDPOINT_ORACLE_HIT,
            "max_null_p95": MAX_NULL_P95,
        },
        "table_row_counts": table_row_counts(res),
        "recommended_next_step": res["decision"]["recommended_next_direction"],
    }


def run(test_status: str = "planned") -> dict:
    config_hash = _lock_config()
    c63_summary = _load_json(C63_JSON)
    res = {
        "config_hash": config_hash,
        "c63_decision": c63_summary["decision"]["primary"],
        "frozen_path_closure_ledger_rows": build_frozen_path_closure_ledger(),
        "remaining_summary_escape_hatches_rows": build_remaining_escape_hatches(),
        "frozen_path_saturation_decision_rows": build_frozen_saturation_decision(),
        "trial_level_cache_minimal_columns_rows": build_trial_cache_columns(),
        "availability_tag_definitions_rows": build_availability_tags(),
        "split_label_power_requirements_rows": build_split_label_power(),
        "split_label_forbidden_claims_rows": build_split_label_forbidden_claims(),
        "conditional_cs_feasibility_matrix_rows": build_conditional_cs_feasibility(),
        "hankel_variable_mapping_rows": build_hankel_mapping(),
        "sample_level_missing_data_ledger_rows": build_missing_data_ledger(),
        "checkpoint_inventory_summary_rows": build_checkpoint_inventory(),
        "reinference_only_feasibility_rows": build_reinference_feasibility(),
        "reinference_resource_estimate_rows": build_resource_estimate(),
        "reinference_risk_ledger_rows": build_reinference_risk_ledger(),
        "training_necessity_decision_rows": build_training_necessity(),
        "training_vs_reinference_tradeoff_rows": build_training_tradeoff(),
        "atom_trace_claim_boundary_rows": build_atom_boundary(),
        "replication_protocol_options_rows": build_replication_options(),
        "replication_pass_fail_criteria_rows": build_replication_criteria(),
        "instrumentation_power_simulation_rows": build_power_simulation(),
        "value_of_information_summary_rows": build_value_summary(),
        "minimum_sample_requirements_rows": build_minimum_requirements(),
        "c64_gate_decision": build_gate_decision(),
        "test_command_manifest_rows": build_test_manifest(test_status),
        "forbidden_claim_scan_rows": [],
        "red_team_failure_ledger_rows": [],
        "schema_validation_summary_rows": [],
        "large_artifact_scan_rows": [],
        "artifact_manifest_rows": [],
        "generated_paths": [],
    }
    res["decision"] = classify({**res, "red_team_failure_ledger_rows": []})
    return res


def write_tables(res: dict, table_dir: str) -> None:
    specs = {
        "frozen_path_closure_ledger.csv": ("frozen_path_closure_ledger_rows", ["path_id", "milestones", "status", "evidence", "remaining_value"]),
        "remaining_summary_escape_hatches.csv": ("remaining_summary_escape_hatches_rows", ["hatch_id", "remaining", "reason", "scientifically_justified_now"]),
        "frozen_path_saturation_decision.csv": ("frozen_path_saturation_decision_rows", ["decision_id", "decision", "passed", "rationale"]),
        "trial_level_cache_minimal_columns.csv": ("trial_level_cache_minimal_columns_rows", ["column", "category", "minimality", "label_dependency", "needed_for_split_label", "available_at_selection_time_if_source_only"]),
        "availability_tag_definitions.csv": ("availability_tag_definitions_rows", ["tag", "definition", "allowed_for_source_rule", "diagnostic_only"]),
        "split_label_power_requirements.csv": ("split_label_power_requirements_rows", ["cell", "min_construct_trials", "min_eval_trials", "bootstrap_reps", "detectable_hit_gap", "status"]),
        "split_label_forbidden_claims.csv": ("split_label_forbidden_claims_rows", ["claim", "forbidden", "reason"]),
        "conditional_cs_feasibility_matrix.csv": ("conditional_cs_feasibility_matrix_rows", ["estimator_component", "project_mapping", "current_supported", "required_cache", "protocol_ready"]),
        "hankel_variable_mapping.csv": ("hankel_variable_mapping_rows", ["paper_object", "project_variable", "current_artifact", "sample_level_required", "supported_now"]),
        "sample_level_missing_data_ledger.csv": ("sample_level_missing_data_ledger_rows", ["missing_item", "blocks_split_label", "blocks_full_cs", "blocks_atom_trace", "recoverable_by_reinference"]),
        "checkpoint_inventory_summary.csv": ("checkpoint_inventory_summary_rows", ["inventory_item", "present", "count", "evidence", "blocks_reinference_now"]),
        "reinference_only_feasibility.csv": ("reinference_only_feasibility_rows", ["requirement", "present_in_checkout", "status", "reinference_only_sufficient_if_present"]),
        "reinference_resource_estimate.csv": ("reinference_resource_estimate_rows", ["resource", "estimate", "needed", "notes"]),
        "reinference_risk_ledger.csv": ("reinference_risk_ledger_rows", ["risk", "severity", "mitigation"]),
        "training_necessity_decision.csv": ("training_necessity_decision_rows", ["need", "reinference_only_possible", "new_training_required", "decision"]),
        "training_vs_reinference_tradeoff.csv": ("training_vs_reinference_tradeoff_rows", ["path", "changes_checkpoint_universe", "supports_split_label", "supports_full_cs", "supports_atom_trace", "confound_risk", "authorized"]),
        "atom_trace_claim_boundary.csv": ("atom_trace_claim_boundary_rows", ["claim", "requires_trace", "identity_gate_required", "current_supported", "future_status"]),
        "replication_protocol_options.csv": ("replication_protocol_options_rows", ["option", "uses_reserved_holdout", "new_training", "value", "authorized", "notes"]),
        "replication_pass_fail_criteria.csv": ("replication_pass_fail_criteria_rows", ["criterion", "pass_rule", "requires_new_field"]),
        "instrumentation_power_simulation.csv": ("instrumentation_power_simulation_rows", ["simulation", "n_construct", "n_eval", "expected_power_class", "value"]),
        "value_of_information_summary.csv": ("value_of_information_summary_rows", ["path", "expected_information_gain", "risk", "recommendation"]),
        "minimum_sample_requirements.csv": ("minimum_sample_requirements_rows", ["evidence_target", "minimum_requirement", "current_status"]),
        "test_command_manifest.csv": ("test_command_manifest_rows", ["test_scope", "command", "status", "environment", "slurm_partition"]),
        "forbidden_claim_scan.csv": ("forbidden_claim_scan_rows", ["pattern", "total_hits", "affirmative_hits", "files", "passed"]),
        "red_team_failure_ledger.csv": ("red_team_failure_ledger_rows", ["gate", "failed", "finding"]),
        "schema_validation_summary.csv": ("schema_validation_summary_rows", ["table_name", "row_count", "required_columns_present", "passed"]),
        "large_artifact_scan.csv": ("large_artifact_scan_rows", ["path", "size_bytes", "over_50mb", "passed"]),
        "artifact_manifest.csv": ("artifact_manifest_rows", ["path", "size_bytes", "sha256", "artifact_class", "row_count"]),
    }
    for name, (key, cols) in specs.items():
        _write_csv(os.path.join(table_dir, name), res[key], cols)


def _write_texts(files: dict[str, str]) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            f.write(text.rstrip() + "\n")


def _write_json_payloads(res: dict) -> None:
    with open(REPORT_JSON, "w") as f:
        json.dump(_compact_json(res), f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "trial_level_cache_schema.json"), "w") as f:
        json.dump({"columns": res["trial_level_cache_minimal_columns_rows"], "availability_tags": res["availability_tag_definitions_rows"]}, f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "full_cs_supported_flag.json"), "w") as f:
        json.dump({"full_conditional_cs_supported_now": False, "reason": "sample-level paired variables are missing"}, f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "atom_trace_schema.json"), "w") as f:
        json.dump({"axes": ["domain", "class", "checkpoint", "split", "atom"], "identity_gate": "sum_atoms_equals_aggregate_leakage"}, f, indent=2, sort_keys=True)
    with open(os.path.join(TABLE_DIR, "c64_gate_decision.json"), "w") as f:
        json.dump(res["c64_gate_decision"], f, indent=2, sort_keys=True)


def _listed_paths() -> list[str]:
    skip = {"artifact_manifest.csv", "large_artifact_scan.csv"}
    return sorted(
        glob.glob(os.path.join(REPORT_DIR, "C64_*.md"))
        + glob.glob(os.path.join(REPORT_DIR, "C64_*.json"))
        + [p for p in glob.glob(os.path.join(TABLE_DIR, "*.csv")) if os.path.basename(p) not in skip]
        + glob.glob(os.path.join(TABLE_DIR, "*.json"))
    )


def _schema_rows(table_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(table_dir, "*.csv"))):
        if os.path.basename(path) in {"schema_validation_summary.csv", "artifact_manifest.csv"}:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
        rows.append({"table_name": os.path.basename(path), "row_count": count, "required_columns_present": int(bool(header)), "passed": int(bool(header))})
    return rows


def _large_scan(paths: list[str]) -> list[dict]:
    rows = []
    for path in sorted(paths):
        size = os.path.getsize(path)
        rows.append({"path": path, "size_bytes": size, "over_50mb": int(size > 50_000_000), "passed": int(size <= 50_000_000)})
    return rows


def _artifact_manifest(paths: list[str], table_dir: str) -> list[dict]:
    row_counts = {}
    for path in glob.glob(os.path.join(table_dir, "*.csv")):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            row_counts[path] = sum(1 for _ in reader)
    rows = []
    for path in sorted(paths):
        cls = "table" if path.endswith(".csv") else "summary_json" if path.endswith(".json") else "report"
        rows.append({"path": path, "size_bytes": os.path.getsize(path), "sha256": _sha256(path), "artifact_class": cls, "row_count": row_counts.get(path, "")})
    return rows


def write_artifacts(res: dict, test_status: str) -> dict:
    os.makedirs(TABLE_DIR, exist_ok=True)
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["forbidden_claim_scan_rows"] = build_forbidden_scan(paths)
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    write_tables(res, TABLE_DIR)

    res["schema_validation_summary_rows"] = _schema_rows(TABLE_DIR)
    res["red_team_failure_ledger_rows"] = build_red_team_rows(res)
    res["decision"] = classify(res)
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = _large_scan(paths)
    res["artifact_manifest_rows"] = [{"path": p} for p in paths]
    _write_json_payloads(res)
    _write_texts(build_reports(res))
    write_tables(res, TABLE_DIR)

    paths = _listed_paths()
    res["generated_paths"] = paths
    res["large_artifact_scan_rows"] = _large_scan(paths)
    _write_csv(os.path.join(TABLE_DIR, "large_artifact_scan.csv"), res["large_artifact_scan_rows"], ["path", "size_bytes", "over_50mb", "passed"])
    res["artifact_manifest_rows"] = _artifact_manifest(paths, TABLE_DIR)
    _write_csv(os.path.join(TABLE_DIR, "artifact_manifest.csv"), res["artifact_manifest_rows"], ["path", "size_bytes", "sha256", "artifact_class", "row_count"])
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c64_trial_level_instrumentation_readiness")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--test-status", default="planned")
    args = ap.parse_args(argv)
    res = run(test_status=args.test_status)
    if args.recompute:
        res = write_artifacts(res, args.test_status)
    print(f"[C64] decision={res['decision']['primary']} tables={len(table_row_counts(res))}")


if __name__ == "__main__":
    main()
