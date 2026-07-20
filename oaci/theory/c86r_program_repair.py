"""C86R metadata-only program reconciliation.

This module materializes the additive C86R registries. It reads committed
protocol metadata and pure C84 subject registries only. It has no EEG, label,
training, forward, active-acquisition, or execution-lock entrypoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from oaci.multidataset.c84_dataset_registry import DATASETS as C84_DATASETS
from oaci.multidataset.c84_dataset_registry import partition_subjects


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c86r_tables"
REPAIR_PROTOCOL = REPORT_DIR / "C86R_ELIGIBILITY_BASELINE_AND_DEVELOPMENT_VIEW_REPAIR_PROTOCOL.json"
EFFECTIVE_MANIFEST = REPORT_DIR / "C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V2.json"
SYNTHETIC_V2 = REPORT_DIR / "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL_V2.json"

CREATED_AT_UTC = "2026-07-17T22:54:43Z"
REPAIR_PROTOCOL_COMMIT = "1ec508e64655edd71f278a0696c4c8d757700e29"
REPAIR_PROTOCOL_SHA256 = "d3550350e2e4f9ff300d03a03f9353fcef09d324b69af21608f5a6f6b45741d3"
C86R_IMPLEMENTATION_COMMIT = "ffe1fa01b6e9395c5fde0f120d348897a270898e"
FINAL_GATE = "C86_UNTOUCHED_COHORT_AGE_ACCESS_OR_INTERFACE_ELIGIBILITY_RECONCILIATION_REQUIRED"
TOTAL_QUERY_GRID: tuple[int | str, ...] = (4, 8, 16, 32, "FULL")

PROTOCOL_STACK: tuple[tuple[str, str, str], ...] = (
    (
        "C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json",
        "430b7b59",
        "d4feac535f8c1144a55d77cd7f322ae961d5c7d5a899dfd15c371484d88fbb7a",
    ),
    (
        "C86P_ACTIVE_ESTIMATOR_OPERATIONALIZATION_PROTOCOL.json",
        "a1a1e736",
        "0cdb05c113a1c681584dec907a002af809aa019c933e042f15065a4f30c1f1dd",
    ),
    (
        "C86P_UNTOUCHED_COHORT_VARIANT_ELIGIBILITY_CORRECTION_PROTOCOL.json",
        "d89ec8d1",
        "5948b76a2d08c45c88e157aace1cc421a8c551b1c763a265376ad25921103c0d",
    ),
    (
        "C86P_HISTORICAL_ACCESS_ELIGIBILITY_CORRECTION_PROTOCOL.json",
        "7e19c99f",
        "fd7e214c6c6675b2a9b071b2bf278e2c4393495b2f4cbe03f82528c3098e7064",
    ),
    (
        "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL.json",
        "0217485f",
        "a80e8cca75eaa4d22b374794c06a9304ef9bb21605ec75f5d6aa53509f86b54b",
    ),
    (
        "C86R_ELIGIBILITY_BASELINE_AND_DEVELOPMENT_VIEW_REPAIR_PROTOCOL.json",
        REPAIR_PROTOCOL_COMMIT,
        REPAIR_PROTOCOL_SHA256,
    ),
)

COMMON_UNTOUCHED_22 = (
    "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "C3", "Cz",
    "C4", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz",
)
COMMON_SOURCE_TARGET_11 = (
    "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
)


class C86RContractError(RuntimeError):
    """Raised when an additive C86R contract does not replay exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C86RContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    _require(values, f"refusing empty C86R table: {path.name}")
    fields = list(values[0])
    _require(all(list(row) == fields for row in values), f"schema drift in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    return sha256_file(path)


def hara_general_k_score(expected_absolute_pairwise_losses: np.ndarray) -> np.ndarray:
    """Return Hara et al.'s general-K sum over unordered model pairs."""
    values = np.asarray(expected_absolute_pairwise_losses, dtype=np.float64)
    _require(values.ndim == 3 and values.shape[1] == values.shape[2], "pairwise tensor must be N x K x K")
    _require(values.shape[1] >= 2, "Hara score requires at least two candidates")
    _require(np.all(np.isfinite(values)) and np.all(values >= 0), "invalid pairwise expectation")
    _require(np.max(np.abs(values - np.swapaxes(values, 1, 2))) <= 1e-15, "pairwise tensor is not symmetric")
    _require(np.max(np.abs(np.diagonal(values, axis1=1, axis2=2))) <= 1e-15, "pairwise diagonal must be zero")
    upper = np.triu_indices(values.shape[1], k=1)
    return np.sum(values[:, upper[0], upper[1]], axis=1, dtype=np.float64)


def project_max_pair_score(expected_absolute_pairwise_losses: np.ndarray) -> np.ndarray:
    """Return the preserved project max-pair heuristic, which is not Hara."""
    values = np.asarray(expected_absolute_pairwise_losses, dtype=np.float64)
    _require(values.ndim == 3 and values.shape[1] == values.shape[2], "pairwise tensor must be N x K x K")
    _require(np.all(np.isfinite(values)) and np.all(values >= 0), "invalid pairwise expectation")
    upper = np.triu_indices(values.shape[1], k=1)
    return np.max(values[:, upper[0], upper[1]], axis=1)


def adult_eligibility(*, minimum_age: float | None, explicit_all_at_least_18: bool) -> str:
    """Apply the locked adults-only rule without imputing from a mean age."""
    if explicit_all_at_least_18 or (minimum_age is not None and minimum_age >= 18.0):
        return "ADULT_ELIGIBILITY_PROVEN_PASS"
    return "AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED"


def _protocol_stack_rows() -> list[dict[str, Any]]:
    rows = []
    for precedence, (name, commit, expected) in enumerate(PROTOCOL_STACK, start=1):
        path = REPORT_DIR / name
        observed = sha256_file(path)
        _require(observed == expected, f"historical protocol drift: {name}")
        rows.append({
            "precedence_low_to_high": precedence,
            "object": name,
            "commit": commit,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "historical_object_modified": 0,
            "operative_role": "ADDITIVE_REPAIR" if name.startswith("C86R_") else "PRESERVED_INPUT",
            "downstream_parent_only_sufficient": 0,
        })
    rows.append({
        "precedence_low_to_high": len(rows) + 1,
        "object": "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL_V2.json",
        "commit": C86R_IMPLEMENTATION_COMMIT,
        "expected_sha256": "GENERATED_AFTER_METHOD_REGISTRY_V2",
        "observed_sha256": "GENERATED_AFTER_METHOD_REGISTRY_V2",
        "historical_object_modified": 0,
        "operative_role": "ADDITIVE_SYNTHETIC_V2",
        "downstream_parent_only_sufficient": 0,
    })
    return rows


def _age_rows() -> list[dict[str, Any]]:
    return [
        {
            "cohort": "Brandl2020", "subjects": 16, "public_age_evidence": "explicit_age_range_22_to_30",
            "minimum_age": "22", "maximum_age": "30", "subject_level_all_adult_proven": 1,
            "evidence_url": "https://doi.org/10.3389/fnins.2020.566147",
            "loader_sha256": "313c912ac42ea89d0a176a7baa972330067b6ca559cdef81acde074757416edd",
            "decision": adult_eligibility(minimum_age=22.0, explicit_all_at_least_18=True),
            "confirmation_role_v2": "PRIMARY_UNTOUCHED_COHORT",
        },
        {
            "cohort": "Kumar2024", "subjects": 18,
            "public_age_evidence": "mean_23.22_SD_3.59_only;public_BIDS_participants_age_values_invalid_2020",
            "minimum_age": "NOT_PROVEN", "maximum_age": "NOT_PROVEN", "subject_level_all_adult_proven": 0,
            "evidence_url": "https://academic.oup.com/pnasnexus/article/3/2/pgae076/7609232;https://github.com/nemarDatasets/nm000177/blob/main/participants.tsv",
            "loader_sha256": "b06d387564032456eb44c2bc24787d09fab4d5be3ae5d755b0178e190c07989d",
            "decision": adult_eligibility(minimum_age=None, explicit_all_at_least_18=False),
            "confirmation_role_v2": "AGE_UNCERTAIN_STRESS_TRACK_ONLY",
        },
        {
            "cohort": "Yang2025_2C", "subjects": 51,
            "public_age_evidence": "public_cohort_metadata_and_loader_report_age_min_17_max_30",
            "minimum_age": "17", "maximum_age": "30", "subject_level_all_adult_proven": 0,
            "evidence_url": "https://eegdash.org/api/dataset/eegdash.dataset.NM000348.html",
            "loader_sha256": "b6a1ea3ee0e415a43d6a4dc04c0d798160b9e740efedd08cf4fd82b35c35220f",
            "decision": adult_eligibility(minimum_age=17.0, explicit_all_at_least_18=False),
            "confirmation_role_v2": "AGE_MIXED_STRESS_TRACK_ONLY",
        },
    ]


def _cohort_rows() -> list[dict[str, Any]]:
    ages = {row["cohort"]: row for row in _age_rows()}
    rows = []
    for cohort, prior_role, access, interface in (
        ("Brandl2020", "C86P_PRIMARY", "CERTIFIABLY_UNTOUCHED", "PASS"),
        ("Kumar2024", "C86P_PRIMARY", "CERTIFIABLY_UNTOUCHED", "PASS"),
        ("Yang2025_2C", "C86P_PRIMARY", "CERTIFIABLY_UNTOUCHED", "PASS"),
        ("Dreyer2023", "C86P_REMOVED", "HISTORICAL_ACCESS_NOT_CERTIFIABLY_ABSENT", "NOT_REEVALUATED"),
    ):
        age = ages.get(cohort)
        adult = age["decision"] if age else "NOT_APPLICABLE_HISTORICAL_ACCESS_FAILURE"
        role = age["confirmation_role_v2"] if age else "C86D_DEVELOPMENT_ONLY"
        selected = int(role == "PRIMARY_UNTOUCHED_COHORT" and access == "CERTIFIABLY_UNTOUCHED")
        rows.append({
            "cohort": cohort, "historical_C86P_role": prior_role, "historical_access_status": access,
            "adult_evidence_status": adult, "metadata_interface_status": interface,
            "authoritative_C86R_role": role, "selected_primary_confirmation": selected,
            "primary_adult_cohort_count": 1, "minimum_primary_cohorts_required": 2,
            "at_least_two_primary_rule": "FAIL", "registered_gate": FINAL_GATE,
            "performance_outcome_used": 0, "adult_rule_relaxed": 0,
        })
    return rows


def _age_truth_rows() -> list[dict[str, Any]]:
    return [
        {"public_evidence": "explicit_minimum_age_at_least_18", "decision": "PASS", "primary_role_allowed": 1},
        {"public_evidence": "explicit_inclusion_requires_age_at_least_18", "decision": "PASS", "primary_role_allowed": 1},
        {"public_evidence": "mean_and_SD_only", "decision": "FAIL_CLOSED", "primary_role_allowed": 0},
        {"public_evidence": "range_includes_17", "decision": "FAIL_CLOSED", "primary_role_allowed": 0},
        {"public_evidence": "subject_table_missing_or_invalid", "decision": "FAIL_CLOSED", "primary_role_allowed": 0},
    ]


def _literature_rows() -> list[dict[str, Any]]:
    return [
        {"source_id": "Kossen2021", "year": 2021, "method": "Active_Testing", "primary_url": "https://proceedings.mlr.press/v139/kossen21a.html", "audited_through": "2026-07-18", "C86_relevance": "active_test_point_acquisition", "fidelity_limit": "A1_is_multicandidate_adaptation"},
        {"source_id": "Farquhar2021", "year": 2021, "method": "LURE", "primary_url": "https://openreview.net/forum?id=JiYq3eqTKY", "audited_through": "2026-07-18", "C86_relevance": "leveled_unbiased_linear_risk_estimation", "fidelity_limit": "unbiasedness_only_for_registered_linear_moments"},
        {"source_id": "Kossen2022", "year": 2022, "method": "ASE_XWED", "primary_url": "https://papers.neurips.cc/paper_files/paper/2022/file/9b9cfd5428153ccfbd4ba34b7e007305-Paper-Conference.pdf", "audited_through": "2026-07-18", "C86_relevance": "surrogate_interpolation_and_weighted_disagreement", "fidelity_limit": "single_model_risk_surrogate_not_81_action_plugin"},
        {"source_id": "Hara2024", "year": 2024, "method": "general_K_VMA", "primary_url": "https://link.springer.com/article/10.1007/s10994-024-06603-1", "audited_through": "2026-07-18", "C86_relevance": "sum_over_all_pairwise_expected_absolute_loss_differences", "fidelity_limit": "linear_loss_objective_only"},
        {"source_id": "Karimi2021", "year": 2021, "method": "Online_Active_Model_Selection", "primary_url": "https://proceedings.mlr.press/v130/reza-karimi21a.html", "audited_through": "2026-07-18", "C86_relevance": "streaming_selective_model_selection", "fidelity_limit": "streaming_setting_differs_from_fixed_finite_pool"},
        {"source_id": "Okanovic2025", "year": 2025, "method": "MODEL_SELECTOR", "primary_url": "https://proceedings.mlr.press/v258/okanovic25a.html", "audited_through": "2026-07-18", "C86_relevance": "pool_based_many_classifier_best_model_identification", "fidelity_limit": "published_accuracy_objective_differs_from_historical_composite"},
        {"source_id": "Kay2025", "year": 2025, "method": "CODA", "primary_url": "https://openaccess.thecvf.com/content/ICCV2025/html/Kay_Consensus-Driven_Active_Model_Selection_ICCV_2025_paper.html", "audited_through": "2026-07-18", "C86_relevance": "consensus_Bayesian_active_model_selection", "fidelity_limit": "published_accuracy_objective_differs_from_historical_composite"},
        {"source_id": "Ashouritaklimi2026", "year": 2026, "method": "PPAT", "primary_url": "https://openreview.net/forum?id=V6wIXluLmp", "audited_through": "2026-07-18", "C86_relevance": "prediction_powered_control_variate_for_active_testing", "fidelity_limit": "emerging_workshop_single_risk_object_no_frozen_81_action_dispatcher"},
    ]


def _method_rows() -> list[dict[str, Any]]:
    return [
        {"method_id": "P0", "name": "passive_uniform_without_replacement", "disposition": "PRIMARY_BASELINE", "query_score": "constant", "objective": "registered_linear_moments_then_composite_plugin", "interface_81_candidates": "EXACT", "production_dispatcher_id": "c86_active_v2:P0_uniform_without_replacement", "confirmation_eligible": 1},
        {"method_id": "P1", "name": "historical_class_stratified_Q0", "disposition": "SECONDARY_BASELINE", "query_score": "class_stratified_uniform", "objective": "historical_Q0", "interface_81_candidates": "EXACT_CLASS_ORACLE", "production_dispatcher_id": "c86_active_v2:P1_class_aware_secondary", "confirmation_eligible": 1},
        {"method_id": "A1", "name": "LURE_active_testing_multicandidate", "disposition": "PRIMARY_BASELINE", "query_score": "mean_expected_candidate_NLL", "objective": "linear_NLL_moments_then_composite_plugin", "interface_81_candidates": "ADAPTATION", "production_dispatcher_id": "c86_active_v2:A1_lure_active_testing", "confirmation_eligible": 1},
        {"method_id": "A2H", "name": "Hara_general_K_variance_minimizing", "disposition": "PRIMARY_BASELINE", "query_score": "sum_over_k_lt_kprime_E_pi_abs_loss_k_minus_loss_kprime", "objective": "linear_pairwise_NLL_differences", "interface_81_candidates": "EXACT_GENERAL_K_SCORE", "production_dispatcher_id": "c86_active_v2:A2H_hara_sum_pairwise", "confirmation_eligible": 1},
        {"method_id": "A2M", "name": "A2M_project_max_pair_heuristic", "disposition": "DEVELOPMENT_ONLY", "query_score": "max_over_pairs_E_pi_abs_loss_difference", "objective": "linear_pairwise_NLL_differences", "interface_81_candidates": "PROJECT_HEURISTIC", "production_dispatcher_id": "c86_development_v2:A2M_max_pair", "confirmation_eligible": 0},
        {"method_id": "A3D", "name": "project_vote_disagreement", "disposition": "DEVELOPMENT_ONLY", "query_score": "one_minus_max_vote_fraction", "objective": "prediction_disagreement", "interface_81_candidates": "PROJECT_ADAPTATION", "production_dispatcher_id": "c86_development_v2:A3D_vote_disagreement", "confirmation_eligible": 0},
        {"method_id": "A4", "name": "project_plausible_best_confidence_set", "disposition": "DEVELOPMENT_ONLY", "query_score": "max_pair_difference_within_plugin_confidence_set", "objective": "project_plugin", "interface_81_candidates": "PROJECT_HEURISTIC", "production_dispatcher_id": "c86_development_v2:A4_plausible_best", "confirmation_eligible": 0},
        {"method_id": "ASE_XWED", "name": "active_surrogate_estimator_XWED", "disposition": "DEVELOPMENT_ONLY", "query_score": "surrogate_weighted_disagreement", "objective": "single_model_risk_interpolation", "interface_81_candidates": "REQUIRES_NEW_MULTIACTION_SURROGATE", "production_dispatcher_id": "NOT_IMPLEMENTED", "confirmation_eligible": 0},
        {"method_id": "ONLINE_AMS", "name": "online_active_model_selection", "disposition": "DEVELOPMENT_ONLY", "query_score": "published_streaming_selective_query", "objective": "best_classifier_in_stream", "interface_81_candidates": "K_SUPPORTED_SETTING_MISMATCH", "production_dispatcher_id": "NOT_IMPLEMENTED", "confirmation_eligible": 0},
        {"method_id": "MODEL_SELECTOR", "name": "MODEL_SELECTOR", "disposition": "SECONDARY_BASELINE", "query_score": "published_pool_based_model_selector", "objective": "best_or_near_best_accuracy_model", "interface_81_candidates": "PREDICTION_INTERFACE_COMPATIBLE_OBJECTIVE_MISMATCH", "production_dispatcher_id": "c86_secondary_v2:MODEL_SELECTOR_fidelity_port", "confirmation_eligible": 1},
        {"method_id": "CODA", "name": "consensus_driven_active_model_selection", "disposition": "SECONDARY_BASELINE", "query_score": "published_consensus_Bayesian_score", "objective": "best_accuracy_model", "interface_81_candidates": "BINARY_PREDICTION_INTERFACE_COMPATIBLE_OBJECTIVE_MISMATCH", "production_dispatcher_id": "c86_secondary_v2:CODA_fidelity_port", "confirmation_eligible": 1},
        {"method_id": "PPAT", "name": "prediction_powered_active_testing", "disposition": "UNAVAILABLE_WITH_EXACT_REASON", "query_score": "prediction_powered_control_variate", "objective": "single_linear_risk", "interface_81_candidates": "NO_LOCKED_MULTIACTION_NONLINEAR_PLUGIN_RULE", "production_dispatcher_id": "UNAVAILABLE_EMERGING_WORKSHOP_METHOD", "confirmation_eligible": 0},
    ]


def _fidelity_rows() -> list[dict[str, Any]]:
    return [
        {"method_id": row["method_id"], "claimed_exact_reference": int(row["method_id"] in {"A2H"}),
         "faithful_object": "general_K_sum_over_pairs_query_score" if row["method_id"] == "A2H" else "NONE_OR_DISCLOSED_ADAPTATION",
         "historical_mislabel_corrected": int(row["method_id"] in {"A2H", "A2M"}),
         "C86H_code_must_freeze_before_access": int(row["confirmation_eligible"] == 1),
         "fidelity_status": (
             "QUERY_SCORE_EXACT_GENERAL_K_HARA" if row["method_id"] == "A2H" else
             "NOT_HARA_PROJECT_HEURISTIC" if row["method_id"] == "A2M" else
             row["interface_81_candidates"]
         )}
        for row in _method_rows()
    ]


def _baseline_availability_rows() -> list[dict[str, Any]]:
    return [
        {"method_id": row["method_id"], "disposition": row["disposition"],
         "implementation_available_at_C86R": int(row["method_id"] in {"P0", "P1", "A1", "A2H", "A2M", "A3D", "A4"}),
         "exact_reason_if_unavailable": (
             "NOT_APPLICABLE" if row["method_id"] in {"P0", "P1", "A1", "A2H", "A2M", "A3D", "A4"}
             else "FUTURE_FIDELITY_PORT_REQUIRED_BEFORE_CONFIRMATION_FREEZE"
             if row["method_id"] in {"MODEL_SELECTOR", "CODA"}
             else "SETTING_OR_OBJECTIVE_REQUIRES_NEW_DEVELOPMENT"
             if row["method_id"] in {"ASE_XWED", "ONLINE_AMS"}
             else "EMERGING_WORKSHOP_METHOD_NO_FROZEN_81_ACTION_PLUGIN_DISPATCHER"
         ),
         "may_be_silently_omitted": 0}
        for row in _method_rows()
    ]


def _queried_information_rows() -> list[dict[str, Any]]:
    return [
        {"object": "clipped_binary_NLL", "per_trial_revealed": 1, "linear_additive_moment": 1, "plugin_nonlinear": 0, "unbiased_LURE_claim": "REGISTERED_LINEAR_MEAN"},
        {"object": "correctness_indicator", "per_trial_revealed": 1, "linear_additive_moment": 1, "plugin_nonlinear": 0, "unbiased_LURE_claim": "REGISTERED_LINEAR_MEAN"},
        {"object": "class_indicator", "per_trial_revealed": 1, "linear_additive_moment": 1, "plugin_nonlinear": 0, "unbiased_LURE_claim": "REGISTERED_LINEAR_MEAN"},
        {"object": "confidence_bin_signed_contribution", "per_trial_revealed": 1, "linear_additive_moment": 1, "plugin_nonlinear": 0, "unbiased_LURE_claim": "SIGNED_BIN_MOMENT_ONLY"},
        {"object": "pairwise_NLL_difference", "per_trial_revealed": 1, "linear_additive_moment": 1, "plugin_nonlinear": 0, "unbiased_LURE_claim": "REGISTERED_LINEAR_DIFFERENCE"},
        {"object": "balanced_accuracy", "per_trial_revealed": 0, "linear_additive_moment": 0, "plugin_nonlinear": 1, "unbiased_LURE_claim": "NONE_RATIO_PLUGIN"},
        {"object": "ECE", "per_trial_revealed": 0, "linear_additive_moment": 0, "plugin_nonlinear": 1, "unbiased_LURE_claim": "NONE_ABSOLUTE_VALUE_PLUGIN"},
        {"object": "candidate_midranks", "per_trial_revealed": 0, "linear_additive_moment": 0, "plugin_nonlinear": 1, "unbiased_LURE_claim": "NONE_CROSS_CANDIDATE_PLUGIN"},
        {"object": "historical_composite_utility", "per_trial_revealed": 0, "linear_additive_moment": 0, "plugin_nonlinear": 1, "unbiased_LURE_claim": "NONE"},
        {"object": "selected_action", "per_trial_revealed": 0, "linear_additive_moment": 0, "plugin_nonlinear": 1, "unbiased_LURE_claim": "NONE_ARGMAX_PLUGIN"},
    ]


def _estimator_claim_rows() -> list[dict[str, Any]]:
    return [
        {"estimator_object": "LURE_linear_moment", "claim": "UNBIASED_UNDER_LOCKED_SAMPLING_AND_POSITIVITY", "scope": "NLL_correctness_class_bin_and_pairwise_linear_terms", "nonlinear_extension_allowed": 0},
        {"estimator_object": "balanced_accuracy_plugin", "claim": "NO_UNBIASEDNESS_CLAIM", "scope": "ratio_of_estimated_class_moments_with_locked_smoothing", "nonlinear_extension_allowed": 0},
        {"estimator_object": "ECE_plugin", "claim": "NO_UNBIASEDNESS_CLAIM", "scope": "absolute_values_of_estimated_signed_bin_moments", "nonlinear_extension_allowed": 0},
        {"estimator_object": "midrank_composite_plugin", "claim": "NO_UNBIASEDNESS_CLAIM", "scope": "cross_candidate_rank_transform_and_arithmetic_mean", "nonlinear_extension_allowed": 0},
        {"estimator_object": "selected_action", "claim": "NO_UNBIASEDNESS_CLAIM", "scope": "first_index_argmax_of_plugin_utility", "nonlinear_extension_allowed": 0},
    ]


def _development_view_rows() -> list[dict[str, Any]]:
    return [
        {"stage": "C86L", "view": "acquisition_pool", "identity": "C84_immutable_construction_trial_IDs_and_matching_frozen_probabilities", "labels_visible": "LABEL_SERVER_QUERIED_ROWS_ONLY", "direct_C84_evaluation_labels": 0, "development_only": 1, "overlap_with_held_outcome": 0},
        {"stage": "C86D", "view": "held_development_outcome", "identity": "accepted_C85U_held_evaluation_candidate_utility_field", "labels_visible": 0, "direct_C84_evaluation_labels": 0, "development_only": 1, "overlap_with_held_outcome": "SELF"},
        {"stage": "C86L_C86D", "view": "same_label_oracle", "identity": "FORBIDDEN", "labels_visible": 0, "direct_C84_evaluation_labels": 0, "development_only": 1, "overlap_with_held_outcome": 0},
    ]


def _development_budget_rows() -> list[dict[str, Any]]:
    bounds = {
        "Lee2019_MI": (50, 50, "25_per_class_locked_metadata"),
        "Cho2017": (100, 100, "50_per_class_locked_metadata"),
        "PhysionetMI": (18, 30, "9_to_15_per_class_locked_metadata"),
    }
    rows = []
    for dataset in ("Lee2019_MI", "Cho2017", "PhysionetMI"):
        targets = partition_subjects(C84_DATASETS[dataset])["targets"]
        lower, upper, basis = bounds[dataset]
        for target in targets:
            for budget in TOTAL_QUERY_GRID:
                if budget == "FULL":
                    available, disposition = 1, "CELL_SPECIFIC_FULL_CONSTRUCTION_POOL"
                elif lower >= int(budget):
                    available, disposition = 1, "AVAILABLE_WITHOUT_REPLACEMENT"
                elif upper < int(budget):
                    available, disposition = 0, "INPUT_UNAVAILABLE"
                else:
                    available, disposition = 0, "INPUT_UNAVAILABLE_FAIL_CLOSED_UNRESOLVED_SUPPORT"
                rows.append({
                    "dataset": dataset, "target_subject_id": target, "budget": budget,
                    "construction_pool_rows_lower_bound": lower, "construction_pool_rows_upper_bound": upper,
                    "support_basis": basis, "available": available, "disposition": disposition,
                    "replacement_sampling": 0, "budget_substitution": 0, "target_deleted": 0,
                })
    _require(len(rows) == 118 * len(TOTAL_QUERY_GRID), "C84 development budget arithmetic drift")
    return rows


def _nonoverlap_rows() -> list[dict[str, Any]]:
    return [
        {"acquisition_view": "C84_construction_label_view", "outcome_view": "C84_evaluation_label_view_under_C85U", "split_rule": "C84_TARGET_SPLIT_V1", "construction_rows": 4773, "evaluation_rows": 4848, "identity_overlap": 0, "label_rows_opened_in_C86R": 0, "status": "PASS_PHYSICALLY_DISJOINT_ACCEPTED_C84S_METADATA"},
        {"acquisition_view": "C84_construction_label_view", "outcome_view": "C85U_candidate_utility_field", "split_rule": "C85U_uses_only_C84_evaluation_trial_IDs", "construction_rows": 4773, "evaluation_rows": 4848, "identity_overlap": 0, "label_rows_opened_in_C86R": 0, "status": "PASS_DERIVED_OUTCOME_DISJOINT"},
        {"acquisition_view": "C84_evaluation_label_view", "outcome_view": "C85U_candidate_utility_field", "split_rule": "SAME_HELD_VIEW", "construction_rows": 0, "evaluation_rows": 4848, "identity_overlap": "SELF", "label_rows_opened_in_C86R": 0, "status": "FORBIDDEN_AS_C86L_ACQUISITION_POOL"},
    ]


def _montage_rows() -> list[dict[str, Any]]:
    native = {
        "Lee2019_MI": 62, "Cho2017": 64, "PhysionetMI": 64,
        "Brandl2020": 63, "Kumar2024": 22, "Yang2025_2C": 59,
    }
    rows = []
    for dataset, count in native.items():
        rows.append({
            "interface_id": "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V1",
            "dataset": dataset, "role": "LEGACY_SOURCE" if dataset in C84_DATASETS else "UNTOUCHED_TARGET",
            "native_EEG_channels": count, "common_channel_count": 11,
            "common_channels": "|".join(COMMON_SOURCE_TARGET_11), "all_channels_source_verified": 1,
            "resample_Hz": 160, "epoch_half_open_seconds": "0|3", "bandpass_Hz": "4|38",
            "events": "left_hand|right_hand", "reuses_C84_20ch_model_bytes": 0,
            "requires_new_candidate_training": 1, "metadata_feasibility": "PASS",
        })
    for dataset, count in (("Brandl2020", 63), ("Kumar2024", 22), ("Yang2025_2C", 59)):
        rows.append({
            "interface_id": "C86_UNTOUCHED_ONLY_22CH_CANDIDATE",
            "dataset": dataset, "role": "UNTOUCHED_TARGET_ONLY",
            "native_EEG_channels": count, "common_channel_count": 22,
            "common_channels": "|".join(COMMON_UNTOUCHED_22), "all_channels_source_verified": 1,
            "resample_Hz": 160, "epoch_half_open_seconds": "0|3", "bandpass_Hz": "4|38",
            "events": "left_hand|right_hand", "reuses_C84_20ch_model_bytes": 0,
            "requires_new_candidate_training": 1,
            "metadata_feasibility": "TARGET_ONLY_NO_COMMON_LEGACY_SOURCE_CONTRACT",
        })
    return rows


def _allocation_rows() -> list[dict[str, Any]]:
    return [
        {"option_id": "O1_LEGACY_SOURCE_ALL_UNTOUCHED_TARGETS", "interface_id": "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V1", "source_training": "C84_fixed_source_panels_12_per_dataset_x3=36_per_panel", "source_audit": "C84_fixed_source_audit_4_per_dataset_x3=12_per_panel", "target_allocation": "all_subjects_in_each_untouched_cohort", "target_counts": "Brandl=16|Kumar=18|Yang2C=51", "target_in_own_training": 0, "same_zoo_across_cohorts": 1, "decision": "PREFERRED_METADATA_FEASIBLE_PENDING_ELIGIBILITY_AND_FUTURE_LOCK"},
        {"option_id": "O2_WITHIN_COHORT_SOURCE_TARGET_SPLIT", "interface_id": "C86_UNTOUCHED_ONLY_22CH_CANDIDATE", "source_training": "would_consume_untouched_cohort_subjects", "source_audit": "would_consume_untouched_cohort_subjects", "target_allocation": "reduced_subject_subset", "target_counts": "not_locked", "target_in_own_training": 0, "same_zoo_across_cohorts": 0, "decision": "REJECT_PRIMARY_REDUCES_TARGET_CLUSTERS_AND_ACTION_IDENTITY"},
        {"option_id": "O3_CROSS_COHORT_LEAVE_ONE_OUT", "interface_id": "C86_UNTOUCHED_ONLY_22CH_CANDIDATE", "source_training": "other_untouched_cohorts", "source_audit": "other_untouched_cohorts", "target_allocation": "one_held_cohort", "target_counts": "16|18|51", "target_in_own_training": 0, "same_zoo_across_cohorts": 0, "decision": "REJECT_PRIMARY_DIFFERENT_ZOO_PER_COHORT"},
    ]


def _inference_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset, targets, adult in (
        ("Brandl2020", 16, 1), ("Kumar2024", 18, 0), ("Yang2025_2C", 51, 0),
    ):
        rows.append({
            "cohort": dataset, "target_subject_clusters": targets, "minimum_clusters_required": 12,
            "cluster_count_pass": int(targets >= 12), "adult_primary_eligible": adult,
            "exact_sign_configurations": str(2**targets), "registered_maxT_draws": 65536,
            "plus_one_Monte_Carlo_min_p_exact": "1/65537",
            "plus_one_Monte_Carlo_min_p_decimal": f"{1/65537:.17g}",
            "favorable_75pct_targets": math.ceil(0.75 * targets), "LOTO_fits": targets,
            "resolution_status": "PASS_IF_COHORT_ELIGIBLE", "scientific_unit": "target_subject",
        })
    return rows


def _resource_rows() -> list[dict[str, Any]]:
    return [
        {"resource": "unique_candidate_units", "estimate": 648, "unit": "models", "envelope": 648, "basis": "2_panels*2_seeds*2_levels*81_shared_zoo", "locked_for_execution": 0},
        {"resource": "training_phases", "estimate": 24, "unit": "phases", "envelope": 24, "basis": "2*2*2*3_regimes", "locked_for_execution": 0},
        {"resource": "three_cohort_target_contexts", "estimate": 680, "unit": "contexts", "envelope": 680, "basis": "(16+18+51)*8", "locked_for_execution": 0},
        {"resource": "three_cohort_candidate_context_slices", "estimate": 55080, "unit": "slices", "envelope": 55080, "basis": "680*81", "locked_for_execution": 0},
        {"resource": "unit_cohort_target_artifacts", "estimate": 1944, "unit": "artifacts", "envelope": 1944, "basis": "648*3", "locked_for_execution": 0},
        {"resource": "raw_download_storage", "estimate": 128, "unit": "GiB", "envelope": 128, "basis": "Yang_public_63.4GB_plus_Kumar_1.2GB_plus_Brandl_and_overhead", "locked_for_execution": 0},
        {"resource": "candidate_checkpoint_optimizer_storage", "estimate": 128, "unit": "GiB", "envelope": 128, "basis": "648_units_conservative", "locked_for_execution": 0},
        {"resource": "target_prediction_field_storage", "estimate": 64, "unit": "GiB", "envelope": 64, "basis": "55080_slices_scaled_from_C84", "locked_for_execution": 0},
        {"resource": "total_scratch_storage", "estimate": 512, "unit": "GiB", "envelope": 512, "basis": "raw_models_outputs_staging_and_replay", "locked_for_execution": 0},
        {"resource": "GPU_time", "estimate": 20, "unit": "GPU_hours_scaled", "envelope": 96, "basis": "24_vs_C84_72_phases;future_canary_required", "locked_for_execution": 0},
        {"resource": "RAM", "estimate": 128, "unit": "GiB", "envelope": 128, "basis": "future_field_generation", "locked_for_execution": 0},
    ]


def _license_rows() -> list[dict[str, Any]]:
    return [
        {"cohort": "Brandl2020", "license": "CC-BY-NC-ND-4.0", "source": "MOABB_loader_metadata_and_dataset_paper", "internal_analysis": "TERMS_REPLAY_REQUIRED", "derived_artifact_redistribution": "RESTRICTED_INSTITUTIONAL_REVIEW_REQUIRED", "attribution_required": 1, "age_status": "PASS", "legal_advice_claimed": 0},
        {"cohort": "Kumar2024", "license": "CC-BY-4.0", "source": "NEMAR_v1.0.1", "internal_analysis": "TERMS_REPLAY_REQUIRED", "derived_artifact_redistribution": "ATTRIBUTION_AND_TERMS_REVIEW", "attribution_required": 1, "age_status": "FAIL_CLOSED", "legal_advice_claimed": 0},
        {"cohort": "Yang2025_2C", "license": "CC-BY-4.0", "source": "NeMAR_NM000348_and_loader_metadata", "internal_analysis": "TERMS_REPLAY_REQUIRED", "derived_artifact_redistribution": "ATTRIBUTION_AND_TERMS_REVIEW", "attribution_required": 1, "age_status": "FAIL_CLOSED", "legal_advice_claimed": 0},
    ]


def _risk_rows() -> list[dict[str, Any]]:
    return [
        {"risk_id": "R1", "risk": "fewer_than_two_adult_eligible_untouched_cohorts", "blocking": 1, "mitigation": "PM_must_supply_public_adult_evidence_or_new_prospective_eligibility_protocol", "status": "OPEN_BLOCKER"},
        {"risk_id": "R2", "risk": "stale_parent_protocol_consumed_downstream", "blocking": 1, "mitigation": "effective_manifest_V2_mandatory", "status": "CLOSED_BY_GUARD"},
        {"risk_id": "R3", "risk": "A2M_mislabeled_as_Hara", "blocking": 1, "mitigation": "A2H_exact_sum_and_A2M_renamed", "status": "CLOSED"},
        {"risk_id": "R4", "risk": "construction_and_held_development_view_collision", "blocking": 1, "mitigation": "construction_pool_only_C85U_held_outcome", "status": "CLOSED_CONTRACT"},
        {"risk_id": "R5", "risk": "future_field_resource_or_license_failure", "blocking": 1, "mitigation": "future_canary_and_terms_replay", "status": "OPEN_FUTURE_STAGE"},
        {"risk_id": "R6", "risk": "nonlinear_plugin_unbiasedness_overclaim", "blocking": 1, "mitigation": "linear_moment_claim_boundary", "status": "CLOSED_CONTRACT"},
    ]


def _failure_rows() -> list[dict[str, Any]]:
    return [
        {"failure_code": "ADULT_ELIGIBLE_COHORT_COUNT_LT_2", "observed": 1, "gate": FINAL_GATE, "automatic_retry": 0},
        {"failure_code": "COMMON_FIELD_INTERFACE_LT_2", "observed": 0, "gate": "C86_CONFIRMATION_FIELD_INTERFACE_OR_TARGET_CLUSTER_INPUT_BLOCKER", "automatic_retry": 0},
        {"failure_code": "BASELINE_OR_EFFECTIVE_MANIFEST_DRIFT", "observed": 0, "gate": "C86_ACTIVE_BASELINE_FIDELITY_DEVELOPMENT_VIEW_OR_EFFECTIVE_PROTOCOL_RECONCILIATION_REQUIRED", "automatic_retry": 0},
    ]


def table_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "c86p_protocol_stack_supersession_ledger.csv": _protocol_stack_rows(),
        "participant_age_eligibility_audit.csv": _age_rows(),
        "final_untouched_cohort_registry_v2.csv": _cohort_rows(),
        "age_uncertainty_truth_table.csv": _age_truth_rows(),
        "active_testing_literature_registry_v2.csv": _literature_rows(),
        "active_method_registry_v2.csv": _method_rows(),
        "reference_fidelity_registry_v2.csv": _fidelity_rows(),
        "baseline_availability_ledger.csv": _baseline_availability_rows(),
        "queried_information_object_contract.csv": _queried_information_rows(),
        "estimator_claim_boundary_v2.csv": _estimator_claim_rows(),
        "C86L_development_view_contract_v2.csv": _development_view_rows(),
        "C84_development_budget_availability.csv": _development_budget_rows(),
        "acquisition_evaluation_nonoverlap_truth_table.csv": _nonoverlap_rows(),
        "common_montage_metadata_audit.csv": _montage_rows(),
        "confirmation_subject_allocation_options.csv": _allocation_rows(),
        "inference_resolution_audit.csv": _inference_rows(),
        "candidate_field_resource_envelope.csv": _resource_rows(),
        "license_and_derived_artifact_policy.csv": _license_rows(),
        "risk_register.csv": _risk_rows(),
        "failure_reason_ledger.csv": _failure_rows(),
    }


def build_synthetic_v2() -> dict[str, Any]:
    parent_path = REPORT_DIR / "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    _require(sha256_file(parent_path) == PROTOCOL_STACK[4][2], "synthetic V1 identity drift")
    value = deepcopy(parent)
    value["schema_version"] = "c86p_synthetic_calibration_operationalization_v2"
    value["status"] = "LOCKED_GENERATIVE_CONTRACT_V2_NOT_SCIENTIFICALLY_EXECUTED"
    value["parent_protocols"] = {
        "synthetic_v1_sha256": PROTOCOL_STACK[4][2],
        "C86R_repair_sha256": REPAIR_PROTOCOL_SHA256,
    }
    value["chronology"].update({
        "registered_synthetic_draws_before_V2": 0,
        "registered_synthetic_results_before_V2": 0,
        "method_registry_update_outcome_informed": False,
    })
    value["primary_dispatcher_contract"] = [
        {"method_id": "P0", "dispatcher_id": "c86_active_v2:P0_uniform_without_replacement", "same_dispatcher_required_in_C86H": True},
        {"method_id": "A1", "dispatcher_id": "c86_active_v2:A1_lure_active_testing", "same_dispatcher_required_in_C86H": True},
        {"method_id": "A2H", "dispatcher_id": "c86_active_v2:A2H_hara_sum_pairwise", "same_dispatcher_required_in_C86H": True},
    ]
    value["method_supersession"] = {
        "A2H": "faithful_general_K_sum_over_pairs_Hara_score",
        "A2M": "development_only_project_max_pair_heuristic",
        "A2M_claimed_as_Hara": False,
    }
    value["queried_information_boundary"] = {
        "linear_objects": ["NLL", "correctness", "class_indicator", "signed_confidence_bin_moments", "pairwise_NLL_differences"],
        "nonlinear_plugins": ["balanced_accuracy", "ECE", "midranks", "composite_utility", "selected_action"],
        "nonlinear_plugin_unbiasedness_claim": "NONE",
    }
    value["publication_rule"].update({
        "C86R_registered_draws": 0,
        "primary_dispatcher_implementation_required_before_registered_execution": True,
    })
    return value


def build_effective_manifest(table_hashes: Mapping[str, str], synthetic_v2_sha256: str) -> dict[str, Any]:
    primary = [row["cohort"] for row in _cohort_rows() if row["selected_primary_confirmation"]]
    _require(primary == ["Brandl2020"], "adult eligibility result drift")
    return {
        "schema_version": "c86_active_testing_effective_program_manifest_v2",
        "created_at_utc": CREATED_AT_UTC,
        "status": "EFFECTIVE_PROGRAM_RESOLVED_WITH_ELIGIBILITY_BLOCKER_NO_DOWNSTREAM_EXECUTION_AUTHORITY",
        "precedence_low_to_high": [name for name, _, _ in PROTOCOL_STACK] + [SYNTHETIC_V2.name],
        "protocol_identities": {
            name: {"commit": commit, "sha256": digest} for name, commit, digest in PROTOCOL_STACK
        } | {SYNTHETIC_V2.name: {"commit": "IMPLEMENTATION_COMMIT_PENDING", "sha256": synthetic_v2_sha256}},
        "authoritative_program": {
            "primary_untouched_confirmation_interfaces": primary,
            "primary_untouched_confirmation_count": len(primary),
            "minimum_required_confirmation_cohorts": 2,
            "at_least_two_cohort_rule": "FAIL",
            "stress_tracks": ["Kumar2024_AGE_UNCERTAIN", "Yang2025_2C_AGE_MIXED"],
            "development_only_cohorts": ["Dreyer2023", "C84_C85_FROZEN_FIELDS"],
            "query_grid": list(TOTAL_QUERY_GRID),
            "candidate_structure": {"ERM": 1, "OACI": 40, "SRC": 40, "total": 81},
            "primary_method_registry": ["P0", "A1", "A2H"],
            "secondary_method_registry": ["P1", "MODEL_SELECTOR", "CODA"],
            "development_method_registry": ["A2M", "A3D", "A4", "ASE_XWED", "ONLINE_AMS"],
            "unavailable_methods": ["PPAT"],
            "Hara_general_K_score": "sum_over_all_unordered_pairwise_expected_absolute_linear_loss_differences",
            "A2M_role": "project_max_pair_heuristic_development_only_not_Hara",
            "linear_unbiasedness_scope": "registered_linear_moments_only",
            "nonlinear_plugin_unbiasedness_claim": "NONE",
            "C86L_acquisition_pool": "C84_immutable_construction_trial_IDs_and_matching_frozen_probabilities",
            "C86L_held_development_outcome": "C85U_held_evaluation_candidate_utility_field",
            "C86L_direct_evaluation_label_access": False,
            "preferred_confirmation_interface": "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V1",
            "synthetic_contract": SYNTHETIC_V2.name,
            "taxonomy": "UNCHANGED_C86_A_TO_E_AND_C86_L1_TO_L4_FROM_PARENT",
            "stage_sequence": ["C86LP", "C86L", "C86DP", "C86D", "C86C_F", "C86H"],
        },
        "table_hashes": dict(sorted(table_hashes.items())),
        "final_gate": FINAL_GATE,
        "downstream_contract": {
            "stale_parent_protocol_alone_accepted": False,
            "effective_manifest_V2_required": True,
            "C86L_authorized": False,
            "C86D_authorized": False,
            "C86C_F_authorized": False,
            "C86H_authorized": False,
            "new_EEG_or_label_access_authorized": False,
            "execution_lock_created": False,
        },
        "protected_counters": {
            "new_EEG_downloads_or_opens": 0,
            "new_target_label_reads": 0,
            "registered_active_policy_runs": 0,
            "new_candidate_training_or_forward": 0,
            "registered_synthetic_results": 0,
            "GPU": 0,
        },
    }


def load_effective_manifest(path: Path = EFFECTIVE_MANIFEST) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    sidecar = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    _require(sha256_file(path) == sidecar, "effective manifest sidecar drift")
    _require(value["schema_version"] == "c86_active_testing_effective_program_manifest_v2", "effective manifest schema drift")
    _require(value["downstream_contract"]["effective_manifest_V2_required"] is True, "effective manifest requirement drift")
    return value


def require_effective_manifest_for_downstream(bound_paths: Sequence[str | Path]) -> None:
    normalized = {Path(path).resolve() for path in bound_paths}
    _require(EFFECTIVE_MANIFEST.resolve() in normalized, "stale parent protocol stack cannot drive a downstream lock")
    load_effective_manifest()


def write_readiness_artifacts(table_dir: Path = TABLE_DIR) -> dict[str, Any]:
    _require(sha256_file(REPAIR_PROTOCOL) == REPAIR_PROTOCOL_SHA256, "C86R repair protocol identity drift")
    rows = table_rows()
    hashes = {name: _write_csv(table_dir / name, value) for name, value in rows.items()}
    synthetic_sha = _write_json(SYNTHETIC_V2, build_synthetic_v2())
    manifest_sha = _write_json(EFFECTIVE_MANIFEST, build_effective_manifest(hashes, synthetic_sha))
    return {
        "tables": len(hashes), "table_hashes": hashes,
        "synthetic_v2_sha256": synthetic_sha, "effective_manifest_sha256": manifest_sha,
        "final_gate": FINAL_GATE, "primary_eligible_cohorts": ["Brandl2020"],
        "new_EEG_or_label_access": 0, "registered_active_or_synthetic_runs": 0,
    }


__all__ = [
    "C86RContractError", "COMMON_SOURCE_TARGET_11", "COMMON_UNTOUCHED_22", "EFFECTIVE_MANIFEST",
    "FINAL_GATE", "SYNTHETIC_V2", "adult_eligibility", "hara_general_k_score",
    "load_effective_manifest", "project_max_pair_score", "require_effective_manifest_for_downstream",
    "table_rows", "write_readiness_artifacts",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize the metadata-only C86R readiness artifacts")
    parser.add_argument("command", choices=("write-readiness", "validate-effective-manifest"))
    args = parser.parse_args(argv)
    if args.command == "write-readiness":
        print(json.dumps(write_readiness_artifacts(), sort_keys=True))
    else:
        print(json.dumps(load_effective_manifest(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
