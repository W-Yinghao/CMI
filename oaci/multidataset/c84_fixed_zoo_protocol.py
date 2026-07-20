"""Generate the metadata-only C84P protocol and audit registries.

No function in this module loads a dataset, resolves a raw data path, or imports
MNE/MOABB/PyTorch.  The generated stage protocols are deliberately non-operative
while the Lee2019 FCz compatibility blocker remains open.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .c84_dataset_registry import (
    DATASETS,
    PRIMARY_CHANNELS,
    SOURCE_AUDIT_SALT,
    SUBJECT_PARTITION_SALT,
    TARGET_SPLIT_SALT,
    dataset_registry_payload,
    partition_subjects,
    source_train_audit_split,
    validate_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84p_tables"
PARENT_HEAD = "2ecc8efd49d6b9d18b50eae3811be8f2ac4cfa25"
CREATED_AT_UTC = "2026-07-13T21:19:22Z"
C83_GATE = "C83_AAAI_EVIDENCE_CLAIM_FIGURE_TABLE_FREEZE_READY_FOR_MANUSCRIPT_AUTHORIZATION"
FAIL_GATE = "C84_DATASET_CHANNEL_EVENT_RESOURCE_OR_PROTOCOL_RECONCILIATION_REQUIRED"
SUCCESS_GATE = "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOLS_LOCKED_READY_FOR_CANARY_AUTHORIZATION"
CHANNEL_BLOCKER = "Lee2019_MI metadata/loader has no FCz; requested 21-channel allowlist is 20/21 available"

DATASET_ORDER = ("Lee2019_MI", "Cho2017", "PhysionetMI")
PANELS = ("A", "B")
SEEDS = (5, 6)
LEVELS = (0, 1)
REGIMES = ("ERM", "OACI", "SRC")
OACI_EPOCHS = tuple(range(4, 200, 5))
SRC_EPOCHS = OACI_EPOCHS
UNITS_PER_ZOO = 81
TOTAL_UNITS = 1944
TOTAL_PHASES = 72
TOTAL_CONTEXTS = 944
TOTAL_CANDIDATE_CONTEXTS = 76464
COMMON_BUDGETS = (1, 2, 4, 8, "FULL")
EXTENDED_BUDGETS = (16, 32)
PRIMARY_ZERO_METHODS = ("U7", "U5", "U11", "U13", "U14", "U15")
METHOD_REGISTRY_SHA256 = "ef48ecf7fcc55188b78b0878d86f07f6239fe4f6c88bbc854829b3a1c7a1a120"
MAX_GIT_BYTES = 50 * 1024 * 1024


class C84ProtocolError(RuntimeError):
    """Raised when a protected C84 execution boundary is not satisfied."""


def utc_now() -> str:
    # The protocol generator is replayable: rerunning it must not mutate hashes.
    return CREATED_AT_UTC


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def write_sha(path: str | Path, digest: str, target_name: str) -> None:
    Path(path).write_text(f"{digest}  {target_name}\n")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"refusing empty C84P table: {path}")
    fieldnames = list(fields or rows[0])
    for index, row in enumerate(rows):
        if set(row) != set(fieldnames):
            raise ValueError(f"C84P schema mismatch in {path}, row {index}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def require_scope_execution_lock(
    *,
    stage_protocol_path: str | Path,
    execution_lock_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    """Fail closed before any future C84C/F/S adapter can import a loader."""
    protocol_path = Path(stage_protocol_path)
    if not protocol_path.is_file():
        raise C84ProtocolError("C84 stage protocol is absent")
    protocol = read_json(protocol_path)
    if protocol.get("status") != "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION":
        raise C84ProtocolError("C84 stage protocol is not execution-ready")
    lock_path = Path(execution_lock_path)
    if not lock_path.is_file():
        raise C84ProtocolError("C84 scope-specific execution lock is absent")
    authorization = Path(authorization_path)
    if not authorization.is_file():
        raise C84ProtocolError("C84 scope-specific authorization record is absent")
    return {
        "protocol": protocol,
        "lock": read_json(lock_path),
        "authorization": read_json(authorization),
    }


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=check)


def subject_partition_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        spec = DATASETS[dataset]
        partition = partition_subjects(spec)
        for panel_name, key in (("A", "source_panel_A"), ("B", "source_panel_B")):
            split = source_train_audit_split(dataset, panel_name, partition[key])
            for role in ("source_training", "source_audit"):
                for subject in split[role]:
                    rows.append({
                        "dataset": dataset,
                        "subject_id": subject,
                        "partition": key,
                        "within_panel_role": role,
                        "subject_hash": hashlib.sha256(
                            f"{SUBJECT_PARTITION_SALT}|{dataset}|{subject}".encode("ascii")
                        ).hexdigest(),
                        "source_audit_hash": hashlib.sha256(
                            f"{SOURCE_AUDIT_SALT}|{dataset}|panel={panel_name}|{subject}".encode("ascii")
                        ).hexdigest(),
                        "outcome_used": 0,
                    })
        for subject in partition["targets"]:
            rows.append({
                "dataset": dataset,
                "subject_id": subject,
                "partition": "target_population",
                "within_panel_role": "target",
                "subject_hash": hashlib.sha256(
                    f"{SUBJECT_PARTITION_SALT}|{dataset}|{subject}".encode("ascii")
                ).hexdigest(),
                "source_audit_hash": "NA",
                "outcome_used": 0,
            })
    if len(rows) != 214:
        raise RuntimeError(f"C84 expected 214 eligible subjects, got {len(rows)}")
    return rows


def _unit_id(dataset: str, panel: str, seed: int, level: int, regime: str, epoch: int) -> str:
    identity = {
        "milestone": "C84", "dataset": dataset, "panel": panel, "seed": seed,
        "level": level, "regime": regime, "epoch": epoch,
    }
    return "c84_" + hashlib.sha256(canonical_bytes(identity)).hexdigest()[:24]


def candidate_units() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for panel in PANELS:
            for seed in SEEDS:
                for level in LEVELS:
                    rows.append({
                        "unit_id": _unit_id(dataset, panel, seed, level, "ERM", 199),
                        "dataset": dataset, "panel": panel, "seed": seed, "level": level,
                        "regime": "ERM", "epoch": 199, "trajectory_order": 0,
                        "canary_subset": int(panel == "A" and seed == 5 and level == 0),
                    })
                    for regime, epochs in (("OACI", OACI_EPOCHS), ("SRC", SRC_EPOCHS)):
                        for order, epoch in enumerate(epochs, start=1):
                            rows.append({
                                "unit_id": _unit_id(dataset, panel, seed, level, regime, epoch),
                                "dataset": dataset, "panel": panel, "seed": seed, "level": level,
                                "regime": regime, "epoch": epoch, "trajectory_order": order,
                                "canary_subset": int(panel == "A" and seed == 5 and level == 0),
                            })
    if len(rows) != TOTAL_UNITS or len({row["unit_id"] for row in rows}) != TOTAL_UNITS:
        raise RuntimeError("C84 unit manifest is not 1,944 unique units")
    return rows


def dataset_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        spec = DATASETS[dataset]
        targets = partition_subjects(spec)["targets"]
        rows.append({
            "dataset": dataset,
            "eligible_subjects": spec.subject_count,
            "excluded_subjects": "|".join(map(str, spec.excluded_subjects)) or "NONE",
            "source_panel_A": 16,
            "source_panel_B": 16,
            "source_training_per_panel": 12,
            "source_audit_per_panel": 4,
            "target_subjects": len(targets),
            "sessions": spec.sessions,
            "native_sfreq_hz": spec.native_sfreq_hz,
            "native_EEG_channels": len(spec.native_eeg_channels),
            "task": "left_hand_vs_right_hand_imagery",
            "class_names": "left_hand|right_hand",
            "task_runs": spec.task_runs,
            "trials_per_class_metadata": spec.trials_per_class,
            "metadata_only": 1,
        })
    return rows


def overlap_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        partition = partition_subjects(DATASETS[dataset])
        pairs = (
            ("source_panel_A", "source_panel_B"),
            ("source_panel_A", "targets"),
            ("source_panel_B", "targets"),
        )
        for left, right in pairs:
            overlap = sorted(set(partition[left]) & set(partition[right]))
            rows.append({
                "dataset": dataset, "left_partition": left, "right_partition": right,
                "overlap_count": len(overlap), "overlap_subjects": "|".join(map(str, overlap)) or "NONE",
                "pass": int(not overlap),
            })
    return rows


def channel_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        spec = DATASETS[dataset]
        for order, channel in enumerate(PRIMARY_CHANNELS, start=1):
            available = channel in spec.native_eeg_channels
            rows.append({
                "dataset": dataset, "order": order, "canonical_channel": channel,
                "metadata_channel": channel if available else "ABSENT",
                "available": int(available), "interpolation_allowed": 0,
                "substitution_allowed": 0, "status": "PASS" if available else "BLOCKER",
            })
    return rows


def channel_alias_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        rule = (
            "strip_trailing_period;uppercase_then_restore_midline_case_from_fixed_map"
            if dataset == "PhysionetMI" else "identity_exact_case"
        )
        rows.append({
            "dataset": dataset,
            "canonicalization_rule": rule,
            "fixed_midline_aliases": "AFZ->AFz|PZ->Pz|FPZ->Fpz|FCZ->FCz|FP1->Fp1|CZ->Cz|OZ->Oz|POZ->POz|IZ->Iz|CPZ->CPz|FP2->Fp2|FZ->Fz" if dataset == "PhysionetMI" else "NONE",
            "scientific_channel_substitution": 0,
            "FCz_to_Fz_substitution": 0,
        })
    return rows


def class_mapping_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        spec = DATASETS[dataset]
        for class_name, native_event in spec.event_mapping:
            rows.append({
                "dataset": dataset, "canonical_class": class_name, "native_event_id": native_event,
                "included": 1, "imagined_only": 1,
                "excluded_event_classes": "online_unlabeled" if dataset == "Lee2019_MI" else (
                    "motor_execution|hands|feet|rest" if dataset == "PhysionetMI" else "real_movement|non_task"
                ),
            })
    return rows


def event_window_rows() -> list[dict[str, Any]]:
    return [{
        "dataset": dataset,
        "event_anchor": "official_MOABB_left/right_imagery_event_onset",
        "native_interval_seconds": "0-4" if dataset == "Lee2019_MI" else "0-3",
        "locked_tmin_seconds": "0.0",
        "locked_tmax_seconds": "3.0",
        "endpoint_semantics": "half_open_[0.0,3.0)",
        "resample_sfreq_hz": 160,
        "expected_n_times": 480,
        "terminal_native_endpoint_action": "drop_if_present_after_resample",
        "padding": "MNE_Epochs.resample_default_pad_edge",
        "event_compatibility": "PASS_METADATA_ONLY",
    } for dataset in DATASET_ORDER]


def preprocessing_rows() -> list[dict[str, Any]]:
    values = (
        ("bandpass", "4.0-38.0_Hz", "MOABB_get_filter_pipeline_IIR"),
        ("reference", "native_dataset_reference", "no_re_reference"),
        ("amplitude_units", "MNE_volts_then_zscore_sample", "dimensionless_model_input"),
        ("normalization", "zscore_per_trial_per_channel", "epsilon_1e-8"),
        ("epoch", "half_open_0.0_to_3.0_seconds", "480_samples"),
        ("resample", "160_Hz", "MNE_1.11_Epochs.resample_fft_npad_auto_window_auto_pad_edge"),
        ("spatial_interpolation", "forbidden", "all_locked_channels_must_exist"),
        ("bad_trials", "MOABB_loader_returned_trials_only", "no_extra_dataset_specific_outcome_filter"),
        ("network_input", "channels_x_time", "locked_channel_order_x_480"),
        ("trial_id", "dataset_subject_session_run_within_recording_ordinal", "stable_before_row_sort"),
    )
    return [{"item": item, "locked_value": value, "implementation_note": note} for item, value, note in values]


def bad_trial_rows() -> list[dict[str, Any]]:
    return [{
        "dataset": dataset,
        "official_flag_metadata": "bad_trial_indices_present" if dataset == "Cho2017" else "none_declared_in_locked_loader_metadata",
        "C84_policy": "retain_exact_trials_returned_by_locked_MOABB_loader;no_extra_flag_parse",
        "nonfinite_or_shape_invalid": "fail_subject_before_outcome_access",
        "outcome_dependent_exclusion": 0,
    } for dataset in DATASET_ORDER]


def candidate_arithmetic_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        rows.extend((
            {"dataset": dataset, "object": "ERM", "count": 8, "formula": "2_panels*2_seeds*2_levels*1"},
            {"dataset": dataset, "object": "OACI", "count": 320, "formula": "2*2*2*40"},
            {"dataset": dataset, "object": "SRC", "count": 320, "formula": "2*2*2*40"},
            {"dataset": dataset, "object": "all_units", "count": 648, "formula": "2*2*2*81"},
            {"dataset": dataset, "object": "training_phases", "count": 24, "formula": "2*2*2*3"},
        ))
    rows.extend((
        {"dataset": "ALL", "object": "all_units", "count": TOTAL_UNITS, "formula": "3*648"},
        {"dataset": "ALL", "object": "training_phases", "count": TOTAL_PHASES, "formula": "3*24"},
        {"dataset": "ALL", "object": "canary_reusable_units", "count": 243, "formula": "3*1_panel*1_seed*1_level*81"},
    ))
    return rows


def context_arithmetic_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        targets = len(partition_subjects(DATASETS[dataset])["targets"])
        contexts = targets * 2 * 2 * 2
        rows.append({
            "dataset": dataset, "target_subjects": targets, "panels": 2, "seeds": 2,
            "levels": 2, "contexts": contexts, "candidates_per_context": 81,
            "candidate_context_evaluations": contexts * 81,
        })
    if sum(row["contexts"] for row in rows) != TOTAL_CONTEXTS:
        raise RuntimeError("C84 target context arithmetic drift")
    return rows


def budget_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASET_ORDER:
        for order, budget in enumerate(COMMON_BUDGETS):
            rows.append({
                "dataset": dataset, "budget": budget, "grid": "three_dataset_primary",
                "ordinal": order, "availability_basis": "metadata_then_C84C_exact_count_gate",
                "enters_cross_dataset_conjunction": 1,
            })
        if dataset != "PhysionetMI":
            for budget in EXTENDED_BUDGETS:
                rows.append({
                    "dataset": dataset, "budget": budget, "grid": "Lee_Cho_secondary",
                    "ordinal": len(COMMON_BUDGETS) + EXTENDED_BUDGETS.index(budget),
                    "availability_basis": "metadata_then_C84C_exact_count_gate",
                    "enters_cross_dataset_conjunction": 0,
                })
    return rows


def selector_rows() -> list[dict[str, Any]]:
    registry_path = REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json"
    if sha256_file(registry_path) != METHOD_REGISTRY_SHA256:
        raise RuntimeError("C84 selector registry parent hash mismatch")
    registry = read_json(registry_path)
    keep = {"B0", "B1", "B2", "B3", "B4O", "B4S", "B5", "S1", *PRIMARY_ZERO_METHODS}
    rows = []
    for method in registry["methods"]:
        if method["id"] not in keep:
            continue
        rows.append({
            "method_id": method["id"], "method_name": method["name"],
            "information_class": method["family"], "formula": method["formula"],
            "score_direction": method["score_direction"], "views": "|".join(method["views"]) or "NONE",
            "primary_role": "zero_label_primary" if method["id"] in PRIMARY_ZERO_METHODS else (
                "strict_source_primary" if method["id"] == "S1" else "control_or_ceiling"
            ),
            "formula_retuned_for_C84": 0, "parent_registry_sha256": METHOD_REGISTRY_SHA256,
        })
    if len(rows) != 14:
        raise RuntimeError(f"C84 expected 14 frozen selectors/controls, got {len(rows)}")
    return rows


def view_rows() -> list[dict[str, Any]]:
    views = (
        ("source_training_view", 1, 0, 0, 0, "training_only"),
        ("source_audit_view", 0, 1, 1, 0, "source_selector_calibration_only"),
        ("target_unlabeled_view", 0, 0, 1, 0, "zero_label_selectors"),
        ("target_construction_view", 0, 0, 0, 1, "Q0_after_full_field_freeze"),
        ("target_evaluation_view", 0, 0, 0, 1, "scoring_after_selection_freeze"),
        ("same_label_oracle_view", 0, 0, 0, 0, "physically_unreachable"),
    )
    return [{
        "view": view, "training_access": train, "source_label_access": source,
        "target_X_access": target_x, "target_label_access": target_y, "purpose": purpose,
        "trial_ID_feature": 0, "row_order_feature": 0,
    } for view, train, source, target_x, target_y, purpose in views]


def physical_split_rows() -> list[dict[str, Any]]:
    return [{
        "dataset": dataset, "split_unit": "dataset_x_target_subject_x_class",
        "sort_key": f"SHA256({TARGET_SPLIT_SALT}|dataset|subject|trial_id)",
        "construction_count": "floor(n_class_trials/2)",
        "evaluation_count": "n-floor(n/2)", "overlap_allowed": 0,
        "minimum_construction_per_class": 8,
        "session_run_role": "dependence_block_and_report_key_only",
        "outcome_used": 0,
    } for dataset in DATASET_ORDER]


def inference_rows() -> list[dict[str, Any]]:
    rows = []
    questions = {
        "E1": "zero_label_improvement_over_S1",
        "E2": "zero_label_noninferiority_to_Q0_B1",
        "E3": "external_label_budget_frontier",
        "E4": "source_panel_and_training_seed_robustness",
        "E5": "cross_dataset_same_method_recurrence",
        "E6": "decision_objective_separation",
    }
    for question, estimand in questions.items():
        rows.append({
            "question": question, "estimand": estimand,
            "highest_group": "dataset", "principal_within_dataset_cluster": "target_subject",
            "repeated_factors": "source_panel|training_seed|level",
            "candidate_trial_pair_iid": 0,
            "resampling": "target_cluster_Rademacher_maxT_65536_fixed_draws" if question in {"E1", "E2", "E3"} else "prelocked_descriptive_or_LOTO",
            "alpha": "0.05", "method_identity_preserved": 1,
        })
    return rows


def taxonomy_truth_rows() -> list[dict[str, Any]]:
    scenarios = (
        ("blocker", "ANY", "ANY", "ANY", 1, "C84-E_multidataset_protocol_field_view_analysis_or_provenance_blocker"),
        ("same_A_method", "U13", "U13", "U13", 0, "C84-A_same_zero_label_selector_matches_B1_across_all_external_datasets"),
        ("same_B_method", "U13", "U13", "U13", 0, "C84-B_same_zero_label_selector_improves_source_across_all_external_datasets_but_not_B1"),
        ("all_stable_C", "NONE", "NONE", "NONE", 0, "C84-C_no_registered_zero_label_selector_materially_improves_source_in_any_external_dataset"),
        ("different_methods", "U13", "U7", "U14", 0, "C84-D_external_dataset_source_panel_seed_or_target_heterogeneous"),
        ("one_dataset_only", "U13", "NONE", "NONE", 0, "C84-D_external_dataset_source_panel_seed_or_target_heterogeneous"),
        ("panel_or_seed_heterogeneous", "U13", "U13", "U13", 0, "C84-D_external_dataset_source_panel_seed_or_target_heterogeneous"),
    )
    return [{
        "scenario": name, "Lee_support": lee, "Cho_support": cho, "Physionet_support": physio,
        "blocker": blocker, "expected_gate": gate,
    } for name, lee, cho, physio, blocker, gate in scenarios]


def resource_rows() -> list[dict[str, Any]]:
    # C78R prospectively estimated 7.5831899579 GPU h for 48 historical phases.
    base_72 = 7.583189957936605 * (72 / 48)
    safety_72 = base_72 * 5.0
    rows = [
        {"resource": "GPU_phase_hours_base", "estimate": f"{base_72:.6f}", "unit": "hours", "envelope": "250", "basis": "C78R_48_phase_base_scaled_to_72", "within_envelope": 1},
        {"resource": "GPU_phase_hours_safety", "estimate": f"{safety_72:.6f}", "unit": "hours", "envelope": "250", "basis": "5x_cross_dataset_IO_and_runtime_factor", "within_envelope": 1},
        {"resource": "GPU_canary_phase_hours_safety", "estimate": f"{safety_72 * 9 / 72:.6f}", "unit": "hours", "envelope": "250", "basis": "9_of_72_phases", "within_envelope": 1},
        {"resource": "CPU_instrumentation", "estimate": "13.5", "unit": "million_row_upper_bound", "envelope": "planning_only", "basis": "metadata trial upper bounds x candidate contexts", "within_envelope": 1},
        {"resource": "raw_download_footprint", "estimate": "180", "unit": "GiB_upper_planning_bound", "envelope": "2048", "basis": "official repository summaries plus margin; no files downloaded", "within_envelope": 1},
        {"resource": "external_derived_payload", "estimate": "450", "unit": "GiB_safety", "envelope": "2048", "basis": "C78 measured row density scaled to <=13.5M rows plus states and 4x margin", "within_envelope": 1},
        {"resource": "combined_external_payload", "estimate": "630", "unit": "GiB_safety", "envelope": "2048", "basis": "download_plus_derived", "within_envelope": 1},
        {"resource": "parallelism", "estimate": "6", "unit": "concurrent_GPU_phases_max", "envelope": "engineering_gate", "basis": "one per dataset-panel without target duplication", "within_envelope": 1},
        {"resource": "retry_envelope", "estimate": "1", "unit": "replacement_per_failed_phase", "envelope": "engineering_only", "basis": "first valid success; all attempts preserved", "within_envelope": 1},
    ]
    for dataset, wall_hours, download_gib, derived_gib in (
        ("Lee2019_MI", 12, 80, 120),
        ("Cho2017", 12, 80, 110),
        ("PhysionetMI", 18, 20, 220),
    ):
        rows.extend((
            {"resource": f"{dataset}_wall_time_safety", "estimate": str(wall_hours), "unit": "hours", "envelope": "planning_only", "basis": "24_phases_with_two_dataset_local_concurrent_workers_and_IO_margin", "within_envelope": 1},
            {"resource": f"{dataset}_raw_download", "estimate": str(download_gib), "unit": "GiB_upper_planning_bound", "envelope": "2048_combined", "basis": "official_repository_metadata_plus_margin;no_files_downloaded", "within_envelope": 1},
            {"resource": f"{dataset}_derived_payload", "estimate": str(derived_gib), "unit": "GiB_safety", "envelope": "2048_combined", "basis": "target_count_and_candidate_context_scaled_planning_bound", "within_envelope": 1},
        ))
    return rows


def risk_rows() -> list[dict[str, Any]]:
    risks = [
        "single_dataset_claim_carried_into_C84", "binary_task_called_exact_four_class_replication",
        "source_target_subject_overlap", "source_panel_overlap", "subject_partition_outcome_driven",
        "Physionet_subject88_not_excluded", "unlabeled_online_Lee_runs_used_as_labeled",
        "motor_execution_entering_Physionet", "class_mapping_drift", "channel_alias_drift",
        "missing_common_channel_silently_dropped", "dataset_specific_filter_tuning",
        "trial_window_misalignment", "sampling_resample_mismatch", "official_bad_trial_policy_inconsistent",
        "target_labels_reaching_training", "construction_evaluation_overlap", "oracle_reachability",
        "candidate_count_differs_across_datasets", "target_specific_retraining_accidentally_used",
        "checkpoint_identity_drift", "new_training_seed_reuse_confusion", "method_formula_or_prior_retuned",
        "different_methods_support_cross_dataset_claim", "subjects_pooled_as_iid_across_datasets",
        "panel_seed_replicates_treated_as_independent_N", "Physionet_low_label_budget_overreach",
        "FULL_called_fixed_label_count", "source_relative_gain_called_low_absolute_regret",
        "top1_called_regret", "dataset_heterogeneity_hidden", "BNCI2014_004_silently_consumed",
        "raw_EEG_or_weights_in_git", "unauthorized_real_data_access", "manuscript_claim_before_C84_review",
    ]
    rows = []
    for risk in risks:
        blocker = risk == "missing_common_channel_silently_dropped"
        rows.append({
            "risk": risk, "blocking": int(blocker),
            "status": "OPEN_BLOCKER" if blocker else "CLOSED_BY_LOCKED_C84P_CONTROL",
            "evidence": CHANNEL_BLOCKER if blocker else "metadata/synthetic protocol control",
            "real_outcome_access": 0,
        })
    return rows


def build_external_protocol() -> dict[str, Any]:
    registry = validate_registry()
    return {
        "schema_version": "c84_multidataset_external_validity_protocol_v1",
        "milestone": "C84P",
        "status": "BLOCKED_CHANNEL_RECONCILIATION_NOT_EXECUTABLE",
        "created_at_utc": utc_now(),
        "parent": {"C83P_HEAD": PARENT_HEAD, "C83P_gate": C83_GATE, "C83_snapshot_mutated": False},
        "epistemic_status": {
            "external_datasets": True, "new_independent_cohorts": True,
            "binary_task_not_exact_four_class_replication": True,
            "prospective_to_all_C84_model_and_selector_outcomes": True,
            "universal_EEG_validity_claim": False,
        },
        "dataset_registry": dataset_registry_payload(),
        "dataset_metadata_validation": registry,
        "subject_partition": {
            "salt": SUBJECT_PARTITION_SALT, "source_panel_A": 16, "source_panel_B": 16,
            "source_training_per_panel": 12, "source_audit_per_panel": 4,
            "target_counts": {code: len(partition_subjects(DATASETS[code])["targets"]) for code in DATASET_ORDER},
            "outcome_driven": False,
        },
        "harmonization": {
            "task": "left_hand_vs_right_hand_motor_imagery",
            "requested_channels": list(PRIMARY_CHANNELS),
            "channel_interpolation": False,
            "channel_substitution": False,
            "open_blocker": CHANNEL_BLOCKER,
            "epoch": "half_open_[0.0,3.0)_after_official_cue_onset",
            "sfreq_hz": 160, "n_times": 480,
            "bandpass_hz": [4.0, 38.0],
            "resample": "MNE_1.11_Epochs.resample_fft_npad_auto_window_auto_pad_edge_then_exact_480",
            "normalization": "zscore_per_trial_per_channel_epsilon_1e-8",
            "spatial_interpolation": False,
        },
        "candidate_field": {
            "panels": list(PANELS), "seeds": list(SEEDS), "levels": list(LEVELS),
            "regimes": list(REGIMES), "units_per_zoo": UNITS_PER_ZOO,
            "units_per_dataset": 648, "total_units": TOTAL_UNITS, "training_phases": TOTAL_PHASES,
            "target_contexts": TOTAL_CONTEXTS, "candidate_context_evaluations": TOTAL_CANDIDATE_CONTEXTS,
            "target_specific_retraining": False,
        },
        "selectors": {
            "parent_method_registry_sha256": METHOD_REGISTRY_SHA256,
            "primary_zero_label": list(PRIMARY_ZERO_METHODS), "strict_source": "S1",
            "labeled_primary": "Q0_B1", "new_method_or_retuning": False,
        },
        "budgets": {"common_primary": list(COMMON_BUDGETS), "Lee_Cho_secondary": list(EXTENDED_BUDGETS), "FULL_cell_specific": True},
        "physical_views": [row["view"] for row in view_rows()],
        "stage_protocols": {
            "C84C": "oaci/reports/C84_CANARY_PROTOCOL.json",
            "C84F": "oaci/reports/C84_FIELD_GENERATION_PROTOCOL.json",
            "C84S": "oaci/reports/C84_SCIENTIFIC_ANALYSIS_PROTOCOL.json",
            "separate_execution_locks_required": True,
        },
        "authorization": {
            "PI_message": "authorizes C84P C84C C84F C84S in intent",
            "C84P_authorization_required": False,
            "C84C_C84F_C84S_authorization_consumable_now": False,
            "reason": "scope-specific protocols are blocked and execution locks do not exist",
            "magic_token_required": False,
        },
        "resource_envelopes": {"GPU_phase_hours": 250, "external_payload_TiB": 2.0, "Git_max_file_MiB": 50},
        "blocking_gate": FAIL_GATE,
    }


def build_canary_protocol(external_sha: str) -> dict[str, Any]:
    canary_targets = {dataset: partition_subjects(DATASETS[dataset])["targets"][0] for dataset in DATASET_ORDER}
    return {
        "schema_version": "c84_canary_protocol_v1", "milestone": "C84C",
        "status": "BLOCKED_NOT_READY_FOR_AUTHORIZATION", "parent_protocol_sha256": external_sha,
        "scope": {"datasets": list(DATASET_ORDER), "panel": "A", "seed": 5, "level": 0,
                  "target_subjects": canary_targets, "units_per_dataset": 81, "total_units": 243,
                  "training_phases": 9, "scientific_metrics": False},
        "reusable_in_C84F": "yes_only_if_exact_unit_ids_configs_views_and_all_engineering_checks_pass",
        "checks": ["loader", "event_mapping", "channel_list", "resampling", "480_sample_shape",
                   "source_target_isolation", "training_identity", "instrumentation_replay", "storage_runtime"],
        "forbidden": ["target_scientific_metric", "construction_labels", "evaluation_labels", "oracle", "GPU_without_scope_lock"],
        "open_blocker": CHANNEL_BLOCKER,
        "future_execution_lock_required": True,
    }


def build_field_protocol(external_sha: str) -> dict[str, Any]:
    units = candidate_units()
    return {
        "schema_version": "c84_field_generation_protocol_v1", "milestone": "C84F",
        "status": "BLOCKED_NOT_READY_FOR_AUTHORIZATION", "parent_protocol_sha256": external_sha,
        "field": {"total_units": len(units), "training_phases": TOTAL_PHASES,
                  "canary_reusable_units": sum(row["canary_subset"] for row in units),
                  "target_contexts": TOTAL_CONTEXTS, "candidate_context_evaluations": TOTAL_CANDIDATE_CONTEXTS},
        "historical_implementation": {
            "C79_field_lock_commit": "35d0c65d76a6a094dd4a73cd3412363a70764f7c",
            "ERM": "exact_historical_stage1_final_anchor", "OACI": "exact_historical_40_checkpoint_trajectory",
            "SRC": "exact_C11_negative_control_commit_2555b36_smooth_temperature_0.1",
            "epochs": 200, "checkpoint_every": 5,
        },
        "waves": {
            "C84C_reuse": "panel_A_seed5_level0_if_validated",
            "A": "remaining_seed5_729_units_27_phases",
            "B": "all_seed6_972_units_36_phases",
            "gate_inputs": "engineering_only_no_target_outcomes",
        },
        "isolation_zero_counts": ["training_target_rows", "training_target_labels", "target_outcome_retention", "target_outcome_retry"],
        "freeze_gate": "C84_MULTI_DATASET_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
        "open_blocker": CHANNEL_BLOCKER,
        "future_execution_lock_required": True,
    }


def build_science_protocol(external_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "c84_scientific_analysis_protocol_v1", "milestone": "C84S",
        "status": "BLOCKED_NOT_READY_FOR_AUTHORIZATION", "parent_protocol_sha256": external_sha,
        "questions": [row["question"] for row in inference_rows()],
        "within_dataset_aggregation": "average_levels_and_four_panel_x_seed_cells_within_target_first",
        "Q1": {"mean_S1_minus_method_min": 0.05, "familywise_p_max": 0.05, "favorable_target_fraction_min": 0.75,
               "worst_target_min": -0.10, "positive_panel_seed_cells_min": 3, "panel_seed_cells": 4},
        "Q2": {"mean_method_minus_Q0B1_max": 0.05, "simultaneous_upper_max": 0.05, "familywise_p_max": 0.05,
               "within_margin_target_fraction_min": 0.75, "worst_target_excess_max": 0.20,
               "within_margin_panel_seed_cells_min": 3, "panel_seed_cells": 4},
        "maxT": {"draws": 65536, "RNG": "SHA256(C84_MAXT_V1|dataset|family)", "principal_cluster": "target_subject",
                 "pooled_three_dataset_pvalue": False},
        "target_composition": {"method_identity_preserved": True, "LOTO_all_datasets": True,
                               "preservation_fraction_min": 0.75},
        "cross_dataset": {
            "A": "intersection_of_dataset_methods_passing_Q1_and_Q2_nonempty",
            "B": "A_intersection_empty_and_intersection_of_dataset_Q1_methods_nonempty",
            "C": "no_primary_zero_label_Q1_pass_in_any_dataset",
            "D": "dataset_panel_seed_or_target_heterogeneity_or_different_method_identity",
            "same_method_required": True,
        },
        "label_frontier": {
            "budgets": list(COMMON_BUDGETS), "closure": "smallest_budget_and_all_larger_qualify",
            "qualification": "source_relative_gain>=0.05|maxT_p<=0.05|75pct_targets|worst>=-0.10|3of4_panel_seed",
            "L1": "all_Bstar_exist_stable_ordinal_distance<=1_and_max_Bstar<=4",
            "L2": "all_Bstar_exist_stable_ordinal_distance<=1_and_max_Bstar_in_{8,FULL}",
            "L3": "all_Bstar_exist_but_distance>1_or_registered_heterogeneity",
            "L4": "one_or_more_Bstar_absent",
        },
        "taxonomy_precedence": [
            "C84-E_multidataset_protocol_field_view_analysis_or_provenance_blocker",
            "C84-D_external_dataset_source_panel_seed_or_target_heterogeneous",
            "C84-A_same_zero_label_selector_matches_B1_across_all_external_datasets",
            "C84-B_same_zero_label_selector_improves_source_across_all_external_datasets_but_not_B1",
            "C84-C_no_registered_zero_label_selector_materially_improves_source_in_any_external_dataset",
        ],
        "open_blocker": CHANNEL_BLOCKER,
        "future_execution_lock_required": True,
    }


def generate() -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "dataset_eligibility_registry.csv": dataset_rows(),
        "dataset_license_and_access.csv": [{
            "dataset": code, "dataset_doi": DATASETS[code].dataset_doi, "paper_doi": DATASETS[code].paper_doi,
            "repository": DATASETS[code].repository, "license": DATASETS[code].license,
            "official_metadata_url": DATASETS[code].official_metadata_url, "downloaded_in_C84P": 0,
            "real_arrays_opened": 0,
        } for code in DATASET_ORDER],
        "subject_partition_registry.csv": subject_partition_rows(),
        "source_target_overlap_audit.csv": overlap_rows(),
        "class_mapping_registry.csv": class_mapping_rows(),
        "event_window_registry.csv": event_window_rows(),
        "channel_allowlist_registry.csv": channel_rows(),
        "channel_alias_registry.csv": channel_alias_rows(),
        "preprocessing_contract.csv": preprocessing_rows(),
        "bad_trial_policy.csv": bad_trial_rows(),
        "candidate_field_arithmetic.csv": candidate_arithmetic_rows(),
        "target_context_arithmetic.csv": context_arithmetic_rows(),
        "common_and_extended_budget_registry.csv": budget_rows(),
        "selector_registry_replay.csv": selector_rows(),
        "information_view_matrix.csv": view_rows(),
        "physical_split_contract.csv": physical_split_rows(),
        "inference_registry.csv": inference_rows(),
        "cross_dataset_taxonomy_truth_table.csv": taxonomy_truth_rows(),
        "resource_estimate.csv": resource_rows(),
        "risk_register.csv": risk_rows(),
        "failure_reason_ledger.csv": [{
            "failure_id": "C84P_CHANNEL_001", "stage": "metadata_compatibility",
            "blocking": 1, "root_cause": CHANNEL_BLOCKER,
            "affected_object": "requested_21_channel_primary_allowlist",
            "real_EEG_or_label_access": 0, "outcome_dependent_decision": 0,
            "required_repair": "PM-approved additive availability-only montage revision; no silent channel drop or FCz/Fz substitution",
            "status": "OPEN",
        }],
    }
    for name, rows in tables.items():
        write_csv(TABLE_DIR / name, rows)

    protocol_path = REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL.json"
    write_json(protocol_path, build_external_protocol())
    protocol_sha = sha256_file(protocol_path)
    write_sha(REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL.sha256", protocol_sha, protocol_path.name)
    stage_builders = (
        ("C84_CANARY_PROTOCOL", build_canary_protocol),
        ("C84_FIELD_GENERATION_PROTOCOL", build_field_protocol),
        ("C84_SCIENTIFIC_ANALYSIS_PROTOCOL", build_science_protocol),
    )
    stage_hashes = {}
    for stem, builder in stage_builders:
        path = REPORT_DIR / f"{stem}.json"
        write_json(path, builder(protocol_sha))
        digest = sha256_file(path)
        write_sha(REPORT_DIR / f"{stem}.sha256", digest, path.name)
        stage_hashes[stem] = digest

    return {
        "protocol_sha256": protocol_sha,
        "stage_hashes": stage_hashes,
        "tables": len(tables),
        "channel_blocker": CHANNEL_BLOCKER,
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "candidate_units_created": 0,
        "execution_locks_created": 0,
        "gate": FAIL_GATE,
    }


def main() -> int:
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
