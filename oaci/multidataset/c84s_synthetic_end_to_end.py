"""Synthetic calibration through the production C84S result-freeze entrypoint."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any

from . import c84s_analysis as analysis
from . import c84s_inference as inference
from . import c84s_label_views as labels
from . import c84s_selection_freeze as selection
from . import c84s_taxonomy as taxonomy
from .c84s_common import C84SContractError, read_csv, require


SELECTION_IDENTITY = {
    "status": "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
    "sha256": "a" * 64,
}
EVALUATION_IDENTITY = {"kind": "evaluation", "manifest_sha256": "b" * 64}


def _target_ids(dataset: str) -> list[str]:
    return [f"{dataset}_T{index:03d}" for index in range(analysis.DATASET_TARGET_COUNTS[dataset])]


def _chosen_method(scenario: str, dataset: str) -> tuple[str | None, str]:
    if scenario == "S1":
        return ("U13", "Q1") if dataset == "Lee2019_MI" else (None, "NULL")
    if scenario in {"S2", "S5", "S6"}:
        return "U13", "Q1"
    if scenario == "S3":
        return "U13", "Q2"
    if scenario == "S4":
        return {"Lee2019_MI": "U5", "Cho2017": "U13", "PhysionetMI": "U14"}[dataset], "Q1"
    return None, "NULL"


def _label_frontier_regret(scenario: str, dataset: str, method: str) -> float:
    default = {
        "Q0_B1": 0.35, "Q0_B2": 0.30, "Q0_B4": 0.25,
        "Q0_B8": 0.20, "Q0_FULL": 0.15,
        "Q0_B16": 0.12, "Q0_B32": 0.10,
    }
    if scenario == "S9":
        default.update({"Q0_B1": 0.80, "Q0_B2": 0.80, "Q0_B4": 0.80, "Q0_B8": 0.30, "Q0_FULL": 0.20})
    elif scenario == "S10":
        if dataset == "Cho2017":
            default.update({"Q0_B1": 0.80, "Q0_B2": 0.80, "Q0_B4": 0.80, "Q0_B8": 0.30, "Q0_FULL": 0.20})
        elif dataset == "PhysionetMI":
            default.update({"Q0_B1": 0.80, "Q0_B2": 0.30, "Q0_B4": 0.25, "Q0_B8": 0.20, "Q0_FULL": 0.15})
    elif scenario == "S11" and dataset == "PhysionetMI":
        default.update({key: 0.80 for key in analysis.PRIMARY_Q0})
    return default[method]


def synthetic_method_context_rows(scenario: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in analysis.DATASET_TARGET_COUNTS:
        selected_method, selected_mode = _chosen_method(scenario, dataset)
        for target_index, target in enumerate(_target_ids(dataset)):
            target_noise = ((target_index % 5) - 2) * 0.001
            for panel_index, panel in enumerate(analysis.PANELS):
                for seed_index, seed in enumerate(analysis.SEEDS):
                    for level in analysis.LEVELS:
                        context_noise = target_noise + (panel_index + seed_index + level) * 0.0002
                        source_regret = 0.82 + context_noise
                        for method in analysis.expected_methods(dataset):
                            if method == "B5":
                                regret = 0.0
                            elif method == "S1":
                                regret = source_regret
                            elif method in analysis.PRIMARY_Q0 or method in analysis.SECONDARY_Q0:
                                regret = _label_frontier_regret(scenario, dataset, method) + context_noise
                            elif method in analysis.PRIMARY_METHODS:
                                regret = 0.79 + context_noise
                                if method == selected_method:
                                    regret = (0.30 if selected_mode == "Q2" else 0.42) + context_noise
                                    if scenario == "S5" and level == 1:
                                        regret = source_regret
                                    if scenario == "S6" and panel == "B" and seed == 6:
                                        regret = source_regret + 0.02
                            else:
                                regret = {
                                    "B0": 0.72, "B1": 0.75, "B2": 0.70,
                                    "B3": 0.71, "B4O": 0.73, "B4S": 0.74,
                                }[method] + context_noise
                            gain = 0.0 if source_regret <= 1e-15 else (source_regret - regret) / source_regret
                            high_topk = scenario == "S13" and method == "U14"
                            high_measurement = scenario == "S12" and method == "U14"
                            top1 = 1.0 if method == "B5" else 0.0
                            top5 = 1.0 if method == "B5" or high_topk else 0.25
                            top10 = 1.0 if method == "B5" or high_topk else 0.40
                            rows.append({
                                "dataset": dataset, "target_subject_id": target,
                                "panel": panel, "training_seed": seed, "level": level,
                                "method_id": method, "standardized_regret": regret,
                                "selected_utility": 1.0 - 0.75 * regret,
                                "source_relative_regret_gain": gain,
                                "top1": top1, "top5": top5, "top10": top10,
                                "coverage": 1.0,
                                "selected_regime": "ORACLE" if method == "B5" else "SYNTHETIC",
                                "catastrophic_failure": int(regret > 0.95),
                                "Spearman": 0.95 if high_measurement else 0.10,
                                "Kendall": 0.85 if high_measurement else 0.08,
                                "pairwise_ordering_accuracy": 0.90 if high_measurement else 0.55,
                                "accuracy_estimation_MAE": None,
                            })
    require(len(rows) == 18608, "synthetic method-context arithmetic drift")
    return rows


def _synthetic_label_registry() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_counts = {"Lee2019_MI": 2200, "Cho2017": 4000, "PhysionetMI": 3421}
    registry: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for dataset, target_count in analysis.DATASET_TARGET_COUNTS.items():
        subjects = _target_ids(dataset)
        base, remainder = divmod(row_counts[dataset], target_count)
        for subject_index, subject in enumerate(subjects):
            count = base + int(subject_index < remainder)
            for index in range(count):
                row = {
                    "dataset": dataset, "target_subject_id": subject,
                    "target_trial_id": f"{dataset}|{subject}|trial-{index:04d}",
                    "session": f"session-{index % 2}", "run": f"run-{index % 4}",
                }
                registry.append(row)
                label_rows.append({**row, "canonical_class_label": index % 2})
    require(len(registry) == len(label_rows) == 9621, "synthetic label-registry arithmetic drift")
    return registry, label_rows


def _run_stage_c(scenario: str, *, blocker: bool = False, failure: str | None = None) -> tuple[dict[str, Any], Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name) / "result"
    try:
        result = analysis.run_analysis_and_freeze(
            synthetic_method_context_rows(scenario),
            selection_freeze_identity=SELECTION_IDENTITY,
            evaluation_view_identity=EVALUATION_IDENTITY,
            final_root=root, draws=64, blocker=blocker, synthetic=True,
            failure_injection_after=failure,
        )
    except BaseException:
        temporary.cleanup()
        raise
    # Keep the temporary directory alive until callers finish inspecting it.
    result["_temporary_directory"] = temporary
    return result, root


@lru_cache(maxsize=1)
def synthetic_calibration_rows() -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []

    def record(scenario: str, expected: str, observed: str, detail: str) -> None:
        output.append({
            "scenario": scenario, "expected": expected, "observed": observed,
            "pass": int(expected == observed), "detail": detail,
            "real_label_access": 0, "real_selector_score": 0,
        })

    expected_gate = {
        "S0": taxonomy.GATE_C, "S1": taxonomy.GATE_D,
        "S2": taxonomy.GATE_B, "S3": taxonomy.GATE_A,
        "S4": taxonomy.GATE_D, "S5": taxonomy.GATE_D,
        "S6": taxonomy.GATE_D,
    }
    for scenario, expected in expected_gate.items():
        result, root = _run_stage_c(scenario)
        require((root / "C84S_RESULT_ARTIFACT_MANIFEST.json").is_file(), "synthetic result manifest absent")
        record(scenario, expected, result["primary_gate"], "production Stage-C gate")
        result.pop("_temporary_directory").cleanup()

    loto = inference.loto_preservation(
        dataset="Lee2019_MI", full_supporting_methods=["U13"],
        omitted_panel_method_sets=[["U13"]] * 16 + [["U5"]] * 6,
    )
    observed = taxonomy.GATE_D if not loto["pass"] else taxonomy.GATE_B
    record("S7", taxonomy.GATE_D, observed, "same-method LOTO below 17/22")

    expected_label = {"S8": "C84-L1", "S9": "C84-L2", "S10": "C84-L3", "S11": "C84-L4"}
    for scenario, expected in expected_label.items():
        result, _ = _run_stage_c(scenario)
        record(scenario, expected, result["label_frontier_tag"], "production label-frontier result")
        result.pop("_temporary_directory").cleanup()

    result, root = _run_stage_c("S12")
    measurement = read_csv(root / "measurement_vs_decision.csv")
    row = next(value for value in measurement if value["dataset"] == "Lee2019_MI" and value["method_id"] == "U14")
    observed = "NO_Q1_SUBSTITUTION" if row["Q1_pass"] == "0" and row["measurement_substitutes_for_regret"] == "0" else "SUBSTITUTED"
    record("S12", "NO_Q1_SUBSTITUTION", observed, "strong association remains measurement-only")
    result.pop("_temporary_directory").cleanup()

    result, root = _run_stage_c("S13")
    topk = read_csv(root / "topk_decision_summary.csv")
    row = next(value for value in topk if value["dataset"] == "Lee2019_MI" and value["method_id"] == "U14")
    observed = "NO_Q1_SUBSTITUTION" if row["Q1_pass"] == "0" and row["endpoint_substitutes_for_regret"] == "0" else "SUBSTITUTED"
    record("S13", "NO_Q1_SUBSTITUTION", observed, "high top-k remains secondary")
    result.pop("_temporary_directory").cleanup()

    duplicated = synthetic_method_context_rows("S0")
    duplicated[-1] = dict(duplicated[0])
    try:
        analysis.validate_method_context_rows(duplicated)
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S14", "REJECTED", observed, "duplicate target-context row")

    chain_rows = synthetic_method_context_rows("S0")
    chain_rows[0] = {**chain_rows[0], "chain": 0}
    try:
        analysis.validate_method_context_rows(chain_rows)
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S15", "REJECTED", observed, "Monte Carlo chain cannot enter target-cluster table")

    try:
        selection.validate_selection_inputs({"evaluation_label_view": "/forbidden"})
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S16", "REJECTED", observed, "evaluation descriptor before selection freeze")

    shared = {"dataset": "D", "target_subject_id": "1", "target_trial_id": "x", "session": "s", "run": "r"}
    try:
        labels.assert_physical_disjointness([shared], [shared])
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S17", "REJECTED", observed, "construction/evaluation overlap")

    registry, label_rows = _synthetic_label_registry()
    drift = [dict(row) for row in label_rows]
    drift[0]["target_trial_id"] = "drift"
    try:
        labels.align_and_split_labels(registry, drift)
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S18", "REJECTED", observed, "label row-order/identity drift")
    _, _, audit = labels.align_and_split_labels(registry, label_rows)
    record("S19", "PASS_9621", f"PASS_{audit['registry_rows']}", "exact synthetic label alignment")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "failed-result"
        try:
            analysis.run_analysis_and_freeze(
                synthetic_method_context_rows("S0"),
                selection_freeze_identity=SELECTION_IDENTITY,
                evaluation_view_identity=EVALUATION_IDENTITY,
                final_root=root, draws=64, synthetic=True,
                failure_injection_after="method_context_decisions.csv",
            )
            observed = "FINAL_VISIBLE"
        except C84SContractError:
            observed = "NO_FINAL_ROOT" if not root.exists() else "FINAL_VISIBLE"
    record("S20", "NO_FINAL_ROOT", observed, "atomic Stage-C publication failure")

    require(len(output) == 21 and all(row["pass"] for row in output), "C84S synthetic S0-S20 failed")
    return tuple(output)
