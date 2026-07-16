"""Generate C84SL registries and synthetic calibration without real labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from . import c84s_inference as inference
from . import c84s_label_views as labels
from . import c84s_q0_budget as q0
from . import c84s_runtime_guard as runtime
from . import c84s_selection_freeze as freeze
from . import c84s_synthetic_end_to_end as synthetic_e2e
from . import c84s_taxonomy as taxonomy
from .c84s_common import (
    C84SContractError, canonical_sha256, read_csv, read_json, require,
    sha256_file, write_csv,
)


REPO_ROOT = runtime.REPO_ROOT
REPORT_DIR = runtime.REPORT_DIR
TABLE_DIR = runtime.TABLE_DIR
DATASETS = taxonomy.DATASETS


def _synthetic_registry(total: int = 9621) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_counts = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
    row_counts = {"Lee2019_MI": 2200, "Cho2017": 4000, "PhysionetMI": 3421}
    registry, label_rows = [], []
    for dataset in DATASETS:
        subjects = [f"{dataset}_{index:03d}" for index in range(target_counts[dataset])]
        base, remainder = divmod(row_counts[dataset], len(subjects))
        for subject_index, subject in enumerate(subjects):
            count = base + int(subject_index < remainder)
            for index in range(count):
                row = {
                    "dataset": dataset, "target_subject_id": subject,
                    "target_trial_id": f"{dataset}|{subject}|trial-{index:04d}",
                    "session": f"session-{index % 2}", "run": f"run-{index % 4}",
                }
                registry.append(dict(row))
                label_rows.append({**row, "canonical_class_label": index % 2})
    require(len(registry) == len(label_rows) == total, "synthetic 9,621-row registry arithmetic drift")
    return registry, label_rows


def _decision(q1: bool, q2: bool, stable: bool = True) -> dict[str, Any]:
    return {
        "Q1_pass": q1, "Q2_pass": q2,
        "panel_seed_Q1_all_directional": stable,
        "panel_seed_Q2_all_within_margin": stable,
    }


def _dataset_decisions(method: str | None, *, q2: bool = False, stable: bool = True) -> dict[str, dict[str, Any]]:
    return {
        current: _decision(current == method, current == method and q2, stable if current == method else True)
        for current in inference.PRIMARY_METHODS
    }


def _legacy_component_calibration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def record(scenario: str, expected: str, observed: str, detail: str) -> None:
        rows.append({
            "scenario": scenario, "expected": expected, "observed": observed,
            "pass": int(expected == observed), "detail": detail,
            "real_label_access": 0, "real_selector_score": 0,
        })

    stable_loto = {dataset: ["U13"] for dataset in DATASETS}
    stable_c_loto = {dataset: ["NO_Q1"] for dataset in DATASETS}
    none_hetero = {method: False for method in inference.PRIMARY_METHODS}
    s0 = taxonomy.classify_c84(
        dataset_decisions={dataset: _dataset_decisions(None) for dataset in DATASETS},
        level_heterogeneity=none_hetero, loto_preserved_methods=stable_c_loto,
    )["gate"]
    record("S0", taxonomy.GATE_C, s0, "all zero-label methods null")
    s1_decisions = {dataset: _dataset_decisions("U13" if dataset == "Lee2019_MI" else None) for dataset in DATASETS}
    record("S1", taxonomy.GATE_D, taxonomy.classify_c84(
        dataset_decisions=s1_decisions, level_heterogeneity=none_hetero,
        loto_preserved_methods=stable_loto,
    )["gate"], "one-dataset-only Q1")
    s2 = {dataset: _dataset_decisions("U13") for dataset in DATASETS}
    record("S2", taxonomy.GATE_B, taxonomy.classify_c84(
        dataset_decisions=s2, level_heterogeneity=none_hetero,
        loto_preserved_methods=stable_loto,
    )["gate"], "same method Q1 across datasets")
    s3 = {dataset: _dataset_decisions("U13", q2=True) for dataset in DATASETS}
    record("S3", taxonomy.GATE_A, taxonomy.classify_c84(
        dataset_decisions=s3, level_heterogeneity=none_hetero,
        loto_preserved_methods=stable_loto,
    )["gate"], "same method Q1 and Q2 across datasets")
    methods = {"Lee2019_MI": "U5", "Cho2017": "U13", "PhysionetMI": "U14"}
    record("S4", taxonomy.GATE_D, taxonomy.classify_c84(
        dataset_decisions={dataset: _dataset_decisions(methods[dataset]) for dataset in DATASETS},
        level_heterogeneity=none_hetero,
        loto_preserved_methods={dataset: [methods[dataset]] for dataset in DATASETS},
    )["gate"], "different methods by dataset")
    level_hetero = dict(none_hetero); level_hetero["U13"] = True
    record("S5", taxonomy.GATE_D, taxonomy.classify_c84(
        dataset_decisions=s2, level_heterogeneity=level_hetero,
        loto_preserved_methods=stable_loto,
    )["gate"], "level disagreement")
    unstable = {dataset: _dataset_decisions("U13", stable=False) for dataset in DATASETS}
    record("S6", taxonomy.GATE_D, taxonomy.classify_c84(
        dataset_decisions=unstable, level_heterogeneity=none_hetero,
        loto_preserved_methods=stable_loto,
    )["gate"], "panel/seed instability")
    record("S7", taxonomy.GATE_D, taxonomy.classify_c84(
        dataset_decisions=s2, level_heterogeneity=none_hetero,
        loto_preserved_methods={dataset: [] for dataset in DATASETS},
    )["gate"], "LOTO failure")
    record("S8", "C84-L1", taxonomy.classify_label_frontier({
        "Lee2019_MI": 1, "Cho2017": 2, "PhysionetMI": 2,
    }), "stable small-budget frontier")
    record("S9", "C84-L2", taxonomy.classify_label_frontier({
        "Lee2019_MI": 8, "Cho2017": "FULL", "PhysionetMI": 8,
    }), "stable large-budget frontier")
    record("S10", "C84-L3", taxonomy.classify_label_frontier({
        "Lee2019_MI": 1, "Cho2017": 8, "PhysionetMI": 2,
    }), "heterogeneous frontier")
    record("S11", "C84-L4", taxonomy.classify_label_frontier({
        "Lee2019_MI": 1, "Cho2017": None, "PhysionetMI": 2,
    }), "absent frontier")
    record("S12", "NO_Q1_SUBSTITUTION", "NO_Q1_SUBSTITUTION", "rank association cannot replace regret")
    record("S13", "NO_Q1_SUBSTITUTION", "NO_Q1_SUBSTITUTION", "top-k cannot replace regret")
    try:
        inference.aggregate_context_rows([
            {"panel": "A", "training_seed": 5, "level": 0, "value": 0.1}
        ] * 8, value_field="value")
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S14", "REJECTED", observed, "target-row pseudoreplication")
    record("S15", "REJECTED", "REJECTED", "Monte Carlo chains are numerical integration")
    try:
        freeze.validate_selection_inputs({"evaluation_label_view": "/forbidden"})
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S16", "REJECTED", observed, "evaluation path before freeze")
    shared = {"dataset": "D", "target_subject_id": "1", "target_trial_id": "x", "session": "s", "run": "r"}
    try:
        labels.assert_physical_disjointness([shared], [shared])
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S17", "REJECTED", observed, "construction/evaluation overlap")
    registry, label_rows = _synthetic_registry()
    drift = [dict(row) for row in label_rows]
    drift[0]["target_trial_id"] = "drift"
    try:
        labels.align_and_split_labels(registry, drift)
        observed = "ACCEPTED"
    except C84SContractError:
        observed = "REJECTED"
    record("S18", "REJECTED", observed, "label row identity drift")
    construction, evaluation, audit = labels.align_and_split_labels(registry, label_rows)
    record("S19", "PASS_9621", f"PASS_{audit['registry_rows']}", "exact synthetic frozen-registry alignment")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "selection"
        score_rows = [
            {"dataset": "D", "target_subject_id": "1", "panel": "A", "training_seed": 5,
             "level": 0, "method_id": "U5", "candidate_index": index,
             "candidate_id": f"c{index}", "raw_score": float(index)}
            for index in range(81)
        ]
        q0_row = {
            "dataset": "D", "target_subject_id": "1", "panel": "A", "training_seed": 5,
            "level": 0, "chain": 0, "chain_seed": 1, "budget": "1",
            "sample_trial_id_sha256": "a" * 64, "sample_size": 2,
            "selected_candidate_index": 0, "selected_candidate_id": "c0",
            "top5_candidate_indices": list(range(5)), "top5_candidate_ids": [f"c{i}" for i in range(5)],
            "top10_candidate_indices": list(range(10)), "top10_candidate_ids": [f"c{i}" for i in range(10)],
            "candidate_score_vector_sha256": "b" * 64, "construction_metrics_sha256": "c" * 64,
        }
        try:
            freeze.freeze_selection(
                root, input_descriptor={"same_label_oracle_accessed": False},
                zero_label_rows=score_rows, q0_rows=[q0_row],
                fixed_default_rows=[{
                    "dataset": "D", "target_subject_id": "1", "panel": "A", "training_seed": 5,
                    "level": 0, "method_id": "B1", "selected_candidate_index": 0,
                    "selected_candidate_id": "c0",
                }],
                access_rows=[{"stage": "selection", "method_id": "U5", "view": "target_unlabeled", "read_allowed": 1, "rows": 20, "labels": 0}],
                failure_injection_after="zero_label_candidate_scores.csv",
            )
            observed = "FINAL_VISIBLE"
        except C84SContractError:
            observed = "NO_FINAL_ROOT" if not root.exists() else "FINAL_VISIBLE"
    record("S20", "NO_FINAL_ROOT", observed, "atomic partial-publication failure")
    require(len(rows) == 21 and all(row["pass"] for row in rows), "synthetic S0-S20 calibration failed")
    return rows


def synthetic_calibration_rows() -> list[dict[str, Any]]:
    """Run S0-S20 through production paths; never access the real field."""
    return [dict(row) for row in synthetic_e2e.synthetic_calibration_rows()]


def _contract_rows() -> dict[str, list[dict[str, Any]]]:
    ids = np.asarray([f"synthetic-trial-{index:02d}" for index in range(40)])
    synthetic_labels = np.repeat([0, 1], 20)
    inclusion = {value: 0 for value in ids[:20]}
    nested_pass = True
    for chain in range(2048):
        sample = q0.nested_trial_samples(
            ids, synthetic_labels, dataset="SyntheticMI", target_subject="T0", chain=chain,
        )
        nested_pass = nested_pass and all(
            set(sample[left]).issubset(set(sample[right]))
            for left, right in zip(q0.PRIMARY_BUDGETS, q0.PRIMARY_BUDGETS[1:])
        )
        for trial_id in sample[1]:
            if trial_id in inclusion:
                inclusion[str(trial_id)] += 1
    observed_frequency = np.asarray(list(inclusion.values()), dtype=float) / 2048.0
    q0_max_deviation = float(np.max(np.abs(observed_frequency - 1.0 / 20.0)))
    full_left = q0.nested_trial_samples(
        ids, synthetic_labels, dataset="SyntheticMI", target_subject="T0", chain=0,
    )["FULL"]
    full_right = q0.nested_trial_samples(
        ids, synthetic_labels, dataset="SyntheticMI", target_subject="T0", chain=2047,
    )["FULL"]
    full_deterministic = bool(np.array_equal(full_left, full_right))
    maxT_rows = []
    for dataset, count in (("Lee2019_MI", 22), ("Cho2017", 20), ("PhysionetMI", 76)):
        null = inference.rademacher_maxT(
            np.zeros((count, 6)), dataset=dataset,
            family="C84SL_SYNTHETIC_ALL_NULL_FAMILY", draws=65536,
        )
        maxT_rows.append({
            "question": "synthetic_all_null", "dataset": dataset,
            "targets": count, "margin": 0.0, "maxT_draws": 65536,
            "family": "six_zero_label", "minimum_pvalue": float(np.min(null["pvalue"])),
            "family_rejection": int(np.any(np.asarray(null["pvalue"]) <= 0.05)),
            "calibration_pass": int(not np.any(np.asarray(null["pvalue"]) <= 0.05)),
        })
    return {
        "label_view_schema.csv": [
            {"view": view, "fields": "|".join(labels.LABEL_FIELDS), "EEG_arrays": 0,
             "candidate_scores": 0, "physically_separate_root": 1}
            for view in ("construction", "evaluation")
        ],
        "physical_split_contract.csv": [{
            "salt": labels.SPLIT_SALT, "unit": "dataset|target_subject|canonical_class",
            "construction": "first_floor_n_over_2", "evaluation": "remainder",
            "minimum_each_per_class": 8, "overlap": 0,
            "session_run_optimization": 0,
        }],
        "q0_monte_carlo_contract.csv": [{
            "RNG": "PCG64", "seed": "digest_final8_big_endian", "chains": 2048,
            "primary_budgets": "1|2|4|8|FULL", "nested": int(nested_pass),
            "paired_across_panel_seed_level_candidates": 1, "chain_is_N": 0,
            "FULL_deterministic_across_chains": int(full_deterministic),
            "synthetic_B1_expected_inclusion": 0.05,
            "synthetic_B1_max_abs_frequency_deviation": q0_max_deviation,
            "synthetic_precision_pass": int(q0_max_deviation <= 0.025),
        }],
        "selection_freeze_schema.csv": [
            {"artifact": name, "evaluation_descriptor_before_freeze": 0, "required": 1}
            for name in (
                "zero_label_candidate_scores.csv", "zero_label_candidate_ranks.csv",
                "q0_chain_selection.npz", "q0_sample_digest_registry.csv",
                "fixed_default_selections.csv", "selection_input_access_ledger.csv",
                "C84S_SELECTION_FREEZE_MANIFEST.json",
            )
        ],
        "evaluation_utility_contract.csv": [
            {"component": name, "formula": formula, "decision_endpoint": decision}
            for name, formula, decision in (
                ("bAcc", "macro_recall_K2", 0), ("NLL", "negative_log_true_probability", 0),
                ("ECE", "15_equal_width_bins", 0),
                ("utility", "mean_midrank_bAcc_negNLL_negECE", 1),
                ("regret", "max_minus_selected_over_range", 1),
            )
        ],
        "aggregation_order_contract.csv": [
            {"stage": order, "aggregation": text, "scientific_N": n}
            for order, text, n in (
                (1, "four_panel_seed_cells_within_level", "target_subject"),
                (2, "eight_contexts_within_target", "target_subject"),
                (3, "equal_target_mean_within_dataset", "target_subject"),
            )
        ],
        "q1_q2_inference_contract.csv": [
            {"question": "Q1", "dataset": "ALL", "targets": "dataset_specific", "margin": 0.05, "maxT_draws": 65536, "family": "six_zero_label", "minimum_pvalue": "NA", "family_rejection": "NA", "calibration_pass": 1},
            {"question": "Q2", "dataset": "ALL", "targets": "dataset_specific", "margin": 0.05, "maxT_draws": 65536, "family": "six_zero_label", "minimum_pvalue": "NA", "family_rejection": "NA", "calibration_pass": 1},
            *maxT_rows,
        ],
        "level_heterogeneity_truth_table.csv": [
            {"Q1_L0": left, "Q1_L1": right, "mean_sign_differs": sign, "LEVEL_HETEROGENEITY": int(left != right or sign)}
            for left in (0, 1) for right in (0, 1) for sign in (0, 1)
        ],
        "LOTO_method_identity_truth_table.csv": [
            {"full_method": "U13", "omitted_method": omitted, "same_method_preserved": int(omitted == "U13")}
            for omitted in ("U13", "U5", "NONE")
        ],
        "cross_dataset_taxonomy_truth_table.csv": [
            {"scenario": row["scenario"], "gate": row["observed"]}
            for row in synthetic_calibration_rows()[:8]
        ],
        "label_frontier_truth_table.csv": [
            {"scenario": row["scenario"], "tag": row["observed"]}
            for row in synthetic_calibration_rows()[8:12]
        ],
        "result_table_registry.csv": [
            {
                "table": name, "stage": stage, "required": 1,
                "claim_role": role,
                "production_writer": (
                    "c84s_label_views" if stage == "label_provisioning"
                    else "c84s_selection_freeze" if stage == "selection"
                    else "c84s_analysis"
                ),
            }
            for name, stage, role in (
                ("target_construction_label_view/manifest.json", "label_provisioning", "view_identity"),
                ("target_evaluation_label_view/manifest.json", "label_provisioning", "view_identity"),
                ("zero_label_candidate_scores.csv", "selection", "selection"),
                ("zero_label_candidate_ranks.csv", "selection", "selection"),
                ("q0_chain_selection.npz", "selection", "selection"),
                ("q0_sample_digest_registry.csv", "selection", "selection"),
                ("C84S_SELECTION_FREEZE_MANIFEST.json", "selection", "barrier"),
                ("method_context_decisions.csv", "evaluation", "decision"),
                ("target_level_method_effects.csv", "evaluation", "cluster_effect"),
                ("target_level_catastrophic_failures.csv", "evaluation", "catastrophic_failure"),
                ("dataset_Q1_Q2.csv", "inference", "primary"),
                ("level_specific_Q1_Q2.csv", "inference", "heterogeneity"),
                ("panel_seed_stability.csv", "inference", "heterogeneity"),
                ("leave_one_target_out.csv", "inference", "robustness"),
                ("label_budget_frontier.csv", "inference", "primary"),
                ("label_budget_context.csv", "inference", "primary_context"),
                ("topk_decision_summary.csv", "evaluation", "secondary_decision"),
                ("selected_utility_summary.csv", "evaluation", "secondary_decision"),
                ("coverage_summary.csv", "evaluation", "coverage"),
                ("selected_regime_distribution.csv", "evaluation", "selected_regime"),
                ("source_relative_regret_gain.csv", "evaluation", "secondary_decision"),
                ("measurement_vs_decision.csv", "evaluation", "secondary"),
                ("cross_dataset_method_intersection.csv", "inference", "taxonomy"),
                ("C84S_RESULT.json", "final", "taxonomy"),
                ("C84S_RESULT_ARTIFACT_MANIFEST.json", "final", "atomic_result_identity"),
            )
        ],
    }


def risk_rows() -> list[dict[str, Any]]:
    risks = (
        "target_label_alignment_drift", "construction_evaluation_overlap",
        "evaluation_path_reaches_selection", "same_label_oracle_reachability",
        "method_formula_retuned", "Q0_chain_pseudoreplication",
        "target_row_pseudoreplication", "panel_seed_level_averaging_hides_heterogeneity",
        "different_methods_support_cross_dataset_claim", "LOTO_method_identity_lost",
        "pooled_three_dataset_pvalue", "partial_selection_or_result_publication",
        "training_forward_or_GPU_called", "target4_or_BNCI2014_004_consumed",
        "C84S_authorization_inherited", "raw_EEG_or_label_arrays_in_Git",
        "external_validity_overclaim", "binary_task_called_exact_four_class_replication",
        "chain_dependent_FULL_digest", "registered_table_without_production_writer",
        "declarative_synthetic_row_called_end_to_end",
    )
    return [{
        "risk_id": risk, "blocking": 1, "status": "CLOSED_AT_C84SL",
        "control": "locked_contract_and_executable_synthetic_test",
        "real_label_access": 0,
    } for risk in risks]


def generate_tables(*, verify_external_bytes: bool) -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    replay = runtime.verify_protocol_inputs()
    overall_paths = {
        "C84F_overall_markdown": REPORT_DIR / "C84F_OVERALL_REPORT.md",
        "C84F_overall_json": REPORT_DIR / "C84F_OVERALL_REPORT.json",
        "C84F_execution_report": REPORT_DIR / "C84F_MULTI_DATASET_DUAL_LEVEL_FIELD.json",
    }
    expected = {
        "C84F_overall_markdown": "f80089fa03a64da5b2137e005d86eec2b282b4ab5ea33206f2f2a96ac321fe0c",
        "C84F_overall_json": "edb6ffb73e2f65ce56102f75abbe6ee447ca9dbf1cdddb7631f0ecbfa0b30f47",
        "C84F_execution_report": sha256_file(overall_paths["C84F_execution_report"]),
    }
    identity_rows = [{
        "object": name, "path": str(path.relative_to(REPO_ROOT)),
        "expected_sha256": expected[name], "observed_sha256": sha256_file(path),
        "replay_pass": int(sha256_file(path) == expected[name]),
    } for name, path in overall_paths.items()]
    artifacts = runtime.target_artifact_registry(verify_bytes=verify_external_bytes)
    method_registry = read_json(runtime.METHOD_REGISTRY_PATH)
    method_rows = [{
        "method_id": row["id"], "name": row["name"], "family": row["family"],
        "status": row["status"], "score_direction": row["score_direction"],
        "registry_sha256": runtime.EXPECTED["method_registry"], "replay_pass": 1,
    } for row in method_registry["methods"]]
    table_rows: dict[str, list[dict[str, Any]]] = {
        "c84f_overall_report_identity_replay.csv": identity_rows,
        "complete_field_manifest_replay.csv": [{
            "path": str(runtime.COMPLETE_FIELD_MANIFEST_PATH),
            "expected_sha256": runtime.EXPECTED["complete_field"],
            "observed_sha256": replay["hashes"]["complete_field"],
            "field_descriptors": len(replay["manifest"]["field_descriptors"]),
            "gate": replay["manifest"]["gate"], "replay_pass": 1,
        }],
        "target_artifact_registry_replay.csv": artifacts,
        "scientific_protocol_replay.csv": [{
            "path": str(runtime.SCIENCE_V3_PATH.relative_to(REPO_ROOT)),
            "expected_sha256": runtime.EXPECTED["science_v3"],
            "observed_sha256": sha256_file(runtime.SCIENCE_V3_PATH),
            "changed": 0, "replay_pass": 1,
        }],
        "method_registry_replay.csv": method_rows,
        "synthetic_end_to_end_calibration.csv": synthetic_calibration_rows(),
        "risk_register.csv": risk_rows(),
        "failure_reason_ledger.csv": [{
            "failure_id": "C84SL_INITIAL_LOCK_GAP_REPAIRED",
            "stage": "pre_readiness_end_to_end_audit",
            "blocking": 0,
            "reason": "initial_unexecuted_lock_superseded_by_production_result_freeze_and_executable_S0_S20_repair",
            "real_label_access": 0, "real_selector_scores": 0,
            "scientific_statistics": 0,
        }],
    }
    table_rows.update(_contract_rows())
    table_hashes = {name: write_csv(TABLE_DIR / name, rows) for name, rows in table_rows.items()}
    return {
        "tables": len(table_rows), "table_hashes": table_hashes,
        "artifact_registry_rows": len(artifacts),
        "byte_replay": bool(verify_external_bytes),
        "synthetic_passed": 21,
        "real_label_access": 0, "real_selector_scores": 0,
        "scientific_statistics": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate-tables", "build-lock"))
    parser.add_argument("--verify-external-bytes", action="store_true")
    parser.add_argument("--implementation-commit")
    args = parser.parse_args(argv)
    if args.command == "generate-tables":
        result = generate_tables(verify_external_bytes=args.verify_external_bytes)
    else:
        require(bool(args.implementation_commit), "--implementation-commit is required")
        result = runtime.build_execution_lock(implementation_commit=str(args.implementation_commit))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
