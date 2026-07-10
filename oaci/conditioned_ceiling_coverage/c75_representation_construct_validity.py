"""C75 no-forward representation/projection construct-validity audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np

from . import c74_cache
from . import c75_data
from . import c75_modeling as modeling
from . import c75_projection
from . import c75_protocol
from . import synthetic_factorization_generator as synthetic


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c75_tables"
STATE_PATH = REPORT_DIR / "C75_REPRESENTATION_CONSTRUCT_ANALYSIS_STATE.json"

PATHS = {
    "P_F1_strict_functional": ("F1", ("F0",), ("F0", "F1")),
    "P_F2_strict_architecture": ("F2", ("F0", "F1"), ("F0", "F1", "F2")),
    "P_F3_target_functional": ("F3", ("F0", "F1"), ("F0", "F1", "F3")),
    "P_F4_target_architecture": ("F4", ("F0", "F1", "F3"), ("F0", "F1", "F3", "F4")),
    "P_F5_construction_positive": ("F5", ("F0", "F1", "F3", "F4"), ("F0", "F1", "F3", "F4", "F5")),
}


def _write_csv(name: str, rows: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _concat(arrays: dict[str, np.ndarray], blocks: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([arrays[block].astype(float) for block in blocks], axis=1)


def _quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if len(finite) else math.nan


def _null_replicate(
    replicate: int, tests: list[dict], targets: np.ndarray, trajectory_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    statistics = []
    for scheme_index, within_trajectory in enumerate((True, False)):
        rng = np.random.default_rng(c75_protocol.RNG_SEED + 1000 + replicate + 100_000 * scheme_index)
        permutation = modeling.blocked_permutation_indices(
            targets, trajectory_ids, rng, within_trajectory=within_trajectory,
        )
        scheme_statistics = []
        for test in tests:
            permuted_full = np.concatenate((test["prior_X"], test["new_X"][permutation]), axis=1)
            result = modeling.crossfit_loto(
                permuted_full, test["y"], targets,
                column_space=True, fixed_alphas=test["full_alphas"],
            )
            scheme_statistics.append(
                modeling.r2(test["y"], result.prediction, targets) - test["prior_R2"]
            )
        statistics.append(np.asarray(scheme_statistics))
    return statistics[0], statistics[1]


def _nested_relevance(arrays: dict[str, np.ndarray], outcome_matrix: np.ndarray, outcome_names: list[str]) -> dict:
    targets = arrays["target_id"].astype(int)
    trajectory_ids = arrays["trajectory_id"].astype(str)
    trajectory_template = arrays["trajectory_template"].astype(str)
    joint_good = outcome_matrix[:, outcome_names.index("primary_joint_good")]
    observed_rows = []
    leave_target_rows = []
    leave_trajectory_rows = []
    actionability_rows = []
    actionability_summary = []
    fold_rows = []
    tests = []
    prediction_store = {}

    for path_name, (new_block, prior_blocks, full_blocks) in PATHS.items():
        prior_X = _concat(arrays, prior_blocks)
        new_X = arrays[new_block].astype(float)
        full_X = _concat(arrays, full_blocks)
        for outcome_index, outcome_name in enumerate(outcome_names):
            y = outcome_matrix[:, outcome_index].astype(float)
            prior_result = modeling.crossfit_loto(prior_X, y, targets, column_space=True)
            full_result = modeling.crossfit_loto(full_X, y, targets, column_space=True)
            prior_R2 = modeling.r2(y, prior_result.prediction, targets)
            full_R2 = modeling.r2(y, full_result.prediction, targets)
            increment = full_R2 - prior_R2
            per_target = modeling.per_target_increment_rows(
                y, prior_result.prediction, full_result.prediction, targets,
            )
            bootstrap = modeling.hierarchical_bootstrap_increment(
                y, prior_result.prediction, full_result.prediction, targets,
                repeats=c75_protocol.BOOTSTRAP_REPLICATES,
                seed=c75_protocol.RNG_SEED + 100 * list(PATHS).index(path_name) + outcome_index,
            )
            observed_rows.append({
                "path": path_name, "new_block": new_block, "prior_blocks": "+".join(prior_blocks),
                "full_blocks": "+".join(full_blocks), "outcome": outcome_name,
                "prior_dimension": prior_X.shape[1], "new_block_dimension": new_X.shape[1],
                "full_dimension": full_X.shape[1], "prior_R2": prior_R2, "full_R2": full_R2,
                "incremental_R2": increment,
                "bootstrap_ci_low": _quantile(bootstrap, 0.025),
                "bootstrap_ci_high": _quantile(bootstrap, 0.975),
                "median_increment_residual_rho": float(np.nanmedian([row["increment_residual_rho"] for row in per_target])),
                "positive_target_count": sum(int(row["positive_increment"]) for row in per_target),
                "outer_crossfit": "leave_one_target_out", "inner_alpha_grid": str(list(c75_protocol.RIDGE_ALPHAS)),
                "target_labels_in_new_block": int(new_block == "F5"),
            })
            for row in per_target:
                leave_target_rows.append({"path": path_name, "new_block": new_block, "outcome": outcome_name, **row})
            for model_name, result in (("prior", prior_result), ("full", full_result)):
                for fold in result.fold_rows:
                    fold_rows.append({"path": path_name, "outcome": outcome_name, "model": model_name, **fold})

            prior_alpha = float(np.median(list(prior_result.alphas.values())))
            full_alpha = float(np.median(list(full_result.alphas.values())))
            for holdout_name, holdout in (("trajectory_template", trajectory_template), ("target_x_trajectory_cell", trajectory_ids)):
                prior_secondary = modeling.crossfit_fixed_holdout(
                    prior_X, y, targets, holdout, alpha=prior_alpha, column_space=True,
                )
                full_secondary = modeling.crossfit_fixed_holdout(
                    full_X, y, targets, holdout, alpha=full_alpha, column_space=True,
                )
                yc = modeling.center_within_groups(y[:, None], targets)[:, 0]
                for group in sorted(set(holdout.tolist())):
                    mask = holdout == group
                    leave_trajectory_rows.append({
                        "path": path_name, "new_block": new_block, "outcome": outcome_name,
                        "holdout_level": holdout_name, "holdout_group": group, "row_count": int(np.sum(mask)),
                        "prior_rho": modeling.safe_spearman(yc[mask], prior_secondary[mask]),
                        "full_rho": modeling.safe_spearman(yc[mask], full_secondary[mask]),
                        "increment_residual_rho": modeling.safe_spearman(
                            yc[mask] - prior_secondary[mask], full_secondary[mask] - prior_secondary[mask]
                        ),
                    })
            if outcome_name == "continuous_joint_utility":
                target_action = modeling.actionability_rows(
                    y, joint_good, prior_result.prediction, full_result.prediction, targets,
                )
                for row in target_action:
                    actionability_rows.append({"path": path_name, "new_block": new_block, **row})
                actionability_summary.append({
                    "path": path_name, "new_block": new_block,
                    **{
                        key: float(np.nanmean([row[key] for row in target_action]))
                        for key in ("delta_spearman", "delta_pairwise", "delta_top1", "delta_top3", "regret_reduction", "delta_joint_good_coverage")
                    },
                    "positive_regret_reduction_targets": sum(row["regret_reduction"] > 0 for row in target_action),
                })
            test = {
                "path": path_name, "new_block": new_block, "outcome": outcome_name,
                "prior_X": prior_X, "new_X": new_X, "y": y,
                "prior_R2": prior_R2, "full_alphas": full_result.alphas,
                "observed": increment,
            }
            tests.append(test)
            prediction_store[(path_name, outcome_name)] = {
                "prior": prior_result.prediction, "full": full_result.prediction,
                "prior_X": prior_X, "new_X": new_X, "full_X": full_X,
                "y": y, "full_alphas": full_result.alphas,
            }
            print(json.dumps({"event": "c75_observed_model", "path": path_name, "outcome": outcome_name, "incremental_R2": increment}), flush=True)

    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    null_results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_null_replicate)(replicate, tests, targets, trajectory_ids)
        for replicate in range(c75_protocol.NULL_REPLICATES)
    )
    primary_null_matrix = np.stack([item[0] for item in null_results])
    secondary_null_matrix = np.stack([item[1] for item in null_results])
    max_stat = np.max(primary_null_matrix, axis=1)
    nested_null_rows = []
    for index, test in enumerate(tests):
        values = primary_null_matrix[:, index]
        secondary_values = secondary_null_matrix[:, index]
        uncorrected_p = (1 + int(np.sum(values >= test["observed"]))) / (1 + len(values))
        max_p = (1 + int(np.sum(max_stat >= test["observed"]))) / (1 + len(max_stat))
        nested_null_rows.append({
            "path": test["path"], "new_block": test["new_block"], "outcome": test["outcome"],
            "observed_incremental_R2": test["observed"],
            "primary_nested_null_mean": float(np.mean(values)),
            "primary_nested_null_p95": float(np.quantile(values, 0.95)),
            "uncorrected_p": uncorrected_p, "max_stat_corrected_p": max_p,
            "secondary_nested_null_mean": float(np.mean(secondary_values)),
            "secondary_nested_null_p95": float(np.quantile(secondary_values, 0.95)),
            "secondary_uncorrected_p": (
                1 + int(np.sum(secondary_values >= test["observed"]))
            ) / (1 + len(secondary_values)),
            "null_replicates": len(values),
            "null_scheme": "permute_new_block_within_target_x_trajectory_keep_prior_and_outcome_fixed",
            "secondary_null_scheme": "permute_new_block_within_target_keep_prior_and_outcome_fixed",
            "outer_fold_alphas_frozen_from_observed_nested_crossfit": 1,
        })
    null_lookup = {(row["path"], row["outcome"]): row for row in nested_null_rows}
    for row in observed_rows:
        null = null_lookup[(row["path"], row["outcome"])]
        row.update({
            "nested_null_p95": null["primary_nested_null_p95"],
            "observed_above_null_p95": int(row["incremental_R2"] > null["primary_nested_null_p95"]),
            "uncorrected_p": null["uncorrected_p"], "max_stat_corrected_p": null["max_stat_corrected_p"],
        })
    max_stat_rows = [
        {"replicate": replicate, "max_incremental_R2": float(value), "family_size": len(tests)}
        for replicate, value in enumerate(max_stat)
    ]
    return {
        "observed": observed_rows, "leave_target": leave_target_rows,
        "leave_trajectory": leave_trajectory_rows, "nested_nulls": nested_null_rows,
        "max_stat": max_stat_rows, "actionability": actionability_rows,
        "actionability_summary": actionability_summary, "folds": fold_rows,
        "prediction_store": prediction_store,
    }


def _redundancy_audit(arrays: dict[str, np.ndarray], utility: np.ndarray) -> dict[str, list[dict]]:
    targets = arrays["target_id"].astype(int)
    specifications = {
        "strict_source": [
            ("B0", _concat(arrays, ("F0",))),
            ("B1_logits_minus_b", np.concatenate((_concat(arrays, ("F0", "F1")), arrays["source_logits_minus_b"]), axis=1)),
            ("B2_plus_Wz", np.concatenate((_concat(arrays, ("F0", "F1")), arrays["source_logits_minus_b"], arrays["source_Wz_summary"]), axis=1)),
            ("B3_plus_z", np.concatenate((_concat(arrays, ("F0", "F1")), arrays["source_logits_minus_b"], arrays["source_Wz_summary"], arrays["source_z_summary"]), axis=1)),
            ("B4_plus_W_geometry", np.concatenate((_concat(arrays, ("F0", "F1")), arrays["source_logits_minus_b"], arrays["source_Wz_summary"], arrays["source_z_summary"], arrays["W_geometry"], arrays["source_alignment"]), axis=1)),
        ],
        "target_unlabeled": [
            ("B0", _concat(arrays, ("F0", "F1"))),
            ("B1_logits_minus_b", np.concatenate((_concat(arrays, ("F0", "F1", "F3")), arrays["target_logits_minus_b"]), axis=1)),
            ("B2_plus_Wz", np.concatenate((_concat(arrays, ("F0", "F1", "F3")), arrays["target_logits_minus_b"], arrays["target_Wz_summary"]), axis=1)),
            ("B3_plus_z", np.concatenate((_concat(arrays, ("F0", "F1", "F3")), arrays["target_logits_minus_b"], arrays["target_Wz_summary"], arrays["target_z_summary"]), axis=1)),
            ("B4_plus_W_geometry", np.concatenate((_concat(arrays, ("F0", "F1", "F3")), arrays["target_logits_minus_b"], arrays["target_Wz_summary"], arrays["target_z_summary"], arrays["W_geometry"], arrays["target_Wz_residual"]), axis=1)),
        ],
    }
    model_rows, specification_rows, scaling_rows, redundancy_rows = [], [], [], []
    for view, stages in specifications.items():
        previous_r2 = math.nan
        previous_prediction = None
        for stage, X in stages:
            result = modeling.crossfit_loto(X, utility, targets, column_space=True)
            score = modeling.r2(utility, result.prediction, targets)
            model_rows.append({
                "view": view, "stage": stage, "dimension": X.shape[1], "column_space_R2": score,
                "incremental_R2": score - previous_r2 if math.isfinite(previous_r2) else math.nan,
                "prediction_delta_max_abs": float(np.max(np.abs(result.prediction - previous_prediction))) if previous_prediction is not None else math.nan,
                "column_space_deduplicated": 1,
            })
            centered = modeling.center_within_groups(X, targets)
            singular = np.linalg.svd((centered - np.mean(centered, axis=0)) / np.where(np.std(centered, axis=0) < 1e-12, 1.0, np.std(centered, axis=0)), full_matrices=False, compute_uv=False)
            rank = int(np.sum(singular > c75_protocol.SVD_RANK_TOLERANCE * max(float(singular[0]), 1.0)))
            scaling_rows.append({
                "view": view, "stage": stage, "raw_columns": X.shape[1], "column_rank": rank,
                "zero_variance_columns": int(np.sum(np.std(centered, axis=0) < 1e-12)),
                "condition_number": float(singular[0] / max(singular[rank - 1], 1e-15)) if rank else math.inf,
                "scaling": "within_target_center_then_training_fold_zscore",
            })
            previous_r2, previous_prediction = score, result.prediction
        B1 = stages[1][1]
        B2 = stages[2][1]
        column_B1 = modeling.crossfit_loto(B1, utility, targets, column_space=True)
        column_B2 = modeling.crossfit_loto(B2, utility, targets, column_space=True)
        naive_B1 = modeling.crossfit_loto(B1, utility, targets, column_space=False)
        naive_B2 = modeling.crossfit_loto(B2, utility, targets, column_space=False)
        exact_left = arrays["source_logits_minus_b"] if view == "strict_source" else arrays["target_logits_minus_b"]
        exact_right = arrays["source_Wz_summary"] if view == "strict_source" else arrays["target_Wz_summary"]
        exact_error = float(np.max(np.abs(exact_right - exact_left)))
        if exact_error != 0.0:
            raise RuntimeError(
                f"C75 canonical Wz/logits-minus-b summary identity failed for {view}: {exact_error}"
            )
        redundancy_rows.append({
            "view": view, "summary_max_abs_Wz_minus_logits_minus_b": exact_error,
            "B1_rank": next(row["column_rank"] for row in scaling_rows if row["view"] == view and row["stage"] == "B1_logits_minus_b"),
            "B2_rank": next(row["column_rank"] for row in scaling_rows if row["view"] == view and row["stage"] == "B2_plus_Wz"),
            "column_space_prediction_delta_max_abs": float(np.max(np.abs(column_B2.prediction - column_B1.prediction))),
            "column_space_incremental_R2": modeling.r2(utility, column_B2.prediction, targets) - modeling.r2(utility, column_B1.prediction, targets),
            "naive_ridge_incremental_R2": modeling.r2(utility, naive_B2.prediction, targets) - modeling.r2(utility, naive_B1.prediction, targets),
            "interpretation": "Wz_exactly_redundant_beyond_logits_and_bias;naive_duplicate_gain_is_regularization_parameterization",
        })
        for held_target in range(1, 10):
            row1 = next(row for row in column_B1.fold_rows if row["held_target"] == held_target)
            row2 = next(row for row in column_B2.fold_rows if row["held_target"] == held_target)
            specification_rows.append({
                "view": view, "held_target": held_target,
                "B1_rank": row1["rank"], "B2_rank": row2["rank"], "rank_gain": row2["rank"] - row1["rank"],
                "B1_columns": row1["raw_columns"], "B2_columns": row2["raw_columns"],
                "rank_tolerance": c75_protocol.SVD_RANK_TOLERANCE,
            })
    return {"redundancy": redundancy_rows, "model": model_rows, "specification": specification_rows, "scaling": scaling_rows}


def _projection_validity(
    arrays: dict[str, np.ndarray], relevance: dict, variance: dict,
) -> dict[str, list[dict] | dict]:
    targets = arrays["target_id"].astype(int)
    split_rows = []
    for target in range(1, 10):
        mask = targets == target
        for class_index in range(4):
            construct = arrays["construct_wz_class"][mask, class_index]
            evaluation = arrays["eval_wz_class"][mask, class_index]
            split_rows.append({
                "target_id": target, "class_index": class_index,
                "pearson": float(np.corrcoef(construct, evaluation)[0, 1]),
                "spearman": modeling.safe_spearman(construct, evaluation),
                "mean_absolute_split_difference": float(np.mean(np.abs(construct - evaluation))),
            })
    F4_rows = [row for row in relevance["observed"] if row["path"] == "P_F4_target_architecture"]
    F4_nulls = [row for row in relevance["nested_nulls"] if row["path"] == "P_F4_target_architecture"]
    primary = next(row for row in F4_rows if row["outcome"] == "continuous_joint_utility")
    stable = float(np.median([row["spearman"] for row in split_rows])) >= 0.90
    incremental = (
        primary["incremental_R2"] >= 0.02
        and primary["observed_above_null_p95"] == 1
        and primary["max_stat_corrected_p"] < 0.05
    )
    classification = "representation_candidate" if stable and incremental else "reliable_nuisance_identity_descriptor" if stable else "construct_failure"
    validity = [{
        "construct": "candidate_specific_target_unlabeled_projection",
        "median_split_spearman": float(np.median([row["spearman"] for row in split_rows])),
        "minimum_split_spearman": float(np.min([row["spearman"] for row in split_rows])),
        "positive_split_cells": sum(row["spearman"] > 0 for row in split_rows),
        "primary_incremental_R2": primary["incremental_R2"],
        "primary_nested_null_p95": primary["nested_null_p95"],
        "primary_max_stat_p": primary["max_stat_corrected_p"],
        "stable": int(stable), "incremental": int(incremental),
        "classification": classification,
        "target_gauge_name_allowed": 0, "mechanism_origin_validated": 0,
    }]
    common_rows = [{
        "component": row["component"], "point_mean": row["point_mean"],
        "ci_low": row["ci_low"], "ci_high": row["ci_high"],
        "causal_interpretation": 0,
    } for row in variance["bootstrap"]]
    return {
        "validity": validity, "split": split_rows,
        "incremental": F4_rows, "nulls": F4_nulls,
        "candidate_common": common_rows, "classification": classification,
    }


def _conditional_observability(
    arrays: dict[str, np.ndarray], outcomes: np.ndarray, relevance: dict,
) -> dict[str, list[dict]]:
    targets = arrays["target_id"].astype(int)
    trajectory = arrays["trajectory_id"].astype(str)
    rows, bandwidth_rows = [], []
    rng = np.random.default_rng(c75_protocol.RNG_SEED + 500)
    permutations = [
        modeling.blocked_permutation_indices(targets, trajectory, rng, within_trajectory=True)
        for _ in range(c75_protocol.NULL_REPLICATES)
    ]
    for path, architecture, functional_blocks in (
        ("strict_source", arrays["F2"].astype(float), ("F0", "F1")),
        ("target_unlabeled", arrays["F4"].astype(float), ("F0", "F1", "F3")),
    ):
        y = outcomes[:, 0]
        functional = _concat(arrays, functional_blocks)
        prior = modeling.crossfit_loto(functional, y, targets, column_space=True)
        residual = modeling.center_within_groups(y[:, None], targets)[:, 0] - prior.prediction
        observed_stats = []
        null_by_bandwidth = []
        for factor in (0.5, 1.0, 2.0):
            observed, observed_bandwidths = modeling.crossfit_kernel_alignment_statistic(
                architecture, residual, targets, factor,
            )
            nulls = np.asarray([
                modeling.crossfit_kernel_alignment_statistic(
                    architecture[permutation], residual, targets, factor,
                )[0]
                for permutation in permutations
            ])
            observed_stats.append(observed)
            null_by_bandwidth.append(nulls)
            bandwidth_rows.append({
                "path": path, "bandwidth_factor": factor, "observed_alignment": observed,
                "null_mean": float(np.mean(nulls)), "null_p95": float(np.quantile(nulls, 0.95)),
                "uncorrected_p": (1 + int(np.sum(nulls >= observed))) / (1 + len(nulls)),
                "null_replicates": len(nulls),
                "median_training_fold_bandwidth": float(np.median(observed_bandwidths)),
                "bandwidth_estimation": "outer_training_targets_only",
            })
        max_null = np.max(np.column_stack(null_by_bandwidth), axis=1)
        max_observed = max(observed_stats)
        linear = next(row for row in relevance["observed"] if row["path"] == ("P_F2_strict_architecture" if path == "strict_source" else "P_F4_target_architecture") and row["outcome"] == "continuous_joint_utility")
        rows.append({
            "path": path, "estimator": "block_aware_RBF_kernel_alignment_on_functional_crossfit_residual",
            "max_observed_alignment": max_observed, "bandwidth_max_stat_p": (1 + int(np.sum(max_null >= max_observed))) / (1 + len(max_null)),
            "linear_incremental_R2": linear["incremental_R2"], "linear_max_stat_p": linear["max_stat_corrected_p"],
            "exact_conditional_CS": 0, "iid_guarantee_claimed": 0,
        })
    contract = [
        {"field": "estimand", "value": "incremental conditional observability of held-out utility beyond functional block"},
        {"field": "primary_estimator", "value": "nested LOTO ridge prediction contrast"},
        {"field": "secondary_estimator", "value": "RBF kernel alignment on cross-fitted functional residual"},
        {"field": "dependence", "value": "target and trajectory blocked; trial rows are not iid"},
        {"field": "exact_conditional_CS", "value": "false; proxy distinction explicit"},
        {"field": "bandwidth", "value": "0.5/1/2 x training-scale median distance; max-stat blocked null"},
    ]
    return {"summary": rows, "bandwidth": bandwidth_rows, "contract": contract}


def _qualification(relevance: dict, redundancy: dict) -> list[dict]:
    rows = []
    for candidate, path, view, label_leak in (
        ("F2_strict_source", "P_F2_strict_architecture", "strict_source", 0),
        ("F4_target_unlabeled", "P_F4_target_architecture", "target_unlabeled", 0),
    ):
        result = next(row for row in relevance["observed"] if row["path"] == path and row["outcome"] == "continuous_joint_utility")
        leave = [row for row in relevance["leave_target"] if row["path"] == path and row["outcome"] == "continuous_joint_utility"]
        duplicate_rank = next(
            row["column_rank"] for row in redundancy["scaling"]
            if row["view"] == view and row["stage"] == "B2_plus_Wz"
        )
        architecture_rank = next(
            row["column_rank"] for row in redundancy["scaling"]
            if row["view"] == view and row["stage"] == "B4_plus_W_geometry"
        )
        nonredundant = int(architecture_rank > duplicate_rank)
        gates = {
            "incremental_R2": result["incremental_R2"] >= 0.02,
            "observed_above_nested_null_p95": bool(result["observed_above_null_p95"]),
            "leave_target_out_median_positive": float(np.nanmedian([row["increment_residual_rho"] for row in leave])) > 0,
            "positive_in_7_of_9_targets": sum(row["positive_increment"] for row in leave) >= 7,
            "max_stat_corrected_p": result["max_stat_corrected_p"] < 0.05,
            "not_redundant_with_logits_probabilities": bool(nonredundant),
            "no_target_label_leakage": label_leak == 0,
        }
        for gate, passed in gates.items():
            rows.append({
                "candidate": candidate, "path": path, "gate": gate, "passed": int(passed),
                "incremental_R2": result["incremental_R2"], "nested_null_p95": result["nested_null_p95"],
                "max_stat_p": result["max_stat_corrected_p"],
                "median_target_rho": float(np.nanmedian([row["increment_residual_rho"] for row in leave])),
                "positive_targets": sum(row["positive_increment"] for row in leave),
                "duplicate_block_rank": duplicate_rank, "architecture_block_rank": architecture_rank,
            })
        rows.append({
            "candidate": candidate, "path": path, "gate": "ALL_REQUIRED", "passed": int(all(gates.values())),
            "incremental_R2": result["incremental_R2"], "nested_null_p95": result["nested_null_p95"],
            "max_stat_p": result["max_stat_corrected_p"],
            "median_target_rho": float(np.nanmedian([row["increment_residual_rho"] for row in leave])),
            "positive_targets": sum(row["positive_increment"] for row in leave),
            "duplicate_block_rank": duplicate_rank, "architecture_block_rank": architecture_rank,
        })
    return rows


def _risk_rows() -> list[dict]:
    risks = {
        "C74_cache_or_view_hash_drift": ("closed", "feature extraction rehashes every allowed descriptor"),
        "same_label_oracle_access": ("closed", "restricted manifest contains five allowed views only"),
        "T3_HO_new_variable_access": ("closed", "T3 IDs excluded and no T3 path consumed"),
        "factorization_origin_overclaim": ("controlled", "GL non-identifiability note and synthetic audit mandatory"),
        "stability_as_prediction": ("controlled", "construct validity requires separate incremental/null gates"),
        "Wz_duplicate_regularization_gain": ("closed", "column-space SVD audit plus naive-ridge sensitivity"),
        "unregistered_feature_search": ("closed", "six low-dimensional blocks frozen in protocol"),
        "target_label_leakage": ("closed", "only F5 uses construction labels; evaluation labels outcomes only"),
        "residual_relabeling_as_gauge": ("closed", "outcome named unexplained candidate-specific residual"),
        "null_hyperparameter_adaptation": ("controlled", "observed nested fold alpha frozen for blocked null"),
        "multiple_blocks_outcomes": ("closed", "max-stat family F1-F5 x six outcomes"),
        "small_target_count": ("controlled", "per-target disclosure and target-cluster bootstrap"),
        "cache_rows_iid": ("controlled", "unit-level modeling and blocked dependence"),
        "counterfactual_as_mechanism": ("closed", "matched nulls and factorization prevent origin claim"),
        "conditional_CS_proxy_overclaim": ("closed", "proxy/exact distinction in estimator contract"),
        "selector_or_checkpoint_artifact": ("closed", "aggregate metrics only; no selected IDs"),
        "raw_cache_in_git": ("closed", "only compact feature cache external"),
        "unauthorized_forward_training_GPU": ("closed", "analysis modules have no model/data forward path"),
    }
    return [{"risk": key, "status": status, "blocking": 0, "evidence": evidence} for key, (status, evidence) in risks.items()]


def analyze() -> dict:
    protocol = c75_data.load_protocol()
    feature_manifest, arrays = c75_data.load_feature_cache()
    targets = arrays["target_id"].astype(int)
    direct_outcomes = arrays["outcomes"].astype(float)
    direct_names = feature_manifest["outcome_names"]
    baseline_X = _concat(arrays, ("F0", "F1", "F3", "F5"))
    baseline = modeling.crossfit_loto(baseline_X, direct_outcomes[:, 0], targets, column_space=True)
    continuous_centered = modeling.center_within_groups(direct_outcomes[:, [0]], targets)[:, 0]
    unexplained_residual = continuous_centered - baseline.prediction
    outcome_matrix = np.column_stack((direct_outcomes, unexplained_residual))
    outcome_names = list(direct_names) + ["unexplained_candidate_specific_residual"]

    relevance = _nested_relevance(arrays, outcome_matrix, outcome_names)
    redundancy = _redundancy_audit(arrays, direct_outcomes[:, 0])
    payloads = c75_projection._target_payloads()
    variance = c75_projection.variance_audit(payloads)
    counterfactual = c75_projection.counterfactual_audit(payloads)
    projection = _projection_validity(arrays, relevance, variance)
    conditional = _conditional_observability(arrays, outcome_matrix, relevance)
    factorization_rows = synthetic.factorization_reparameterization_audit()
    synthetic_rows, synthetic_summary = synthetic.construct_validity_benchmark()
    qualification = _qualification(relevance, redundancy)
    qualified = [row["candidate"] for row in qualification if row["gate"] == "ALL_REQUIRED" and row["passed"] == 1]

    factorization_catalog = []
    for row in csv.DictReader(open(c75_protocol.TABLE_DIR / "factorization_status_ledger.csv")):
        factorization_catalog.append({
            **row,
            "C75_implication": "mechanism_origin_not_unique" if row["general_invertible_A_invariant"] == "0" else "function_level_identifiable",
        })
    strict_primary = next(row for row in relevance["observed"] if row["path"] == "P_F2_strict_architecture" and row["outcome"] == "continuous_joint_utility")
    target_primary = next(row for row in relevance["observed"] if row["path"] == "P_F4_target_architecture" and row["outcome"] == "continuous_joint_utility")
    strict_adversary = [{
        "candidate": "F2_strict_source", "incremental_R2": strict_primary["incremental_R2"],
        "nested_null_p95": strict_primary["nested_null_p95"], "max_stat_p": strict_primary["max_stat_corrected_p"],
        "qualification_passed": int("F2_strict_source" in qualified),
        "conclusion": "escape_hatch_candidate" if "F2_strict_source" in qualified else "no_registered_strict_source_representation_escape_hatch_found",
        "universal_source_failure_claimed": 0,
    }]
    t3_decision = [{
        "qualified_candidate_count": len(qualified), "qualified_candidates": ";".join(qualified),
        "C76_protocol_created": 0,
        "T3_HO_campaign_justified": int(bool(qualified)),
        "T3_HO_z_Wz_touched": 0,
        "decision": "candidate_ready_for_locked_C76_protocol" if qualified else "T3_HO_representation_campaign_not_justified",
        "projected_GiB_avoided": 18.70788759738207 if not qualified else 0.0,
    }]

    tables = {
        "Wz_logit_redundancy.csv": redundancy["redundancy"],
        "redundancy_model_specification_audit.csv": redundancy["specification"] + redundancy["model"],
        "feature_scaling_audit.csv": redundancy["scaling"],
        "cross_fitted_incremental_relevance.csv": relevance["observed"],
        "leave_target_out_relevance.csv": relevance["leave_target"],
        "leave_trajectory_out_relevance.csv": relevance["leave_trajectory"],
        "nested_block_nulls.csv": relevance["nested_nulls"],
        "max_stat_null_distribution.csv": relevance["max_stat"],
        "actionability_target_ledger.csv": relevance["actionability"],
        "actionability_increment_summary.csv": relevance["actionability_summary"],
        "nested_crossfit_fold_audit.csv": relevance["folds"],
        "projection_construct_validity.csv": projection["validity"],
        "projection_split_stability.csv": projection["split"],
        "projection_incremental_prediction.csv": projection["incremental"],
        "projection_null_calibration.csv": projection["nulls"],
        "candidate_vs_target_common_projection.csv": projection["candidate_common"],
        "projection_variance_estimand.csv": variance["estimand"],
        "projection_variance_bootstrap.csv": variance["bootstrap"],
        "projection_variance_by_target_class.csv": variance["by_target_class"],
        "counterfactual_identity_vs_mechanism.csv": counterfactual["identity"],
        "counterfactual_matched_nulls.csv": counterfactual["nulls"],
        "counterfactual_blocked_effects.csv": counterfactual["blocked"],
        "strict_source_representation_adversary.csv": strict_adversary,
        "representation_conditional_observability.csv": conditional["summary"],
        "conditional_estimator_contract.csv": conditional["contract"],
        "bandwidth_nested_null_audit.csv": conditional["bandwidth"],
        "synthetic_reparameterization_audit.csv": factorization_rows,
        "synthetic_construct_validity.csv": synthetic_rows,
        "synthetic_false_positive_control.csv": synthetic_summary,
        "factorization_invariance_catalog.csv": factorization_catalog,
        "t3_qualification_decision.csv": qualification,
        "t3_ho_decision.csv": t3_decision,
        "risk_register.csv": _risk_rows(),
        "failure_reason_ledger.csv": [{
            "reason": "none_blocking", "active": 1, "feature_cache_units": feature_manifest["unit_count"],
            "qualified_candidates": ";".join(qualified), "T3_HO_touched": 0,
            "notes": "C75 completed over restricted C74 T2 views only",
        }],
    }
    for name, rows in tables.items():
        _write_csv(name, rows)

    stable = projection["classification"] != "construct_failure"
    if qualified:
        primary = "C75-C_strict_source_representation_escape_hatch_candidate" if "F2_strict_source" in qualified else "C75-B_target_unlabeled_projection_candidate"
        final_gate_candidate = "T3_HO_REPRESENTATION_CAMPAIGN_READY_BUT_NOT_AUTHORIZED"
    else:
        primary = "C75-A_stable_projection_construct_functionally_redundant_nonpredictive" if stable else "C75-D_factorization_nonidentifiable_functional_logit_level_only"
        final_gate_candidate = "T3_HO_REPRESENTATION_CAMPAIGN_NOT_JUSTIFIED"
    state = {
        "schema_version": "c75_representation_construct_analysis_state_v1",
        "protocol_sha256": c75_protocol.sha256(c75_protocol.PROTOCOL_PATH),
        "feature_cache_manifest_sha256": c75_protocol.sha256(c75_data.feature_manifest_path(protocol)),
        "feature_cache_units": feature_manifest["unit_count"],
        "same_label_oracle_accessed": feature_manifest["same_label_oracle_accessed"],
        "T3_HO_z_Wz_accessed": feature_manifest["T3_HO_z_Wz_accessed"],
        "Wz_plus_b_logits_max_abs": feature_manifest["Wz_plus_b_logits_max_abs"],
        "unexplained_residual_name": "unexplained_candidate_specific_residual",
        "factorization_nonidentifiability_formalized": True,
        "projection_classification": projection["classification"],
        "strict_source_primary": strict_primary,
        "target_unlabeled_primary": target_primary,
        "qualified_candidates": qualified,
        "C76_protocol_created": False,
        "primary_candidate": primary,
        "final_gate_candidate": final_gate_candidate,
        "representation_mechanism_claimed": False,
        "target_gauge_claimed": False,
        "selector_or_checkpoint_artifact": False,
        "diagnostic_only_non_deployable": True,
    }
    c74_cache.atomic_json(STATE_PATH, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "analyze"))
    args = parser.parse_args(argv)
    result = c75_data.extract_feature_cache() if args.command == "extract" else analyze()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
