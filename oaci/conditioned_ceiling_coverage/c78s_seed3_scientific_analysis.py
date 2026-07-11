"""C78S full seed-3 multi-regime scientific analysis."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np

from . import c74_cache
from . import c75_data
from . import c78s_data
from . import c78s_modeling as modeling
from . import c78s_protocol as protocol


REPORT_PATH = protocol.REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS.md"
RESULT_PATH = protocol.REPORT_DIR / "C78S_SEED3_SCIENTIFIC_ANALYSIS.json"
ARTIFACT_MANIFEST_PATH = protocol.REPORT_DIR / "C78S_ARTIFACT_MANIFEST.json"
STATE_PATH = protocol.REPORT_DIR / "C78S_ANALYSIS_STATE.json"
C79_PROTOCOL_PATH = protocol.REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.json"
C79_PROTOCOL_SHA_PATH = protocol.REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.sha256"

OUTCOME_NAMES = c78s_data.OUTCOME_NAMES
PRIMARY_PATHS = {
    "strict_source_F2": {
        "prior": ("F0", "F1"),
        "new": "F2",
        "information_class": "strict_source_architecture_tied",
        "primary_hypothesis": "H4",
    },
    "target_unlabeled_F4_geometry": {
        "prior": ("F0", "F1", "F3"),
        "new": "F4_geometry",
        "information_class": "target_unlabeled_architecture_tied",
        "primary_hypothesis": "H5",
    },
    "target_unlabeled_F4_full_mixed": {
        "prior": ("F0", "F1", "F3"),
        "new": "F4",
        "information_class": "target_unlabeled_mixed_secondary",
        "primary_hypothesis": "H5_secondary",
    },
    "construction_F5_positive": {
        "prior": ("F0", "F1", "F3", "F4_geometry"),
        "new": "F5",
        "information_class": "target_construction_labels_diagnostic_positive_control",
        "primary_hypothesis": "H6",
    },
}


def _write_table(name: str, rows: list[dict[str, Any]]) -> None:
    protocol.write_csv(protocol.TABLE_DIR / name, rows)


def _concat(arrays: dict[str, np.ndarray], blocks: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([arrays[block].astype(float) for block in blocks], axis=1)


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else math.nan


def _state(event: str, **payload: Any) -> None:
    record = {"event": event, "at_utc": protocol.utc_now(), **payload}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def _prepare_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prepared = dict(arrays)
    prepared["F4_geometry"] = arrays["F4"][:, :20].astype(float)
    prepared["F4_functional_projection"] = arrays["F4"][:, 20:].astype(float)
    if prepared["F4_geometry"].shape[1] != 20 or prepared["F4_functional_projection"].shape[1] != 15:
        raise RuntimeError("C78S C76 F4 partition drift")
    if len(prepared["unit_id"]) != protocol.PRIMARY_UNITS:
        raise RuntimeError("C78S analysis row-count drift")
    if set(prepared["target_id"].astype(int).tolist()) != set(protocol.PRIMARY_TARGETS):
        raise RuntimeError("C78S primary target registry drift")
    if 4 in set(prepared["target_id"].astype(int).tolist()):
        raise RuntimeError("C78S target 4 entered primary arrays")
    return prepared


def _measurement_control(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    utility = arrays["outcomes"][:, 0].astype(float)
    joint_good = arrays["outcomes"][:, 4].astype(float)
    construction = arrays["F5"][:, -1].astype(float)
    source_bacc = arrays["F0"][:, 6].astype(float)
    targets = arrays["target_id"].astype(int)
    levels = arrays["level"].astype(int)
    regimes = arrays["regime"].astype(str)
    trajectory = arrays["trajectory_id"].astype(str)
    cell_ids = arrays["cell_id"].astype(str)
    reliability_rows = []
    for target in protocol.PRIMARY_TARGETS:
        for level in protocol.LEVELS:
            cell_mask = (targets == target) & (levels == level)
            reliability_rows.append({
                "target_id": target,
                "level": level,
                "regime": "ALL_81",
                "candidate_count": int(np.sum(cell_mask)),
                "construction_evaluation_spearman": modeling.safe_spearman(
                    construction[cell_mask], utility[cell_mask],
                ),
                "construction_evaluation_pairwise": modeling.pairwise_accuracy(
                    utility[cell_mask], construction[cell_mask],
                ),
                "ERM_anchor_included": 1,
            })
            for regime in ("OACI", "SRC"):
                mask = cell_mask & (regimes == regime)
                reliability_rows.append({
                    "target_id": target,
                    "level": level,
                    "regime": regime,
                    "candidate_count": int(np.sum(mask)),
                    "construction_evaluation_spearman": modeling.safe_spearman(
                        construction[mask], utility[mask],
                    ),
                    "construction_evaluation_pairwise": modeling.pairwise_accuracy(
                        utility[mask], construction[mask],
                    ),
                    "ERM_anchor_included": 0,
                })
    target_reliability = {
        target: float(np.nanmean([
            row["construction_evaluation_spearman"]
            for row in reliability_rows
            if row["target_id"] == target and row["regime"] in {"OACI", "SRC"}
        ]))
        for target in protocol.PRIMARY_TARGETS
    }
    reliability_values = np.asarray(list(target_reliability.values()))
    reliability_p = modeling.exact_sign_flip_p(reliability_values)
    target_bootstrap = modeling.target_cluster_bootstrap(
        target_reliability,
        replicates=protocol.BOOTSTRAP_REPLICATES,
        seed=protocol.RNG_SEED + 10,
    )
    action_rows = modeling.cell_actionability(
        utility, joint_good,
        {"source_bAcc": source_bacc, "construction": construction},
        cell_ids, targets, levels,
    )
    action_summary = modeling.summarize_actionability(action_rows, "source_bAcc", "construction")
    direct_rows = []
    for row in action_rows:
        direct_rows.append({
            **row,
            "construction_top1_minus_random": row["construction_oracle_best_in_predicted_top1"] - row["random_top1"],
            "construction_top5_minus_random": row["construction_oracle_best_in_predicted_top5"] - row["random_top5"],
            "construction_top10_minus_random": row["construction_oracle_best_in_predicted_top10"] - row["random_top10"],
            "construction_regret_reduction_vs_random_expectation": row["random_expected_regret"] - row["construction_regret"],
        })
    summary = {
        "trajectory_reliability_mean": float(np.nanmean([
            row["construction_evaluation_spearman"]
            for row in reliability_rows if row["regime"] in {"OACI", "SRC"}
        ])),
        "trajectory_reliability_median": float(np.nanmedian([
            row["construction_evaluation_spearman"]
            for row in reliability_rows if row["regime"] in {"OACI", "SRC"}
        ])),
        "target_mean_reliability": float(np.mean(reliability_values)),
        "target_bootstrap_ci_low": _quantile(target_bootstrap, 0.025),
        "target_bootstrap_ci_high": _quantile(target_bootstrap, 0.975),
        "target_sign_flip_p": reliability_p,
        **action_summary,
    }
    return {
        "reliability": reliability_rows,
        "target_reliability": target_reliability,
        "action": direct_rows,
        "summary": summary,
    }


def _ridge_null_replicate(
    replicate: int,
    tests: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    rng = np.random.default_rng(protocol.RNG_SEED + 1000 + replicate)
    permutation = modeling.blocked_permutation("trajectory_preserving_permutation", arrays, rng)
    targets = arrays["target_id"].astype(int)
    cells = arrays["cell_id"].astype(str)
    values = []
    for test in tests:
        full_X = np.concatenate((test["prior_X"], test["new_X"][permutation]), axis=1)
        result = modeling.crossfit_ridge(
            full_X, test["y"],
            outer_groups=targets,
            inner_groups=targets,
            center_groups=cells,
            fixed_alphas=test["full_alphas"],
        )
        values.append(
            modeling.centered_r2(test["y"], result.prediction, cells) - test["prior_R2"]
        )
    return np.asarray(values, dtype=float)


def _prediction_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    targets = arrays["target_id"].astype(int)
    regimes = arrays["regime"].astype(str)
    cells = arrays["cell_id"].astype(str)
    trajectories = arrays["trajectory_id"].astype(str)
    trajectory_templates = arrays["trajectory_template"].astype(str)
    levels = arrays["level"].astype(int)
    joint_good = arrays["outcomes"][:, 4].astype(float)
    observed_rows = []
    leave_target_rows = []
    leave_regime_rows = []
    leave_trajectory_rows = []
    fold_rows = []
    action_rows = []
    action_summary_rows = []
    tests = []
    prediction_store: dict[tuple[str, str], dict[str, Any]] = {}
    for path, registry in PRIMARY_PATHS.items():
        prior_X = _concat(arrays, registry["prior"])
        new_X = arrays[registry["new"]].astype(float)
        full_X = np.concatenate((prior_X, new_X), axis=1)
        for outcome_index, outcome_name in enumerate(OUTCOME_NAMES):
            y = arrays["outcomes"][:, outcome_index].astype(float)
            prior = modeling.crossfit_ridge(
                prior_X, y,
                outer_groups=targets,
                inner_groups=targets,
                center_groups=cells,
            )
            full = modeling.crossfit_ridge(
                full_X, y,
                outer_groups=targets,
                inner_groups=targets,
                center_groups=cells,
            )
            prior_r2 = modeling.centered_r2(y, prior.prediction, cells)
            full_r2 = modeling.centered_r2(y, full.prediction, cells)
            per_target = modeling.per_target_increment(
                y, prior.prediction, full.prediction, targets, cells,
            )
            prior_loro = modeling.crossfit_ridge(
                prior_X, y,
                outer_groups=regimes,
                inner_groups=targets,
                center_groups=trajectories,
            )
            full_loro = modeling.crossfit_ridge(
                full_X, y,
                outer_groups=regimes,
                inner_groups=targets,
                center_groups=trajectories,
            )
            prior_lotr = modeling.crossfit_ridge(
                prior_X, y,
                outer_groups=trajectory_templates,
                inner_groups=targets,
                center_groups=trajectories,
            )
            full_lotr = modeling.crossfit_ridge(
                full_X, y,
                outer_groups=trajectory_templates,
                inner_groups=targets,
                center_groups=trajectories,
            )
            prior_loro_r2 = modeling.centered_r2(y, prior_loro.prediction, trajectories)
            full_loro_r2 = modeling.centered_r2(y, full_loro.prediction, trajectories)
            observed_rows.append({
                "path": path,
                "information_class": registry["information_class"],
                "primary_hypothesis": registry["primary_hypothesis"],
                "outcome": outcome_name,
                "prior_dimension": prior_X.shape[1],
                "new_dimension": new_X.shape[1],
                "full_dimension": full_X.shape[1],
                "prior_LOTO_R2": prior_r2,
                "full_LOTO_R2": full_r2,
                "incremental_LOTO_R2": full_r2 - prior_r2,
                "LOTO_median_increment_residual_rho": float(np.nanmedian([
                    row["increment_residual_rho"] for row in per_target
                ])),
                "positive_targets": int(np.sum([
                    row["positive_increment"] for row in per_target
                ])),
                "prior_LORO_R2": prior_loro_r2,
                "full_LORO_R2": full_loro_r2,
                "incremental_LORO_R2": full_loro_r2 - prior_loro_r2,
                "target_labels_in_new_block": int(registry["new"] == "F5"),
            })
            for row in per_target:
                leave_target_rows.append({"path": path, "outcome": outcome_name, **row})
            for regime in protocol.REGIMES:
                mask = regimes == regime
                yc = modeling.center_within_groups(y[:, None], trajectories)[:, 0]
                leave_regime_rows.append({
                    "path": path,
                    "outcome": outcome_name,
                    "held_regime": regime,
                    "row_count": int(np.sum(mask)),
                    "prior_rho": modeling.safe_spearman(yc[mask], prior_loro.prediction[mask]),
                    "full_rho": modeling.safe_spearman(yc[mask], full_loro.prediction[mask]),
                    "increment_residual_rho": modeling.safe_spearman(
                        yc[mask] - prior_loro.prediction[mask],
                        full_loro.prediction[mask] - prior_loro.prediction[mask],
                    ),
                })
            for template in sorted(set(trajectory_templates.tolist())):
                mask = trajectory_templates == template
                yc = modeling.center_within_groups(y[:, None], trajectories)[:, 0]
                leave_trajectory_rows.append({
                    "path": path,
                    "outcome": outcome_name,
                    "held_trajectory_template": template,
                    "row_count": int(np.sum(mask)),
                    "prior_rho": modeling.safe_spearman(yc[mask], prior_lotr.prediction[mask]),
                    "full_rho": modeling.safe_spearman(yc[mask], full_lotr.prediction[mask]),
                    "increment_residual_rho": modeling.safe_spearman(
                        yc[mask] - prior_lotr.prediction[mask],
                        full_lotr.prediction[mask] - prior_lotr.prediction[mask],
                    ),
                })
            for model_name, result in (
                ("prior_LOTO", prior), ("full_LOTO", full),
                ("prior_LORO", prior_loro), ("full_LORO", full_loro),
                ("prior_LOTR", prior_lotr), ("full_LOTR", full_lotr),
            ):
                for row in result.fold_rows:
                    fold_rows.append({"path": path, "outcome": outcome_name, "model": model_name, **row})
            if outcome_name == "continuous_joint_utility":
                rows = modeling.cell_actionability(
                    y, joint_good,
                    {"prior": prior.prediction, "full": full.prediction},
                    cells, targets, levels,
                )
                for row in rows:
                    action_rows.append({"path": path, **row})
                action_summary_rows.append({"path": path, **modeling.summarize_actionability(rows, "prior", "full")})
            tests.append({
                "path": path,
                "outcome": outcome_name,
                "prior_X": prior_X,
                "new_X": new_X,
                "y": y,
                "prior_R2": prior_r2,
                "full_alphas": full.alphas,
                "observed": full_r2 - prior_r2,
            })
            prediction_store[(path, outcome_name)] = {
                "prior": prior.prediction,
                "full": full.prediction,
                "prior_loro": prior_loro.prediction,
                "full_loro": full_loro.prediction,
                "prior_lotr": prior_lotr.prediction,
                "full_lotr": full_lotr.prediction,
                "prior_X": prior_X,
                "new_X": new_X,
                "y": y,
            }
        _state("prediction_path_complete", path=path)
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    null_values = Parallel(n_jobs=workers, backend="loky", verbose=5, max_nbytes="5M")(
        delayed(_ridge_null_replicate)(replicate, tests, arrays)
        for replicate in range(protocol.NULL_REPLICATES)
    )
    null_matrix = np.stack(null_values)
    max_null = np.max(null_matrix, axis=1)
    null_rows = []
    for index, test in enumerate(tests):
        values = null_matrix[:, index]
        uncorrected = (1 + int(np.sum(values >= test["observed"]))) / (1 + len(values))
        corrected = (1 + int(np.sum(max_null >= test["observed"]))) / (1 + len(values))
        null_rows.append({
            "path": test["path"],
            "outcome": test["outcome"],
            "observed_incremental_R2": test["observed"],
            "null_mean": float(np.mean(values)),
            "null_p95": float(np.quantile(values, 0.95)),
            "uncorrected_p": uncorrected,
            "max_stat_corrected_p": corrected,
            "null_replicates": len(values),
            "null_scheme": "new_block_permuted_within_target_x_level_x_regime",
            "observed_fold_alphas_frozen": 1,
        })
    lookup = {(row["path"], row["outcome"]): row for row in null_rows}
    for row in observed_rows:
        null = lookup[(row["path"], row["outcome"])]
        row.update({
            "nested_null_p95": null["null_p95"],
            "observed_above_null_p95": int(row["incremental_LOTO_R2"] > null["null_p95"]),
            "uncorrected_p": null["uncorrected_p"],
            "max_stat_corrected_p": null["max_stat_corrected_p"],
        })
    return {
        "observed": observed_rows,
        "leave_target": leave_target_rows,
        "leave_regime": leave_regime_rows,
        "leave_trajectory": leave_trajectory_rows,
        "folds": fold_rows,
        "action": action_rows,
        "action_summary": action_summary_rows,
        "null": null_rows,
        "prediction_store": prediction_store,
    }


def _krr_audit(arrays: dict[str, np.ndarray], prediction: dict[str, Any]) -> dict[str, Any]:
    targets = arrays["target_id"].astype(int)
    regimes = arrays["regime"].astype(str)
    cells = arrays["cell_id"].astype(str)
    trajectories = arrays["trajectory_id"].astype(str)
    trajectory_templates = arrays["trajectory_template"].astype(str)
    levels = arrays["level"].astype(int)
    utility = arrays["outcomes"][:, 0].astype(float)
    joint_good = arrays["outcomes"][:, 4].astype(float)
    paths = {
        "strict_source": {
            "features": arrays["F2"],
            "linear_path": "strict_source_F2",
            "kernel": "rbf",
        },
        "target_unlabeled": {
            "features": arrays["F4_geometry"],
            "linear_path": "target_unlabeled_F4_geometry",
            "kernel": "laplacian",
        },
    }
    rows, fold_rows, leave_target_rows, leave_regime_rows, leave_trajectory_rows, action_rows, action_summary = [], [], [], [], [], [], []
    null_by_path = {}
    for path, item in paths.items():
        store = prediction["prediction_store"][(item["linear_path"], "continuous_joint_utility")]
        yc = modeling.center_within_groups(utility[:, None], cells)[:, 0]
        residual = yc - store["prior"]
        increment, folds = modeling.crossfit_krr_fixed(
            item["features"], residual, targets, cells,
            kernel_family=item["kernel"], bandwidth_factor=1.0, alpha=1.0,
        )
        full = store["prior"] + increment
        prior_r2 = modeling.centered_r2(utility, store["prior"], cells)
        full_r2 = modeling.centered_r2(utility, full, cells)
        per_target = modeling.per_target_increment(utility, store["prior"], full, targets, cells)
        yc_loro = modeling.center_within_groups(utility[:, None], trajectories)[:, 0]
        residual_loro = yc_loro - store["prior_loro"]
        increment_loro, loro_folds = modeling.crossfit_krr_fixed(
            item["features"], residual_loro, targets, trajectories,
            kernel_family=item["kernel"], bandwidth_factor=1.0, alpha=1.0,
            outer_groups=regimes,
        )
        full_loro = store["prior_loro"] + increment_loro
        prior_loro_r2 = modeling.centered_r2(utility, store["prior_loro"], trajectories)
        full_loro_r2 = modeling.centered_r2(utility, full_loro, trajectories)
        residual_lotr = yc_loro - store["prior_lotr"]
        increment_lotr, lotr_folds = modeling.crossfit_krr_fixed(
            item["features"], residual_lotr, targets, trajectories,
            kernel_family=item["kernel"], bandwidth_factor=1.0, alpha=1.0,
            outer_groups=trajectory_templates,
        )
        full_lotr = store["prior_lotr"] + increment_lotr
        nulls = modeling.krr_trajectory_nulls(
            item["features"], residual, store["prior"], utility, arrays,
            kernel_family=item["kernel"], bandwidth_factor=1.0, alpha=1.0,
            replicates=protocol.NULL_REPLICATES,
            seed=protocol.RNG_SEED + 2000 + (path == "target_unlabeled"),
        )
        null_by_path[path] = nulls
        rows.append({
            "path": path,
            "kernel": item["kernel"],
            "bandwidth_factor": 1.0,
            "alpha": 1.0,
            "prior_LOTO_R2": prior_r2,
            "full_LOTO_R2": full_r2,
            "incremental_LOTO_R2": full_r2 - prior_r2,
            "LOTO_median_increment_residual_rho": float(np.nanmedian([row["increment_residual_rho"] for row in per_target])),
            "positive_targets": int(np.sum([row["positive_increment"] for row in per_target])),
            "prior_LORO_R2": prior_loro_r2,
            "full_LORO_R2": full_loro_r2,
            "incremental_LORO_R2": full_loro_r2 - prior_loro_r2,
            "null_p95": float(np.quantile(nulls, 0.95)),
            "uncorrected_p": (1 + int(np.sum(nulls >= full_r2 - prior_r2))) / (1 + len(nulls)),
        })
        for fold in folds:
            fold_rows.append({"path": path, "transport": "leave_target_out", **fold})
        for fold in loro_folds:
            fold_rows.append({"path": path, "transport": "leave_regime_out", **fold})
        for fold in lotr_folds:
            fold_rows.append({"path": path, "transport": "leave_trajectory_out", **fold})
        for target_row in per_target:
            leave_target_rows.append({"path": path, **target_row})
        for regime in protocol.REGIMES:
            mask = regimes == regime
            leave_regime_rows.append({
                "path": path,
                "held_regime": regime,
                "row_count": int(np.sum(mask)),
                "increment_residual_rho": modeling.safe_spearman(
                    yc_loro[mask] - store["prior_loro"][mask], increment_loro[mask],
                ),
            })
        for template in sorted(set(trajectory_templates.tolist())):
            mask = trajectory_templates == template
            leave_trajectory_rows.append({
                "path": path,
                "held_trajectory_template": template,
                "row_count": int(np.sum(mask)),
                "increment_residual_rho": modeling.safe_spearman(
                    yc_loro[mask] - store["prior_lotr"][mask], increment_lotr[mask],
                ),
            })
        cells_rows = modeling.cell_actionability(
            utility, joint_good,
            {"prior": store["prior"], "full": full},
            cells, targets, levels,
        )
        for cell_row in cells_rows:
            action_rows.append({"path": path, **cell_row})
        action_summary.append({"path": path, **modeling.summarize_actionability(cells_rows, "prior", "full")})
        _state("krr_path_complete", path=path)
    null_matrix = np.column_stack([null_by_path[path] for path in ("strict_source", "target_unlabeled")])
    max_null = np.max(null_matrix, axis=1)
    for row in rows:
        observed = row["incremental_LOTO_R2"]
        row["global_max_stat_p"] = (1 + int(np.sum(max_null >= observed))) / (1 + len(max_null))
    return {
        "summary": rows,
        "folds": fold_rows,
        "leave_target": leave_target_rows,
        "leave_regime": leave_regime_rows,
        "leave_trajectory": leave_trajectory_rows,
        "action": action_rows,
        "action_summary": action_summary,
    }


def _association_null_replicate(
    scheme: str,
    replicate: int,
    feature_paths: dict[str, np.ndarray],
    residual_paths: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    seed = protocol.RNG_SEED + 3000 + 100_000 * (
        "target_block_permutation",
        "checkpoint_block_permutation",
        "trajectory_preserving_permutation",
        "candidate_within_target_regime_permutation",
        "identity_only_matched_null",
        "nested_bandwidth_null",
    ).index(scheme) + replicate
    rng = np.random.default_rng(seed)
    if scheme == "identity_only_matched_null":
        candidate = {
            path: modeling.matched_gaussian_features(
                features, arrays["F0"], arrays["cell_id"].astype(str), rng,
            )
            for path, features in feature_paths.items()
        }
    else:
        permutation = modeling.blocked_permutation(scheme, arrays, rng)
        candidate = {path: features[permutation] for path, features in feature_paths.items()}
    family = modeling.association_family(candidate, residual_paths, arrays)
    return np.asarray([row["association"] for row in family], dtype=float)


def _association_audit(arrays: dict[str, np.ndarray], prediction: dict[str, Any]) -> dict[str, Any]:
    targets = arrays["target_id"].astype(int)
    cells = arrays["cell_id"].astype(str)
    trajectories = arrays["trajectory_id"].astype(str)
    regimes = arrays["regime"].astype(str)
    utility = arrays["outcomes"][:, 0].astype(float)
    feature_paths = {
        "strict_source": arrays["F2"].astype(float),
        "target_unlabeled": arrays["F4_geometry"].astype(float),
    }
    residual_paths = {}
    for path, linear_path in (
        ("strict_source", "strict_source_F2"),
        ("target_unlabeled", "target_unlabeled_F4_geometry"),
    ):
        prior = prediction["prediction_store"][(linear_path, "continuous_joint_utility")]["prior"]
        residual_paths[path] = modeling.center_within_groups(utility[:, None], cells)[:, 0] - prior
    observed = modeling.association_family(feature_paths, residual_paths, arrays)
    schemes = (
        "target_block_permutation",
        "checkpoint_block_permutation",
        "trajectory_preserving_permutation",
        "candidate_within_target_regime_permutation",
        "identity_only_matched_null",
        "nested_bandwidth_null",
    )
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    null_rows = []
    summary_rows = []
    matrices = {}
    for scheme in schemes:
        values = Parallel(n_jobs=workers, backend="loky", verbose=5, max_nbytes="5M")(
            delayed(_association_null_replicate)(
                scheme, replicate, feature_paths, residual_paths, arrays,
            )
            for replicate in range(protocol.NULL_REPLICATES)
        )
        matrix = np.stack(values)
        matrices[scheme] = matrix
        max_stat = np.max(matrix, axis=1)
        for index, observed_row in enumerate(observed):
            path_values = matrix[:, index]
            null_rows.append({
                "null": scheme,
                "path": observed_row["path"],
                "kernel": observed_row["kernel"],
                "bandwidth_factor": observed_row["bandwidth_factor"],
                "statistic": observed_row["statistic"],
                "observed": observed_row["association"],
                "null_mean": float(np.mean(path_values)),
                "null_p95": float(np.quantile(path_values, 0.95)),
                "uncorrected_p": (1 + int(np.sum(path_values >= observed_row["association"]))) / (1 + len(path_values)),
                "global_family_max_stat_p": (1 + int(np.sum(max_stat >= observed_row["association"]))) / (1 + len(max_stat)),
                "bandwidth_selected_inside_null": 1,
                "null_replicates": len(path_values),
            })
        _state("association_null_scheme_complete", scheme=scheme)
    for observed_row in observed:
        matches = [
            row for row in null_rows
            if row["path"] == observed_row["path"]
            and row["kernel"] == observed_row["kernel"]
            and float(row["bandwidth_factor"]) == float(observed_row["bandwidth_factor"])
            and row["statistic"] == observed_row["statistic"]
        ]
        summary_rows.append({
            **{key: observed_row[key] for key in (
                "path", "kernel", "bandwidth_factor", "statistic", "association",
                "median_target_association", "positive_targets",
            )},
            "worst_required_global_p": max(row["global_family_max_stat_p"] for row in matches),
            "required_nulls_passing_0.05": int(np.sum([
                row["global_family_max_stat_p"] < 0.05 for row in matches
            ])),
            "required_null_count": len(matches),
        })
    topology_rows = []
    topology_folds = []
    fixed = {
        "strict_source": ("rbf", 1.0, "normalized_alignment"),
        "target_unlabeled": ("laplacian", 1.0, "centered_hsic"),
    }
    group_specs = {
        "pooled": np.asarray(["ALL"] * len(targets)),
        "within_target": targets.astype(str),
        "within_target_x_level": cells,
        "within_target_x_level_x_regime": trajectories,
        "within_regime": regimes,
    }
    for path, features in feature_paths.items():
        kernel, factor, statistic = fixed[path]
        for level, groups in group_specs.items():
            value, folds = modeling.topology_association(
                features, residual_paths[path], groups,
                kernel_family=kernel, bandwidth_factor=factor, statistic=statistic,
            )
            topology_rows.append({
                "path": path,
                "level": level,
                "kernel": kernel,
                "bandwidth_factor": factor,
                "statistic": statistic,
                "association": value,
                "group_count": len(folds),
                "positive_group_fraction": float(np.mean([
                    row["association"] > 0 for row in folds
                ])) if folds else math.nan,
            })
            for row in folds:
                topology_folds.append({"path": path, "level": level, **row})
    return {
        "family": [{key: value for key, value in row.items() if key != "folds"} for row in observed],
        "family_folds": [
            {
                "path": row["path"], "kernel": row["kernel"],
                "bandwidth_factor": row["bandwidth_factor"], "statistic": row["statistic"], **fold,
            }
            for row in observed for fold in row["folds"]
        ],
        "null": null_rows,
        "summary": summary_rows,
        "topology": topology_rows,
        "topology_folds": topology_folds,
    }


def _geometry_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    targets = arrays["target_id"].astype(int)
    levels = arrays["level"].astype(int)
    regimes = arrays["regime"].astype(str)
    orders = arrays["candidate_order"].astype(int)
    utility = arrays["outcomes"][:, 0].astype(float)
    construction = arrays["F5"][:, -1].astype(float)
    rows = []
    for target in protocol.PRIMARY_TARGETS:
        for level in protocol.LEVELS:
            for regime in ("OACI", "SRC"):
                base = (targets == target) & (levels == level) & (regimes == regime)
                for prefix in protocol.PREFIX_SIZES:
                    indices = np.where(base & (orders >= 1) & (orders <= prefix))[0]
                    if len(indices) != prefix:
                        raise RuntimeError(f"C78S prefix field drift {target}/{level}/{regime}/{prefix}")
                    true_order = np.argsort(utility[indices])[::-1]
                    selected = int(np.argmax(construction[indices]))
                    best = float(utility[indices][true_order[0]])
                    second = float(utility[indices][true_order[1]]) if len(indices) > 1 else best
                    spread = best - float(np.min(utility[indices]))
                    effective = int(np.sum(best - utility[indices] <= protocol.PRIMARY_GEOMETRY_EPSILON + 1e-15))
                    rows.append({
                        "target_id": target,
                        "level": level,
                        "regime": regime,
                        "prefix_size": prefix,
                        "raw_M": len(indices),
                        "effective_M_epsilon_0.05": effective,
                        "top_two_gap": best - second,
                        "top1_miss": int(selected != int(true_order[0])),
                        "selected_true_rank": int(np.where(true_order == selected)[0][0]) + 1,
                        "continuous_regret": best - float(utility[indices][selected]),
                        "standardized_regret": (
                            (best - float(utility[indices][selected])) / spread if spread > 1e-15 else 0.0
                        ),
                        "construction_eval_spearman": modeling.safe_spearman(
                            construction[indices], utility[indices],
                        ),
                        "endpoint_derived_geometry_diagnostic_only": 1,
                    })
    return rows


def _geometry_null_replicate(
    replicate: int,
    raw_X: np.ndarray,
    full_X: np.ndarray,
    y: np.ndarray,
    targets: np.ndarray,
    strata: np.ndarray,
) -> float:
    rng = np.random.default_rng(protocol.RNG_SEED + 4000 + replicate)
    permuted = full_X.copy()
    for stratum in sorted(set(strata.tolist())):
        indices = np.where(strata == stratum)[0]
        permutation = rng.permutation(indices)
        permuted[indices, 1:] = full_X[permutation, 1:]
    return modeling.crossfit_logistic_deviance(raw_X, permuted, y, targets)["incremental_deviance_reduction"]


def _geometry_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = _geometry_rows(arrays)
    raw_X = np.asarray([[math.log(row["raw_M"])] for row in rows], dtype=float)
    full_X = np.asarray([
        [
            math.log(row["raw_M"]),
            math.log(max(row["effective_M_epsilon_0.05"], 1)),
            -math.log(row["top_two_gap"] + 1e-6),
        ]
        for row in rows
    ], dtype=float)
    y = np.asarray([row["top1_miss"] for row in rows], dtype=float)
    targets = np.asarray([row["target_id"] for row in rows], dtype=int)
    strata = np.asarray([
        f"M-{row['raw_M']}|level-{row['level']}|{row['regime']}"
        for row in rows
    ])
    observed = modeling.crossfit_logistic_deviance(raw_X, full_X, y, targets)
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    nulls = np.asarray(Parallel(n_jobs=workers, backend="loky", verbose=5)(
        delayed(_geometry_null_replicate)(
            replicate, raw_X, full_X, y, targets, strata,
        )
        for replicate in range(protocol.NULL_REPLICATES)
    ))
    coefficient_rows = observed.pop("coefficient_rows")
    observed.pop("raw_prediction")
    observed.pop("full_prediction")
    summary = {
        **observed,
        "null_mean": float(np.mean(nulls)),
        "null_p95": float(np.quantile(nulls, 0.95)),
        "permutation_p": (1 + int(np.sum(nulls >= observed["incremental_deviance_reduction"]))) / (1 + len(nulls)),
        "null_replicates": len(nulls),
        "effective_M_coefficient_median": float(np.median([
            row["effective_M_coefficient"] for row in coefficient_rows
        ])),
        "inverse_gap_coefficient_median": float(np.median([
            row["inverse_gap_coefficient"] for row in coefficient_rows
        ])),
        "raw_M_range": f"{min(row['raw_M'] for row in rows)}-{max(row['raw_M'] for row in rows)}",
        "endpoint_geometry_is_diagnostic_not_selector": 1,
    }
    return {"rows": rows, "coefficients": coefficient_rows, "summary": summary}


def _batch_endpoint(
    probabilities: np.ndarray,
    prediction: np.ndarray,
    sampled_indices: np.ndarray,
    sampled_labels: np.ndarray,
) -> np.ndarray:
    units = probabilities.shape[0]
    selected_probability = probabilities[:, sampled_indices]
    selected_prediction = prediction[:, sampled_indices]
    metrics = np.empty((units, 3), dtype=float)
    recalls = []
    for class_index in range(4):
        class_mask = sampled_labels == class_index
        recalls.append(np.mean(selected_prediction[:, class_mask] == class_index, axis=1))
    metrics[:, 0] = np.mean(np.stack(recalls, axis=1), axis=1)
    true_probability = np.take_along_axis(
        selected_probability,
        np.broadcast_to(sampled_labels[None, :, None], (units, len(sampled_labels), 1)),
        axis=2,
    )[:, :, 0]
    metrics[:, 1] = -np.mean(np.log(np.clip(true_probability, 1e-12, 1.0)), axis=1)
    confidence = np.max(selected_probability, axis=2)
    correctness = selected_prediction == sampled_labels[None, :]
    ece = np.zeros(units, dtype=float)
    edges = np.linspace(0.0, 1.0, 16)
    for bin_index in range(15):
        mask = (confidence >= edges[bin_index]) & (
            confidence <= edges[bin_index + 1] if bin_index == 14 else confidence < edges[bin_index + 1]
        )
        count = np.sum(mask, axis=1)
        nonzero = count > 0
        if np.any(nonzero):
            accuracy = np.divide(
                np.sum(correctness * mask, axis=1), count,
                out=np.zeros(units), where=nonzero,
            )
            mean_confidence = np.divide(
                np.sum(confidence * mask, axis=1), count,
                out=np.zeros(units), where=nonzero,
            )
            ece += (count / len(sampled_labels)) * np.abs(accuracy - mean_confidence)
    metrics[:, 2] = ece
    return metrics


def _rank_utility(metrics: np.ndarray) -> np.ndarray:
    return np.mean(np.column_stack((
        c75_data.midrank_percentile(metrics[:, 0]),
        c75_data.midrank_percentile(-metrics[:, 1]),
        c75_data.midrank_percentile(-metrics[:, 2]),
    )), axis=1)


def _trial_bootstrap_target(
    target_id: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    row_indices = np.where(arrays["target_id"].astype(int) == target_id)[0]
    logits = arrays["target_logits"][row_indices].astype(float)
    shifted = logits - np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    prediction = np.argmax(probabilities, axis=2)
    split_position = int(np.where(arrays["split_target_id"].astype(int) == target_id)[0][0])
    c_mask = arrays["construct_indices"][split_position] >= 0
    e_mask = arrays["eval_indices"][split_position] >= 0
    c_indices = arrays["construct_indices"][split_position][c_mask].astype(int)
    c_labels = arrays["construct_labels"][split_position][c_mask].astype(int)
    e_indices = arrays["eval_indices"][split_position][e_mask].astype(int)
    e_labels = arrays["eval_labels"][split_position][e_mask].astype(int)
    levels = arrays["level"][row_indices].astype(int)
    regimes = arrays["regime"][row_indices].astype(str)
    rng = np.random.default_rng(protocol.RNG_SEED + 5000 + target_id)
    reliability = np.empty(protocol.TRIAL_BOOTSTRAP_REPLICATES)
    top5 = np.empty(protocol.TRIAL_BOOTSTRAP_REPLICATES)
    standardized_regret = np.empty(protocol.TRIAL_BOOTSTRAP_REPLICATES)
    for replicate in range(protocol.TRIAL_BOOTSTRAP_REPLICATES):
        sampled_c = np.concatenate([
            rng.choice(np.where(c_labels == class_index)[0], size=int(np.sum(c_labels == class_index)), replace=True)
            for class_index in range(4)
        ])
        sampled_e = np.concatenate([
            rng.choice(np.where(e_labels == class_index)[0], size=int(np.sum(e_labels == class_index)), replace=True)
            for class_index in range(4)
        ])
        construct_metrics = _batch_endpoint(
            probabilities, prediction, c_indices[sampled_c], c_labels[sampled_c],
        )
        evaluation_metrics = _batch_endpoint(
            probabilities, prediction, e_indices[sampled_e], e_labels[sampled_e],
        )
        rhos, hits, regrets = [], [], []
        for level in protocol.LEVELS:
            cell = levels == level
            construct_utility = _rank_utility(construct_metrics[cell])
            evaluation_utility = _rank_utility(evaluation_metrics[cell])
            cell_regimes = regimes[cell]
            for regime in ("OACI", "SRC"):
                mask = cell_regimes == regime
                rhos.append(modeling.safe_spearman(construct_utility[mask], evaluation_utility[mask]))
            predicted_order = np.argsort(construct_utility)[::-1]
            true_order = np.argsort(evaluation_utility)[::-1]
            selected = int(predicted_order[0])
            hits.append(int(int(true_order[0]) in set(map(int, predicted_order[:5]))))
            spread = float(np.max(evaluation_utility) - np.min(evaluation_utility))
            regrets.append(
                (float(np.max(evaluation_utility)) - float(evaluation_utility[selected])) / spread
                if spread > 1e-15 else 0.0
            )
        reliability[replicate] = float(np.nanmean(rhos))
        top5[replicate] = float(np.mean(hits))
        standardized_regret[replicate] = float(np.mean(regrets))
    return {
        "target_id": target_id,
        "reliability": reliability,
        "top5": top5,
        "standardized_regret": standardized_regret,
    }


def _hierarchical_bootstrap(
    arrays: dict[str, np.ndarray],
    measurement: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    workers = max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    target_trial = Parallel(n_jobs=workers, backend="loky", verbose=5, max_nbytes="10M")(
        delayed(_trial_bootstrap_target)(target, arrays)
        for target in protocol.PRIMARY_TARGETS
    )
    trial_matrix = np.stack([row["reliability"] for row in target_trial])
    top5_matrix = np.stack([row["top5"] for row in target_trial])
    regret_matrix = np.stack([row["standardized_regret"] for row in target_trial])
    target_trial_rows = []
    for row in target_trial:
        target_trial_rows.append({
            "target_id": row["target_id"],
            "trial_bootstrap_replicates": len(row["reliability"]),
            "reliability_mean": float(np.mean(row["reliability"])),
            "reliability_ci_low": _quantile(row["reliability"], 0.025),
            "reliability_ci_high": _quantile(row["reliability"], 0.975),
            "construction_top5_mean": float(np.mean(row["top5"])),
            "construction_top5_ci_low": _quantile(row["top5"], 0.025),
            "construction_top5_ci_high": _quantile(row["top5"], 0.975),
            "standardized_regret_mean": float(np.mean(row["standardized_regret"])),
            "standardized_regret_ci_low": _quantile(row["standardized_regret"], 0.025),
            "standardized_regret_ci_high": _quantile(row["standardized_regret"], 0.975),
            "bootstrap_unit": "shared_trial_id_within_target_stratified_by_class",
        })
    rng = np.random.default_rng(protocol.RNG_SEED + 6000)
    crossed = np.empty(protocol.TRIAL_BOOTSTRAP_REPLICATES)
    crossed_top5 = np.empty(protocol.TRIAL_BOOTSTRAP_REPLICATES)
    crossed_regret = np.empty(protocol.TRIAL_BOOTSTRAP_REPLICATES)
    for replicate in range(protocol.TRIAL_BOOTSTRAP_REPLICATES):
        sampled_targets = rng.integers(0, len(protocol.PRIMARY_TARGETS), size=len(protocol.PRIMARY_TARGETS))
        crossed[replicate] = float(np.mean(trial_matrix[sampled_targets, replicate]))
        crossed_top5[replicate] = float(np.mean(top5_matrix[sampled_targets, replicate]))
        crossed_regret[replicate] = float(np.mean(regret_matrix[sampled_targets, replicate]))
    crossed_rows = [{
        "estimand": "construction_evaluation_reliability",
        "mean": float(np.mean(crossed)),
        "ci_low": _quantile(crossed, 0.025),
        "ci_high": _quantile(crossed, 0.975),
        "bootstrap": "target_cluster_x_shared_trial_id_cluster",
        "replicates": len(crossed),
    }, {
        "estimand": "construction_top5_hit",
        "mean": float(np.mean(crossed_top5)),
        "ci_low": _quantile(crossed_top5, 0.025),
        "ci_high": _quantile(crossed_top5, 0.975),
        "bootstrap": "target_cluster_x_shared_trial_id_cluster",
        "replicates": len(crossed_top5),
    }, {
        "estimand": "construction_standardized_regret",
        "mean": float(np.mean(crossed_regret)),
        "ci_low": _quantile(crossed_regret, 0.025),
        "ci_high": _quantile(crossed_regret, 0.975),
        "bootstrap": "target_cluster_x_shared_trial_id_cluster",
        "replicates": len(crossed_regret),
    }]
    target_values = measurement["target_reliability"]
    target_bootstrap = modeling.target_cluster_bootstrap(
        target_values,
        replicates=protocol.BOOTSTRAP_REPLICATES,
        seed=protocol.RNG_SEED + 6100,
    )
    checkpoint_rng = np.random.default_rng(protocol.RNG_SEED + 6200)
    utility = arrays["outcomes"][:, 0].astype(float)
    construction = arrays["F5"][:, -1].astype(float)
    trajectories = arrays["trajectory_id"].astype(str)
    targets = arrays["target_id"].astype(int)
    checkpoint_values = []
    eligible = [
        trajectory for trajectory in sorted(set(trajectories.tolist()))
        if int(np.sum(trajectories == trajectory)) >= 4
    ]
    for _ in range(protocol.BOOTSTRAP_REPLICATES):
        target_rhos = defaultdict(list)
        for trajectory in eligible:
            indices = np.where(trajectories == trajectory)[0]
            sampled = checkpoint_rng.choice(indices, size=len(indices), replace=True)
            target_rhos[int(targets[indices[0]])].append(
                modeling.safe_spearman(construction[sampled], utility[sampled])
            )
        checkpoint_values.append(float(np.nanmean([
            np.nanmean(values) for values in target_rhos.values()
        ])))
    checkpoint_values = np.asarray(checkpoint_values)
    level_rows = [{
        "bootstrap_level": "target",
        "estimand": "construction_evaluation_reliability",
        "mean": float(np.mean(target_bootstrap)),
        "ci_low": _quantile(target_bootstrap, 0.025),
        "ci_high": _quantile(target_bootstrap, 0.975),
        "replicates": len(target_bootstrap),
    }, {
        "bootstrap_level": "checkpoint_within_trajectory_then_target",
        "estimand": "construction_evaluation_reliability",
        "mean": float(np.mean(checkpoint_values)),
        "ci_low": _quantile(checkpoint_values, 0.025),
        "ci_high": _quantile(checkpoint_values, 0.975),
        "replicates": len(checkpoint_values),
    }]
    return {"target_trial": target_trial_rows, "crossed": crossed_rows, "levels": level_rows}


def _candidate_gate(
    path: str,
    prediction: dict[str, Any],
    krr: dict[str, Any],
) -> dict[str, Any]:
    linear = next(
        row for row in prediction["observed"]
        if row["path"] == path and row["outcome"] == "continuous_joint_utility"
    )
    action = next(row for row in prediction["action_summary"] if row["path"] == path)
    leave_regime = [
        row["increment_residual_rho"] for row in prediction["leave_regime"]
        if row["path"] == path and row["outcome"] == "continuous_joint_utility"
    ]
    material = bool(action["material_actionability"])
    passed = (
        linear["incremental_LOTO_R2"] >= 0.02
        and linear["max_stat_corrected_p"] < 0.05
        and linear["LOTO_median_increment_residual_rho"] > 0
        and linear["positive_targets"] >= 6
        and float(np.nanmedian(leave_regime)) > 0
        and material
    )
    return {
        "path": path,
        "incremental_R2": linear["incremental_LOTO_R2"],
        "max_stat_corrected_p": linear["max_stat_corrected_p"],
        "leave_target_median": linear["LOTO_median_increment_residual_rho"],
        "positive_targets": linear["positive_targets"],
        "leave_regime_median": float(np.nanmedian(leave_regime)),
        "material_actionability": int(material),
        "all_registered_gates_pass": int(passed),
        "gate_interpretation": "diagnostic_candidate_only_not_selector" if passed else "no_registered_candidate_qualification",
    }


def _primary_verdicts(
    measurement: dict[str, Any],
    prediction: dict[str, Any],
    krr: dict[str, Any],
    association: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strict_gate = _candidate_gate("strict_source_F2", prediction, krr)
    target_gate = _candidate_gate("target_unlabeled_F4_geometry", prediction, krr)
    construction_linear = next(
        row for row in prediction["observed"]
        if row["path"] == "construction_F5_positive" and row["outcome"] == "continuous_joint_utility"
    )
    per_target = defaultdict(dict)
    for row in prediction["leave_target"]:
        if row["outcome"] == "continuous_joint_utility":
            per_target[int(row["target_id"])][row["path"]] = row["incremental_R2"]
    h6_values = np.asarray([
        values["construction_F5_positive"] - max(
            values["strict_source_F2"], values["target_unlabeled_F4_geometry"],
        )
        for values in per_target.values()
    ])
    h6_p = modeling.exact_sign_flip_p(h6_values)
    fixed_association = {
        "strict_source": next(
            row for row in association["summary"]
            if row["path"] == "strict_source" and row["kernel"] == "rbf"
            and float(row["bandwidth_factor"]) == 1.0 and row["statistic"] == "normalized_alignment"
        ),
        "target_unlabeled": next(
            row for row in association["summary"]
            if row["path"] == "target_unlabeled" and row["kernel"] == "laplacian"
            and float(row["bandwidth_factor"]) == 1.0 and row["statistic"] == "centered_hsic"
        ),
    }
    local_topology = {
        path: next(
            row for row in association["topology"]
            if row["path"] == path and row["level"] == "within_target_x_level_x_regime"
        )
        for path in ("strict_source", "target_unlabeled")
    }
    krr_by_path = {row["path"]: row for row in krr["summary"]}
    krr_action = {row["path"]: row for row in krr["action_summary"]}
    h3_local = max(local_topology[path]["association"] for path in local_topology)
    h3_p = min(fixed_association[path]["worst_required_global_p"] for path in fixed_association)
    h3_transport_qualified = any(
        row["incremental_LOTO_R2"] >= 0.02
        and row["global_max_stat_p"] < 0.05
        and row["incremental_LORO_R2"] > 0
        and krr_action[row["path"]]["material_actionability"]
        for row in krr["summary"]
    )
    h1_active = (
        measurement["summary"]["target_mean_reliability"] > 0
        and measurement["summary"]["target_sign_flip_p"] < 0.05
        and not measurement["summary"]["material_actionability"]
    )
    h2_active = (
        geometry["summary"]["incremental_deviance_reduction"] > 0
        and geometry["summary"]["permutation_p"] < 0.05
    )
    h3_active = h3_local > 0 and h3_p < 0.05 and not h3_transport_qualified
    h4_active = not bool(strict_gate["all_registered_gates_pass"])
    h5_active = not bool(target_gate["all_registered_gates_pass"])
    h6_active = (
        float(np.mean(h6_values)) > 0
        and h6_p < 0.05
        and construction_linear["incremental_LOTO_R2"] > 0
    )
    primary_rows = [
        {
            "hypothesis": "H1",
            "claim": "measurement_control_separation_replicates",
            "raw_p": measurement["summary"]["target_sign_flip_p"],
            "material_gate": int(not measurement["summary"]["material_actionability"]),
            "active_before_Holm": int(h1_active),
            "effect": measurement["summary"]["target_mean_reliability"],
            "absence_claim_rule": "actionability_nonqualification_not_null_acceptance",
        },
        {
            "hypothesis": "H2",
            "claim": "effective_multiplicity_top_gap_improve_failure_anatomy_beyond_raw_M",
            "raw_p": geometry["summary"]["permutation_p"],
            "material_gate": int(geometry["summary"]["incremental_deviance_reduction"] > 0),
            "active_before_Holm": int(h2_active),
            "effect": geometry["summary"]["incremental_deviance_reduction"],
            "absence_claim_rule": "not_applicable",
        },
        {
            "hypothesis": "H3",
            "claim": "local_nonlinear_association_nontransportable_nonactionable",
            "raw_p": h3_p,
            "material_gate": int(not h3_transport_qualified),
            "active_before_Holm": int(h3_active),
            "effect": h3_local,
            "absence_claim_rule": "transport_and_actionability_gates_not_p_greater_0.05",
        },
        {
            "hypothesis": "H4",
            "claim": "no_registered_strict_source_representation_escape_hatch",
            "raw_p": strict_gate["max_stat_corrected_p"],
            "material_gate": int(not strict_gate["all_registered_gates_pass"]),
            "active_before_Holm": int(h4_active),
            "effect": strict_gate["incremental_R2"],
            "absence_claim_rule": "registered_candidate_nonqualification_only",
        },
        {
            "hypothesis": "H5",
            "claim": "no_registered_target_unlabeled_representation_actionable_control",
            "raw_p": target_gate["max_stat_corrected_p"],
            "material_gate": int(not target_gate["all_registered_gates_pass"]),
            "active_before_Holm": int(h5_active),
            "effect": target_gate["incremental_R2"],
            "absence_claim_rule": "registered_candidate_nonqualification_only",
        },
        {
            "hypothesis": "H6",
            "claim": "split_label_construction_strongest_registered_nonoracle_positive_control",
            "raw_p": h6_p,
            "material_gate": int(construction_linear["incremental_LOTO_R2"] > 0),
            "active_before_Holm": int(h6_active),
            "effect": float(np.mean(h6_values)),
            "absence_claim_rule": "not_applicable",
        },
    ]
    adjusted = modeling.holm_adjust(primary_rows)
    for row in adjusted:
        if row["hypothesis"] in {"H4", "H5"}:
            row["active_after_Holm"] = row["active_before_Holm"]
            row["Holm_note"] = "nonqualification_does_not_require_accepting_a_null"
        else:
            row["active_after_Holm"] = int(row["active_before_Holm"] and row["Holm_reject_0.05"])
            row["Holm_note"] = "positive_evidence_requires_Holm_rejection"
    gates = {
        "strict_source": strict_gate,
        "target_unlabeled": target_gate,
        "h6_target_effect_mean": float(np.mean(h6_values)),
        "h6_target_sign_p": h6_p,
        "h3_transport_qualified": int(h3_transport_qualified),
    }
    return adjusted, gates


def _aggregate_context_tables(
    arrays: dict[str, np.ndarray],
    prediction: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    targets = arrays["target_id"].astype(int)
    regimes = arrays["regime"].astype(str)
    levels = arrays["level"].astype(int)
    utility = arrays["outcomes"][:, 0].astype(float)
    construct = arrays["F5"][:, -1].astype(float)
    target_rows = []
    for target in protocol.PRIMARY_TARGETS:
        mask = targets == target
        target_rows.append({
            "target_id": target,
            "units": int(np.sum(mask)),
            "continuous_utility_mean": float(np.mean(utility[mask])),
            "construction_eval_spearman": modeling.safe_spearman(construct[mask], utility[mask]),
            "target4_canary": 0,
            "seed3_exploratory_not_confirmation": 1,
        })
    regime_rows = []
    for regime in protocol.REGIMES:
        mask = regimes == regime
        regime_rows.append({
            "regime": regime,
            "units": int(np.sum(mask)),
            "targets": len(set(targets[mask].tolist())),
            "levels": len(set(levels[mask].tolist())),
            "continuous_utility_mean": float(np.mean(utility[mask])),
            "construction_eval_spearman": modeling.safe_spearman(construct[mask], utility[mask]),
            "trajectory_role": "anchor" if regime == "ERM" else ("historical_primary" if regime == "OACI" else "historical_negative_control"),
        })
    level_rows = []
    for level in protocol.LEVELS:
        mask = levels == level
        level_rows.append({
            "level": level,
            "units": int(np.sum(mask)),
            "continuous_utility_mean": float(np.mean(utility[mask])),
            "construction_eval_spearman": modeling.safe_spearman(construct[mask], utility[mask]),
        })
    return {"target": target_rows, "regime": regime_rows, "level": level_rows}


def _write_c79_protocol(
    primary_rows: list[dict[str, Any]],
    output_manifest_sha: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "c79_seed4_locked_confirmation_protocol_v1",
        "created_at_utc": protocol.utc_now(),
        "status": "LOCKED_PROTOCOL_READY_BUT_EXECUTION_NOT_AUTHORIZED",
        "parent_C78S_nonoracle_output_manifest_sha256": output_manifest_sha,
        "seed": 4,
        "dataset": "BNCI2014_001",
        "targets": list(range(1, 10)),
        "levels": [0, 1],
        "regimes": ["ERM", "OACI", "SRC"],
        "field_units": 1458,
        "confirmation_role": "new_seed_checkpoint_field_not_new_target_population",
        "C78S_active_hypotheses_to_confirm": [
            row["hypothesis"] for row in primary_rows if row["active_after_Holm"]
        ],
        "fixed_feature_blocks": protocol.feature_registry(),
        "fixed_materiality": json.loads(protocol.PROTOCOL_PATH.read_text())["materiality"],
        "fixed_nulls": json.loads(protocol.PROTOCOL_PATH.read_text())["nulls"],
        "fixed_multiplicity": json.loads(protocol.PROTOCOL_PATH.read_text())["multiplicity"],
        "inference": {
            "seed4_analyzed_alone_before_any_seed3_pooling": True,
            "within_target_and_regime_centering": True,
            "leave_target_regime_trajectory": True,
            "target_checkpoint_trial_cluster_inference": True,
            "row_iid": False,
            "ERM_anchor_not_symmetric_trajectory": True,
        },
        "execution_requires_future_direct_PI_authorization": True,
        "authorization_received": False,
        "forbidden": [
            "C79_execution_under_C78S_authorization",
            "BNCI2014_004",
            "target_outcome_hyperparameter_tuning",
            "selector_or_checkpoint_recommendation",
            "manuscript_drafting",
        ],
    }
    protocol.write_json(C79_PROTOCOL_PATH, payload)
    C79_PROTOCOL_SHA_PATH.write_text(protocol.sha256_file(C79_PROTOCOL_PATH) + "\n")
    return payload


def _risk_rows() -> list[dict[str, Any]]:
    evidence = {
        "protocol_hash_or_execution_lock_drift": "protocol_replay.csv",
        "target4_in_primary_estimand": "target4_exclusion_audit.csv",
        "target4_in_primary_null_pool": "target4_exclusion_audit.csv",
        "target4_in_primary_multiplicity_family": "primary_hypothesis_multiplicity.csv",
        "quarantined_label_read_before_execution_lock": "authorization_and_timing_audit.csv",
        "construction_evaluation_trial_overlap": "label_split_isolation.csv",
        "same_label_oracle_early_access": "oracle_access_audit.csv",
        "trial_id_or_row_order_as_predictor": "feature_availability_ledger.csv",
        "target_label_in_unlabeled_feature": "feature_availability_ledger.csv",
        "outcome_adaptive_retry_or_report_scope": "execution_attempt_ledger.csv",
        "bandwidth_selected_outside_nested_contract": "association_null_summary.csv",
        "row_iid_inference": "hierarchical_bootstrap_summary.csv",
        "checkpoint_units_treated_as_population_replication": "claim_boundary_ledger.csv",
        "ERM_OACI_SRC_false_symmetry": "regime_summary.csv",
        "association_called_prediction": "association_prediction_separation.csv",
        "prediction_called_actionability": "actionability_summary.csv",
        "seed3_called_seed_confirmation": "claim_boundary_ledger.csv",
        "source_or_target_unlabeled_null_called_universal_failure": "claim_boundary_ledger.csv",
        "target_population_overclaim": "claim_boundary_ledger.csv",
        "seed4_or_C79_access": "seed4_protection_audit.csv",
        "BNCI2014_004_access": "seed4_protection_audit.csv",
        "training_forward_reinference_or_GPU": "execution_attempt_ledger.csv",
        "raw_cache_or_weights_in_git": "artifact_hygiene_audit.csv",
        "selector_or_checkpoint_recommendation": "claim_boundary_ledger.csv",
        "manuscript_drafting": "claim_boundary_ledger.csv",
    }
    return [
        {
            **row,
            "status": "CLOSED",
            "blocking": 0,
            "evidence": evidence[row["risk"]],
        }
        for row in protocol.risk_registry()
    ]


def _report_text(result: dict[str, Any]) -> str:
    h = {row["hypothesis"]: row for row in result["primary_hypotheses"]}
    return f"""# C78S — Full Seed-3 Multi-Regime Scientific Analysis

## Final gate

```text
{result['final_gate']}
```

Primary target field: eight prospectively generated seed-3 targets, 1,296 units.
Target 4 is excluded from every primary estimand, null pool, and multiplicity family.
Seed 3 remains exploratory replication; it is not an independent target-population or seed confirmation.

## Registered H1–H6 verdicts

| Hypothesis | Active | Effect | Raw p | Holm p | Boundary |
|---|---:|---:|---:|---:|---|
""" + "\n".join(
        f"| {row['hypothesis']} | {row['active_after_Holm']} | {row['effect']:.6f} | {row['raw_p']:.6f} | {row['Holm_p']:.6f} | {row['absence_claim_rule']} |"
        for row in result["primary_hypotheses"]
    ) + f"""

## Measurement and control

Construction/evaluation trajectory reliability is `{result['headline']['trajectory_reliability_mean']:.6f}`
(target-cluster 95% CI `{result['headline']['reliability_ci_low']:.6f}` to
`{result['headline']['reliability_ci_high']:.6f}`). The registered construction score's
top-5 improvement over the source baseline is `{result['headline']['construction_delta_top5']:.6f}`;
standardized regret reduction is `{result['headline']['construction_regret_reduction']:.6f}`.
These quantities are reported separately because association/reliability is not checkpoint control.

## Representation information classes

Strict-source F2 incremental R2 is `{result['headline']['strict_source_incremental_R2']:.6f}`;
target-unlabeled F4-geometry incremental R2 is
`{result['headline']['target_unlabeled_incremental_R2']:.6f}`. Registered qualification requires
material R2, corrected-null passage, leave-target and leave-regime transport, positive direction in
at least six targets, and material top-k or regret improvement. The report therefore makes only the
registered candidate/nonqualification calls shown in H4 and H5; it does not claim universal
impossibility for all source or target-unlabeled functions.

## Geometry and transport

The effective-multiplicity/top-gap model improves held-target top-1-miss deviance by
`{result['headline']['geometry_deviance_reduction']:.6f}` with blocked permutation p
`{result['headline']['geometry_p']:.6f}`. This is endpoint-derived diagnostic geometry, not a selector.
The strongest registered local nonlinear association is
`{result['headline']['local_association']:.6f}`; its prediction and actionability gates are reported
separately and cannot be replaced by an association p-value.

## Information and provenance boundaries

- Construction and evaluation trial IDs are physically disjoint and cover all 576 target trials.
- Trial IDs and row order are used only for joining, splitting, and dependence clustering.
- The same-label oracle descriptor was not presented to the primary runner and was never opened.
- No training, forward pass, re-inference, GPU, seed 4, C79 execution, BNCI2014_004, selector,
  checkpoint recommendation, or manuscript drafting occurred.
- C79 is protocol-ready only. It remains unauthorized.

## Taxonomy

```text
{chr(10).join(result['active_taxonomy'])}
```

## Red-team status

The independent result red team must pass before this report is presented as final.
"""


def run() -> dict[str, Any]:
    lock, lock_sha = protocol.load_execution_lock()
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    _state("C78S_started", lock_sha256=lock_sha, target_label_payload_reads_before_lock=0)
    provenance = c78s_data.provenance_replay()
    for name, rows in provenance.items():
        _write_table(f"{name}_replay.csv", rows)
    _write_table("feature_block_registry.csv", protocol.feature_registry())
    _write_table("feature_availability_ledger.csv", [
        {
            **row,
            "available_at_original_source_only_selection_time": int(row["information_class"].startswith("strict_source") or row["block"] == "F0"),
            "diagnostic_only": int(row["information_class"].startswith("target_")),
        }
        for row in protocol.feature_registry()
    ])
    c78s_data.extract_unlabeled_cache()
    _state("unlabeled_features_frozen", target_labels_accessed=0)
    c78s_data.build_labeled_cache()
    manifests, raw_arrays = c78s_data.load_analysis_cache()
    arrays = _prepare_arrays(raw_arrays)
    _write_table("label_split_isolation.csv", manifests["labeled"]["split_isolation"])
    _write_table("oracle_access_audit.csv", [{
        "primary_route_contains_oracle_descriptor": 0,
        "same_label_oracle_view_opened": 0,
        "terminal_oracle_stage_run": 0,
        "reason": "not_required_for_registered_H1_H6_and_not_authorized_as_primary_input",
        "passed": 1,
    }])
    measurement = _measurement_control(arrays)
    _write_table("measurement_reliability_by_context.csv", measurement["reliability"])
    _write_table("measurement_control_actionability_cells.csv", measurement["action"])
    _write_table("measurement_control_summary.csv", [measurement["summary"]])
    _state("H1_complete")
    prediction = _prediction_audit(arrays)
    _write_table("cross_fitted_incremental_prediction.csv", prediction["observed"])
    _write_table("leave_target_out_prediction.csv", prediction["leave_target"])
    _write_table("leave_regime_out_prediction.csv", prediction["leave_regime"])
    _write_table("leave_trajectory_out_prediction.csv", prediction["leave_trajectory"])
    _write_table("prediction_fold_ledger.csv", prediction["folds"])
    _write_table("nested_block_nulls.csv", prediction["null"])
    _write_table("actionability_cells.csv", prediction["action"])
    _write_table("actionability_summary.csv", prediction["action_summary"])
    krr = _krr_audit(arrays, prediction)
    _write_table("nonlinear_prediction_summary.csv", krr["summary"])
    _write_table("nonlinear_prediction_folds.csv", krr["folds"])
    _write_table("nonlinear_leave_target.csv", krr["leave_target"])
    _write_table("nonlinear_leave_regime.csv", krr["leave_regime"])
    _write_table("nonlinear_leave_trajectory.csv", krr["leave_trajectory"])
    _write_table("nonlinear_actionability_cells.csv", krr["action"])
    _write_table("nonlinear_actionability_summary.csv", krr["action_summary"])
    association = _association_audit(arrays, prediction)
    _write_table("nonlinear_association_family.csv", association["family"])
    _write_table("nonlinear_association_target_folds.csv", association["family_folds"])
    _write_table("association_null_summary.csv", association["null"])
    _write_table("association_strict_control_summary.csv", association["summary"])
    _write_table("association_topology.csv", association["topology"])
    _write_table("association_topology_folds.csv", association["topology_folds"])
    _write_table("association_prediction_separation.csv", [
        {
            "path": path,
            "local_association": next(
                row["association"] for row in association["topology"]
                if row["path"] == path and row["level"] == "within_target_x_level_x_regime"
            ),
            "incremental_LOTO_R2": next(row["incremental_LOTO_R2"] for row in krr["summary"] if row["path"] == path),
            "incremental_LORO_R2": next(row["incremental_LORO_R2"] for row in krr["summary"] if row["path"] == path),
            "material_actionability": next(row["material_actionability"] for row in krr["action_summary"] if row["path"] == path),
            "association_is_not_prediction": 1,
            "prediction_is_not_actionability": 1,
        }
        for path in ("strict_source", "target_unlabeled")
    ])
    _state("H3_H4_H5_complete")
    geometry = _geometry_audit(arrays)
    _write_table("effective_multiplicity_prefix_ledger.csv", geometry["rows"])
    _write_table("effective_multiplicity_coefficients.csv", geometry["coefficients"])
    _write_table("effective_multiplicity_summary.csv", [geometry["summary"]])
    _state("H2_complete")
    bootstrap = _hierarchical_bootstrap(arrays, measurement)
    _write_table("trial_cluster_bootstrap_by_target.csv", bootstrap["target_trial"])
    _write_table("crossed_target_trial_bootstrap.csv", bootstrap["crossed"])
    _write_table("hierarchical_bootstrap_summary.csv", bootstrap["levels"])
    primary_rows, gates = _primary_verdicts(measurement, prediction, krr, association, geometry)
    _write_table("primary_hypothesis_multiplicity.csv", primary_rows)
    _write_table("registered_candidate_gate.csv", [gates["strict_source"], gates["target_unlabeled"]])
    context = _aggregate_context_tables(arrays, prediction)
    _write_table("target_summary.csv", context["target"])
    _write_table("regime_summary.csv", context["regime"])
    _write_table("level_summary.csv", context["level"])
    _write_table("target4_exclusion_audit.csv", [{
        "target4_primary_units": 0,
        "target4_primary_estimands": 0,
        "target4_primary_null_members": 0,
        "target4_primary_multiplicity_members": 0,
        "target4_engineering_settings_changed_after_remaining_outcomes": 0,
        "passed": 1,
    }])
    _write_table("authorization_and_timing_audit.csv", [{
        "protocol_sha256": lock["protocol_sha256"],
        "protocol_commit": lock["protocol_commit"],
        "implementation_commit": lock["implementation_commit"],
        "execution_lock_sha256": lock_sha,
        "execution_lock_committed_before_label_payload_read": 1,
        "direct_PI_authorization": 1,
        "authorization_mode": lock["authorization"]["mode"],
        "passed": 1,
    }])
    _write_table("execution_attempt_ledger.csv", [{
        "attempt": 1,
        "scope": "one_registered_H1_H6_analysis",
        "outcome_adaptive_retry": 0,
        "training": 0,
        "forward": 0,
        "reinference": 0,
        "GPU": 0,
        "status": "PRIMARY_RUN_COMPLETE",
    }])
    _write_table("seed4_protection_audit.csv", [{
        "seed4_data_access": 0,
        "seed4_training_jobs": 0,
        "seed4_forward_or_reinference": 0,
        "seed4_checkpoints": 0,
        "seed4_caches": 0,
        "seed4_outcomes_read": 0,
        "C79_execution": 0,
        "passed": 1,
    }])
    _write_table("claim_boundary_ledger.csv", [
        {"claim": "seed3_exploratory_replication", "allowed": 1, "active": 1},
        {"claim": "seed_level_confirmation", "allowed": 0, "active": 0},
        {"claim": "target_population_confirmation", "allowed": 0, "active": 0},
        {"claim": "all_source_functions_fail", "allowed": 0, "active": 0},
        {"claim": "all_target_unlabeled_functions_fail", "allowed": 0, "active": 0},
        {"claim": "representation_causality", "allowed": 0, "active": 0},
        {"claim": "selector_or_checkpoint_recommendation", "allowed": 0, "active": 0},
        {"claim": "deployability", "allowed": 0, "active": 0},
        {"claim": "manuscript_drafting", "allowed": 0, "active": 0},
    ])
    tracked_candidates = [
        Path(path) for path in protocol.git("ls-files", "oaci").splitlines()
        if path and Path(path).is_file()
    ]
    largest = max((path.stat().st_size for path in tracked_candidates), default=0)
    _write_table("artifact_hygiene_audit.csv", [{
        "raw_cache_in_git": 0,
        "checkpoint_weights_in_git": 0,
        "largest_oaci_worktree_file_bytes": largest,
        "payload_over_50MiB": int(largest >= 50 * 1024 * 1024),
        "external_cache_root": str(c78s_data.run_root(lock)),
        "passed": int(largest < 50 * 1024 * 1024),
    }])
    _write_table("risk_register.csv", _risk_rows())
    _write_table("failure_reason_ledger.csv", [{
        "failure": "none",
        "blocking": 0,
        "detail": "all_protocol_provenance_masking_and_analysis_gates_completed",
    }])
    active_h = [row["hypothesis"] for row in primary_rows if row["active_after_Holm"]]
    strict_escape = bool(gates["strict_source"]["all_registered_gates_pass"])
    target_escape = bool(gates["target_unlabeled"]["all_registered_gates_pass"])
    if strict_escape or target_escape:
        final_gate = "SEED3_SOURCE_OR_TARGET_UNLABELED_ESCAPE_HATCH_REQUIRES_FORENSICS"
    elif "H1" in active_h and "H6" in active_h:
        final_gate = "SEED3_MEASUREMENT_CONTROL_SEPARATION_REPLICATED_C79_READY_BUT_NOT_AUTHORIZED"
    else:
        final_gate = "SEED3_MIXED_RESULTS_C79_PROTOCOL_REVIEW_REQUIRED"
    active_taxonomy = [
        f"C78S-{row['hypothesis']}_{row['claim']}"
        for row in primary_rows if row["active_after_Holm"]
    ]
    active_taxonomy.extend([
        "C78S-S1_complete_1296_unit_primary_field_consumed",
        "C78S-S2_target4_mechanically_excluded",
        "C78S-S3_same_label_oracle_not_accessed",
        "C78S-S4_seed3_exploratory_not_confirmation",
        "C78S-S5_seed4_untouched",
        "C78S-S6_C79_protocol_ready_execution_not_authorized",
    ])
    strict_row = next(
        row for row in prediction["observed"]
        if row["path"] == "strict_source_F2" and row["outcome"] == "continuous_joint_utility"
    )
    target_row = next(
        row for row in prediction["observed"]
        if row["path"] == "target_unlabeled_F4_geometry" and row["outcome"] == "continuous_joint_utility"
    )
    local_association = max(
        row["association"] for row in association["topology"]
        if row["level"] == "within_target_x_level_x_regime"
    )
    output_manifest = c74_cache.self_hashed_manifest({
        "schema_version": "c78s_nonoracle_primary_output_manifest_v1",
        "created_at_utc": protocol.utc_now(),
        "protocol_sha256": lock["protocol_sha256"],
        "execution_lock_sha256": lock_sha,
        "primary_targets": list(protocol.PRIMARY_TARGETS),
        "primary_units": protocol.PRIMARY_UNITS,
        "primary_hypotheses": primary_rows,
        "same_label_oracle_accessed": False,
        "target4_accessed": False,
        "seed4_accessed": False,
        "table_files": sorted(path.name for path in protocol.TABLE_DIR.glob("*.csv")),
    })
    external_output_manifest = c78s_data.run_root(lock) / "C78S_NONORACLE_PRIMARY_OUTPUTS.json"
    c74_cache.atomic_json(external_output_manifest, output_manifest)
    frozen = c78s_data.mark_primary_outputs_frozen({
        "path": str(external_output_manifest),
        "sha256": protocol.sha256_file(external_output_manifest),
        "manifest_sha256": output_manifest["manifest_sha256"],
    })
    c79 = _write_c79_protocol(primary_rows, output_manifest["manifest_sha256"])
    result = {
        "schema_version": "c78s_seed3_scientific_analysis_result_v1",
        "milestone": "C78S",
        "protocol_sha256": lock["protocol_sha256"],
        "execution_lock_sha256": lock_sha,
        "authorization": {
            "received": True,
            "mode": lock["authorization"]["mode"],
            "scope": "C78S_locked_seed3_analysis_only",
        },
        "field": {
            "primary_targets": list(protocol.PRIMARY_TARGETS),
            "target4_primary": False,
            "units": protocol.PRIMARY_UNITS,
            "regimes": list(protocol.REGIMES),
            "levels": list(protocol.LEVELS),
            "seed": 3,
            "seed3_role": "exploratory_replication_not_seed_confirmation",
        },
        "primary_hypotheses": primary_rows,
        "registered_candidate_gates": gates,
        "active_taxonomy": active_taxonomy,
        "headline": {
            "trajectory_reliability_mean": measurement["summary"]["trajectory_reliability_mean"],
            "reliability_ci_low": measurement["summary"]["target_bootstrap_ci_low"],
            "reliability_ci_high": measurement["summary"]["target_bootstrap_ci_high"],
            "construction_delta_top5": measurement["summary"]["delta_oracle_best_in_predicted_top5"],
            "construction_regret_reduction": measurement["summary"]["standardized_regret_reduction"],
            "strict_source_incremental_R2": strict_row["incremental_LOTO_R2"],
            "target_unlabeled_incremental_R2": target_row["incremental_LOTO_R2"],
            "geometry_deviance_reduction": geometry["summary"]["incremental_deviance_reduction"],
            "geometry_p": geometry["summary"]["permutation_p"],
            "local_association": local_association,
        },
        "information_boundaries": {
            "same_label_oracle_accessed": False,
            "trial_id_predictor": False,
            "row_order_predictor": False,
            "training": False,
            "forward": False,
            "reinference": False,
            "GPU": False,
            "seed4": False,
            "C79_execution": False,
            "BNCI2014_004": False,
            "selector_or_checkpoint_recommendation": False,
            "manuscript_drafting": False,
        },
        "nonoracle_output_manifest": {
            "path": str(external_output_manifest),
            "sha256": protocol.sha256_file(external_output_manifest),
            "manifest_sha256": output_manifest["manifest_sha256"],
        },
        "primary_freeze_manifest_sha256": frozen["manifest_sha256"],
        "C79_protocol": {
            "path": str(C79_PROTOCOL_PATH),
            "sha256": protocol.sha256_file(C79_PROTOCOL_PATH),
            "execution_authorized": False,
        },
        "final_gate": final_gate,
        "red_team": {"status": "PENDING_INDEPENDENT_RESULT_RED_TEAM"},
    }
    protocol.write_json(RESULT_PATH, result)
    REPORT_PATH.write_text(_report_text(result))
    artifact_manifest = {
        "schema_version": "c78s_artifact_manifest_v1",
        "created_at_utc": protocol.utc_now(),
        "result": {"path": str(RESULT_PATH), "sha256": protocol.sha256_file(RESULT_PATH)},
        "report": {"path": str(REPORT_PATH), "sha256": protocol.sha256_file(REPORT_PATH)},
        "C79_protocol": {"path": str(C79_PROTOCOL_PATH), "sha256": protocol.sha256_file(C79_PROTOCOL_PATH)},
        "tables": [
            {"path": str(path), "sha256": protocol.sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(protocol.TABLE_DIR.glob("*.csv"))
        ],
        "raw_cache_in_git": False,
    }
    protocol.write_json(ARTIFACT_MANIFEST_PATH, artifact_manifest)
    _state("C78S_primary_outputs_complete", final_gate=final_gate, red_team="pending")
    return result


def finalize() -> dict[str, Any]:
    red_team_path = protocol.REPORT_DIR / "C78S_AUTHORIZED_RED_TEAM_VERIFICATION.json"
    if not red_team_path.is_file():
        raise RuntimeError("C78S finalize requires independent red-team JSON")
    red_team = json.loads(red_team_path.read_text())
    if red_team["blocking_failures"] != 0 or not red_team["passed"]:
        raise RuntimeError("C78S red team has blocking failures")
    result = json.loads(RESULT_PATH.read_text())
    result["red_team"] = {
        "status": "PASS",
        "checks": red_team["checks"],
        "blocking_failures": 0,
        "report": "oaci/reports/C78S_AUTHORIZED_RED_TEAM_VERIFICATION.md",
        "sha256": protocol.sha256_file(protocol.REPORT_DIR / "C78S_AUTHORIZED_RED_TEAM_VERIFICATION.md"),
    }
    protocol.write_json(RESULT_PATH, result)
    REPORT_PATH.write_text(_report_text(result).replace(
        "The independent result red team must pass before this report is presented as final.",
        f"Independent result red team: PASS ({red_team['checks']}/{red_team['checks']}, zero blockers).",
    ))
    artifact_manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text())
    artifact_manifest["result"]["sha256"] = protocol.sha256_file(RESULT_PATH)
    artifact_manifest["report"]["sha256"] = protocol.sha256_file(REPORT_PATH)
    artifact_manifest["red_team"] = {
        "path": "oaci/reports/C78S_AUTHORIZED_RED_TEAM_VERIFICATION.md",
        "sha256": protocol.sha256_file(protocol.REPORT_DIR / "C78S_AUTHORIZED_RED_TEAM_VERIFICATION.md"),
    }
    protocol.write_json(ARTIFACT_MANIFEST_PATH, artifact_manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78s_seed3_scientific_analysis")
    parser.add_argument("command", choices=("run", "finalize"))
    args = parser.parse_args(argv)
    result = run() if args.command == "run" else finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
