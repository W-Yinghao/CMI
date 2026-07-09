"""TTA_MECH_01 real existing-baseline replay / mechanism audit.

This runner replays frozen feature baselines over the CEDAR_01F handoff
artifacts. It is a mechanism audit, not a new method, deployment selection, or
P1/P2 training step.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cedar_eeg.data.feature_handoff import validate_handoff_manifest
from cedar_eeg.data.feature_schema import sha256_file
from cedar_eeg.data.load_frozen_features import FrozenFeatureBundle, load_frozen_feature_npz
from tta_mech_eeg.audit_axes.schema import audit_axis_schema_payload
from tta_mech_eeg.baselines.registry import ALLOWED_BASELINES, registry_payload, stable_hash
from tta_mech_eeg.data.artifact_inventory import build_artifact_inventory
from tta_mech_eeg.data.handoff_schema import DEFAULT_CEDAR01F_HANDOFF
from tta_mech_eeg.red_team.baseline_universe_freeze import validate_baseline_universe
from tta_mech_eeg.red_team.no_new_method_guard import validate_no_new_method


EXPECTED_BASELINE_REGISTRY_HASH = "0d8a00cdb2d0bf810a20c58056323c21c1fde8807fb12e65ebc9ff8334748da7"
EXPECTED_ARTIFACT_INVENTORY_HASH = "6323097829f2d3277275f392ea33329a4197e7032969bb3ebf87d1ad2e090cb2"


@dataclass(frozen=True)
class ReadoutState:
    class_labels: tuple[int, ...]
    source_prior: np.ndarray
    source_mean: np.ndarray
    source_cov: np.ndarray
    source_prototypes: np.ndarray
    pooled_var: np.ndarray
    weight: np.ndarray
    bias: np.ndarray


def _rows(bundle: FrozenFeatureBundle, role_name: str) -> dict[str, np.ndarray]:
    role = np.asarray(bundle.role).astype(str)
    keep = role == role_name
    if not np.any(keep):
        raise ValueError(f"artifact {bundle.metadata.get('path')} has no {role_name} rows")
    return {
        "z": bundle.z[keep],
        "y": bundle.y[keep],
        "domain": bundle.domain[keep],
        "groups": bundle.groups[keep],
    }


def _cov(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    return cov + np.eye(cov.shape[0]) * eps


def _matrix_power_psd(mat: np.ndarray, power: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(mat, dtype=np.float64))
    vals = np.maximum(vals, 1e-8)
    return (vecs * (vals ** power)[None, :]) @ vecs.T


def _fit_readout(z_source: np.ndarray, y_source: np.ndarray) -> ReadoutState:
    z = np.asarray(z_source, dtype=np.float64)
    y = np.asarray(y_source).astype(np.int64, copy=False)
    labels = tuple(int(x) for x in sorted(np.unique(y)))
    if len(labels) < 2:
        raise ValueError("source readout requires at least two classes")
    prototypes = []
    counts = []
    for label in labels:
        rows = z[y == label]
        prototypes.append(rows.mean(axis=0))
        counts.append(len(rows))
    proto = np.vstack(prototypes)
    counts_arr = np.asarray(counts, dtype=np.float64)
    prior = counts_arr / counts_arr.sum()
    pooled_var = np.maximum(z.var(axis=0), 1e-4)
    weight = (proto / pooled_var).T
    bias = -0.5 * np.sum((proto * proto) / pooled_var, axis=1) + np.log(np.maximum(prior, 1e-8))
    return ReadoutState(
        class_labels=labels,
        source_prior=prior,
        source_mean=z.mean(axis=0),
        source_cov=_cov(z),
        source_prototypes=proto,
        pooled_var=pooled_var,
        weight=weight,
        bias=bias,
    )


def _logits(state: ReadoutState, z: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) @ state.weight + state.bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(proba: np.ndarray) -> np.ndarray:
    return -np.sum(proba * np.log(np.maximum(proba, 1e-8)), axis=1)


def _margin(proba: np.ndarray) -> np.ndarray:
    sorted_p = np.sort(proba, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


def _array_hash(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(arr))
    payload = {
        "shape": tuple(int(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "sha256": __import__("hashlib").sha256(arr.tobytes()).hexdigest(),
    }
    return stable_hash(payload)


def _encode_y(y: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(labels)}
    return np.asarray([mapping[int(v)] for v in y], dtype=np.int64)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[int, ...]) -> float:
    recalls = []
    for label in labels:
        keep = y_true == label
        if np.any(keep):
            recalls.append(float((y_pred[keep] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[int, ...]) -> float:
    scores = []
    for label in labels:
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        denom = (2.0 * tp + fp + fn)
        scores.append(0.0 if denom <= 0.0 else float((2.0 * tp) / denom))
    return float(np.mean(scores)) if scores else float("nan")


def _nll(y_true: np.ndarray, proba: np.ndarray, labels: tuple[int, ...]) -> float:
    encoded = _encode_y(y_true, labels)
    picked = np.clip(proba[np.arange(len(encoded)), encoded], 1e-8, 1.0)
    return float(-np.log(picked).mean())


def _ece(y_true: np.ndarray, pred_labels: np.ndarray, proba: np.ndarray, *, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    correct = (pred_labels == y_true).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            keep = (conf >= lo) & (conf <= hi)
        else:
            keep = (conf >= lo) & (conf < hi)
        if np.any(keep):
            ece += float(keep.mean()) * abs(float(correct[keep].mean()) - float(conf[keep].mean()))
    return float(ece)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-8)
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-8)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _coral_distance(z: np.ndarray, state: ReadoutState) -> float:
    cov = _cov(z)
    return float(np.linalg.norm(cov - state.source_cov, ord="fro"))


def _prototype_distance(state: ReadoutState, z: np.ndarray, pred_idx: np.ndarray) -> float:
    distances = []
    for class_idx, proto in enumerate(state.source_prototypes):
        keep = pred_idx == class_idx
        if np.any(keep):
            distances.append(float(np.linalg.norm(z[keep].mean(axis=0) - proto)))
    return float(np.mean(distances)) if distances else float("nan")


def _base_probs(state: ReadoutState, z: np.ndarray) -> np.ndarray:
    return _softmax(_logits(state, z))


def _replay_baseline(
    baseline: str,
    state: ReadoutState,
    z_source: np.ndarray,
    z_target: np.ndarray,
    base_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    z_target = np.asarray(z_target, dtype=np.float64)
    details: dict[str, Any] = {
        "baseline": baseline,
        "new_method_introduced": False,
        "target_labels_used": False,
    }
    if baseline == "ERM_NO_ADAPT":
        z_after = z_target
        proba = base_proba
        details["operation"] = "identity"
    elif baseline == "TTA_CONTROL_REPLAY":
        entropy_ratio = float(_entropy(base_proba).mean() / max(np.log(len(state.class_labels)), 1e-8))
        mean_conf = float(base_proba.max(axis=1).mean())
        log_t = float(np.clip(0.25 * (mean_conf - 0.70) - 0.10 * (entropy_ratio - 0.50), -0.25, 0.25))
        proba = _softmax(_logits(state, z_target) / max(np.exp(log_t), 1e-8))
        z_after = z_target
        details.update({"operation": "temperature_replay", "log_t": log_t})
    elif baseline == "SPDIM":
        shift = state.source_mean - z_target.mean(axis=0)
        z_after = z_target + shift[None, :]
        proba = _base_probs(state, z_after)
        details.update({"operation": "feature_recentering", "recenter_norm": float(np.linalg.norm(shift))})
    elif baseline == "MATCHED_CORAL":
        mu_t = z_target.mean(axis=0)
        source_sqrt = _matrix_power_psd(state.source_cov, 0.5)
        target_inv_sqrt = _matrix_power_psd(_cov(z_target), -0.5)
        z_after = (z_target - mu_t[None, :]) @ target_inv_sqrt @ source_sqrt + state.source_mean[None, :]
        proba = _base_probs(state, z_after)
        details.update(
            {
                "operation": "matched_coral",
                "recenter_norm": float(np.linalg.norm(z_after.mean(axis=0) - z_target.mean(axis=0))),
            }
        )
    elif baseline == "T3A":
        pred_idx = base_proba.argmax(axis=1)
        conf = base_proba.max(axis=1)
        templates = []
        selected_counts = []
        for class_idx, source_proto in enumerate(state.source_prototypes):
            keep = pred_idx == class_idx
            if np.any(keep):
                class_conf = conf[keep]
                threshold = float(np.quantile(class_conf, 0.70))
                selected = z_target[keep][class_conf >= threshold]
                if len(selected) == 0:
                    template = source_proto
                else:
                    template = 0.5 * source_proto + 0.5 * selected.mean(axis=0)
                selected_counts.append(int(len(selected)))
            else:
                template = source_proto
                selected_counts.append(0)
            templates.append(template)
        template_arr = np.vstack(templates)
        # Distance-to-template logits are a replay of the existing T3A family,
        # not a trainable adapter or new objective.
        dist = ((z_target[:, None, :] - template_arr[None, :, :]) ** 2 / state.pooled_var[None, None, :]).sum(axis=2)
        proba = _softmax(-0.5 * dist + np.log(np.maximum(state.source_prior, 1e-8))[None, :])
        z_after = z_target
        details.update({"operation": "classifier_template_adjustment", "selected_counts": selected_counts})
    else:
        raise ValueError(f"baseline is not in frozen universe: {baseline}")
    return z_after, proba, details


def _target_y_for_scenario(y: np.ndarray, scenario: str, seed: int, key: str) -> np.ndarray | None:
    if scenario == "true_target_y_final_only":
        return y
    if scenario == "target_y_removed":
        return None
    if scenario == "target_y_permuted":
        rng = np.random.default_rng(int(stable_hash({"seed": seed, "key": key})[:8], 16))
        return rng.permutation(y)
    raise ValueError(f"unknown target-label scenario: {scenario}")


def _final_metrics(y_true: np.ndarray | None, proba: np.ndarray, labels: tuple[int, ...]) -> dict[str, Any] | None:
    if y_true is None:
        return None
    pred_idx = proba.argmax(axis=1)
    pred_labels = np.asarray([labels[idx] for idx in pred_idx], dtype=np.int64)
    return {
        "target_bacc": _balanced_accuracy(y_true, pred_labels, labels),
        "target_macro_f1": _macro_f1(y_true, pred_labels, labels),
        "target_nll": _nll(y_true, proba, labels),
        "target_ece": _ece(y_true, pred_labels, proba),
        "n_eval": int(len(y_true)),
    }


def _artifact_key(record: dict[str, Any]) -> str:
    return f"{record.get('dataset')}|{record.get('backbone')}|seed{record.get('seed')}|fold{record.get('fold_id')}"


def _run_artifact_scenario(
    *,
    record: dict[str, Any],
    bundle: FrozenFeatureBundle,
    scenario: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source = _rows(bundle, "source_train")
    target = _rows(bundle, "target_audit")
    state = _fit_readout(source["z"], source["y"])
    base_proba = _base_probs(state, target["z"])
    base_pred_idx = base_proba.argmax(axis=1)
    base_pred_marginal = base_proba.mean(axis=0)
    y_final = _target_y_for_scenario(target["y"], scenario, seed, _artifact_key(record))
    scenario_records = []
    metric_rows = []
    audit_rows = []
    for baseline in ALLOWED_BASELINES:
        z_after, proba_after, details = _replay_baseline(baseline, state, source["z"], target["z"], base_proba)
        pred_idx = proba_after.argmax(axis=1)
        pred_labels = np.asarray([state.class_labels[idx] for idx in pred_idx], dtype=np.int64)
        final = _final_metrics(y_final, proba_after, state.class_labels)

        entropy_before = float(_entropy(base_proba).mean())
        entropy_after = float(_entropy(proba_after).mean())
        mean_conf_before = float(base_proba.max(axis=1).mean())
        mean_conf_after = float(proba_after.max(axis=1).mean())
        margin_before = float(_margin(base_proba).mean())
        margin_after = float(_margin(proba_after).mean())
        pred_marginal_after = proba_after.mean(axis=0)
        label_counts = np.bincount(pred_idx, minlength=len(state.class_labels)).astype(np.int64)
        dominance = float(label_counts.max() / max(1, label_counts.sum()))
        coral_before = _coral_distance(target["z"], state)
        coral_after = _coral_distance(z_after, state)
        covariance_shift = float(np.linalg.norm(_cov(z_after) - _cov(target["z"]), ord="fro"))
        recenter_norm = float(np.linalg.norm(z_after.mean(axis=0) - target["z"].mean(axis=0)))
        prototype_dist = _prototype_distance(state, z_after, pred_idx)
        premetric = {
            "artifact_key": _artifact_key(record),
            "baseline": baseline,
            "predictions_hash": _array_hash(pred_idx),
            "probabilities_hash": _array_hash(proba_after),
            "adapted_feature_hash": _array_hash(z_after),
            "details": details,
            "entropy_after": entropy_after,
            "mean_confidence_after": mean_conf_after,
            "predicted_marginal_after": pred_marginal_after.astype(float).tolist(),
            "coral_distance_after": coral_after,
            "source_replay_axis": "NOT_AVAILABLE_IN_THIS_REPLAY",
            "BN_axis": "NOT_TESTED_IN_FROZEN_FEATURE_REPLAY",
        }
        premetric_hash = stable_hash(premetric)
        final_hash = stable_hash(final) if final is not None else None
        scenario_records.append(
            {
                "artifact_key": _artifact_key(record),
                "baseline": baseline,
                "premetric_output_hash": premetric_hash,
                "predictions_hash": premetric["predictions_hash"],
                "probabilities_hash": premetric["probabilities_hash"],
                "adapted_feature_hash": premetric["adapted_feature_hash"],
                "final_metrics_hash": final_hash,
            }
        )
        if scenario == "true_target_y_final_only":
            if final is None:
                raise ValueError("true_target_y_final_only scenario missing final metrics")
            common = {
                "dataset": record.get("dataset"),
                "backbone": record.get("backbone"),
                "seed": record.get("seed"),
                "fold_id": record.get("fold_id"),
                "baseline": baseline,
                "artifact_path": record.get("path"),
                "source_replay_axis": "NOT_AVAILABLE_IN_THIS_REPLAY",
                "BN_axis": "NOT_TESTED_IN_FROZEN_FEATURE_REPLAY",
            }
            metric_rows.append(
                {
                    **common,
                    **final,
                    "prediction_change_rate_vs_erm": float((pred_idx != base_pred_idx).mean()),
                    "mean_confidence": mean_conf_after,
                    "bacc_nll_divergence_flag": False,
                }
            )
            audit_rows.append(
                {
                    **common,
                    "entropy_before": entropy_before,
                    "entropy_after": entropy_after,
                    "delta_entropy": entropy_after - entropy_before,
                    "mean_max_probability_before": mean_conf_before,
                    "mean_max_probability_after": mean_conf_after,
                    "margin_before": margin_before,
                    "margin_after": margin_after,
                    "source_label_prior": json.dumps(state.source_prior.astype(float).tolist(), sort_keys=True),
                    "target_predicted_marginal_before": json.dumps(base_pred_marginal.astype(float).tolist(), sort_keys=True),
                    "target_predicted_marginal_after": json.dumps(pred_marginal_after.astype(float).tolist(), sort_keys=True),
                    "KL_mean_p_target_source_prior": _kl(pred_marginal_after, state.source_prior),
                    "single_class_dominance": dominance,
                    "class_balance_collapse_guard": bool(dominance < 0.98),
                    "feature_mean_shift_norm": float(np.linalg.norm(target["z"].mean(axis=0) - state.source_mean)),
                    "target_recenter_norm": recenter_norm,
                    "CORAL_distance_before": coral_before,
                    "CORAL_distance_after": coral_after,
                    "covariance_shift": covariance_shift,
                    "class_prototype_distance_if_available": prototype_dist,
                    "SPDIM_recentering_magnitude": recenter_norm if baseline == "SPDIM" else 0.0,
                    "pre_metric_output_hash": premetric_hash,
                    "details_hash": stable_hash(details),
                }
            )
    if scenario == "true_target_y_final_only" and metric_rows:
        erm = next(row for row in metric_rows if row["baseline"] == "ERM_NO_ADAPT")
        for row in metric_rows:
            row["bacc_nll_divergence_flag"] = bool(
                float(row["target_bacc"]) > float(erm["target_bacc"])
                and float(row["target_nll"]) > float(erm["target_nll"])
            )
    scenario_payload = {
        "scenario": scenario,
        "artifact_key": _artifact_key(record),
        "baseline_order": list(ALLOWED_BASELINES),
        "records": scenario_records,
    }
    return scenario_payload, metric_rows, audit_rows


def _validate_artifacts(handoff_manifest: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    handoff = validate_handoff_manifest(handoff_manifest)
    records = []
    for rec in handoff.get("per_artifact_hashes", []):
        observed = sha256_file(rec["path"])
        records.append(
            {
                "path": rec["path"],
                "dataset": rec.get("dataset"),
                "backbone": rec.get("backbone"),
                "fold_id": rec.get("fold_id"),
                "expected_sha256": rec.get("file_sha256"),
                "observed_sha256": observed,
                "status": "PASS" if observed == rec.get("file_sha256") else "FAIL",
            }
        )
    validation = {
        "handoff_manifest": str(handoff_manifest),
        "handoff_hash": sha256_file(handoff_manifest),
        "expected_artifacts": 18,
        "loaded_artifacts": len(records),
        "per_artifact_hash_check": "PASS" if len(records) == 18 and all(r["status"] == "PASS" for r in records) else "FAIL",
        "records": records,
    }
    if validation["per_artifact_hash_check"] != "PASS":
        raise ValueError("artifact hash validation failed")
    return handoff, validation


def _scenario_signature(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_key": rec["artifact_key"],
            "baseline": rec["baseline"],
            "premetric_output_hash": rec["premetric_output_hash"],
            "predictions_hash": rec["predictions_hash"],
            "probabilities_hash": rec["probabilities_hash"],
            "adapted_feature_hash": rec["adapted_feature_hash"],
        }
        for rec in scenario["records"]
    ]


def _validate_target_noninterference(scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks = []
    warnings: list[str] = []
    required = ("true_target_y_final_only", "target_y_removed", "target_y_permuted")
    missing = [name for name in required if name not in scenarios]
    if missing:
        raise AssertionError(f"missing target-label scenario: {missing}")
    ref = _scenario_signature(scenarios["true_target_y_final_only"])
    for name in ("target_y_removed", "target_y_permuted"):
        if _scenario_signature(scenarios[name]) != ref:
            raise AssertionError(f"target labels changed premetric replay outputs for {name}")
    checks.extend(
        [
            "premetric_outputs_identical",
            "baseline_outputs_identical_before_final_metric",
            "audit_premetric_artifacts_identical",
        ]
    )
    if any(rec.get("final_metrics_hash") is not None for rec in scenarios["target_y_removed"]["records"]):
        raise AssertionError("removed-target-y scenario emitted final metrics")
    checks.append("removed_target_y_has_no_final_metrics")
    final_metric_diffs = sum(
        left["final_metrics_hash"] != right["final_metrics_hash"]
        for left, right in zip(
            scenarios["true_target_y_final_only"]["records"],
            scenarios["target_y_permuted"]["records"],
        )
    )
    checks.append("permuted_target_y_differs_only_after_final_metric")
    return {
        "passed": True,
        "checks": checks,
        "warnings": warnings,
        "final_metric_hash_differences_vs_permuted_y": int(final_metric_diffs),
    }


def _validate_replay_determinism(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    checks = []
    warnings: list[str] = []
    if _scenario_signature(left) != _scenario_signature(right):
        raise AssertionError("replay determinism signature mismatch")
    checks.append("premetric_replay_hashes_identical")
    return {"passed": True, "checks": checks, "warnings": warnings}


def _aggregate(per_fold: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_lookup = {
        (row["backbone"], row["baseline"], row["fold_id"]): row
        for row in audit_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_fold:
        grouped.setdefault((str(row["backbone"]), str(row["baseline"])), []).append(row)
    aggregates = []
    lookup = {}
    for (backbone, baseline), rows in sorted(grouped.items()):
        bacc = np.asarray([float(r["target_bacc"]) for r in rows])
        nll = np.asarray([float(r["target_nll"]) for r in rows])
        ece = np.asarray([float(r["target_ece"]) for r in rows])
        macro_f1 = np.asarray([float(r["target_macro_f1"]) for r in rows])
        pred_change = np.asarray([float(r["prediction_change_rate_vs_erm"]) for r in rows])
        audit_subset = [audit_lookup[(r["backbone"], r["baseline"], r["fold_id"])] for r in rows]
        rec = {
            "backbone": backbone,
            "baseline": baseline,
            "n_folds": len(rows),
            "mean_target_bacc": float(bacc.mean()),
            "worst_fold_bacc": float(bacc.min()),
            "mean_target_macro_f1": float(macro_f1.mean()),
            "mean_target_nll": float(nll.mean()),
            "mean_target_ece": float(ece.mean()),
            "mean_prediction_change_rate_vs_erm": float(pred_change.mean()),
            "mean_delta_entropy": float(np.mean([float(r["delta_entropy"]) for r in audit_subset])),
            "mean_balance_KL": float(np.mean([float(r["KL_mean_p_target_source_prior"]) for r in audit_subset])),
            "mean_CORAL_delta": float(
                np.mean([float(r["CORAL_distance_after"]) - float(r["CORAL_distance_before"]) for r in audit_subset])
            ),
            "mean_recenter_norm": float(np.mean([float(r["target_recenter_norm"]) for r in audit_subset])),
            "source_replay_axis": "NOT_AVAILABLE_IN_THIS_REPLAY",
            "BN_axis": "NOT_TESTED_IN_FROZEN_FEATURE_REPLAY",
        }
        aggregates.append(rec)
        lookup[(backbone, baseline)] = rec
    for rec in aggregates:
        erm = lookup[(rec["backbone"], "ERM_NO_ADAPT")]
        rec["delta_bacc_vs_erm"] = float(rec["mean_target_bacc"] - erm["mean_target_bacc"])
        rec["delta_nll_vs_erm"] = float(rec["mean_target_nll"] - erm["mean_target_nll"])
        rec["delta_ece_vs_erm"] = float(rec["mean_target_ece"] - erm["mean_target_ece"])
        rec["bacc_NLL_divergence_flag"] = bool(rec["delta_bacc_vs_erm"] > 0.0 and rec["delta_nll_vs_erm"] > 0.0)
    return aggregates


def _dominant_axis(rec: dict[str, Any]) -> str:
    baseline = str(rec["baseline"])
    if baseline == "ERM_NO_ADAPT":
        return "reference_no_adaptation"
    if baseline == "TTA_CONTROL_REPLAY":
        if rec["delta_nll_vs_erm"] < -0.01 and abs(rec["delta_bacc_vs_erm"]) < 0.005:
            return "calibration_temperature_effect"
        if rec["mean_delta_entropy"] < -0.01:
            return "entropy_confidence"
        return "neutral_entropy_replay"
    if baseline in {"MATCHED_CORAL", "SPDIM"}:
        if rec["mean_CORAL_delta"] < -0.01:
            if rec["delta_bacc_vs_erm"] > 0.0:
                return "geometry_alignment_with_accuracy_gain"
            return "geometry_alignment_without_accuracy_gain"
        return "geometry_no_clear_alignment"
    if baseline == "T3A":
        if rec["mean_prediction_change_rate_vs_erm"] < 0.01:
            return "classifier_template_neutral"
        if rec["delta_bacc_vs_erm"] > 0.0:
            return "classifier_template_adjustment_with_accuracy_gain"
        return "classifier_template_adjustment_without_accuracy_gain"
    return "unclassified"


def _mechanism_summary(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for rec in aggregates:
        rows.append({**rec, "dominant_observable_axis": _dominant_axis(rec)})
    outcome = "MECHANISM_INFORMATIVE_PASS" if all(r["dominant_observable_axis"] != "unclassified" for r in rows) else "MECHANISM_INCONCLUSIVE_PASS"
    return {
        "outcome": outcome,
        "selection_for_deployment": False,
        "new_method_claim": False,
        "p1_p2_training_claim": False,
        "rows": rows,
        "source_replay_axis": "NOT_AVAILABLE_IN_THIS_REPLAY",
        "BN_axis": "NOT_TESTED_IN_FROZEN_FEATURE_REPLAY",
    }


def build_replay_payload(
    *,
    handoff_manifest: str | Path = DEFAULT_CEDAR01F_HANDOFF,
    seed: int = 0,
) -> dict[str, Any]:
    registry = registry_payload()
    registry_result = validate_baseline_universe(registry).to_dict()
    if registry["baseline_registry_hash"] != EXPECTED_BASELINE_REGISTRY_HASH:
        raise ValueError("baseline registry hash changed from TTA_MECH_00A")
    inventory = build_artifact_inventory(handoff_manifest)
    if inventory["artifact_inventory_hash"] != EXPECTED_ARTIFACT_INVENTORY_HASH:
        raise ValueError("artifact inventory hash changed from TTA_MECH_00A")
    handoff, artifact_validation = _validate_artifacts(handoff_manifest)
    no_new_method = validate_no_new_method(
        {
            "project": "TTA-MECH-EEG",
            "phase": "TTA_MECH_01",
            "new_method_claim": False,
            "active_baselines": list(ALLOWED_BASELINES),
            "audit_axes": [axis["name"] for axis in audit_axis_schema_payload()["axes"]],
        }
    ).to_dict()

    scenario_names = ("true_target_y_final_only", "target_y_removed", "target_y_permuted")
    scenarios = {
        name: {"scenario": name, "records": []}
        for name in scenario_names
    }
    repeat_true = {"scenario": "true_target_y_final_only", "records": []}
    per_fold_metrics: list[dict[str, Any]] = []
    audit_axes_table: list[dict[str, Any]] = []
    records = sorted(
        handoff.get("per_artifact_hashes", []),
        key=lambda r: (str(r.get("backbone")), int(r.get("fold_id")), str(r.get("path"))),
    )
    for record in records:
        bundle = load_frozen_feature_npz(record["path"])
        for scenario in scenario_names:
            payload, metrics, audit_rows = _run_artifact_scenario(
                record=record,
                bundle=bundle,
                scenario=scenario,
                seed=seed,
            )
            scenarios[scenario]["records"].extend(payload["records"])
            if scenario == "true_target_y_final_only":
                per_fold_metrics.extend(metrics)
                audit_axes_table.extend(audit_rows)
        repeat_payload, _, _ = _run_artifact_scenario(
            record=record,
            bundle=bundle,
            scenario="true_target_y_final_only",
            seed=seed,
        )
        repeat_true["records"].extend(repeat_payload["records"])

    target_noninterference = _validate_target_noninterference(scenarios)
    replay_determinism = _validate_replay_determinism(scenarios["true_target_y_final_only"], repeat_true)
    aggregate = _aggregate(per_fold_metrics, audit_axes_table)
    mechanism_summary = _mechanism_summary(aggregate)
    red_team = {
        "target_label_noninterference": target_noninterference,
        "baseline_universe_hash": {
            "passed": True,
            "checks": ["baseline_registry_hash_matches_tta_mech00a"],
            "warnings": [],
            "baseline_registry_hash": registry["baseline_registry_hash"],
        },
        "artifact_inventory_hash": {
            "passed": True,
            "checks": ["artifact_inventory_hash_matches_tta_mech00a"],
            "warnings": [],
            "artifact_inventory_hash": inventory["artifact_inventory_hash"],
        },
        "artifact_hash_validation": {
            "passed": artifact_validation["per_artifact_hash_check"] == "PASS",
            "checks": ["handoff_manifest_validated", "all_18_artifact_hashes_pass"],
            "warnings": [],
        },
        "replay_determinism": replay_determinism,
        "no_new_method_guard": no_new_method,
        "baseline_universe_freeze": registry_result,
    }
    red_team_failures = [name for name, item in red_team.items() if not item.get("passed")]
    status = "FAIL" if red_team_failures else mechanism_summary["outcome"]
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_01_existing_baseline_real_replay_mechanism_audit",
        "status": status,
        "preflight_baseline_commit": "3c4d0fe",
        "real_eeg_replay_run": True,
        "target_metrics_final_only": True,
        "baseline_selected_for_deployment": False,
        "new_method_introduced": False,
        "p1_p2_training": False,
        "cmi_pruning_mask_surgery_safety_router_active": False,
        "baseline_registry_hash": registry["baseline_registry_hash"],
        "artifact_inventory_hash": inventory["artifact_inventory_hash"],
        "artifact_validation": artifact_validation,
        "red_team": red_team,
        "per_fold_metrics": per_fold_metrics,
        "audit_axes_table": audit_axes_table,
        "aggregate_baseline_table": aggregate,
        "mechanism_summary": mechanism_summary,
        "target_label_scenarios": scenarios,
    }
    payload["tta_mech01_payload_hash"] = stable_hash(payload)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(payload: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_registry_hash.txt").write_text(payload["baseline_registry_hash"] + "\n")
    (out / "artifact_inventory_hash.txt").write_text(payload["artifact_inventory_hash"] + "\n")
    json_outputs = {
        "run_manifest.json": {
            "project": payload["project"],
            "phase": payload["phase"],
            "status": payload["status"],
            "preflight_baseline_commit": payload["preflight_baseline_commit"],
            "real_eeg_replay_run": payload["real_eeg_replay_run"],
            "target_metrics_final_only": payload["target_metrics_final_only"],
            "baseline_selected_for_deployment": payload["baseline_selected_for_deployment"],
            "new_method_introduced": payload["new_method_introduced"],
            "p1_p2_training": payload["p1_p2_training"],
            "cmi_pruning_mask_surgery_safety_router_active": payload["cmi_pruning_mask_surgery_safety_router_active"],
            "baseline_registry_hash": payload["baseline_registry_hash"],
            "artifact_inventory_hash": payload["artifact_inventory_hash"],
            "tta_mech01_payload_hash": payload["tta_mech01_payload_hash"],
        },
        "mechanism_summary.json": payload["mechanism_summary"],
        "target_label_quarantine.json": payload["red_team"]["target_label_noninterference"],
        "replay_determinism.json": payload["red_team"]["replay_determinism"],
        "red_team.json": payload["red_team"],
        "no_new_method_guard.json": payload["red_team"]["no_new_method_guard"],
        "artifact_validation.json": payload["artifact_validation"],
        "tta_mech01_summary.json": payload,
    }
    for name, data in json_outputs.items():
        with (out / name).open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    _write_csv(out / "per_fold_metrics.csv", payload["per_fold_metrics"])
    _write_csv(out / "audit_axes_table.csv", payload["audit_axes_table"])
    _write_csv(out / "aggregate_baseline_table.csv", payload["aggregate_baseline_table"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff-manifest", default=str(DEFAULT_CEDAR01F_HANDOFF))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/tta_mech/tta_mech01_bnci2014_001_seed0")
    args = ap.parse_args()
    payload = build_replay_payload(handoff_manifest=args.handoff_manifest, seed=args.seed)
    write_outputs(payload, args.out_dir)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "phase": payload["phase"],
                "out_dir": args.out_dir,
                "red_team_failures": [name for name, item in payload["red_team"].items() if not item.get("passed")],
                "baseline_registry_hash": payload["baseline_registry_hash"],
                "artifact_inventory_hash": payload["artifact_inventory_hash"],
                "tta_mech01_payload_hash": payload["tta_mech01_payload_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
