"""No-forward C74 T2 instrumentation smoke analyses and artifact assembly."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy import stats

from . import c74_cache as cache
from . import c74_t2_source_wz_instrumentation as runner


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c74_tables"
ANALYSIS_STATE = REPORT_DIR / "C74_T2_SOURCE_WZ_ANALYSIS_STATE.json"
RNG_SEED = 740013
RIDGE_ALPHA = 1.0
NULL_REPLICATES = 200
CONSTRUCT_FEASIBILITY_MIN_MEDIAN_SPEARMAN = 0.20
CONSTRUCT_FEASIBILITY_MIN_POSITIVE_FRACTION = 0.75
PRIMARY_SMOKE_POINTER = "PRIMARY_SMOKE_INPUT_POINTER.json"


def _write_csv(name: str, rows: list[dict], columns: list[str] | None = None) -> None:
    path = TABLE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _descriptor(manifest: dict, kind: str) -> dict:
    matches = [item for item in manifest["shards"] if item["kind"] == kind]
    if len(matches) != 1:
        raise RuntimeError(f"C74 unit {manifest['unit_id']} has {len(matches)} {kind} shards")
    return matches[0]


def _load(descriptor: dict, fields: set[str]) -> dict[str, np.ndarray]:
    if set(descriptor["fields"]) != fields:
        raise RuntimeError(f"C74 analysis schema mismatch for {descriptor['path']}")
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        return {name: shard[name] for name in shard.files}


def _all_manifests() -> list[dict]:
    protocol = cache.load_locked_protocol()
    for stage, expected_gate in (
        ("P0_pilot", "P0_PILOT_ALL_GATES_PASSED"),
        ("P1_expansion", "P1_EXPANSION_ALL_GATES_PASSED"),
    ):
        gate_path = cache.stage_gate_path(protocol, stage)
        if not gate_path.is_file() or json.loads(gate_path.read_text()).get("final_gate") != expected_gate:
            raise RuntimeError(f"C74 analysis blocked by missing {stage} aggregate gate")
    manifests = []
    for stage in ("P0_pilot", "P1_expansion"):
        for target_id in range(1, 10):
            for row in cache.stage_rows(stage, target_id):
                path = cache.unit_directory(protocol, stage, target_id, row["c74_unit_id"]) / "unit_manifest.json"
                manifest = cache.verify_unit_manifest(path, rehash_payloads=False)
                if manifest["checkpoint_id"] != row["checkpoint_id"]:
                    raise RuntimeError(f"C74 analysis unit mismatch: {path}")
                manifests.append(manifest)
    if len(manifests) != 216 or len({item["unit_id"] for item in manifests}) != 216:
        raise RuntimeError("C74 analysis requires exactly 216 unique T2 units")
    return sorted(manifests, key=lambda item: (item["target_id"], item["seed"], item["level"], item["candidate_order"]))


def _write_primary_smoke_input(manifests: list[dict], protocol: dict) -> dict:
    """Create a restricted consumer manifest with no oracle descriptor/path."""
    allowed_kinds = {
        "checkpoint_Wb", "strict_source_trial", "target_unlabeled_representation",
        "target_construction_labels", "target_evaluation_labels",
    }
    restricted = []
    for manifest in manifests:
        entry = {key: value for key, value in manifest.items() if key != "shards"}
        entry["shards"] = [
            descriptor for descriptor in manifest["shards"] if descriptor["kind"] in allowed_kinds
        ]
        if {descriptor["kind"] for descriptor in entry["shards"]} != allowed_kinds:
            raise RuntimeError(f"C74 restricted-view assembly failed for {manifest['unit_id']}")
        restricted.append(entry)
    payload = {
        "schema_version": "c74_primary_smoke_input_v1",
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "forbidden_view": "same" + "_label_oracle",
        "unit_count": len(restricted),
        "units": restricted,
    }
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if "same_label_oracle" in body:
        # The split literal above keeps the policy label out of descriptors;
        # the policy field itself is removed before the hard payload check.
        payload.pop("forbidden_view")
        body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if "same_label_oracle" in body:
        raise RuntimeError("C74 oracle path leaked into primary smoke input")
    views_dir = cache.run_root(protocol) / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    provisional = views_dir / ".primary_smoke_input.json"
    digest = cache.atomic_json(provisional, payload)
    final = views_dir / f"primary_smoke_input_sha256_{digest[:16]}.json"
    if final.exists() and cache.sha256_file(final) != digest:
        raise RuntimeError(f"C74 primary-input immutable collision: {final}")
    if final.exists():
        provisional.unlink(missing_ok=True)
    else:
        os.replace(provisional, final)
    pointer = {
        "schema_version": "c74_primary_smoke_pointer_v1",
        "path": str(final), "sha256": digest, "unit_count": len(restricted),
        "oracle_descriptor_present": False,
    }
    pointer_path = views_dir / PRIMARY_SMOKE_POINTER
    cache.atomic_json(pointer_path, pointer)
    return {"pointer_path": str(pointer_path), **pointer}


def _primary_smoke_manifests(protocol: dict) -> list[dict]:
    pointer_path = cache.run_root(protocol) / "views" / PRIMARY_SMOKE_POINTER
    if not pointer_path.is_file():
        raise RuntimeError("C74 primary smoke input is not prepared")
    pointer = json.loads(pointer_path.read_text())
    path = Path(pointer["path"])
    if cache.sha256_file(path) != pointer["sha256"]:
        raise RuntimeError("C74 primary smoke input hash mismatch")
    body = path.read_text()
    if "same_label_oracle" in body:
        raise RuntimeError("C74 primary smoke input contains oracle descriptor/path")
    payload = json.loads(body)
    manifests = payload["units"]
    allowed = {
        "checkpoint_Wb", "strict_source_trial", "target_unlabeled_representation",
        "target_construction_labels", "target_evaluation_labels",
    }
    if len(manifests) != 216 or any({item["kind"] for item in manifest["shards"]} != allowed for manifest in manifests):
        raise RuntimeError("C74 primary smoke restricted-view contract failed")
    return manifests


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in sorted(set(int(value) for value in y_true)):
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _nll(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    values = probabilities[np.arange(len(y_true)), y_true.astype(int)]
    return float(-np.mean(np.log(np.clip(values, 1e-12, 1.0))))


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)


def _label_indices(target_trial_ids: np.ndarray, labels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    index = {str(trial_id): i for i, trial_id in enumerate(target_trial_ids)}
    requested = [str(trial_id) for trial_id in labels["target_trial_id"]]
    if len(requested) != len(set(requested)) or any(trial_id not in index for trial_id in requested):
        raise RuntimeError("C74 target-label view trial-ID integrity failed")
    return np.asarray([index[trial_id] for trial_id in requested], dtype=int), labels["target_class_label"].astype(int)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def _extract_unit_features(manifest: dict) -> dict:
    source = _load(_descriptor(manifest, "strict_source_trial"), runner.SOURCE_FIELDS)
    target = _load(
        _descriptor(manifest, "target_unlabeled_representation"),
        runner.TARGET_UNLABELED_FIELDS,
    )
    construction = _load(
        _descriptor(manifest, "target_construction_labels"),
        runner.CONSTRUCTION_FIELDS,
    )
    evaluation = _load(
        _descriptor(manifest, "target_evaluation_labels"),
        runner.EVALUATION_FIELDS,
    )
    wb = _load(_descriptor(manifest, "checkpoint_Wb"), runner.CHECKPOINT_FIELDS)

    construct_idx, construct_y = _label_indices(target["target_trial_id"], construction)
    evaluation_idx, evaluation_y = _label_indices(target["target_trial_id"], evaluation)
    if set(construct_idx.tolist()) & set(evaluation_idx.tolist()):
        raise RuntimeError(f"C74 construction/evaluation overlap in {manifest['unit_id']}")
    if len(construct_idx) + len(evaluation_idx) != len(target["target_trial_id"]):
        raise RuntimeError(f"C74 construction/evaluation partition incomplete in {manifest['unit_id']}")

    source_y = source["source_class_label"].astype(int)
    source_pred = source["predicted_class"].astype(int)
    target_probabilities = target["probabilities"]
    source_z_norm = np.linalg.norm(source["z"], axis=1)
    target_z_norm = np.linalg.norm(target["z"], axis=1)
    source_features = np.asarray([
        _balanced_accuracy(source_y, source_pred),
        _nll(source_y, source["probabilities"]),
        float(np.mean(np.max(source["probabilities"], axis=1))),
        float(np.mean(_entropy(source["probabilities"]))),
    ])
    construction_features = np.asarray([
        _balanced_accuracy(construct_y, target["predicted_class"][construct_idx]),
        _nll(construct_y, target_probabilities[construct_idx]),
    ])
    shared_features = np.asarray([
        float(np.mean(np.max(target_probabilities, axis=1))),
        float(np.mean(_entropy(target_probabilities))),
        float(np.mean(np.linalg.norm(target["logits"], axis=1))),
    ])
    source_representation_features = np.concatenate((
        [float(np.mean(source_z_norm)), float(np.std(source_z_norm))],
        np.mean(source["Wz"], axis=0), np.std(source["Wz"], axis=0),
    ))
    target_representation_features = np.concatenate((
        [float(np.mean(target_z_norm)), float(np.std(target_z_norm))],
        np.mean(target["Wz"], axis=0), np.std(target["Wz"], axis=0),
    ))
    construction_wz_class_mean = np.asarray([
        float(np.mean(target["Wz"][construct_idx[construct_y == label], label]))
        for label in range(4)
    ])
    evaluation_wz_class_mean = np.asarray([
        float(np.mean(target["Wz"][evaluation_idx[evaluation_y == label], label]))
        for label in range(4)
    ])
    return {
        "manifest": manifest,
        "source_features": source_features,
        "construction_features": construction_features,
        "shared_features": shared_features,
        "source_representation_features": source_representation_features,
        "target_representation_features": target_representation_features,
        "evaluation_bAcc": _balanced_accuracy(
            evaluation_y, target["predicted_class"][evaluation_idx].astype(int)
        ),
        "construction_wz_class_mean": construction_wz_class_mean,
        "evaluation_wz_class_mean": evaluation_wz_class_mean,
        "target_trial_ids": target["target_trial_id"],
        "target_Wz": target["Wz"],
        "target_logits": target["logits"],
        "bias": wb["b"],
        "evaluation_indices": evaluation_idx,
        "evaluation_labels": evaluation_y,
    }


def _ridge_loto_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    predictions = np.empty_like(y, dtype=float)
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        mean = np.mean(X[train], axis=0)
        scale = np.std(X[train], axis=0)
        scale[scale < 1e-10] = 1.0
        train_x = (X[train] - mean) / scale
        test_x = (X[test] - mean) / scale
        train_y = y[train]
        centered_y = train_y - np.mean(train_y)
        beta = np.linalg.solve(
            train_x.T @ train_x + RIDGE_ALPHA * np.eye(train_x.shape[1]),
            train_x.T @ centered_y,
        )
        predictions[test] = np.mean(train_y) + test_x @ beta
    return predictions


def _prediction_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / denominator if denominator else float("nan")
    return {
        "loto_R2": r2,
        "loto_spearman": _safe_spearman(y, prediction),
        "loto_MAE": float(np.mean(np.abs(y - prediction))),
    }


def _incremental_prediction(records: list[dict]) -> list[dict]:
    groups = np.asarray([int(record["manifest"]["target_id"]) for record in records])
    y = np.asarray([record["evaluation_bAcc"] for record in records])
    feature_blocks = [
        ("source_summaries", "strict_source", "source_features"),
        ("plus_construction_summaries", "target_split_label_diagnostic", "construction_features"),
        ("plus_shared_target_calibration", "target_unlabeled", "shared_features"),
        ("plus_source_trial_z_Wz", "strict_source", "source_representation_features"),
        ("plus_target_unlabeled_z_Wz", "target_unlabeled", "target_representation_features"),
    ]
    cumulative = []
    rows = []
    previous_r2 = float("nan")
    rng = np.random.default_rng(RNG_SEED)
    permutation_indices = []
    for _ in range(NULL_REPLICATES):
        perm = np.arange(len(y))
        for target_id in sorted(set(groups.tolist())):
            indices = np.where(groups == target_id)[0]
            perm[indices] = rng.permutation(indices)
        permutation_indices.append(perm)
    for order, (name, availability, key) in enumerate(feature_blocks, 1):
        cumulative.append(np.stack([record[key] for record in records]))
        X = np.concatenate(cumulative, axis=1)
        prediction = _ridge_loto_predictions(X, y, groups)
        metrics = _prediction_metrics(y, prediction)
        null_r2 = []
        for perm in permutation_indices:
            null_prediction = _ridge_loto_predictions(X, y[perm], groups)
            null_r2.append(_prediction_metrics(y[perm], null_prediction)["loto_R2"])
        rows.append({
            "order": order,
            "model": name,
            "new_block_availability": availability,
            "feature_count": X.shape[1],
            **metrics,
            "incremental_R2": metrics["loto_R2"] - previous_r2 if math.isfinite(previous_r2) else float("nan"),
            "target_blocked_null_replicates": NULL_REPLICATES,
            "target_blocked_null_R2_mean": float(np.mean(null_r2)),
            "target_blocked_null_R2_p95": float(np.quantile(null_r2, 0.95)),
            "exceeds_null_p95": int(metrics["loto_R2"] > float(np.quantile(null_r2, 0.95))),
            "ridge_alpha_locked": RIDGE_ALPHA,
            "crossfit": "leave_one_target_out",
            "diagnostic_only": 1,
        })
        previous_r2 = metrics["loto_R2"]
    return rows


def _projection_variance(records: list[dict]) -> list[dict]:
    rows = []
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["manifest"]["target_id"])].append(record)
    for target_id, target_records in sorted(grouped.items()):
        reference_ids = target_records[0]["target_trial_ids"]
        if any(not np.array_equal(reference_ids, record["target_trial_ids"]) for record in target_records):
            raise RuntimeError(f"C74 target trial alignment failed for target {target_id}")
        stack = np.stack([record["target_Wz"] for record in target_records])
        for class_index in range(stack.shape[2]):
            values = stack[:, :, class_index].astype(float)
            grand = float(np.mean(values))
            candidate = np.mean(values, axis=1, keepdims=True) - grand
            trial = np.mean(values, axis=0, keepdims=True) - grand
            residual = values - grand - candidate - trial
            total_ss = float(np.sum((values - grand) ** 2))
            candidate_ss = float(values.shape[1] * np.sum(candidate ** 2))
            trial_ss = float(values.shape[0] * np.sum(trial ** 2))
            residual_ss = float(np.sum(residual ** 2))
            rows.append({
                "target_id": target_id,
                "class_index": class_index,
                "layer": "classifier_projection_Wz",
                "candidate_count": values.shape[0],
                "trial_count": values.shape[1],
                "total_variance": float(np.var(values)),
                "target_common_trial_fraction": trial_ss / total_ss if total_ss else float("nan"),
                "checkpoint_candidate_fraction": candidate_ss / total_ss if total_ss else float("nan"),
                "candidate_x_trial_residual_fraction": residual_ss / total_ss if total_ss else float("nan"),
                "accounting_sum": (candidate_ss + trial_ss + residual_ss) / total_ss if total_ss else float("nan"),
                "target_labels_used": 0,
                "interpretation": "descriptive_projection_variance_not_gauge",
            })
    return rows


def _split_stability(records: list[dict]) -> list[dict]:
    rows = []
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["manifest"]["target_id"])].append(record)
    for target_id, target_records in sorted(grouped.items()):
        for class_index in range(4):
            construction = np.asarray([record["construction_wz_class_mean"][class_index] for record in target_records])
            evaluation = np.asarray([record["evaluation_wz_class_mean"][class_index] for record in target_records])
            rows.append({
                "target_id": target_id,
                "class_index": class_index,
                "candidate_count": len(target_records),
                "pearson": _safe_pearson(construction, evaluation),
                "spearman": _safe_spearman(construction, evaluation),
                "construction_mean": float(np.mean(construction)),
                "evaluation_mean": float(np.mean(evaluation)),
                "mean_absolute_split_difference": float(np.mean(np.abs(construction - evaluation))),
                "construction_labels_used": 1,
                "evaluation_labels_used": 1,
                "same_label_oracle_used": 0,
                "diagnostic_only": 1,
            })
    return rows


def _pair_flip_fraction(reference: np.ndarray, alternative: np.ndarray) -> float:
    flips = 0
    comparable = 0
    for i in range(len(reference)):
        for j in range(i + 1, len(reference)):
            left = np.sign(reference[i] - reference[j])
            right = np.sign(alternative[i] - alternative[j])
            if left and right:
                comparable += 1
                flips += int(left != right)
    return flips / comparable if comparable else float("nan")


def _utility_from_logits(logits: np.ndarray, eval_indices: np.ndarray, eval_y: np.ndarray) -> float:
    return _balanced_accuracy(eval_y, np.argmax(logits[eval_indices], axis=1))


def _counterfactuals(records: list[dict]) -> list[dict]:
    rows = []
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["manifest"]["target_id"])].append(record)
    for target_id, target_records in sorted(grouped.items()):
        target_records = sorted(target_records, key=lambda record: record["manifest"]["unit_id"])
        trial_ids = target_records[0]["target_trial_ids"]
        if any(not np.array_equal(trial_ids, record["target_trial_ids"]) for record in target_records):
            raise RuntimeError(f"C74 counterfactual trial alignment failed for target {target_id}")
        eval_indices = target_records[0]["evaluation_indices"]
        eval_y = target_records[0]["evaluation_labels"]
        if any(
            not np.array_equal(eval_indices, record["evaluation_indices"])
            or not np.array_equal(eval_y, record["evaluation_labels"])
            for record in target_records
        ):
            raise RuntimeError(f"C74 evaluation-label view mismatch for target {target_id}")
        wz = np.stack([record["target_Wz"] for record in target_records]).astype(float)
        bias = np.stack([record["bias"] for record in target_records]).astype(float)
        common = np.mean(wz, axis=0, keepdims=True)
        residual = wz - common
        original_logits = wz + bias[:, None, :]
        original_utility = np.asarray([
            _utility_from_logits(logits, eval_indices, eval_y) for logits in original_logits
        ])
        stored_error = max(
            float(np.max(np.abs(original_logits[index] - target_records[index]["target_logits"])))
            for index in range(len(target_records))
        )
        rng = np.random.default_rng(RNG_SEED + target_id)
        permutation = rng.permutation(len(target_records))
        trajectory_residual = residual.copy()
        by_trajectory = defaultdict(list)
        for index, record in enumerate(target_records):
            by_trajectory[record["manifest"]["trajectory_id"]].append(index)
        for indices in by_trajectory.values():
            shuffled = rng.permutation(indices)
            trajectory_residual[indices] = residual[shuffled]
        random_residual = rng.normal(size=residual.shape)
        random_residual -= np.mean(random_residual, axis=0, keepdims=True)
        scale = np.linalg.norm(residual) / max(np.linalg.norm(random_residual), 1e-12)
        random_residual *= scale
        variants = {
            "I0_original": residual,
            "shrink_candidate_residual_alpha_0.5": 0.5 * residual,
            "replace_with_target_common_alpha_0": np.zeros_like(residual),
            "candidate_permuted_residual": residual[permutation],
            "trajectory_preserving_shuffle": trajectory_residual,
            "magnitude_matched_random_residual": random_residual,
        }
        original_best = float(np.max(original_utility))
        for variant, variant_residual in variants.items():
            logits = common + variant_residual + bias[:, None, :]
            utility = np.asarray([_utility_from_logits(value, eval_indices, eval_y) for value in logits])
            rows.append({
                "target_id": target_id,
                "counterfactual": variant,
                "candidate_count": len(target_records),
                "mean_evaluation_bAcc": float(np.mean(utility)),
                "utility_spearman_vs_original": _safe_spearman(original_utility, utility),
                "pairwise_rank_flip_fraction": _pair_flip_fraction(original_utility, utility),
                "top1_agreement": int(int(np.argmax(original_utility)) == int(np.argmax(utility))),
                "original_best_bAcc": original_best,
                "counterfactual_best_bAcc": float(np.max(utility)),
                "best_utility_delta": float(np.max(utility) - original_best),
                "original_Wz_plus_b_vs_stored_logits_max_abs": stored_error,
                "evaluation_labels_used": 1,
                "same_label_oracle_used": 0,
                "hyperparameter_tuned_on_evaluation": 0,
                "diagnostic_only": 1,
            })
    return rows


def _content_and_abi_tables(manifests: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    abi_rows = []
    identity_rows = []
    wb_rows = []
    content_rows = []
    for manifest in manifests:
        identity = manifest["identity"]
        abi_rows.append({
            "unit_id": manifest["unit_id"], "target_id": manifest["target_id"],
            "model_factory": manifest["model_factory"], "input_shape": str(manifest["input_shape"]),
            "representation_shape": str(manifest["representation_shape"]),
            "W_shape": str(manifest["W_shape"]), "b_shape": str(manifest["b_shape"]),
            "checkpoint_file_sha256": manifest["checkpoint_file_sha256"],
            "state_hash_after": identity["state_hash_after"],
            "state_hash_unchanged": int(identity["state_hash_after"] == manifest["checkpoint_id"]),
            "passed": int(manifest["all_gates_passed"]),
        })
        identity_rows.append({
            "unit_id": manifest["unit_id"], "target_id": manifest["target_id"],
            **identity,
        })
        wb = _descriptor(manifest, "checkpoint_Wb")
        wb_rows.append({
            "unit_id": manifest["unit_id"], "target_id": manifest["target_id"],
            "W_shape": str(manifest["W_shape"]), "b_shape": str(manifest["b_shape"]),
            "external_path": wb["path"], "sha256": wb["sha256"],
            "size_bytes": wb["size_bytes"], "stored_once_per_unit": 1,
        })
        for shard in manifest["shards"]:
            content_rows.append({
                "unit_id": manifest["unit_id"], "target_id": manifest["target_id"],
                "stage": manifest["stage"], "view_kind": shard["kind"],
                "external_path": shard["path"], "sha256": shard["sha256"],
                "size_bytes": shard["size_bytes"], "row_count": shard["row_count"],
                "fields": ";".join(shard["fields"]), "git_tracked": 0,
            })
    return abi_rows, identity_rows, wb_rows, content_rows


def _materialize_view_manifests(manifests: list[dict], protocol: dict) -> list[dict]:
    view_specs = [
        ("strict_source_trial_view", "strict_source_trial", 1, 0, 0, 1, 0, "C75_strict_source_diagnostic"),
        ("target_unlabeled_representation_view", "target_unlabeled_representation", 0, 0, 0, 0, 1, "C75_target_unlabeled_diagnostic"),
        ("target_construction_view", "target_construction_labels", 0, 1, 0, 0, 0, "C75_split_construction_diagnostic"),
        ("target_evaluation_view", "target_evaluation_labels", 0, 1, 1, 0, 0, "C75_heldout_evaluation_diagnostic"),
        ("same_label_oracle_view", "same_label_oracle", 0, 1, 1, 0, 0, "NO_PRIMARY_SMOKE_CONSUMER"),
    ]
    output = []
    views_dir = cache.run_root(protocol) / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    for view_name, kind, source_labels, target_labels, eval_labels, strict_source, target_unlabeled, consumer in view_specs:
        entries = []
        for manifest in manifests:
            shard = _descriptor(manifest, kind)
            entries.append({
                "unit_id": manifest["unit_id"], "target_id": manifest["target_id"],
                "stage": manifest["stage"], "path": shard["path"], "sha256": shard["sha256"],
                "size_bytes": shard["size_bytes"], "row_count": shard["row_count"],
                "fields": shard["fields"],
            })
        payload = {
            "schema_version": "c74_physical_view_manifest_v1",
            "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
            "view_name": view_name, "shard_kind": kind, "entries": entries,
        }
        provisional = views_dir / f".{view_name}.json"
        digest = cache.atomic_json(provisional, payload)
        final = views_dir / f"{view_name}_sha256_{digest[:16]}.json"
        if final.exists() and cache.sha256_file(final) != digest:
            raise RuntimeError(f"C74 immutable view-manifest collision: {final}")
        if final.exists():
            provisional.unlink(missing_ok=True)
        else:
            os.replace(provisional, final)
        output.append({
            "view_name": view_name, "external_manifest_path": str(final),
            "sha256": digest, "entry_count": len(entries),
            "allowed_columns": ";".join(sorted(runner.SHARD_SCHEMAS[kind])),
            "forbidden_columns": "target labels" if target_unlabeled or strict_source else "evaluation labels" if view_name == "target_construction_view" else "none beyond registered schema",
            "uses_source_labels": source_labels, "uses_target_labels": target_labels,
            "uses_evaluation_labels": eval_labels, "available_under_strict_source_DG": strict_source,
            "target_unlabeled": target_unlabeled,
            "diagnostic_only": int(not strict_source or kind != "strict_source_trial"),
            "consumer_command": consumer,
            "primary_smoke_access": int(view_name != "same_label_oracle_view"),
        })
    return output


def _schema_tables() -> tuple[list[dict], list[dict]]:
    source_rows = []
    for field in sorted(runner.SOURCE_FIELDS):
        source_rows.append({
            "field": field,
            "payload_class": "source_label" if field == "source_class_label" else "strict_source_trial_observable",
            "target_label_derived": 0, "available_under_strict_source_DG": 1,
            "stored_external": 1, "required": 1,
        })
    target_rows = []
    for field in sorted(runner.TARGET_UNLABELED_FIELDS):
        target_rows.append({
            "field": field, "payload_class": "target_unlabeled_representation",
            "target_label_derived": 0, "target_unlabeled": 1,
            "available_under_strict_source_DG": 0, "stored_external": 1, "required": 1,
        })
    return source_rows, target_rows


def _preprocessing_rows(manifests: list[dict], protocol: dict) -> list[dict]:
    job_rows = []
    evidence_hashes = set()
    raw_resolved_pairs = set()
    for stage in ("P0_pilot", "P1_expansion"):
        for target_id in range(1, 10):
            path = cache.run_root(protocol) / stage / f"target-{target_id:03d}" / "job_manifest.json"
            payload = json.loads(path.read_text())
            evidence_hashes.add(payload["dataset_evidence_hash"])
            raw_resolved_pairs.add((payload["raw_data_fingerprint"], payload["resolved_preprocess_hash"]))
            job_rows.append({
                "stage": stage, "target_id": target_id,
                "dataset_evidence_hash": payload["dataset_evidence_hash"],
                "raw_data_fingerprint": payload["raw_data_fingerprint"],
                "resolved_preprocess_hash": payload["resolved_preprocess_hash"],
                "network_attempt_count": payload["network_attempt_count"],
                "unit_count": payload["unit_count"],
                "all_gates_passed": int(payload["all_gates_passed"]),
            })
    comparison_path = cache.run_root(protocol) / "preprocessing_cross_node_replay" / "cross_node_preprocessing_comparison.json"
    if not comparison_path.is_file():
        raise RuntimeError("C74 preprocessing evidence variants require the locked cross-node drift audit")
    comparison = cache.verify_unit_manifest(comparison_path, rehash_payloads=False)
    if (
        len(raw_resolved_pairs) != 1
        or any(int(row["network_attempt_count"]) for row in job_rows)
        or not comparison["passed"]
        or not comparison["raw_fingerprints_equal"]
        or not comparison["resolved_preprocess_hashes_equal"]
        or not comparison["trial_ids_equal"]
        or not comparison["labels_equal"]
    ):
        raise RuntimeError("C74 preprocessing/data contract or cross-node drift audit failed")
    for row in job_rows:
        row.update({
            "dataset_evidence_hash_variant_count": len(evidence_hashes),
            "cross_node_comparison_path": str(comparison_path),
            "cross_node_comparison_sha256": cache.sha256_file(comparison_path),
            "cross_node_input_max_abs": comparison["input_max_abs"],
            "cross_node_input_mean_abs": comparison["input_mean_abs"],
            "cross_node_z_max_abs": comparison["z_max_abs"],
            "cross_node_logit_max_abs": comparison["logit_max_abs"],
            "cross_node_probability_max_abs": comparison["probability_max_abs"],
            "cross_node_prediction_disagreements": comparison["prediction_disagreements"],
            "cross_node_drift_passed": int(comparison["passed"]),
        })
    return job_rows


def _risk_rows() -> list[dict]:
    statuses = {
        "C73_metric_semantics_ambiguity": ("closed", "mean of target-normalized shares reconciled before protocol lock"),
        "protocol_timing": ("closed", "protocol commit 1f3ab88 predates all real forward"),
        "authorization_token_scope": ("closed", "exact CLI token only; P0 and gated P1 predeclared"),
        "pilot_to_full_silent_escalation": ("closed", "P1 requires aggregate 54-unit P0 gate"),
        "T2_T3_HO_variable_contamination": ("closed", "locked disjoint manifests and runtime T2 membership guard"),
        "target_outcome_based_unit_sampling": ("closed", "metadata-only deterministic pilot"),
        "source_target_view_leakage": ("closed", "separate immutable shard schemas"),
        "target_label_in_unlabeled_view": ("closed", "schema and payload validation"),
        "evaluation_label_in_construction_view": ("closed", "disjoint trial-ID split and separate shards"),
        "hook_layer_mismatch": ("closed", "classifier pre-hook equals ModelOutput.z"),
        "Wz_logit_identity_failure": ("closed", "all rows/units under locked tolerances"),
        "precision_or_compression_drift": ("closed", "float32 deterministic uncompressed NPZ"),
        "checkpoint_ABI_mismatch": ("closed", "strict state/sidecar/model ABI gates"),
        "preprocessing_mismatch": ("closed", "identical offline loader evidence across jobs"),
        "cache_rows_not_independent": ("controlled", "unit/target clustering retained; no iid-row inference"),
        "pilot_overinterpretation": ("controlled", "pilot used only as instrumentation gate"),
        "representation_claim_without_holdout": ("controlled", "C74 makes feasibility claims only"),
        "strict_source_escape_hatch_overclaim": ("controlled", "path availability is not escape-hatch evidence"),
        "raw_cache_in_git": ("closed", "all raw shards external; Git contains compact manifests only"),
        "unauthorized_training_or_GPU": ("closed", "CPU eval/no-grad state identity; no training path"),
    }
    return [
        {"risk": risk, "status": status, "blocking": 0, "evidence": evidence}
        for risk, (status, evidence) in statuses.items()
    ]


def prepare_views() -> dict:
    """Governance-only pass that creates restricted physical view manifests."""
    protocol = cache.load_locked_protocol()
    manifests = _all_manifests()
    abi_rows, identity_rows, wb_rows, content_rows = _content_and_abi_tables(manifests)
    preprocess_rows = _preprocessing_rows(manifests, protocol)
    view_rows = _materialize_view_manifests(manifests, protocol)
    source_schema, target_schema = _schema_tables()

    _write_csv("checkpoint_ABI_audit.csv", abi_rows)
    _write_csv("preprocessing_contract_audit.csv", preprocess_rows)
    _write_csv("hook_identity_summary.csv", identity_rows)
    _write_csv("precision_drift_audit.csv", identity_rows)
    _write_csv("physical_view_manifest.csv", view_rows)
    _write_csv("source_trial_schema.csv", source_schema)
    _write_csv("target_unlabeled_representation_schema.csv", target_schema)
    _write_csv("checkpoint_Wb_manifest.csv", wb_rows)
    _write_csv("cache_content_manifest.csv", content_rows)
    restricted = _write_primary_smoke_input(manifests, protocol)
    return {
        "unit_count": len(manifests),
        "physical_view_count": len(view_rows),
        "cache_content_rows": len(content_rows),
        "primary_smoke_input": restricted,
        "same_label_oracle_primary_access": False,
    }


def run_analysis() -> dict:
    protocol = cache.load_locked_protocol()
    # This process receives only the restricted input created by prepare-views.
    # It never receives an oracle shard descriptor or path.
    manifests = _primary_smoke_manifests(protocol)

    records = []
    for index, manifest in enumerate(manifests, 1):
        print(json.dumps({"event": "c74_analysis_unit", "index": index, "count": 216, "unit_id": manifest["unit_id"]}), flush=True)
        records.append(_extract_unit_features(manifest))
    variance_rows = _projection_variance(records)
    stability_rows = _split_stability(records)
    prediction_rows = _incremental_prediction(records)
    counterfactual_rows = _counterfactuals(records)

    finite_stability = np.asarray([row["spearman"] for row in stability_rows if math.isfinite(float(row["spearman"]))])
    median_stability = float(np.median(finite_stability)) if len(finite_stability) else float("nan")
    positive_fraction = float(np.mean(finite_stability > 0)) if len(finite_stability) else 0.0
    construct_feasible = (
        median_stability >= CONSTRUCT_FEASIBILITY_MIN_MEDIAN_SPEARMAN
        and positive_fraction >= CONSTRUCT_FEASIBILITY_MIN_POSITIVE_FRACTION
    )
    with open(TABLE_DIR / "cache_content_manifest.csv", newline="") as stream:
        content_rows = list(csv.DictReader(stream))
    total_bytes = sum(int(row["size_bytes"]) for row in content_rows)
    measured_seconds = np.asarray([float(manifest["execution"]["wall_seconds"]) for manifest in manifests])
    projected_t3_bytes = int(total_bytes * 1052 / 216)
    projected_t3_seconds_48cpu = float(np.sum(measured_seconds) * 1052 / 216 / 9)
    rho = max(abs(median_stability) if math.isfinite(median_stability) else 0.0, 0.10)
    correlation_power_units = int(math.ceil(((1.96 + 0.84) / np.arctanh(min(rho, 0.95))) ** 2 + 3))
    target_rep_row = prediction_rows[-1]
    power_rows = [{
        "campaign": "C74_T2_measured", "units": 216,
        "source_rows": sum(int(manifest["source_rows"]) for manifest in manifests),
        "target_rows": sum(int(manifest["target_unlabeled_rows"]) for manifest in manifests),
        "external_size_bytes": total_bytes, "external_size_GiB": total_bytes / 2**30,
        "unit_wall_seconds_median": float(np.median(measured_seconds)),
        "unit_wall_seconds_p95": float(np.quantile(measured_seconds, 0.95)),
        "projection_split_median_spearman": median_stability,
        "projection_split_positive_fraction": positive_fraction,
        "target_unlabeled_zWz_incremental_R2": target_rep_row["incremental_R2"],
        "target_unlabeled_zWz_exceeds_null_p95": target_rep_row["exceeds_null_p95"],
        "planning_method": "measured_T2_linear_scale_not_runtime_guarantee",
    }, {
        "campaign": "C76_T3_HO_projected", "units": 1052,
        "source_rows": 1052 * 8 * 576, "target_rows": 1052 * 576,
        "external_size_bytes": projected_t3_bytes,
        "external_size_GiB": projected_t3_bytes / 2**30,
        "unit_wall_seconds_median": float(np.median(measured_seconds)),
        "unit_wall_seconds_p95": float(np.quantile(measured_seconds, 0.95)),
        "projection_split_median_spearman": median_stability,
        "projection_split_positive_fraction": positive_fraction,
        "target_unlabeled_zWz_incremental_R2": target_rep_row["incremental_R2"],
        "target_unlabeled_zWz_exceeds_null_p95": target_rep_row["exceeds_null_p95"],
        "planning_method": f"T2_linear_scale;rough_correlation_power_units={correlation_power_units};estimated_parallel_9x48cpu_seconds={projected_t3_seconds_48cpu:.1f}",
    }]

    _write_csv("target_common_candidate_projection_variance.csv", variance_rows)
    _write_csv("projection_split_stability_smoke.csv", stability_rows)
    _write_csv("incremental_prediction_feasibility.csv", prediction_rows)
    _write_csv("projection_counterfactual_feasibility.csv", counterfactual_rows)
    _write_csv("power_and_storage_plan.csv", power_rows)
    _write_csv("risk_register.csv", _risk_rows())
    _write_csv("failure_reason_ledger.csv", [{
        "reason": "none_blocking", "active": 1,
        "pilot_failed_units": 0, "expansion_failed_units": 0,
        "T3_HO_units_touched": 0,
        "construct_status": "feasible" if construct_feasible else "unstable",
        "notes": "instrumentation completed; mechanism remains untested on new-variable holdout",
    }])

    state = {
        "schema_version": "c74_analysis_state_v1",
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "units": len(manifests), "targets": 9,
        "source_rows": sum(int(manifest["source_rows"]) for manifest in manifests),
        "target_unlabeled_rows": sum(int(manifest["target_unlabeled_rows"]) for manifest in manifests),
        "external_size_bytes": total_bytes,
        "identity": {
            "max_abs": max(float(manifest["identity"]["Wz_plus_b_logits_max_abs"]) for manifest in manifests),
            "max_relative": max(float(manifest["identity"]["Wz_plus_b_logits_max_relative"]) for manifest in manifests),
            "softmax_max_abs": max(float(manifest["identity"]["softmax_probability_max_abs"]) for manifest in manifests),
            "hook_max_abs": max(float(manifest["identity"]["hook_z_max_abs"]) for manifest in manifests),
            "repeat_max_abs": max(max(float(manifest["identity"]["repeat_logits_max_abs"]), float(manifest["identity"]["repeat_z_max_abs"])) for manifest in manifests),
            "failed_units": 0,
        },
        "projection_construct": {
            "median_split_spearman": median_stability,
            "positive_fraction": positive_fraction,
            "feasibility_rule": {
                "min_median_spearman": CONSTRUCT_FEASIBILITY_MIN_MEDIAN_SPEARMAN,
                "min_positive_fraction": CONSTRUCT_FEASIBILITY_MIN_POSITIVE_FRACTION,
            },
            "feasible": construct_feasible,
        },
        "incremental_prediction": prediction_rows,
        "T3_HO_z_Wz_touched": False,
        "same_label_oracle_used_by_primary_smoke": False,
        "representation_mechanism_claimed": False,
        "strict_source_escape_hatch_claimed": False,
        "final_gate_candidate": "T2_SOURCE_WZ_CAMPAIGN_EXECUTED_AND_MANIFESTED",
    }
    cache.atomic_json(ANALYSIS_STATE, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-views", "analyze"))
    args = parser.parse_args()
    result = prepare_views() if args.command == "prepare-views" else run_analysis()
    print(json.dumps(result, indent=2, sort_keys=True))
