"""C75 restricted-view feature extraction over manifested C74 T2 caches."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from . import c74_analysis
from . import c74_cache
from . import c74_t2_source_wz_instrumentation as c74_runner
from . import c75_protocol


EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c75-representation-construct")
SPECTRAL_SALT = "C75_SPECTRAL_SUPPORT_V1"
ALLOWED_KINDS = {
    "checkpoint_Wb", "strict_source_trial", "target_unlabeled_representation",
    "target_construction_labels", "target_evaluation_labels",
}


def load_protocol() -> dict:
    expected = c75_protocol.PROTOCOL_SHA_PATH.read_text().strip()
    observed = c75_protocol.sha256(c75_protocol.PROTOCOL_PATH)
    if expected != observed:
        raise RuntimeError(f"C75 protocol hash drift: {observed} != {expected}")
    protocol = json.loads(c75_protocol.PROTOCOL_PATH.read_text())
    for item in protocol["locked_registry_tables"].values():
        if c75_protocol.sha256(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C75 locked registry drift: {item['path']}")
    return protocol


def run_root(protocol: dict) -> Path:
    return EXTERNAL_ROOT / f"protocol_{c75_protocol.sha256(c75_protocol.PROTOCOL_PATH)[:16]}"


def feature_manifest_path(protocol: dict) -> Path:
    return run_root(protocol) / "feature_cache_manifest.json"


def _descriptor(manifest: dict, kind: str) -> dict:
    matches = [item for item in manifest["shards"] if item["kind"] == kind]
    if len(matches) != 1:
        raise RuntimeError(f"C75 {manifest['unit_id']} expected one {kind} shard")
    return matches[0]


def _load_fields(descriptor: dict, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        missing = set(fields) - set(shard.files)
        if missing:
            raise RuntimeError(f"C75 missing fields {sorted(missing)} in {descriptor['path']}")
        return {field: shard[field] for field in fields}


def _softmax_entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def endpoint_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float | np.ndarray]:
    shifted = logits.astype(float) - np.max(logits.astype(float), axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = np.argmax(probabilities, axis=1)
    recalls = np.asarray([
        float(np.mean(predicted[labels == class_index] == class_index))
        for class_index in range(4)
    ])
    true_probability = np.asarray([
        float(np.mean(probabilities[labels == class_index, class_index]))
        for class_index in range(4)
    ])
    bacc = float(np.mean(recalls))
    nll = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))))
    confidence = np.max(probabilities, axis=1)
    correctness = (predicted == labels).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        mask = (confidence >= edges[index]) & (confidence <= edges[index + 1] if index == 14 else confidence < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correctness[mask])) - float(np.mean(confidence[mask])))
    return {
        "bAcc": bacc, "NLL": nll, "ECE": float(ece), "recall": recalls,
        "true_probability": true_probability, "mean_confidence": float(np.mean(confidence)),
        "mean_entropy": float(np.mean(_softmax_entropy(probabilities))),
        "mean_margin": float(np.mean(_margin(probabilities))),
    }


def midrank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones_like(values)
    return (stats.rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def _sha_key(value: str) -> str:
    return hashlib.sha256(f"{SPECTRAL_SALT}|{value}".encode()).hexdigest()


def _source_spectral_indices(trial_ids: np.ndarray, domains: np.ndarray, labels: np.ndarray) -> np.ndarray:
    selected = []
    for domain in sorted(set(map(str, domains))):
        for class_index in range(4):
            indices = np.where((domains.astype(str) == domain) & (labels == class_index))[0]
            ordered = sorted(indices.tolist(), key=lambda index: _sha_key(str(trial_ids[index])))
            if len(ordered) < 8:
                raise RuntimeError(f"C75 source spectral support short for {domain} class {class_index}")
            selected.extend(ordered[:8])
    if len(selected) != 256 or len(set(selected)) != 256:
        raise RuntimeError("C75 source spectral support must contain 256 unique trials")
    return np.asarray(selected, dtype=int)


def _target_spectral_indices(trial_ids: np.ndarray) -> np.ndarray:
    ordered = sorted(range(len(trial_ids)), key=lambda index: _sha_key(str(trial_ids[index])))
    if len(ordered) < 256:
        raise RuntimeError("C75 target spectral support has fewer than 256 trials")
    return np.asarray(ordered[:256], dtype=int)


def z_features(z: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(z.astype(float), axis=1)
    moments = np.asarray([
        float(np.mean(norms)), float(np.std(norms)),
        float(np.quantile(norms, 0.25)), float(np.quantile(norms, 0.75)),
    ])
    centered = z[support].astype(float) - np.mean(z[support].astype(float), axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular ** 2 / max(len(support) - 1, 1)
    trace = float(np.sum(eigenvalues))
    if trace <= 1e-15:
        spectral = np.zeros(6, dtype=float)
    else:
        probabilities = eigenvalues / trace
        positive = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
        stable_rank = float(trace / max(float(eigenvalues[0]), 1e-15))
        spectral = np.asarray([
            trace, effective_rank, stable_rank,
            float(eigenvalues[0] / trace), float(np.sum(eigenvalues[:5]) / trace),
            float(np.sum(eigenvalues[:10]) / trace),
        ])
    return moments, spectral


def W_features(W: np.ndarray, bias: np.ndarray) -> np.ndarray:
    W = W.astype(float)
    bias = bias.astype(float)
    row_norms = np.linalg.norm(W, axis=1)
    normalized = W / np.maximum(row_norms[:, None], 1e-15)
    cosine = normalized @ normalized.T
    offdiag = cosine[np.triu_indices(4, 1)]
    singular = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    squared = singular ** 2
    probabilities = squared / max(float(np.sum(squared)), 1e-15)
    positive = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    return np.asarray([
        float(np.linalg.norm(W)), float(np.linalg.norm(bias)),
        float(np.mean(row_norms)), float(np.std(row_norms)),
        float(np.mean(offdiag)), float(np.max(np.abs(offdiag))),
        float(singular[0] / max(singular[-1], 1e-15)), effective_rank,
        float(squared[0] / max(float(np.sum(squared)), 1e-15)),
        float(np.linalg.norm(bias) / max(np.linalg.norm(W), 1e-15)),
    ])


def alignment_features(z: np.ndarray, W: np.ndarray) -> np.ndarray:
    z = z.astype(float)
    W = W.astype(float)
    denominator = np.maximum(np.linalg.norm(z, axis=1, keepdims=True) * np.linalg.norm(W, axis=1)[None, :], 1e-15)
    alignment = (z @ W.T) / denominator
    return np.concatenate((np.mean(alignment, axis=0), [float(np.mean(np.max(np.abs(alignment), axis=1)))]))


def projection_summary(Wz: np.ndarray) -> np.ndarray:
    canonical = np.asarray(Wz, dtype=float)
    return np.concatenate((np.mean(canonical, axis=0), np.std(canonical, axis=0)))


def _functional_features(
    logits: np.ndarray, probabilities: np.ndarray, predicted: np.ndarray,
    labels: np.ndarray, common_predicted: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
    metrics = endpoint_metrics(logits, labels)
    confidence = np.max(probabilities, axis=1)
    occupancy = np.asarray([float(np.mean(predicted == class_index)) for class_index in range(4)])
    predicted_confidence = np.asarray([
        float(np.mean(confidence[predicted == class_index])) if np.any(predicted == class_index) else 0.0
        for class_index in range(4)
    ])
    values = np.concatenate((
        [float(np.mean(confidence)), float(np.mean(_softmax_entropy(probabilities))),
         float(np.mean(_margin(probabilities))), float(np.mean(np.linalg.norm(logits, axis=1)))],
        occupancy, predicted_confidence, np.asarray(metrics["recall"]),
        np.asarray(metrics["true_probability"]), np.mean(probabilities, axis=0),
        [float(np.mean(predicted != common_predicted))],
    ))
    if len(values) != 25:
        raise RuntimeError(f"C75 F1 dimension drift: {len(values)}")
    return values, metrics


def _target_functional_features(
    logits: np.ndarray, probabilities: np.ndarray, predicted: np.ndarray,
    common_logits: np.ndarray, common_probabilities: np.ndarray, common_predicted: np.ndarray,
) -> np.ndarray:
    probability_shift = np.linalg.norm(probabilities - common_probabilities, axis=1)
    logit_shift = np.linalg.norm(logits - common_logits, axis=1)
    values = np.concatenate((
        [float(np.mean(np.max(probabilities, axis=1))), float(np.mean(_softmax_entropy(probabilities))),
         float(np.mean(_margin(probabilities))), float(np.mean(np.linalg.norm(logits, axis=1)))],
        [float(np.mean(predicted == class_index)) for class_index in range(4)],
        np.mean(probabilities, axis=0),
        [float(np.mean(predicted != common_predicted)),
         float(np.mean(probability_shift)), float(np.std(probability_shift)),
         float(np.mean(logit_shift)), float(np.std(logit_shift)),
         float(np.sqrt(np.mean((logits - common_logits) ** 2)))],
    ))
    if len(values) != 18:
        raise RuntimeError(f"C75 F3 dimension drift: {len(values)}")
    return values


def _view_replay(protocol: dict) -> dict:
    view_rows = list(csv_dicts(c75_protocol.C74_VIEW_MANIFEST))
    failures = []
    for row in view_rows:
        path = Path(row["external_manifest_path"])
        if not path.is_file() or c75_protocol.sha256(path) != row["sha256"]:
            failures.append(row["view_name"])
    if failures:
        raise RuntimeError(f"C75 C74 view-manifest replay failed: {failures}")
    return {row["view_name"]: row["sha256"] for row in view_rows}


def csv_dicts(path: str | Path):
    import csv
    with open(path, newline="") as stream:
        yield from csv.DictReader(stream)


def extract_feature_cache() -> dict:
    protocol = load_protocol()
    view_hashes = _view_replay(protocol)
    manifests = c74_analysis._primary_smoke_manifests(json.loads(c75_protocol.C74_PROTOCOL.read_text()))
    if len(manifests) != 216 or any({item["kind"] for item in manifest["shards"]} != ALLOWED_KINDS for manifest in manifests):
        raise RuntimeError("C75 restricted C74 input contract failed")
    t3_ids = {row["checkpoint_id"] for row in csv_dicts(c75_protocol.C74_T3_UNITS)}
    if any(manifest["checkpoint_id"] in t3_ids for manifest in manifests):
        raise RuntimeError("C75 T3-HO contamination")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for manifest in manifests:
        grouped[int(manifest["target_id"])].append(manifest)
    rows = []
    max_wz_identity = 0.0
    max_raw_logits_minus_b_error = 0.0
    payloads_rehashed = 0
    for target_id, target_manifests in sorted(grouped.items()):
        target_manifests = sorted(target_manifests, key=lambda item: (item["seed"], item["level"], item["candidate_order"], item["unit_id"]))
        small = []
        for manifest in target_manifests:
            descriptors = {item["kind"]: item for item in manifest["shards"]}
            for kind, descriptor in descriptors.items():
                c74_cache.verify_shard(descriptor, required_fields=c74_runner.SHARD_SCHEMAS[kind])
                payloads_rehashed += 1
            source = _load_fields(descriptors["strict_source_trial"], (
                "source_trial_id", "source_domain_id", "source_class_label", "logits",
                "probabilities", "predicted_class", "Wz", "Wz_plus_b",
            ))
            target = _load_fields(descriptors["target_unlabeled_representation"], (
                "target_trial_id", "logits", "probabilities", "predicted_class", "Wz", "Wz_plus_b",
            ))
            Wb = _load_fields(descriptors["checkpoint_Wb"], ("W", "b"))
            construction = _load_fields(descriptors["target_construction_labels"], ("target_trial_id", "target_class_label"))
            evaluation = _load_fields(descriptors["target_evaluation_labels"], ("target_trial_id", "target_class_label"))
            max_wz_identity = max(
                max_wz_identity,
                float(np.max(np.abs(source["Wz_plus_b"] - source["logits"]))),
                float(np.max(np.abs(target["Wz_plus_b"] - target["logits"]))),
            )
            small.append({"manifest": manifest, "descriptors": descriptors, "source": source, "target": target, "Wb": Wb, "construction": construction, "evaluation": evaluation})

        source_ids = small[0]["source"]["source_trial_id"]
        target_ids = small[0]["target"]["target_trial_id"]
        if any(not np.array_equal(source_ids, item["source"]["source_trial_id"]) for item in small):
            raise RuntimeError(f"C75 source trial alignment failed for target {target_id}")
        if any(
            not np.array_equal(small[0]["source"][field], item["source"][field])
            for item in small
            for field in ("source_domain_id", "source_class_label")
        ):
            raise RuntimeError(f"C75 source domain/label alignment failed for target {target_id}")
        if any(not np.array_equal(target_ids, item["target"]["target_trial_id"]) for item in small):
            raise RuntimeError(f"C75 target trial alignment failed for target {target_id}")
        if any(
            not np.array_equal(small[0][view][field], item[view][field])
            for item in small
            for view in ("construction", "evaluation")
            for field in ("target_trial_id", "target_class_label")
        ):
            raise RuntimeError(f"C75 target label-view alignment failed for target {target_id}")
        source_common_prob = np.mean(np.stack([item["source"]["probabilities"] for item in small]), axis=0)
        source_common_pred = np.argmax(source_common_prob, axis=1)
        target_common_prob = np.mean(np.stack([item["target"]["probabilities"] for item in small]), axis=0)
        target_common_logits = np.mean(np.stack([item["target"]["logits"] for item in small]), axis=0)
        target_common_Wz = np.mean(np.stack([item["target"]["Wz"] for item in small]), axis=0)
        target_common_pred = np.argmax(target_common_prob, axis=1)
        trajectory_max = defaultdict(int)
        for item in small:
            trajectory_max[item["manifest"]["trajectory_id"]] = max(
                trajectory_max[item["manifest"]["trajectory_id"]], int(item["manifest"]["candidate_order"])
            )

        target_rows = []
        for item in small:
            manifest = item["manifest"]
            source, target, Wb = item["source"], item["target"], item["Wb"]
            source_labels = source["source_class_label"].astype(int)
            F1, source_metrics = _functional_features(
                source["logits"], source["probabilities"], source["predicted_class"].astype(int),
                source_labels, source_common_pred,
            )
            max_order = trajectory_max[manifest["trajectory_id"]]
            F0 = np.concatenate((
                np.eye(3)[int(manifest["seed"])], np.eye(2)[int(manifest["level"])],
                [int(manifest["candidate_order"]) / max(max_order, 1),
                 float(source_metrics["bAcc"]), float(source_metrics["NLL"]), float(source_metrics["ECE"])],
            ))
            F3 = _target_functional_features(
                target["logits"], target["probabilities"], target["predicted_class"].astype(int),
                target_common_logits, target_common_prob, target_common_pred,
            )
            source_z = _load_fields(item["descriptors"]["strict_source_trial"], ("z",))["z"]
            target_z = _load_fields(item["descriptors"]["target_unlabeled_representation"], ("z",))["z"]
            source_support = _source_spectral_indices(
                source["source_trial_id"], source["source_domain_id"], source_labels,
            )
            target_support = _target_spectral_indices(target["target_trial_id"])
            source_moments, source_spectrum = z_features(source_z, source_support)
            target_moments, target_spectrum = z_features(target_z, target_support)
            Wgeometry = W_features(Wb["W"], Wb["b"])
            source_alignment = alignment_features(source_z, Wb["W"])
            F2 = np.concatenate((source_moments, source_spectrum, Wgeometry, source_alignment))
            wz_mean_std = np.concatenate((np.mean(target["Wz"], axis=0), np.std(target["Wz"], axis=0)))
            residual = target["Wz"].astype(float) - target_common_Wz.astype(float)
            residual_norm = np.linalg.norm(residual, axis=1)
            residual_features = np.concatenate((
                np.mean(residual, axis=0),
                [float(np.mean(residual_norm)), float(np.std(residual_norm)), float(np.sqrt(np.mean(residual ** 2)))],
            ))
            F4 = np.concatenate((target_moments, target_spectrum, Wgeometry, wz_mean_std, residual_features))
            if tuple(map(len, (F0, F1, F2, F3, F4))) != (9, 25, 25, 18, 35):
                raise RuntimeError("C75 registered feature dimension drift")

            index = {str(trial_id): idx for idx, trial_id in enumerate(target["target_trial_id"])}
            construct_idx = np.asarray([index[str(value)] for value in item["construction"]["target_trial_id"]], dtype=int)
            eval_idx = np.asarray([index[str(value)] for value in item["evaluation"]["target_trial_id"]], dtype=int)
            construct_labels = item["construction"]["target_class_label"].astype(int)
            eval_labels = item["evaluation"]["target_class_label"].astype(int)
            if set(construct_idx.tolist()) & set(eval_idx.tolist()) or len(set(construct_idx.tolist()) | set(eval_idx.tolist())) != 576:
                raise RuntimeError(f"C75 construction/evaluation split failed for {manifest['unit_id']}")
            construct_metrics = endpoint_metrics(target["logits"][construct_idx], construct_labels)
            eval_metrics = endpoint_metrics(target["logits"][eval_idx], eval_labels)
            split_construct = np.asarray([
                float(np.mean(target["Wz"][construct_idx[construct_labels == class_index], class_index]))
                for class_index in range(4)
            ])
            split_eval = np.asarray([
                float(np.mean(target["Wz"][eval_idx[eval_labels == class_index], class_index]))
                for class_index in range(4)
            ])
            raw_source_projection = source["logits"].astype(float) - Wb["b"].astype(float)[None, :]
            raw_target_projection = target["logits"].astype(float) - Wb["b"].astype(float)[None, :]
            max_raw_logits_minus_b_error = max(
                max_raw_logits_minus_b_error,
                float(np.max(np.abs(raw_source_projection - source["Wz"].astype(float)))),
                float(np.max(np.abs(raw_target_projection - target["Wz"].astype(float)))),
            )
            # Once Wz+b==logits has passed, use the stored Wz as the canonical
            # logits-minus-b coordinate. This prevents float32 subtraction noise
            # from creating a spurious extra rank in the duplicate-block audit.
            source_logits_minus_b = source["Wz"].astype(float)
            target_logits_minus_b = target["Wz"].astype(float)
            target_rows.append({
                "unit_id": manifest["unit_id"], "target_id": target_id,
                "seed": int(manifest["seed"]), "level": int(manifest["level"]),
                "candidate_order": int(manifest["candidate_order"]),
                "trajectory_id": manifest["trajectory_id"],
                "trajectory_template": f"seed-{manifest['seed']}|level-{manifest['level']}|{manifest['regime']}",
                "F0": F0, "F1": F1, "F2": F2, "F3": F3, "F4": F4,
                "construct_metrics": construct_metrics, "eval_metrics": eval_metrics,
                "construct_wz_class": split_construct, "eval_wz_class": split_eval,
                "source_logits_minus_b": projection_summary(source_logits_minus_b),
                "source_Wz_summary": projection_summary(source_logits_minus_b),
                "target_logits_minus_b": projection_summary(target_logits_minus_b),
                "target_Wz_summary": projection_summary(target_logits_minus_b),
                "source_z_summary": np.concatenate((source_moments, source_spectrum)),
                "target_z_summary": np.concatenate((target_moments, target_spectrum)),
                "W_geometry": Wgeometry,
                "source_alignment": source_alignment,
                "target_Wz_residual": residual_features,
            })

        construct_oriented = np.column_stack((
            midrank_percentile(np.asarray([row["construct_metrics"]["bAcc"] for row in target_rows])),
            midrank_percentile(-np.asarray([row["construct_metrics"]["NLL"] for row in target_rows])),
            midrank_percentile(-np.asarray([row["construct_metrics"]["ECE"] for row in target_rows])),
        ))
        eval_oriented = np.column_stack((
            midrank_percentile(np.asarray([row["eval_metrics"]["bAcc"] for row in target_rows])),
            midrank_percentile(-np.asarray([row["eval_metrics"]["NLL"] for row in target_rows])),
            midrank_percentile(-np.asarray([row["eval_metrics"]["ECE"] for row in target_rows])),
        ))
        for index, row in enumerate(target_rows):
            construct = row["construct_metrics"]
            F5 = np.concatenate((
                [float(construct["bAcc"]), float(construct["NLL"]), float(construct["ECE"])],
                np.asarray(construct["recall"]), np.asarray(construct["true_probability"]),
                [float(construct["mean_confidence"]), float(construct["mean_entropy"]),
                 float(construct["mean_margin"]), float(np.mean(construct_oriented[index]))],
            ))
            if len(F5) != 15:
                raise RuntimeError("C75 F5 dimension drift")
            row["F5"] = F5
            row["outcomes"] = np.asarray([
                float(np.mean(eval_oriented[index])), float(row["eval_metrics"]["bAcc"]),
                -float(row["eval_metrics"]["NLL"]), -float(row["eval_metrics"]["ECE"]),
                float(np.all(eval_oriented[index] >= 0.75)),
            ])
            rows.append(row)
        print(json.dumps({"event": "c75_feature_target_complete", "target_id": target_id, "units": len(target_rows)}), flush=True)

    if max_wz_identity != 0.0:
        raise RuntimeError(f"C75 exact Wz+b/logit replay failed: {max_wz_identity}")

    rows = sorted(rows, key=lambda row: (row["target_id"], row["seed"], row["level"], row["candidate_order"], row["unit_id"]))
    arrays: dict[str, np.ndarray] = {
        "unit_id": np.asarray([row["unit_id"] for row in rows], dtype="<U32"),
        "target_id": np.asarray([row["target_id"] for row in rows], dtype=np.int16),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int16),
        "level": np.asarray([row["level"] for row in rows], dtype=np.int16),
        "candidate_order": np.asarray([row["candidate_order"] for row in rows], dtype=np.int32),
        "trajectory_id": np.asarray([row["trajectory_id"] for row in rows], dtype="<U80"),
        "trajectory_template": np.asarray([row["trajectory_template"] for row in rows], dtype="<U80"),
        "outcomes": np.stack([row["outcomes"] for row in rows]),
    }
    for block in ("F0", "F1", "F2", "F3", "F4", "F5"):
        arrays[block] = np.stack([row[block] for row in rows])
    for key in (
        "construct_wz_class", "eval_wz_class", "source_logits_minus_b", "source_Wz_summary",
        "target_logits_minus_b", "target_Wz_summary", "source_z_summary", "target_z_summary",
        "W_geometry", "source_alignment", "target_Wz_residual",
    ):
        arrays[key] = np.stack([row[key] for row in rows])

    root = run_root(protocol)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = c74_cache.write_content_addressed_npz(root, "c75_registered_feature_cache", arrays)
    payload = c74_cache.self_hashed_manifest({
        "schema_version": "c75_registered_feature_cache_manifest_v1",
        "protocol_sha256": c75_protocol.sha256(c75_protocol.PROTOCOL_PATH),
        "parent_C74_result_commit": c75_protocol.PARENT_COMMIT,
        "unit_count": len(rows), "target_count": len(grouped),
        "view_manifest_hashes": view_hashes,
        "allowed_view_kinds": sorted(ALLOWED_KINDS),
        "same_label_oracle_accessed": False, "T3_HO_z_Wz_accessed": False,
        "payload_descriptors_rehashed": payloads_rehashed,
        "Wz_plus_b_logits_max_abs": max_wz_identity,
        "raw_logits_minus_b_Wz_max_abs": max_raw_logits_minus_b_error,
        "feature_dimensions": {block: int(arrays[block].shape[1]) for block in ("F0", "F1", "F2", "F3", "F4", "F5")},
        "outcome_names": ["continuous_joint_utility", "bAcc", "negNLL", "negECE", "primary_joint_good"],
        "descriptor": descriptor,
    })
    path = feature_manifest_path(protocol)
    c74_cache.atomic_json(path, payload)
    return {"feature_manifest_path": str(path), **payload}


def load_feature_cache() -> tuple[dict, dict[str, np.ndarray]]:
    protocol = load_protocol()
    manifest = c74_cache.verify_unit_manifest(feature_manifest_path(protocol), rehash_payloads=False)
    descriptor = manifest["descriptor"]
    c74_cache.verify_shard(descriptor)
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    if len(arrays["unit_id"]) != 216 or manifest["T3_HO_z_Wz_accessed"]:
        raise RuntimeError("C75 feature cache contract failed")
    return manifest, arrays


if __name__ == "__main__":
    print(json.dumps(extract_feature_cache(), indent=2, sort_keys=True))
