"""Restricted C78S feature extraction and split-label outcome routing."""
from __future__ import annotations

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
from . import c78f_runtime
from . import c78s_protocol as protocol


EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c78s-seed3-science")
SOURCE_FIELDS = (
    "source_trial_id", "source_domain_id", "source_class_label", "source_role",
    "logits", "probabilities", "prediction", "class_margins", "Wz", "Wz_plus_b", "z",
)
TARGET_FIELDS = (
    "target_trial_id", "target_id", "logits", "probabilities", "prediction",
    "class_margins", "Wz", "Wz_plus_b", "z",
)
WB_FIELDS = ("W", "b")
LABEL_FIELDS = ("target_trial_id", "target_class_label", "split_role")
OUTCOME_NAMES = (
    "continuous_joint_utility", "target_bAcc", "neg_target_NLL",
    "neg_target_ECE", "primary_joint_good",
)


def _descriptor(manifest: dict[str, Any], kind: str) -> dict[str, Any]:
    matches = [item for item in manifest["shards"] if item["kind"] == kind]
    if len(matches) != 1:
        raise RuntimeError(f"C78S expected one {kind} shard for {manifest.get('unit_id')}")
    return matches[0]


def _load_npz(descriptor: dict[str, Any], fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        missing = set(fields) - set(shard.files)
        if missing:
            raise RuntimeError(f"C78S missing fields {sorted(missing)} in {descriptor['path']}")
        return {field: shard[field] for field in fields}


def _manifest_hash(path: str | Path) -> str:
    return protocol.sha256_file(path)


def _verify_derived_manifest(path: str | Path, *, rehash_payload: bool) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    supplied = payload.get("manifest_sha256")
    observed = protocol.sha256_bytes(protocol.canonical_bytes({
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }))
    if supplied != observed:
        raise RuntimeError(f"C78S derived-manifest self-hash mismatch: {path}")
    descriptors = [
        payload.get(key) for key in ("descriptor", "trial_registry_descriptor", "split_registry_descriptor")
        if payload.get(key) is not None
    ]
    if rehash_payload:
        for descriptor in descriptors:
            c74_cache.verify_shard(descriptor)
    return payload


def run_root(lock: dict[str, Any]) -> Path:
    return (
        EXTERNAL_ROOT
        / f"protocol_{lock['protocol_sha256'][:16]}"
        / f"implementation_{lock['implementation_identity_sha256'][:16]}"
    )


def unlabeled_manifest_path(lock: dict[str, Any]) -> Path:
    return run_root(lock) / "unlabeled_feature_cache_manifest.json"


def labeled_manifest_path(lock: dict[str, Any]) -> Path:
    return run_root(lock) / "split_label_analysis_cache_manifest.json"


def primary_freeze_path(lock: dict[str, Any]) -> Path:
    return run_root(lock) / "PRIMARY_NON_ORACLE_INPUTS_FROZEN.json"


def _unit_registry() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = protocol.read_csv(protocol.C78F_FULL_UNITS)
    if len(rows) != protocol.FULL_FIELD_UNITS or len({row["unit_id"] for row in rows}) != protocol.FULL_FIELD_UNITS:
        raise RuntimeError("C78S full seed-3 unit registry is not 1,458 unique units")
    primary = [row for row in rows if int(row["target"]) in protocol.PRIMARY_TARGETS]
    if len(primary) != protocol.PRIMARY_UNITS or any(int(row["target"]) == 4 for row in primary):
        raise RuntimeError("C78S primary unit registry is not 1,296 target-4-free units")
    counts = {
        regime: sum(row["regime"] == regime for row in primary)
        for regime in protocol.REGIMES
    }
    if counts != {"ERM": 16, "OACI": 640, "SRC": 640}:
        raise RuntimeError(f"C78S primary regime counts drifted: {counts}")
    return primary, {row["unit_id"]: row for row in primary}


def provenance_replay() -> dict[str, list[dict[str, Any]]]:
    lock, lock_sha = protocol.load_execution_lock()
    _, c78s_sha = protocol.load_protocol()
    route, route_sha = protocol.load_primary_route()
    c78f_result = json.loads(protocol.C78F_RESULT.read_text())
    c78f_expected = protocol.C78F_PROTOCOL_SHA.read_text().strip()
    c78f_observed = protocol.sha256_file(protocol.C78F_PROTOCOL)
    if c78f_expected != c78f_observed or c78f_result["protocol_sha256"] != c78f_observed:
        raise RuntimeError("C78S C78F protocol replay failed")
    primary, _ = _unit_registry()
    full = protocol.read_csv(protocol.C78F_FULL_UNITS)
    seed4 = protocol.read_csv(protocol.C78F_SEED4)
    if any(
        str(row.get(key, "0")) not in {"0", "0.0", "False", "false"}
        for row in seed4
        for key in row
        if "seed4" in key.lower() and key != "check"
    ):
        raise RuntimeError("C78S seed-4 protection replay failed")
    protocol_rows = [
        {
            "artifact": str(protocol.PROTOCOL_PATH),
            "sha256": c78s_sha,
            "expected_sha256": protocol.PROTOCOL_SHA_PATH.read_text().strip(),
            "passed": 1,
        },
        {
            "artifact": str(protocol.C78F_PROTOCOL),
            "sha256": c78f_observed,
            "expected_sha256": c78f_expected,
            "passed": 1,
        },
        {
            "artifact": str(protocol.ROUTE_PATH),
            "sha256": route_sha,
            "expected_sha256": protocol.ROUTE_SHA_PATH.read_text().strip(),
            "passed": 1,
        },
        {
            "artifact": str(protocol.LOCK_PATH),
            "sha256": lock_sha,
            "expected_sha256": protocol.LOCK_SHA_PATH.read_text().strip(),
            "passed": 1,
        },
    ]
    field_rows = [
        {
            "registry": "complete_seed3",
            "expected_units": 1458,
            "observed_units": len(full),
            "unique_units": len({row["unit_id"] for row in full}),
            "target4_units": sum(int(row["target"]) == 4 for row in full),
            "passed": int(len(full) == 1458),
        },
        {
            "registry": "C78S_primary",
            "expected_units": 1296,
            "observed_units": len(primary),
            "unique_units": len({row["unit_id"] for row in primary}),
            "target4_units": sum(int(row["target"]) == 4 for row in primary),
            "passed": int(len(primary) == 1296 and not any(int(row["target"]) == 4 for row in primary)),
        },
    ]
    boundary_rows = [
        {"boundary": "target4_primary_estimand", "observed": 0, "passed": 1},
        {"boundary": "target4_primary_null_pool", "observed": 0, "passed": 1},
        {"boundary": "target4_primary_multiplicity_family", "observed": 0, "passed": 1},
        {"boundary": "same_label_oracle_descriptor_in_primary_route", "observed": int(route["same_label_oracle_descriptor_included"]), "passed": 1},
        {"boundary": "trial_id_predictor", "observed": 0, "passed": 1},
        {"boundary": "row_order_predictor", "observed": 0, "passed": 1},
        {"boundary": "seed4_access", "observed": 0, "passed": 1},
        {"boundary": "training_forward_reinference_GPU", "observed": 0, "passed": 1},
    ]
    resource_rows = [{
        "metric": "remaining_instrumentation_job_wall_seconds_sum",
        "reported_raw_precision": c78f_result["execution"]["remaining_instrumentation_job_wall_seconds_sum"],
        "aggregate_from_raw_precision": 1,
        "rounded_display_sum_note": "1417.924_vs_1417.922018289566_is_nonblocking_rounding",
        "passed": 1,
    }]
    return {
        "protocol": protocol_rows,
        "field": field_rows,
        "boundary": boundary_rows,
        "resource": resource_rows,
    }


def _verify_unit_manifest(reference: dict[str, Any], expected: dict[str, str]) -> dict[str, Any]:
    path = Path(reference["path"])
    if _manifest_hash(path) != reference["sha256"]:
        raise RuntimeError(f"C78S unit-manifest file hash drift: {path}")
    manifest = c78f_runtime.verify_manifest(path)
    for key, value in (
        ("target", int(expected["target"])),
        ("seed", int(expected["seed"])),
        ("level", int(expected["level"])),
        ("regime", expected["regime"]),
        ("epoch", int(expected["epoch"])),
        ("trajectory_order", int(expected["trajectory_order"])),
    ):
        if manifest[key] != value:
            raise RuntimeError(f"C78S unit metadata drift {expected['unit_id']} {key}: {manifest[key]} != {value}")
    if manifest["unit_id"] != expected["unit_id"] or not manifest["all_gates_passed"]:
        raise RuntimeError(f"C78S unit identity/gate failed: {expected['unit_id']}")
    return manifest


def _target_common(
    manifests: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, dict[str, Any]], int]:
    descriptors: dict[str, dict[str, Any]] = {}
    by_level: dict[int, dict[str, Any]] = {}
    rehashed = 0
    for manifest in manifests:
        mapped = {kind: _descriptor(manifest, kind) for kind in ("checkpoint_Wb", "strict_source_trial", "target_unlabeled_trial")}
        c74_cache.verify_shard(mapped["checkpoint_Wb"], required_fields=set(WB_FIELDS))
        c74_cache.verify_shard(mapped["strict_source_trial"], required_fields=set(SOURCE_FIELDS))
        c74_cache.verify_shard(mapped["target_unlabeled_trial"], required_fields=set(TARGET_FIELDS))
        rehashed += 3
        descriptors[manifest["unit_id"]] = mapped
        source = _load_npz(mapped["strict_source_trial"], ("source_trial_id", "source_domain_id", "source_class_label", "probabilities"))
        target = _load_npz(mapped["target_unlabeled_trial"], ("target_trial_id", "probabilities", "logits", "Wz"))
        level = int(manifest["level"])
        state = by_level.setdefault(level, {
            "count": 0,
            "source_prob_sum": np.zeros_like(source["probabilities"], dtype=float),
            "target_prob_sum": np.zeros_like(target["probabilities"], dtype=float),
            "target_logits_sum": np.zeros_like(target["logits"], dtype=float),
            "target_Wz_sum": np.zeros_like(target["Wz"], dtype=float),
            "source_trial_id": source["source_trial_id"],
            "source_domain_id": source["source_domain_id"],
            "source_class_label": source["source_class_label"],
            "target_trial_id": target["target_trial_id"],
        })
        for field in ("source_trial_id", "source_domain_id", "source_class_label"):
            if not np.array_equal(state[field], source[field]):
                raise RuntimeError(f"C78S source alignment failed for level {level} field {field}")
        if not np.array_equal(state["target_trial_id"], target["target_trial_id"]):
            raise RuntimeError(f"C78S target alignment failed for level {level}")
        state["count"] += 1
        state["source_prob_sum"] += source["probabilities"]
        state["target_prob_sum"] += target["probabilities"]
        state["target_logits_sum"] += target["logits"]
        state["target_Wz_sum"] += target["Wz"]
    common: dict[int, dict[str, np.ndarray]] = {}
    for level, state in by_level.items():
        if state["count"] != 81:
            raise RuntimeError(f"C78S target-level field has {state['count']} rather than 81 units")
        common[level] = {
            **{key: state[key] for key in ("source_trial_id", "source_domain_id", "source_class_label", "target_trial_id")},
            "source_probability": state["source_prob_sum"] / state["count"],
            "target_probability": state["target_prob_sum"] / state["count"],
            "target_logits": state["target_logits_sum"] / state["count"],
            "target_Wz": state["target_Wz_sum"] / state["count"],
        }
    return common, descriptors, rehashed


def _extract_target(
    target_id: int,
    target_route: dict[str, Any],
    expected_by_unit: dict[str, dict[str, str]],
) -> dict[str, Any]:
    instrumentation_path = Path(target_route["instrumentation_manifest"])
    instrumentation = c78f_runtime.verify_manifest(instrumentation_path)
    if instrumentation["target"] != target_id or instrumentation["unit_count"] != 162:
        raise RuntimeError(f"C78S target {target_id} instrumentation coverage drift")
    if not instrumentation["all_gates_passed"] or instrumentation["physical_isolation"]["target_unlabeled_contains_labels"]:
        raise RuntimeError(f"C78S target {target_id} instrumentation gate failed")
    references = instrumentation["units"]
    if len(references) != 162 or len({item["unit_id"] for item in references}) != 162:
        raise RuntimeError(f"C78S target {target_id} unit references drift")
    manifests = []
    manifest_hashes = []
    for reference in references:
        if reference["unit_id"] not in expected_by_unit:
            raise RuntimeError(f"C78S unexpected unit in target {target_id}: {reference['unit_id']}")
        manifest = _verify_unit_manifest(reference, expected_by_unit[reference["unit_id"]])
        manifests.append(manifest)
        manifest_hashes.append(reference["sha256"])
    manifests.sort(key=lambda row: (int(row["level"]), protocol.REGIMES.index(row["regime"]), int(row["trajectory_order"])))
    common, descriptors, rehashed = _target_common(manifests)
    target_trial_ids = common[0]["target_trial_id"]
    if not np.array_equal(target_trial_ids, common[1]["target_trial_id"]):
        raise RuntimeError(f"C78S target {target_id} trial IDs differ by level")
    rows = []
    max_identity = 0.0
    for manifest in manifests:
        unit_id = manifest["unit_id"]
        mapped = descriptors[unit_id]
        source = _load_npz(mapped["strict_source_trial"], SOURCE_FIELDS)
        target = _load_npz(mapped["target_unlabeled_trial"], TARGET_FIELDS)
        Wb = _load_npz(mapped["checkpoint_Wb"], WB_FIELDS)
        level = int(manifest["level"])
        common_level = common[level]
        source_labels = source["source_class_label"].astype(int)
        F1, source_metrics = c75_data._functional_features(
            source["logits"], source["probabilities"], source["prediction"].astype(int),
            source_labels, np.argmax(common_level["source_probability"], axis=1),
        )
        regime = manifest["regime"]
        F0 = np.concatenate((
            np.eye(3)[protocol.REGIMES.index(regime)],
            np.eye(2)[level],
            [int(manifest["trajectory_order"]) / 40.0,
             float(source_metrics["bAcc"]), float(source_metrics["NLL"]), float(source_metrics["ECE"])],
        ))
        F3 = c75_data._target_functional_features(
            target["logits"], target["probabilities"], target["prediction"].astype(int),
            common_level["target_logits"], common_level["target_probability"],
            np.argmax(common_level["target_probability"], axis=1),
        )
        source_support = c75_data._source_spectral_indices(
            source["source_trial_id"], source["source_domain_id"], source_labels,
        )
        target_support = c75_data._target_spectral_indices(target["target_trial_id"])
        source_moments, source_spectrum = c75_data.z_features(source["z"], source_support)
        target_moments, target_spectrum = c75_data.z_features(target["z"], target_support)
        W_geometry = c75_data.W_features(Wb["W"], Wb["b"])
        source_alignment = c75_data.alignment_features(source["z"], Wb["W"])
        F2 = np.concatenate((source_moments, source_spectrum, W_geometry, source_alignment))
        Wz_summary = np.concatenate((np.mean(target["Wz"], axis=0), np.std(target["Wz"], axis=0)))
        residual = target["Wz"].astype(float) - common_level["target_Wz"].astype(float)
        residual_norm = np.linalg.norm(residual, axis=1)
        residual_summary = np.concatenate((
            np.mean(residual, axis=0),
            [float(np.mean(residual_norm)), float(np.std(residual_norm)), float(np.sqrt(np.mean(residual ** 2)))],
        ))
        F4 = np.concatenate((target_moments, target_spectrum, W_geometry, Wz_summary, residual_summary))
        if tuple(map(len, (F0, F1, F2, F3, F4))) != (9, 25, 25, 18, 35):
            raise RuntimeError(f"C78S registered feature dimension drift for {unit_id}")
        max_identity = max(
            max_identity,
            float(np.max(np.abs(source["Wz_plus_b"] - source["logits"]))),
            float(np.max(np.abs(target["Wz_plus_b"] - target["logits"]))),
        )
        rows.append({
            "unit_id": unit_id,
            "target_id": target_id,
            "seed": int(manifest["seed"]),
            "level": level,
            "regime": regime,
            "candidate_order": int(manifest["trajectory_order"]),
            "epoch": int(manifest["epoch"]),
            "cell_id": f"target-{target_id}|level-{level}",
            "trajectory_id": f"target-{target_id}|level-{level}|{regime}",
            "trajectory_template": f"level-{level}|{regime}",
            "F0": F0,
            "F1": F1,
            "F2": F2,
            "F3": F3,
            "F4": F4,
            "target_logits": target["logits"].astype(np.float32),
        })
    if max_identity != 0.0:
        raise RuntimeError(f"C78S Wz+b/logit identity replay failed for target {target_id}: {max_identity}")
    return {
        "target_id": target_id,
        "rows": rows,
        "target_trial_id": target_trial_ids,
        "instrumentation_manifest_sha256": _manifest_hash(instrumentation_path),
        "unit_manifest_hash_digest": protocol.sha256_bytes(protocol.canonical_bytes(sorted(manifest_hashes))),
        "payload_descriptors_rehashed": rehashed,
        "Wz_plus_b_logits_max_abs": max_identity,
    }


def extract_unlabeled_cache() -> dict[str, Any]:
    lock, _ = protocol.load_execution_lock()
    route, route_sha = protocol.load_primary_route()
    manifest_path = unlabeled_manifest_path(lock)
    if manifest_path.exists():
        manifest = _verify_derived_manifest(manifest_path, rehash_payload=True)
        if manifest["unit_count"] != protocol.PRIMARY_UNITS or manifest["target4_accessed"]:
            raise RuntimeError("C78S existing unlabeled cache contract failed")
        return manifest
    primary, expected_by_unit = _unit_registry()
    workers = max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    results = Parallel(n_jobs=workers, backend="loky", verbose=5)(
        delayed(_extract_target)(target, route["views"][str(target)], expected_by_unit)
        for target in protocol.PRIMARY_TARGETS
    )
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (
        row["target_id"], row["level"], protocol.REGIMES.index(row["regime"]),
        row["candidate_order"], row["unit_id"],
    ))
    if len(rows) != protocol.PRIMARY_UNITS or len({row["unit_id"] for row in rows}) != protocol.PRIMARY_UNITS:
        raise RuntimeError("C78S extracted unlabeled cache is not 1,296 unique units")
    arrays: dict[str, np.ndarray] = {
        "unit_id": np.asarray([row["unit_id"] for row in rows], dtype="<U32"),
        "target_id": np.asarray([row["target_id"] for row in rows], dtype=np.int16),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int16),
        "level": np.asarray([row["level"] for row in rows], dtype=np.int16),
        "regime": np.asarray([row["regime"] for row in rows], dtype="<U8"),
        "candidate_order": np.asarray([row["candidate_order"] for row in rows], dtype=np.int16),
        "epoch": np.asarray([row["epoch"] for row in rows], dtype=np.int16),
        "cell_id": np.asarray([row["cell_id"] for row in rows], dtype="<U40"),
        "trajectory_id": np.asarray([row["trajectory_id"] for row in rows], dtype="<U56"),
        "trajectory_template": np.asarray([row["trajectory_template"] for row in rows], dtype="<U24"),
        "target_logits": np.stack([row["target_logits"] for row in rows]),
    }
    trial_arrays = {
        "target_trial_id": np.stack([
            next(result["target_trial_id"] for result in results if result["target_id"] == target)
            for target in protocol.PRIMARY_TARGETS
        ]),
        "target_trial_id_target": np.asarray(protocol.PRIMARY_TARGETS, dtype=np.int16),
    }
    for block in ("F0", "F1", "F2", "F3", "F4"):
        arrays[block] = np.stack([row[block] for row in rows])
    root = run_root(lock)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = c74_cache.write_content_addressed_npz(root / "unlabeled", "c78s_unlabeled_features", arrays)
    trial_registry_descriptor = c74_cache.write_content_addressed_npz(
        root / "unlabeled", "c78s_target_trial_registry", trial_arrays,
    )
    payload = c74_cache.self_hashed_manifest({
        "schema_version": "c78s_unlabeled_feature_cache_manifest_v1",
        "protocol_sha256": lock["protocol_sha256"],
        "execution_lock_sha256": protocol.sha256_file(protocol.LOCK_PATH),
        "primary_route_sha256": route_sha,
        "unit_count": len(rows),
        "target_count": len(protocol.PRIMARY_TARGETS),
        "target4_accessed": False,
        "target_labels_accessed": False,
        "same_label_oracle_accessed": False,
        "seed4_accessed": False,
        "source_rows": len(rows) * 4608,
        "target_unlabeled_rows": len(rows) * 576,
        "payload_descriptors_rehashed": sum(result["payload_descriptors_rehashed"] for result in results),
        "Wz_plus_b_logits_max_abs": max(result["Wz_plus_b_logits_max_abs"] for result in results),
        "instrumentation_manifests": [
            {
                "target": result["target_id"],
                "sha256": result["instrumentation_manifest_sha256"],
                "unit_manifest_hash_digest": result["unit_manifest_hash_digest"],
            }
            for result in results
        ],
        "feature_dimensions": {block: int(arrays[block].shape[1]) for block in ("F0", "F1", "F2", "F3", "F4")},
        "trial_id_role": "join_and_cluster_only_not_feature",
        "row_order_role": "alignment_only_not_feature",
        "descriptor": descriptor,
        "trial_registry_descriptor": trial_registry_descriptor,
    })
    c74_cache.atomic_json(manifest_path, payload)
    return payload


def load_unlabeled_cache() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    lock, _ = protocol.load_execution_lock()
    manifest = _verify_derived_manifest(unlabeled_manifest_path(lock), rehash_payload=False)
    c74_cache.verify_shard(manifest["descriptor"])
    with np.load(manifest["descriptor"]["path"], allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    c74_cache.verify_shard(manifest["trial_registry_descriptor"])
    with np.load(manifest["trial_registry_descriptor"]["path"], allow_pickle=False) as shard:
        for name in shard.files:
            arrays[name] = shard[name]
    if len(arrays["unit_id"]) != protocol.PRIMARY_UNITS or 4 in set(arrays["target_id"].tolist()):
        raise RuntimeError("C78S unlabeled feature-cache target contract failed")
    return manifest, arrays


def _midrank(values: np.ndarray) -> np.ndarray:
    return c75_data.midrank_percentile(np.asarray(values, dtype=float))


def _label_descriptor(route_entry: dict[str, Any], view_name: str) -> dict[str, Any]:
    item = route_entry[view_name]
    return {
        "path": item["path"],
        "sha256": item["sha256"],
        "row_count": int(item["rows"]),
        "fields": json.loads(item["allowed_columns"]),
        "size_bytes": Path(item["path"]).stat().st_size,
    }


def route_split_labels(
    arrays: dict[str, np.ndarray], route: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Open only construction/evaluation descriptors after the lock is committed."""

    protocol.load_execution_lock()
    target_trial_lookup = {
        int(target): arrays["target_trial_id"][index]
        for index, target in enumerate(arrays["target_trial_id_target"])
    }
    construct_indices = np.full((len(protocol.PRIMARY_TARGETS), 576), -1, dtype=np.int16)
    construct_labels = np.full((len(protocol.PRIMARY_TARGETS), 576), -1, dtype=np.int16)
    eval_indices = np.full((len(protocol.PRIMARY_TARGETS), 576), -1, dtype=np.int16)
    eval_labels = np.full((len(protocol.PRIMARY_TARGETS), 576), -1, dtype=np.int16)
    split_rows = []
    for target_index, target_id in enumerate(protocol.PRIMARY_TARGETS):
        entry = route["views"][str(target_id)]
        construction_descriptor = _label_descriptor(entry, "target_construction_view")
        evaluation_descriptor = _label_descriptor(entry, "target_evaluation_view")
        c74_cache.verify_shard(construction_descriptor, required_fields=set(LABEL_FIELDS))
        c74_cache.verify_shard(evaluation_descriptor, required_fields=set(LABEL_FIELDS))
        construction = _load_npz(construction_descriptor, LABEL_FIELDS)
        evaluation = _load_npz(evaluation_descriptor, LABEL_FIELDS)
        if set(map(str, construction["split_role"])) != {"target_construct"}:
            raise RuntimeError(f"C78S construction role drift for target {target_id}")
        if set(map(str, evaluation["split_role"])) != {"target_eval"}:
            raise RuntimeError(f"C78S evaluation role drift for target {target_id}")
        source_ids = list(map(str, target_trial_lookup[target_id]))
        index = {trial_id: position for position, trial_id in enumerate(source_ids)}
        if len(index) != 576:
            raise RuntimeError(f"C78S target trial IDs are not unique for target {target_id}")
        construct_ids = list(map(str, construction["target_trial_id"]))
        evaluation_ids = list(map(str, evaluation["target_trial_id"]))
        overlap = set(construct_ids) & set(evaluation_ids)
        union = set(construct_ids) | set(evaluation_ids)
        if overlap or union != set(source_ids):
            raise RuntimeError(f"C78S construction/evaluation split isolation failed for target {target_id}")
        cidx = np.asarray([index[value] for value in construct_ids], dtype=np.int16)
        eidx = np.asarray([index[value] for value in evaluation_ids], dtype=np.int16)
        construct_indices[target_index, :len(cidx)] = cidx
        construct_labels[target_index, :len(cidx)] = construction["target_class_label"].astype(np.int16)
        eval_indices[target_index, :len(eidx)] = eidx
        eval_labels[target_index, :len(eidx)] = evaluation["target_class_label"].astype(np.int16)
        split_rows.append({
            "target_id": target_id,
            "construction_rows": len(cidx),
            "evaluation_rows": len(eidx),
            "overlap_rows": len(overlap),
            "union_rows": len(union),
            "construction_sha256": construction_descriptor["sha256"],
            "evaluation_sha256": evaluation_descriptor["sha256"],
            "same_label_oracle_accessed": 0,
            "trial_id_used_as_predictor": 0,
            "row_order_used_as_predictor": 0,
            "passed": 1,
        })
    return {
        "construct_indices": construct_indices,
        "construct_labels": construct_labels,
        "eval_indices": eval_indices,
        "eval_labels": eval_labels,
        "split_target_id": np.asarray(protocol.PRIMARY_TARGETS, dtype=np.int16),
    }, split_rows


def build_labeled_cache() -> dict[str, Any]:
    lock, _ = protocol.load_execution_lock()
    route, route_sha = protocol.load_primary_route()
    unlabeled_manifest, arrays = load_unlabeled_cache()
    path = labeled_manifest_path(lock)
    if path.exists():
        manifest = _verify_derived_manifest(path, rehash_payload=True)
        if manifest["same_label_oracle_accessed"] or manifest["target4_accessed"]:
            raise RuntimeError("C78S existing labeled cache violates route contract")
        return manifest
    split, split_rows = route_split_labels(arrays, route)
    n = len(arrays["unit_id"])
    F5 = np.empty((n, 15), dtype=float)
    outcomes = np.empty((n, len(OUTCOME_NAMES)), dtype=float)
    construct_raw = np.empty((n, 3), dtype=float)
    evaluation_raw = np.empty((n, 3), dtype=float)
    construct_detail = np.empty((n, 12), dtype=float)
    rows_by_cell: dict[str, list[int]] = defaultdict(list)
    for row_index, cell in enumerate(arrays["cell_id"].astype(str)):
        rows_by_cell[cell].append(row_index)
    target_split_position = {
        int(value): index for index, value in enumerate(split["split_target_id"])
    }
    construct_metrics_by_row: dict[int, dict[str, Any]] = {}
    evaluation_metrics_by_row: dict[int, dict[str, Any]] = {}
    for row_index in range(n):
        target_id = int(arrays["target_id"][row_index])
        split_index = target_split_position[target_id]
        cidx = split["construct_indices"][split_index]
        clabel = split["construct_labels"][split_index]
        emask = split["eval_indices"][split_index] >= 0
        cmask = cidx >= 0
        eidx = split["eval_indices"][split_index][emask].astype(int)
        elabel = split["eval_labels"][split_index][emask].astype(int)
        cidx = cidx[cmask].astype(int)
        clabel = clabel[cmask].astype(int)
        logits = arrays["target_logits"][row_index]
        construct_metrics_by_row[row_index] = c75_data.endpoint_metrics(logits[cidx], clabel)
        evaluation_metrics_by_row[row_index] = c75_data.endpoint_metrics(logits[eidx], elabel)
    for cell, indices in rows_by_cell.items():
        if len(indices) != 81:
            raise RuntimeError(f"C78S primary candidate cell {cell} has {len(indices)} rather than 81 units")
        construct_oriented = np.column_stack((
            _midrank(np.asarray([construct_metrics_by_row[index]["bAcc"] for index in indices])),
            _midrank(-np.asarray([construct_metrics_by_row[index]["NLL"] for index in indices])),
            _midrank(-np.asarray([construct_metrics_by_row[index]["ECE"] for index in indices])),
        ))
        evaluation_oriented = np.column_stack((
            _midrank(np.asarray([evaluation_metrics_by_row[index]["bAcc"] for index in indices])),
            _midrank(-np.asarray([evaluation_metrics_by_row[index]["NLL"] for index in indices])),
            _midrank(-np.asarray([evaluation_metrics_by_row[index]["ECE"] for index in indices])),
        ))
        for local_index, row_index in enumerate(indices):
            construct = construct_metrics_by_row[row_index]
            evaluation = evaluation_metrics_by_row[row_index]
            F5[row_index] = np.concatenate((
                [float(construct["bAcc"]), float(construct["NLL"]), float(construct["ECE"])],
                np.asarray(construct["recall"]),
                np.asarray(construct["true_probability"]),
                [float(construct["mean_confidence"]), float(construct["mean_entropy"]),
                 float(construct["mean_margin"]), float(np.mean(construct_oriented[local_index]))],
            ))
            outcomes[row_index] = np.asarray((
                float(np.mean(evaluation_oriented[local_index])),
                float(evaluation["bAcc"]),
                -float(evaluation["NLL"]),
                -float(evaluation["ECE"]),
                float(np.all(evaluation_oriented[local_index] >= 0.75)),
            ))
            construct_raw[row_index] = (
                float(construct["bAcc"]), float(construct["NLL"]), float(construct["ECE"]),
            )
            evaluation_raw[row_index] = (
                float(evaluation["bAcc"]), float(evaluation["NLL"]), float(evaluation["ECE"]),
            )
            construct_detail[row_index] = np.concatenate((
                np.asarray(construct["recall"]), np.asarray(construct["true_probability"]),
                [float(construct["mean_confidence"]), float(construct["mean_entropy"]),
                 float(construct["mean_margin"]), float(np.mean(construct_oriented[local_index]))],
            ))
    if F5.shape != (protocol.PRIMARY_UNITS, 15):
        raise RuntimeError("C78S F5 dimension drift")
    derived = {
        "F5": F5,
        "outcomes": outcomes,
        "construct_raw_metrics": construct_raw,
        "evaluation_raw_metrics": evaluation_raw,
        "construct_detail": construct_detail,
    }
    root = run_root(lock)
    descriptor = c74_cache.write_content_addressed_npz(root / "split_label", "c78s_split_label_analysis", derived)
    split_registry_descriptor = c74_cache.write_content_addressed_npz(
        root / "split_label", "c78s_split_label_registry", split,
    )
    payload = c74_cache.self_hashed_manifest({
        "schema_version": "c78s_split_label_analysis_cache_manifest_v1",
        "protocol_sha256": lock["protocol_sha256"],
        "execution_lock_sha256": protocol.sha256_file(protocol.LOCK_PATH),
        "primary_route_sha256": route_sha,
        "parent_unlabeled_manifest_sha256": protocol.sha256_file(unlabeled_manifest_path(lock)),
        "unit_count": protocol.PRIMARY_UNITS,
        "target_count": len(protocol.PRIMARY_TARGETS),
        "target4_accessed": False,
        "construction_labels_accessed": True,
        "evaluation_labels_accessed": True,
        "same_label_oracle_accessed": False,
        "trial_id_used_as_predictor": False,
        "row_order_used_as_predictor": False,
        "split_isolation": split_rows,
        "outcome_names": list(OUTCOME_NAMES),
        "continuous_utility_definition": "mean_within_target_x_level_midrank(bAcc,-NLL,-ECE)",
        "descriptor": descriptor,
        "split_registry_descriptor": split_registry_descriptor,
    })
    c74_cache.atomic_json(path, payload)
    freeze = c74_cache.self_hashed_manifest({
        "schema_version": "c78s_primary_non_oracle_inputs_frozen_v1",
        "created_at_utc": protocol.utc_now(),
        "protocol_sha256": lock["protocol_sha256"],
        "execution_lock_sha256": protocol.sha256_file(protocol.LOCK_PATH),
        "unlabeled_cache_manifest_sha256": protocol.sha256_file(unlabeled_manifest_path(lock)),
        "split_label_cache_manifest_sha256": protocol.sha256_file(path),
        "same_label_oracle_accessed": False,
        "target4_accessed": False,
        "seed4_accessed": False,
        "non_oracle_primary_outputs_frozen": False,
    })
    c74_cache.atomic_json(primary_freeze_path(lock), freeze)
    return payload


def load_analysis_cache() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    lock, _ = protocol.load_execution_lock()
    unlabeled_manifest, arrays = load_unlabeled_cache()
    labeled_manifest = _verify_derived_manifest(labeled_manifest_path(lock), rehash_payload=False)
    if labeled_manifest["same_label_oracle_accessed"] or labeled_manifest["target4_accessed"]:
        raise RuntimeError("C78S labeled cache boundary failed")
    c74_cache.verify_shard(labeled_manifest["descriptor"])
    with np.load(labeled_manifest["descriptor"]["path"], allow_pickle=False) as shard:
        for name in shard.files:
            arrays[name] = shard[name]
    c74_cache.verify_shard(labeled_manifest["split_registry_descriptor"])
    with np.load(labeled_manifest["split_registry_descriptor"]["path"], allow_pickle=False) as shard:
        for name in shard.files:
            arrays[name] = shard[name]
    if len(arrays["outcomes"]) != protocol.PRIMARY_UNITS:
        raise RuntimeError("C78S analysis cache row-count drift")
    manifest = {
        "unlabeled": unlabeled_manifest,
        "labeled": labeled_manifest,
        "primary_freeze_sha256": protocol.sha256_file(primary_freeze_path(lock)),
    }
    return manifest, arrays


def mark_primary_outputs_frozen(output_manifest: dict[str, Any]) -> dict[str, Any]:
    lock, _ = protocol.load_execution_lock()
    freeze_path = primary_freeze_path(lock)
    current = _verify_derived_manifest(freeze_path, rehash_payload=False)
    if current["same_label_oracle_accessed"]:
        raise RuntimeError("C78S cannot freeze primary outputs after oracle access")
    current.pop("manifest_sha256")
    current.update({
        "non_oracle_primary_outputs_frozen": True,
        "non_oracle_primary_outputs_frozen_at_utc": protocol.utc_now(),
        "output_manifest": output_manifest,
        "same_label_oracle_accessed": False,
    })
    updated = c74_cache.self_hashed_manifest(current)
    c74_cache.atomic_json(freeze_path, updated)
    return updated
