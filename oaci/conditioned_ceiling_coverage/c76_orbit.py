"""C76 function-preserving latent-orbit extraction over restricted C74 T2 views."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np

from . import c74_analysis
from . import c74_cache
from . import c74_t2_source_wz_instrumentation as c74_runner
from . import c75_data
from . import c75_protocol
from . import c76_protocol


EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c76-representation-association")
DIMENSION = 800
PROBE_COUNT = 8
IDENTITY_ROWS_PER_VIEW = 32


@dataclass(frozen=True)
class Transform:
    orbit: str
    name: str
    family: str
    scope: str
    replicate: int
    blocks: np.ndarray | None
    inverse_blocks: np.ndarray | None
    permutation: np.ndarray | None
    signs: np.ndarray | None
    condition_number: float
    transform_hash: str


def load_protocol() -> dict:
    expected = c76_protocol.PROTOCOL_SHA_PATH.read_text().strip()
    observed = c76_protocol.sha256(c76_protocol.PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError(f"C76 protocol hash drift: {observed} != {expected}")
    protocol = json.loads(c76_protocol.PROTOCOL_PATH.read_text())
    for item in protocol["locked_registry_tables"].values():
        if c76_protocol.sha256(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C76 locked registry drift: {item['path']}")
    return protocol


def run_root(protocol: dict) -> Path:
    return EXTERNAL_ROOT / f"protocol_{c76_protocol.sha256(c76_protocol.PROTOCOL_PATH)[:16]}"


def orbit_manifest_path(protocol: dict) -> Path:
    return run_root(protocol) / "orbit_feature_cache_manifest.json"


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _hash_transform(blocks, inverse_blocks, permutation, signs) -> str:
    digest = hashlib.sha256()
    for value in (blocks, inverse_blocks, permutation, signs):
        if value is not None:
            digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def _block_inverse(blocks: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(blocks)
    a, b = blocks[:, 0, 0], blocks[:, 0, 1]
    c, d = blocks[:, 1, 0], blocks[:, 1, 1]
    determinant = a * d - b * c
    inverse[:, 0, 0] = d / determinant
    inverse[:, 0, 1] = -b / determinant
    inverse[:, 1, 0] = -c / determinant
    inverse[:, 1, 1] = a / determinant
    return inverse


def _block_condition(blocks: np.ndarray) -> float:
    return float(max(np.linalg.cond(block) for block in blocks))


def _transform_seed(orbit: str, replicate: int, unit_id: str, scope: str) -> int:
    unit_key = "GLOBAL" if scope == "global" else unit_id
    return _seed(str(c76_protocol.RNG_SEED + 100), orbit, str(replicate), unit_key)


def make_transform(orbit: str, replicate: int, unit_id: str) -> Transform:
    registry = {row["orbit"]: row for row in c76_protocol.orbit_registry()}
    row = registry[orbit]
    rng = np.random.default_rng(_transform_seed(orbit, replicate, unit_id, row["scope"]))
    blocks = inverse = permutation = signs = None
    if orbit == "O0":
        blocks = np.tile(np.eye(2), (DIMENSION // 2, 1, 1))
    elif row["family"] == "orthogonal":
        angles = rng.uniform(-math.pi, math.pi, size=DIMENSION // 2)
        cosine, sine = np.cos(angles), np.sin(angles)
        blocks = np.empty((DIMENSION // 2, 2, 2), dtype=float)
        blocks[:, 0, 0], blocks[:, 0, 1] = cosine, -sine
        blocks[:, 1, 0], blocks[:, 1, 1] = sine, cosine
    elif row["family"] == "diagonal":
        scales = np.exp(rng.uniform(math.log(0.7), math.log(1.4), size=DIMENSION))
        blocks = np.zeros((DIMENSION // 2, 2, 2), dtype=float)
        blocks[:, 0, 0], blocks[:, 1, 1] = scales[0::2], scales[1::2]
    elif row["family"] == "nonorthogonal":
        blocks = np.zeros((DIMENSION // 2, 2, 2), dtype=float)
        for index in range(DIMENSION // 2):
            while True:
                a, d = np.exp(rng.uniform(math.log(0.85), math.log(1.15), size=2))
                shear = rng.uniform(-0.35, 0.35)
                candidate = np.asarray([[a, shear], [0.0, d]])
                if np.linalg.cond(candidate) <= float(row["condition_bound"]):
                    blocks[index] = candidate
                    break
    elif row["family"] == "signed_permutation":
        permutation = rng.permutation(DIMENSION)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=DIMENSION)
    else:  # pragma: no cover - registry invariant
        raise ValueError(orbit)
    if blocks is not None:
        inverse = _block_inverse(blocks)
        condition = _block_condition(blocks)
    else:
        condition = 1.0
    transform_hash = _hash_transform(blocks, inverse, permutation, signs)
    return Transform(
        orbit=orbit, name=row["name"], family=row["family"], scope=row["scope"],
        replicate=replicate, blocks=blocks, inverse_blocks=inverse,
        permutation=permutation, signs=signs, condition_number=condition,
        transform_hash=transform_hash,
    )


def orbit_variants() -> list[tuple[str, int]]:
    result = []
    for row in c76_protocol.orbit_registry():
        result.extend((row["orbit"], replicate) for replicate in range(int(row["replicates"])))
    return result


def apply_z(z: np.ndarray, transform: Transform) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if transform.permutation is not None:
        return z[:, transform.permutation] * transform.signs[None, :]
    reshaped = z.reshape(len(z), DIMENSION // 2, 2)
    return np.einsum("pij,npj->npi", transform.blocks, reshaped, optimize=True).reshape(len(z), DIMENSION)


def apply_vector(vector: np.ndarray, transform: Transform) -> np.ndarray:
    return apply_z(np.asarray(vector, dtype=float)[None, :], transform)[0]


def apply_W(W: np.ndarray, transform: Transform) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    if transform.permutation is not None:
        return W[:, transform.permutation] * transform.signs[None, :]
    reshaped = W.reshape(W.shape[0], DIMENSION // 2, 2)
    return np.einsum("cpi,pij->cpj", reshaped, transform.inverse_blocks, optimize=True).reshape(W.shape)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _coordinate_directions() -> np.ndarray:
    rng = np.random.default_rng(_seed("C76_FIXED_COORDINATE_PROBES", str(c76_protocol.RNG_SEED)))
    directions = rng.choice(np.asarray([-1.0, 1.0]), size=(PROBE_COUNT, DIMENSION))
    return directions / math.sqrt(DIMENSION)


def coordinate_features(mean_z: np.ndarray, W: np.ndarray, transform: Transform) -> np.ndarray:
    directions = _coordinate_directions()
    transformed_mean = apply_vector(mean_z, transform)
    transformed_W = apply_W(W, transform)
    return np.concatenate((
        transformed_mean @ directions[:4].T,
        np.asarray([transformed_W[class_index] @ directions[4 + class_index] for class_index in range(4)]),
    ))


def _alignment_from_invariant_Wz(Wz: np.ndarray, transformed_z: np.ndarray, transformed_W: np.ndarray) -> np.ndarray:
    denominator = np.maximum(
        np.linalg.norm(transformed_z, axis=1, keepdims=True)
        * np.linalg.norm(transformed_W, axis=1)[None, :],
        1e-15,
    )
    alignment = np.asarray(Wz, dtype=float) / denominator
    return np.concatenate((
        np.mean(alignment, axis=0),
        [float(np.mean(np.max(np.abs(alignment), axis=1)))],
    ))


def _descriptor(manifest: dict, kind: str) -> dict:
    matches = [item for item in manifest["shards"] if item["kind"] == kind]
    if len(matches) != 1:
        raise RuntimeError(f"C76 expected one {kind} shard for {manifest['unit_id']}")
    return matches[0]


def _load(descriptor: dict, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        return {field: shard[field] for field in fields}


def _unit_rows(manifest: dict, baseline: dict) -> list[dict]:
    descriptors = {item["kind"]: item for item in manifest["shards"]}
    if set(descriptors) != c75_data.ALLOWED_KINDS:
        raise RuntimeError(f"C76 restricted-view failure for {manifest['unit_id']}")
    for kind, descriptor in descriptors.items():
        c74_cache.verify_shard(descriptor, required_fields=c74_runner.SHARD_SCHEMAS[kind])
    source = _load(descriptors["strict_source_trial"], (
        "source_trial_id", "source_domain_id", "source_class_label", "z", "Wz",
    ))
    target = _load(descriptors["target_unlabeled_representation"], (
        "target_trial_id", "z", "Wz",
    ))
    Wb = _load(descriptors["checkpoint_Wb"], ("W", "b"))
    source_support = c75_data._source_spectral_indices(
        source["source_trial_id"], source["source_domain_id"], source["source_class_label"].astype(int),
    )
    target_support = c75_data._target_spectral_indices(target["target_trial_id"])
    source_identity = source_support[:IDENTITY_ROWS_PER_VIEW]
    target_identity = target_support[:IDENTITY_ROWS_PER_VIEW]
    source_mean = np.mean(source["z"].astype(float), axis=0)
    target_mean = np.mean(target["z"].astype(float), axis=0)
    baseline_F2 = baseline["F2"].astype(float)
    baseline_F4 = baseline["F4"].astype(float)
    rows = []
    for orbit, replicate in orbit_variants():
        transform = make_transform(orbit, replicate, manifest["unit_id"])
        transformed_W = apply_W(Wb["W"], transform)
        if transform.family in {"identity", "orthogonal", "signed_permutation"}:
            F2 = baseline_F2.copy()
            F4 = baseline_F4.copy()
        else:
            source_z = apply_z(source["z"], transform)
            target_z = apply_z(target["z"], transform)
            source_moments, source_spectrum = c75_data.z_features(source_z, source_support)
            target_moments, target_spectrum = c75_data.z_features(target_z, target_support)
            Wgeometry = c75_data.W_features(transformed_W, Wb["b"])
            source_alignment = _alignment_from_invariant_Wz(source["Wz"], source_z, transformed_W)
            F2 = np.concatenate((source_moments, source_spectrum, Wgeometry, source_alignment))
            F4 = np.concatenate((target_moments, target_spectrum, Wgeometry, baseline_F4[20:]))
        sample_source_z = apply_z(source["z"][source_identity], transform)
        sample_target_z = apply_z(target["z"][target_identity], transform)
        source_projection = sample_source_z @ transformed_W.T
        target_projection = sample_target_z @ transformed_W.T
        # C74 separately establishes stored Wz+b==logits exactly.  This gate
        # isolates numerical orbit algebra by comparing each transform against
        # the same float64 identity-coordinate matrix product.
        source_reference = (
            source["z"][source_identity].astype(float) @ Wb["W"].astype(float).T
        )
        target_reference = (
            target["z"][target_identity].astype(float) @ Wb["W"].astype(float).T
        )
        projection_error = max(
            float(np.max(np.abs(source_projection - source_reference))),
            float(np.max(np.abs(target_projection - target_reference))),
        )
        source_logits = source_projection + Wb["b"].astype(float)[None, :]
        source_reference_logits = source_reference + Wb["b"].astype(float)[None, :]
        target_logits = target_projection + Wb["b"].astype(float)[None, :]
        target_reference_logits = target_reference + Wb["b"].astype(float)[None, :]
        probability_error = max(
            float(np.max(np.abs(_softmax(source_logits) - _softmax(source_reference_logits)))),
            float(np.max(np.abs(_softmax(target_logits) - _softmax(target_reference_logits)))),
        )
        prediction_disagreement = int(
            np.sum(np.argmax(source_logits, axis=1) != np.argmax(source_reference_logits, axis=1))
            + np.sum(np.argmax(target_logits, axis=1) != np.argmax(target_reference_logits, axis=1))
        )
        rows.append({
            "orbit": orbit, "orbit_name": transform.name, "family": transform.family,
            "scope": transform.scope, "replicate": replicate,
            "condition_number": transform.condition_number,
            "transform_hash": transform.transform_hash,
            "F2": F2, "F4": F4,
            "G4S": coordinate_features(source_mean, Wb["W"], transform),
            "G4T": coordinate_features(target_mean, Wb["W"], transform),
            "projection_max_abs_error": projection_error,
            "probability_max_abs_error": probability_error,
            "prediction_disagreements": prediction_disagreement,
        })
    return rows


def extract_orbit_cache() -> dict:
    protocol = load_protocol()
    c75_manifest, c75_arrays = c75_data.load_feature_cache()
    manifests = c74_analysis._primary_smoke_manifests(json.loads(c76_protocol.C74_PROTOCOL.read_text()))
    if len(manifests) != 216:
        raise RuntimeError("C76 expected 216 restricted T2 manifests")
    t3_ids = {row["checkpoint_id"] for row in c75_data.csv_dicts(c76_protocol.C74_T3_UNITS)}
    if any(manifest["checkpoint_id"] in t3_ids for manifest in manifests):
        raise RuntimeError("C76 T3-HO contamination")
    baseline_index = {str(unit_id): index for index, unit_id in enumerate(c75_arrays["unit_id"])}
    manifest_by_unit = {manifest["unit_id"]: manifest for manifest in manifests}
    if set(baseline_index) != set(manifest_by_unit):
        raise RuntimeError("C76 C74/C75 unit mapping mismatch")
    workers = max(1, min(int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    unit_results = Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_unit_rows)(
            manifest_by_unit[unit_id],
            {"F2": c75_arrays["F2"][index], "F4": c75_arrays["F4"][index]},
        )
        for unit_id, index in sorted(baseline_index.items(), key=lambda item: item[1])
    )
    flat = []
    ordered_units = list(c75_arrays["unit_id"].astype(str))
    for unit_index, rows in enumerate(unit_results):
        for row in rows:
            flat.append({"unit_index": unit_index, **row})
    expected_rows = 216 * len(orbit_variants())
    if len(flat) != expected_rows:
        raise RuntimeError(f"C76 orbit row count drift: {len(flat)} != {expected_rows}")
    max_projection = max(row["projection_max_abs_error"] for row in flat)
    max_probability = max(row["probability_max_abs_error"] for row in flat)
    disagreements = sum(row["prediction_disagreements"] for row in flat)
    if max_projection > c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE or max_probability > c76_protocol.FUNCTIONAL_IDENTITY_TOLERANCE or disagreements:
        raise RuntimeError(
            f"C76 orbit identity failed: projection={max_projection};prob={max_probability};pred={disagreements}"
        )
    global_hash_failures = 0
    for orbit, replicate in orbit_variants():
        selected = [row["transform_hash"] for row in flat if row["orbit"] == orbit and row["replicate"] == replicate]
        scope = next(row["scope"] for row in flat if row["orbit"] == orbit and row["replicate"] == replicate)
        if scope == "global" and len(set(selected)) != 1:
            global_hash_failures += 1
        if scope == "checkpoint" and len(set(selected)) != 216:
            global_hash_failures += 1
    if global_hash_failures:
        raise RuntimeError(f"C76 transform scope/hash failure: {global_hash_failures}")
    arrays = {
        "unit_index": np.asarray([row["unit_index"] for row in flat], dtype=np.int16),
        "unit_id": np.asarray([ordered_units[row["unit_index"]] for row in flat], dtype="<U32"),
        "target_id": np.asarray([c75_arrays["target_id"][row["unit_index"]] for row in flat], dtype=np.int16),
        "trajectory_id": np.asarray([c75_arrays["trajectory_id"][row["unit_index"]] for row in flat], dtype="<U80"),
        "orbit": np.asarray([row["orbit"] for row in flat], dtype="<U2"),
        "orbit_name": np.asarray([row["orbit_name"] for row in flat], dtype="<U64"),
        "family": np.asarray([row["family"] for row in flat], dtype="<U24"),
        "scope": np.asarray([row["scope"] for row in flat], dtype="<U12"),
        "replicate": np.asarray([row["replicate"] for row in flat], dtype=np.int16),
        "condition_number": np.asarray([row["condition_number"] for row in flat], dtype=float),
        "transform_hash": np.asarray([row["transform_hash"] for row in flat], dtype="<U64"),
        "F2": np.stack([row["F2"] for row in flat]),
        "F4": np.stack([row["F4"] for row in flat]),
        "G4S": np.stack([row["G4S"] for row in flat]),
        "G4T": np.stack([row["G4T"] for row in flat]),
        "projection_max_abs_error": np.asarray([row["projection_max_abs_error"] for row in flat], dtype=float),
        "probability_max_abs_error": np.asarray([row["probability_max_abs_error"] for row in flat], dtype=float),
        "prediction_disagreements": np.asarray([row["prediction_disagreements"] for row in flat], dtype=np.int16),
    }
    root = run_root(protocol)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = c74_cache.write_content_addressed_npz(root, "c76_orbit_feature_cache", arrays)
    payload = c74_cache.self_hashed_manifest({
        "schema_version": "c76_orbit_feature_cache_manifest_v1",
        "protocol_sha256": c76_protocol.sha256(c76_protocol.PROTOCOL_PATH),
        "parent_C75_feature_manifest_sha256": c76_protocol.sha256(c75_data.feature_manifest_path(json.loads(c75_protocol.PROTOCOL_PATH.read_text()))),
        "C75_feature_payload_sha256": c75_manifest["descriptor"]["sha256"],
        "unit_count": 216, "target_count": 9,
        "orbit_variant_count": len(orbit_variants()), "orbit_row_count": len(flat),
        "C74_descriptors_rehashed": 216 * 5,
        "functional_identity_max_abs": max_projection,
        "probability_identity_max_abs": max_probability,
        "prediction_disagreements": disagreements,
        "transform_scope_hash_failures": global_hash_failures,
        "T3_HO_z_Wz_accessed": False, "same_label_oracle_accessed": False,
        "forward_passes": 0, "training": 0, "GPU": False,
        "descriptor": descriptor,
    })
    path = orbit_manifest_path(protocol)
    c74_cache.atomic_json(path, payload)
    return {"orbit_manifest_path": str(path), **payload}


def load_orbit_cache() -> tuple[dict, dict[str, np.ndarray]]:
    protocol = load_protocol()
    manifest = c74_cache.verify_unit_manifest(orbit_manifest_path(protocol), rehash_payloads=False)
    c74_cache.verify_shard(manifest["descriptor"])
    with np.load(manifest["descriptor"]["path"], allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    if manifest["unit_count"] != 216 or manifest["T3_HO_z_Wz_accessed"]:
        raise RuntimeError("C76 orbit cache contract failed")
    return manifest, arrays


if __name__ == "__main__":
    print(json.dumps(extract_orbit_cache(), indent=2, sort_keys=True))
