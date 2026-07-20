"""Independent, read-only acceptance replay for the frozen C85U field.

This module deliberately does not import the C85U writer or historical-replay
implementation. It derives all public evidence from the persisted bytes. The
public certificate contains identities, counts, replay maxima, and status only.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
C85U_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
    "c85u-v2-77382c16a593f7c2-91a428488a634268"
)
SELECTION_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/"
    "stage_b_selection_freeze"
)
RESULT_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5/"
    "stage_c_scientific_result"
)

EXPECTED = {
    "lock_sha256": "77382c16a593f7c2bdeb4dcacdfa21df11dcfd59982e9bfb982d6b88f5f04d1d",
    "lock_commit": "672670d05e9d7adfbe12673d4a64bfd499413162",
    "authorization_sha256": "024d95b6364651d6faa7b7cbeb5e0a1d896fe56e122d3b4ad2d6ba284ac1b6db",
    "consumption_sha256": "f2ae41730a005d5622280ad7617efcd198ab308805604894cddb25d8eb5726b9",
    "protected_replay_sha256": "9013c5223bf271edcefe477443add3c1404381d5bf9c884b8d283ac3bc94651e",
    "u1_manifest_sha256": "95bdbc04f05103a090d46dd4419dc12c766ab45f807c8466ebf883a1171b05c6",
    "u1_handoff_sha256": "2fad474bff7d80c55243a825ae3c15fa0afab45e8c841343d2a0aa20cefe1975",
    "index_sha256": "83bddf56290c4e06a306d64dadfc9611115a177f479d433fe0e4485b0c181509",
    "u2_result_sha256": "84177e80c9883611ef0bc0e9d27a4c38867a45db9b0458d7b090c422b23c39be",
    "u2_handoff_sha256": "bb0b7bbad5b198eefdfc4a38f1bf353b96b60cecdcbf47815f2ff8ccbda789dc",
    "result_sha256": "d19b11c24a811c1e8677cc0681d3d57bcb437a1d43702a5df8b2e1c92d43f83c",
    "acceptance_manifest_sha256": "dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620",
    "completion_sha256": "5d8bdc9888106f6382531f52150613744cce3e15fbc73e0b557b3eaa89e7a129",
    "lifecycle_sha256": "c7ade7f29723fdcaa4f4472e2b431a1ec3c581e07f157f1c4d1d49d60125b2b7",
    "selection_manifest_sha256": "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4",
    "candidate_ranks_sha256": "b0b16731e80f7c1f7c11747b7bfb52b5da4c84861421104092542f533854214f",
    "fixed_actions_sha256": "ab3662e62d7eaf2f4962fe720f3710f06c648ee7701811137ee17fe3977b0bc8",
    "q0_index_sha256": "e76146401a0047db527cdf373e2211943cca5f5a4fca59ba150ef6c2b83a45d6",
    "result_manifest_sha256": "516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5",
    "method_context_sha256": "984d72201c2224fdaff600b0492fe507f68773b228a5e75dfd70130c0afa13d6",
}
EXPECTED_CONTEXTS = 944
EXPECTED_CANDIDATE_ROWS = 76_464
EXPECTED_METHOD_ROWS = 18_432
EXPECTED_Q0_SHARDS = 944
EXPECTED_FINITE_Q0_RECORDS = 8_749_056
EXPECTED_U1_BYTES = 44_003_342
CHAINS = 2_048
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

SCORE_METHODS = ("S1", "U5", "U7", "U11", "U13", "U14", "U15")
FIXED_METHODS = ("B1", "B2", "B3", "B4O", "B4S")
COMMON_METHODS = (
    "B0", *FIXED_METHODS, "B5", *SCORE_METHODS,
    "Q0_B1", "Q0_B2", "Q0_B4", "Q0_B8", "Q0_FULL",
)
INDEX_FIELDS = (
    "context_id", "dataset", "target_subject_id", "panel", "training_seed",
    "level", "candidate_index", "candidate_id", "regime",
    "trajectory_order", "epoch", "evaluation_trial_count",
    "balanced_accuracy", "NLL", "ECE", "bAcc_midrank_percentile",
    "negative_NLL_midrank_percentile", "negative_ECE_midrank_percentile",
    "composite_utility", "utility_rank_midrank",
    "canonical_utility_order_position", "standardized_regret",
    "is_canonical_best", "is_in_canonical_top5", "is_in_canonical_top10",
    "context_artifact_path", "context_artifact_sha256",
)
CONTEXT_SCALARS = {
    "schema_version", "context_id", "dataset", "target_subject_id", "panel",
    "training_seed", "level", "evaluation_trial_count",
    "candidate_id_order_sha256", "evaluation_trial_id_sha256",
    "evaluation_label_view_manifest_sha256", "target_artifact_input_sha256",
    "metric_matrix_sha256", "utility_vector_sha256", "best_candidate_id",
    "best_candidate_index", "utility_min", "utility_max", "utility_spread",
    "exact_comaximizer_count",
}
CONTEXT_VECTORS = {
    "candidate_index", "candidate_id", "regime", "trajectory_order", "epoch",
    "target_artifact_sha256", "balanced_accuracy", "NLL", "ECE",
    "bAcc_midrank_percentile", "negative_NLL_midrank_percentile",
    "negative_ECE_midrank_percentile", "composite_utility",
    "utility_rank_midrank", "canonical_utility_order_position",
    "standardized_regret", "is_canonical_best", "is_in_canonical_top5",
    "is_in_canonical_top10",
}
FLOAT_VECTORS = {
    "balanced_accuracy", "NLL", "ECE", "bAcc_midrank_percentile",
    "negative_NLL_midrank_percentile", "negative_ECE_midrank_percentile",
    "composite_utility", "utility_rank_midrank", "standardized_regret",
}
LIFECYCLE_STAGES = (
    "PREFLIGHT_STARTED", "PREFLIGHT_COMPLETED", "AUTHORIZATION_CONSUMED",
    "PROTECTED_INPUT_REPLAY_STARTED", "PROTECTED_INPUT_REPLAY_COMPLETED",
    "STAGE_U1_STARTED", "STAGE_U1_COMPLETED", "STAGE_U2_STARTED",
    "STAGE_U2_COMPLETED", "ACCEPTANCE_MANIFEST_STARTED",
    "ACCEPTANCE_MANIFEST_COMPLETED", "ATOMIC_ACCEPTANCE_COMMIT_READY",
)


class C85EP2ReplayError(RuntimeError):
    """Raised when accepted C85U bytes fail independent replay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C85EP2ReplayError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_sha(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    _require(path.is_file(), f"missing JSON: {path}")
    if expected_sha is not None:
        _require(_sha256_file(path) == expected_sha, f"SHA-256 drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _scalar(payload: Mapping[str, np.ndarray], field: str) -> Any:
    value = np.asarray(payload[field])
    _require(value.shape == (), f"scalar shape drift: {field}")
    return value.item()


def _midrank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 1 and np.all(np.isfinite(values)), "midrank input drift")
    if len(values) <= 1:
        return np.ones_like(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return (ranks - 1.0) / (len(values) - 1.0)


def _expected_methods(dataset: str) -> tuple[str, ...]:
    if dataset == "Lee2019_MI":
        return COMMON_METHODS + ("Q0_B16",)
    if dataset == "Cho2017":
        return COMMON_METHODS + ("Q0_B16", "Q0_B32")
    _require(dataset == "PhysionetMI", f"unknown dataset: {dataset}")
    return COMMON_METHODS


def _finite_budgets(dataset: str) -> tuple[int, ...]:
    if dataset == "Lee2019_MI":
        return (1, 2, 4, 8, 16)
    if dataset == "Cho2017":
        return (1, 2, 4, 8, 16, 32)
    _require(dataset == "PhysionetMI", f"unknown dataset: {dataset}")
    return (1, 2, 4, 8)


@dataclass(frozen=True)
class _Context:
    context_id: str
    dataset: str
    target_subject_id: str
    panel: str
    training_seed: int
    level: int
    candidate_ids: tuple[str, ...]
    regimes: tuple[str, ...]
    utility: np.ndarray

    @property
    def key_prefix(self) -> tuple[str, str, str, int, int]:
        return (
            self.dataset, self.target_subject_id, self.panel,
            self.training_seed, self.level,
        )


def _replay_context_payload(
    path: Path, row: Mapping[str, Any],
) -> tuple[_Context, dict[str, Any], dict[str, np.ndarray]]:
    _require(path.is_file(), f"context artifact absent: {path}")
    artifact_sha = _sha256_file(path)
    _require(artifact_sha == str(row["sha256"]), "context artifact SHA drift")
    _require(path.stat().st_size == int(row["bytes"]), "context artifact size drift")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    _require(set(payload) == CONTEXT_SCALARS | CONTEXT_VECTORS,
             "context payload field-set drift")
    _require(str(_scalar(payload, "schema_version")) == "c85u_candidate_utility_context_v1",
             "context payload schema drift")
    for field, value in payload.items():
        array = np.asarray(value)
        _require(not array.dtype.hasobject, f"object dtype forbidden: {field}")
        if np.issubdtype(array.dtype, np.number):
            _require(np.all(np.isfinite(array)), f"nonfinite field: {field}")
    for field in CONTEXT_VECTORS:
        _require(np.asarray(payload[field]).shape == (81,), f"vector shape drift: {field}")
    for field in FLOAT_VECTORS:
        _require(np.asarray(payload[field]).dtype == np.dtype("<f8"),
                 f"float dtype drift: {field}")
    exact_dtypes = {
        "candidate_index": "<i2", "trajectory_order": "<i2", "epoch": "<i2",
        "canonical_utility_order_position": "<i2", "best_candidate_index": "<i2",
        "exact_comaximizer_count": "<i2", "training_seed": "<i8", "level": "<i8",
        "evaluation_trial_count": "<i8", "is_canonical_best": "u1",
        "is_in_canonical_top5": "u1", "is_in_canonical_top10": "u1",
    }
    for field, dtype in exact_dtypes.items():
        _require(np.asarray(payload[field]).dtype == np.dtype(dtype),
                 f"exact dtype drift: {field}")

    candidate_index = np.asarray(payload["candidate_index"], dtype=np.int64)
    candidate_ids = tuple(map(str, np.asarray(payload["candidate_id"]).tolist()))
    target_hashes = tuple(map(str, np.asarray(payload["target_artifact_sha256"]).tolist()))
    _require(np.array_equal(candidate_index, np.arange(81)), "candidate index drift")
    _require(len(set(candidate_ids)) == 81, "candidate ID uniqueness drift")
    _require(all(SHA_RE.fullmatch(value) for value in target_hashes),
             "target artifact SHA shape drift")
    order_sha = hashlib.sha256("\n".join(candidate_ids).encode("ascii")).hexdigest()
    _require(str(_scalar(payload, "candidate_id_order_sha256")) == order_sha,
             "candidate order digest drift")
    target_input_sha = _canonical_sha([
        {"unit_id": unit_id, "sha256": sha}
        for unit_id, sha in zip(candidate_ids, target_hashes, strict=True)
    ])
    _require(str(_scalar(payload, "target_artifact_input_sha256")) == target_input_sha,
             "target input digest drift")
    for field in (
        "evaluation_trial_id_sha256", "evaluation_label_view_manifest_sha256",
        "metric_matrix_sha256", "utility_vector_sha256",
    ):
        _require(SHA_RE.fullmatch(str(_scalar(payload, field))) is not None,
                 f"SHA field drift: {field}")

    metrics = np.column_stack((
        payload["balanced_accuracy"], payload["NLL"], payload["ECE"],
    )).astype("<f8", copy=False)
    expected_oriented = np.column_stack((
        _midrank_percentile(metrics[:, 0]),
        _midrank_percentile(-metrics[:, 1]),
        _midrank_percentile(-metrics[:, 2]),
    )).astype("<f8", copy=False)
    observed_oriented = np.column_stack((
        payload["bAcc_midrank_percentile"],
        payload["negative_NLL_midrank_percentile"],
        payload["negative_ECE_midrank_percentile"],
    )).astype("<f8", copy=False)
    utility = np.asarray(payload["composite_utility"], dtype="<f8")
    utility_expected = np.mean(expected_oriented, axis=1)
    midrank_error = float(np.max(np.abs(observed_oriented - expected_oriented)))
    utility_error = float(np.max(np.abs(utility - utility_expected)))
    _require(midrank_error == 0.0, "oriented midrank replay mismatch")
    _require(utility_error <= 1e-12, "composite utility replay mismatch")
    _require(str(_scalar(payload, "metric_matrix_sha256")) == _array_sha(metrics),
             "metric matrix digest drift")
    _require(str(_scalar(payload, "utility_vector_sha256")) == _array_sha(utility),
             "utility vector digest drift")

    order = np.lexsort((np.arange(81), -utility))
    positions = np.empty(81, dtype=np.int16)
    positions[order] = np.arange(1, 82, dtype=np.int16)
    spread = float(np.max(utility) - np.min(utility))
    regret = (
        np.zeros(81, dtype=np.float64)
        if spread <= 1e-15 else (np.max(utility) - utility) / spread
    )
    best = int(order[0])
    expected_best = (np.arange(81) == best).astype(np.uint8)
    expected_top5 = np.isin(np.arange(81), order[:5]).astype(np.uint8)
    expected_top10 = np.isin(np.arange(81), order[:10]).astype(np.uint8)
    utility_rank = _midrank_percentile(utility)
    regret_error = float(np.max(np.abs(regret - payload["standardized_regret"])))
    rank_error = float(np.max(np.abs(utility_rank - payload["utility_rank_midrank"])))
    _require(regret_error <= 1e-12 and rank_error == 0.0, "rank/regret replay mismatch")
    _require(np.array_equal(positions, payload["canonical_utility_order_position"]),
             "canonical order position drift")
    _require(np.array_equal(expected_best, payload["is_canonical_best"]),
             "canonical best flag drift")
    _require(np.array_equal(expected_top5, payload["is_in_canonical_top5"]),
             "canonical top5 flag drift")
    _require(np.array_equal(expected_top10, payload["is_in_canonical_top10"]),
             "canonical top10 flag drift")
    _require(int(_scalar(payload, "best_candidate_index")) == best and
             str(_scalar(payload, "best_candidate_id")) == candidate_ids[best],
             "best candidate identity drift")
    _require(float(_scalar(payload, "utility_min")) == float(np.min(utility)) and
             float(_scalar(payload, "utility_max")) == float(np.max(utility)) and
             float(_scalar(payload, "utility_spread")) == spread,
             "utility scalar replay drift")
    _require(int(_scalar(payload, "exact_comaximizer_count")) ==
             int(np.sum(utility == np.max(utility))), "co-maximizer count drift")
    _require(str(_scalar(payload, "context_id")) == str(row["context_id"]),
             "context manifest identity drift")
    _require(str(_scalar(payload, "metric_matrix_sha256")) == str(row["metric_matrix_sha256"]) and
             str(_scalar(payload, "utility_vector_sha256")) == str(row["utility_vector_sha256"]),
             "context manifest semantic digest drift")
    context = _Context(
        context_id=str(_scalar(payload, "context_id")),
        dataset=str(_scalar(payload, "dataset")),
        target_subject_id=str(_scalar(payload, "target_subject_id")),
        panel=str(_scalar(payload, "panel")),
        training_seed=int(_scalar(payload, "training_seed")),
        level=int(_scalar(payload, "level")),
        candidate_ids=candidate_ids,
        regimes=tuple(map(str, np.asarray(payload["regime"]).tolist())),
        utility=utility.copy(),
    )
    replay = {
        "artifact_sha256": artifact_sha,
        "artifact_bytes": path.stat().st_size,
        "midrank_max_abs": midrank_error,
        "utility_max_abs": utility_error,
        "regret_max_abs": regret_error,
        "rank_max_abs": rank_error,
    }
    return context, replay, payload


def _replay_u1(root: Path) -> tuple[dict[str, Any], dict[str, _Context]]:
    u1 = root / "stage_u1_candidate_utility_v2"
    manifest_path = u1 / "C85U_CANDIDATE_UTILITY_MANIFEST.json"
    sidecar = (u1 / "C85U_CANDIDATE_UTILITY_MANIFEST.sha256").read_text(encoding="ascii").split()[0]
    _require(sidecar == _sha256_file(manifest_path), "U1 compatibility manifest sidecar drift")
    manifest = _load_json(manifest_path)
    v2_path = u1 / "C85U_CANDIDATE_UTILITY_MANIFEST_V2.json"
    v2 = _load_json(v2_path, EXPECTED["u1_manifest_sha256"])
    handoff = _load_json(u1 / "C85U_STAGE_U1_HANDOFF.json", EXPECTED["u1_handoff_sha256"])
    _require(manifest.get("schema_version") == "c85u_complete_utility_manifest_v1" and
             manifest.get("status") == "COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN",
             "U1 compatibility manifest state drift")
    _require(v2.get("schema_version") == "c85u_complete_utility_manifest_v2" and
             handoff.get("schema_version") == "c85u_stage_u1_handoff_v2",
             "U1 V2 schema drift")
    for value in (v2, handoff):
        _require(value.get("execution_lock_sha256") == EXPECTED["lock_sha256"],
                 "U1 lock binding drift")
        _require(value.get("attempt_id") == handoff.get("attempt_id") and
                 value.get("authorization_binding_sha256") == handoff.get("authorization_binding_sha256"),
                 "U1 attempt binding drift")
    _require(handoff.get("U1_manifest_sha256") == EXPECTED["u1_manifest_sha256"],
             "U1 handoff manifest linkage drift")
    _require(v2.get("protected_replay_sha256") == EXPECTED["protected_replay_sha256"] and
             handoff.get("protected_replay_sha256") == EXPECTED["protected_replay_sha256"],
             "U1 protected replay linkage drift")

    artifacts = manifest.get("context_artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == EXPECTED_CONTEXTS,
             "U1 context artifact count drift")
    contexts: dict[str, _Context] = {}
    payloads: dict[str, tuple[dict[str, np.ndarray], Mapping[str, Any]]] = {}
    totals = {"artifact_bytes": 0, "midrank": 0.0, "utility": 0.0, "regret": 0.0, "rank": 0.0}
    registry_rows: list[dict[str, str]] = []
    for row in artifacts:
        context, replay, payload = _replay_context_payload(u1 / str(row["path"]), row)
        _require(context.context_id not in contexts, "duplicate U1 context")
        contexts[context.context_id] = context
        payloads[context.context_id] = (payload, row)
        totals["artifact_bytes"] += int(replay["artifact_bytes"])
        for public, source in (("midrank", "midrank_max_abs"), ("utility", "utility_max_abs"),
                               ("regret", "regret_max_abs"), ("rank", "rank_max_abs")):
            totals[public] = max(float(totals[public]), float(replay[source]))
        registry_rows.append({
            "context_id": context.context_id,
            "sha256": str(row["sha256"]),
            "metric_matrix_sha256": str(row["metric_matrix_sha256"]),
            "utility_vector_sha256": str(row["utility_vector_sha256"]),
        })
    _require(_canonical_sha(registry_rows) == manifest.get("context_registry_sha256"),
             "U1 context registry digest drift")

    index_meta = manifest.get("candidate_utility_index")
    _require(isinstance(index_meta, dict), "U1 candidate index metadata missing")
    index_path = u1 / str(index_meta["path"])
    _require(_sha256_file(index_path) == EXPECTED["index_sha256"] == index_meta.get("sha256"),
             "U1 candidate index SHA drift")
    candidate_keys: set[tuple[str, int]] = set()
    index_rows = 0
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == INDEX_FIELDS, "U1 index field-order drift")
        for row in reader:
            context_id = str(row["context_id"])
            candidate = int(row["candidate_index"])
            key = (context_id, candidate)
            _require(context_id in payloads and 0 <= candidate < 81 and key not in candidate_keys,
                     "U1 index candidate identity drift")
            candidate_keys.add(key)
            payload, artifact = payloads[context_id]
            scalar_fields = (
                "dataset", "target_subject_id", "panel", "training_seed", "level",
                "evaluation_trial_count",
            )
            _require(all(str(row[field]) == str(_scalar(payload, field)) for field in scalar_fields),
                     "U1 index scalar linkage drift")
            for field in ("candidate_id", "regime"):
                _require(str(row[field]) == str(payload[field][candidate]),
                         f"U1 index string linkage drift: {field}")
            for field in (
                "trajectory_order", "epoch", "canonical_utility_order_position",
                "is_canonical_best", "is_in_canonical_top5", "is_in_canonical_top10",
            ):
                _require(int(row[field]) == int(payload[field][candidate]),
                         f"U1 index integer linkage drift: {field}")
            for field in FLOAT_VECTORS:
                _require(float(row[field]) == float(payload[field][candidate]),
                         f"U1 index float linkage drift: {field}")
            _require(row["context_artifact_path"] == str(artifact["path"]) and
                     row["context_artifact_sha256"] == str(artifact["sha256"]),
                     "U1 index artifact linkage drift")
            index_rows += 1
    _require(index_rows == EXPECTED_CANDIDATE_ROWS and len(candidate_keys) == index_rows,
             "U1 candidate index coverage drift")
    _require(manifest.get("contexts") == EXPECTED_CONTEXTS and
             manifest.get("candidate_rows") == EXPECTED_CANDIDATE_ROWS and
             v2.get("contexts") == EXPECTED_CONTEXTS and
             v2.get("candidate_rows") == EXPECTED_CANDIDATE_ROWS,
             "U1 manifest arithmetic drift")
    _require(all(int(value) == 0 for value in manifest["protected_counters"].values()) and
             all(int(value) == 0 for value in manifest["forbidden_output_payloads"].values()) and
             all(int(value) == 0 for value in v2["forbidden_access_counters"].values()),
             "U1 forbidden/protected counter nonzero")
    actual_tree_bytes = sum(path.stat().st_size for path in u1.rglob("*") if path.is_file())
    _require(actual_tree_bytes == EXPECTED_U1_BYTES == int(v2["actual_total_output_bytes"]),
             "U1 complete byte total drift")
    return ({
        "status": "PASS",
        "contexts": len(contexts),
        "candidate_rows": index_rows,
        "context_artifacts": len(artifacts),
        "complete_tree_bytes": actual_tree_bytes,
        "context_artifact_bytes": int(totals["artifact_bytes"]),
        "candidate_index_sha256": EXPECTED["index_sha256"],
        "compatibility_manifest_sha256": sidecar,
        "v2_manifest_sha256": EXPECTED["u1_manifest_sha256"],
        "handoff_sha256": EXPECTED["u1_handoff_sha256"],
        "maximum_midrank_replay_error": totals["midrank"],
        "maximum_utility_replay_error": totals["utility"],
        "maximum_regret_replay_error": totals["regret"],
        "maximum_rank_replay_error": totals["rank"],
        "forbidden_payload_fields": 0,
    }, contexts)


def _load_score_orders(contexts: Mapping[str, _Context]) -> dict[str, dict[str, np.ndarray]]:
    path = SELECTION_ROOT / "candidate_ranks.csv"
    _require(_sha256_file(path) == EXPECTED["candidate_ranks_sha256"], "candidate ranks SHA drift")
    orders = {
        context_id: {method: np.full(81, -1, dtype=np.int16) for method in SCORE_METHODS}
        for context_id in contexts
    }
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context_id = row["context_id"]
            method = row["method_id"]
            candidate = int(row["candidate_index"])
            rank = int(row["rank"])
            _require(context_id in contexts and method in SCORE_METHODS and
                     0 <= candidate < 81 and 1 <= rank <= 81,
                     "candidate rank identity/range drift")
            _require(row["candidate_id"] == contexts[context_id].candidate_ids[candidate],
                     "candidate rank candidate linkage drift")
            _require(orders[context_id][method][rank - 1] == -1, "duplicate candidate rank")
            orders[context_id][method][rank - 1] = candidate
            rows += 1
    _require(rows == EXPECTED_CONTEXTS * len(SCORE_METHODS) * 81,
             "candidate rank row coverage drift")
    _require(all(np.array_equal(np.sort(order), np.arange(81))
                 for methods in orders.values() for order in methods.values()),
             "candidate rank order is not a permutation")
    return orders


def _load_fixed_actions(contexts: Mapping[str, _Context]) -> dict[str, dict[str, int]]:
    path = SELECTION_ROOT / "fixed_default_selections.csv"
    _require(_sha256_file(path) == EXPECTED["fixed_actions_sha256"], "fixed action SHA drift")
    fixed = {context_id: {} for context_id in contexts}
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context_id = row["context_id"]
            method = row["method_id"]
            candidate = int(row["selected_candidate_index"])
            _require(context_id in contexts and method in FIXED_METHODS and
                     method not in fixed[context_id] and 0 <= candidate < 81,
                     "fixed action identity/range drift")
            _require(row["selected_candidate_id"] == contexts[context_id].candidate_ids[candidate],
                     "fixed action candidate linkage drift")
            fixed[context_id][method] = candidate
            rows += 1
    _require(rows == EXPECTED_CONTEXTS * len(FIXED_METHODS) and
             all(set(value) == set(FIXED_METHODS) for value in fixed.values()),
             "fixed action coverage drift")
    return fixed


def _fixed_endpoint(context: _Context, selected: int) -> dict[str, Any]:
    utility = context.utility
    best = int(np.lexsort((np.arange(81), -utility))[0])
    spread = float(np.max(utility) - np.min(utility))
    regret = 0.0 if spread <= 1e-15 else float((np.max(utility) - utility[selected]) / spread)
    hit = float(selected == best)
    return {
        "selected_utility": float(utility[selected]), "standardized_regret": regret,
        "top1": hit, "top5": hit, "top10": hit,
        "selected_regime": context.regimes[selected],
    }


def _order_endpoint(context: _Context, order: np.ndarray) -> dict[str, Any]:
    selected = int(order[0])
    best = int(np.lexsort((np.arange(81), -context.utility))[0])
    endpoint = _fixed_endpoint(context, selected)
    endpoint.update({
        "top1": float(np.any(order[:1] == best)),
        "top5": float(np.any(order[:5] == best)),
        "top10": float(np.any(order[:10] == best)),
    })
    return endpoint


def _q0_endpoint(context: _Context, orders: np.ndarray) -> dict[str, Any]:
    utility = context.utility
    best = int(np.lexsort((np.arange(81), -utility))[0])
    selected = orders[:, 0].astype(np.int64)
    selected_utility = utility[selected]
    spread = float(np.max(utility) - np.min(utility))
    regrets = np.zeros(len(orders)) if spread <= 1e-15 else (
        np.max(utility) - selected_utility
    ) / spread
    regimes = np.asarray(context.regimes)[selected]
    return {
        "selected_utility": float(np.mean(selected_utility)),
        "standardized_regret": float(np.mean(regrets)),
        "top1": float(np.mean(np.any(orders[:, :1] == best, axis=1))),
        "top5": float(np.mean(np.any(orders[:, :5] == best, axis=1))),
        "top10": float(np.mean(np.any(orders[:, :10] == best, axis=1))),
        "selected_regime": str(regimes[0]) if len(orders) == 1 else "STOCHASTIC_Q0",
    }


def _q0_payload(path: Path, expected_sha: str, context: _Context) -> dict[str, np.ndarray]:
    _require(_sha256_file(path) == expected_sha, "Q0 shard SHA drift")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "schema_version", "dataset", "target_subject_id", "panel", "training_seed",
        "level", "context_id", "candidate_ids", "finite_chain", "finite_chain_seed",
        "finite_budget_code", "finite_sample_size", "finite_sample_digest",
        "finite_selected_index", "finite_candidate_order", "finite_candidate_score_digest",
        "finite_construction_metric_digest", "FULL_sample_size", "FULL_sample_digest",
        "FULL_selected_index", "FULL_candidate_order", "FULL_candidate_score_digest",
        "FULL_construction_metric_digest",
    }
    _require(set(payload) == required, "Q0 shard field-set drift")
    text = lambda field: np.asarray(payload[field]).item().decode("ascii")
    _require(text("schema_version") == "c84sr3_q0_context_shard_v2" and
             text("context_id") == context.context_id and text("dataset") == context.dataset and
             text("target_subject_id") == context.target_subject_id and
             text("panel") == context.panel and
             int(np.asarray(payload["training_seed"]).item()) == context.training_seed and
             int(np.asarray(payload["level"]).item()) == context.level,
             "Q0 shard context identity drift")
    candidate_ids = tuple(value.decode("ascii") for value in payload["candidate_ids"].tolist())
    _require(candidate_ids == context.candidate_ids, "Q0 candidate identity/order drift")
    budgets = _finite_budgets(context.dataset)
    count = len(budgets) * CHAINS
    orders = np.asarray(payload["finite_candidate_order"])
    _require(orders.dtype == np.uint8 and orders.shape == (count, 81),
             "Q0 finite order shape/dtype drift")
    _require(np.all(np.sort(orders, axis=1) == np.arange(81, dtype=np.uint8)),
             "Q0 finite order is not a permutation")
    _require(np.array_equal(payload["finite_selected_index"], orders[:, 0]),
             "Q0 finite selected/order mismatch")
    codes = np.asarray(payload["finite_budget_code"])
    chains = np.asarray(payload["finite_chain"])
    samples = np.asarray(payload["finite_sample_size"])
    for budget in budgets:
        mask = codes == budget
        _require(np.array_equal(np.sort(chains[mask]), np.arange(CHAINS, dtype=np.uint16)) and
                 np.all(samples[mask] == 2 * budget),
                 f"Q0 chain/sample coverage drift: {context.dataset}/B{budget}")
    full = np.asarray(payload["FULL_candidate_order"])
    _require(full.dtype == np.uint8 and full.shape == (1, 81) and
             np.array_equal(np.sort(full[0]), np.arange(81, dtype=np.uint8)) and
             int(payload["FULL_selected_index"][0]) == int(full[0, 0]),
             "Q0 FULL order drift")
    return payload


def _replay_u2(root: Path, contexts: Mapping[str, _Context]) -> dict[str, Any]:
    selection_manifest = _load_json(
        SELECTION_ROOT / "C84S_SELECTION_FREEZE_MANIFEST_V3.json",
        EXPECTED["selection_manifest_sha256"],
    )
    _require(selection_manifest.get("contexts") == EXPECTED_CONTEXTS and
             selection_manifest.get("Q0_records") == 8_750_000 and
             selection_manifest.get("evaluation_label_descriptor_received") is False,
             "selection manifest state drift")
    orders = _load_score_orders(contexts)
    fixed = _load_fixed_actions(contexts)
    q0_index_path = SELECTION_ROOT / "q0_selection_shard_index.csv"
    _require(_sha256_file(q0_index_path) == EXPECTED["q0_index_sha256"], "Q0 index SHA drift")
    shards: dict[str, dict[str, str]] = {}
    with q0_index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            _require(row["context_id"] in contexts and row["context_id"] not in shards,
                     "Q0 index context drift")
            shards[row["context_id"]] = dict(row)
    _require(set(shards) == set(contexts), "Q0 shard index coverage drift")

    method_path = RESULT_ROOT / "method_context_decisions.csv"
    _require(_sha256_file(method_path) == EXPECTED["method_context_sha256"],
             "historical method-context SHA drift")
    historical: dict[tuple[str, str, str, int, int, str], dict[str, str]] = {}
    with method_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["dataset"], row["target_subject_id"], row["panel"],
                int(row["training_seed"]), int(row["level"]), row["method_id"],
            )
            _require(key not in historical, "duplicate historical method-context row")
            historical[key] = dict(row)
    _require(len(historical) == EXPECTED_METHOD_ROWS, "historical method-context row drift")

    maxima = {field: 0.0 for field in (
        "selected_utility", "standardized_regret", "top1", "top5", "top10",
    )}
    compared = 0
    finite_records = 0
    regime_mismatches = 0
    context_replay_rows: list[dict[str, Any]] = []
    for context_id in sorted(contexts):
        context = contexts[context_id]
        row = shards[context_id]
        q0 = _q0_payload(SELECTION_ROOT / row["path"], row["sha256"], context)
        endpoints: dict[str, dict[str, Any]] = {}
        utility = context.utility
        spread = float(np.max(utility) - np.min(utility))
        regrets = np.zeros(81) if spread <= 1e-15 else (np.max(utility) - utility) / spread
        endpoints["B0"] = {
            "selected_utility": float(np.mean(utility)),
            "standardized_regret": float(np.mean(regrets)),
            "top1": 1.0 / 81.0, "top5": 5.0 / 81.0, "top10": 10.0 / 81.0,
            "selected_regime": "ANALYTIC_UNIFORM_RANDOM",
        }
        best = int(np.lexsort((np.arange(81), -utility))[0])
        endpoints["B5"] = {
            "selected_utility": float(utility[best]), "standardized_regret": 0.0,
            "top1": 1.0, "top5": 1.0, "top10": 1.0,
            "selected_regime": context.regimes[best],
        }
        for method, selected in fixed[context_id].items():
            endpoints[method] = _fixed_endpoint(context, selected)
        for method, order in orders[context_id].items():
            endpoints[method] = _order_endpoint(context, order)
        codes = np.asarray(q0["finite_budget_code"])
        finite_orders = np.asarray(q0["finite_candidate_order"])
        for budget in _finite_budgets(context.dataset):
            selected_orders = finite_orders[codes == budget]
            _require(len(selected_orders) == CHAINS, "Q0 finite budget row drift")
            endpoints[f"Q0_B{budget}"] = _q0_endpoint(context, selected_orders)
        endpoints["Q0_FULL"] = _q0_endpoint(context, np.asarray(q0["FULL_candidate_order"]))
        _require(set(endpoints) == set(_expected_methods(context.dataset)),
                 "method set replay drift")
        for method, observed in endpoints.items():
            key = (*context.key_prefix, method)
            _require(key in historical, "historical method row absent")
            expected = historical[key]
            for field in maxima:
                difference = abs(float(observed[field]) - float(expected[field]))
                maxima[field] = max(maxima[field], difference)
                _require(difference <= 1e-12, f"historical endpoint mismatch: {method}/{field}")
            if str(observed["selected_regime"]) != str(expected["selected_regime"]):
                regime_mismatches += 1
            compared += 1
        finite_records += len(codes)
        context_replay_rows.append({
            "context_id": context_id,
            "method_count": len(endpoints),
            "utility_vector_sha256": _array_sha(utility),
            "q0_shard_sha256": row["sha256"],
        })
    _require(compared == EXPECTED_METHOD_ROWS and finite_records == EXPECTED_FINITE_Q0_RECORDS,
             "U2 replay arithmetic drift")
    _require(regime_mismatches == 0 and all(value == 0.0 for value in maxima.values()),
             "U2 endpoint replay is not byte-equivalent")
    u2 = root / "stage_u2_historical_replay_v2"
    result = _load_json(u2 / "C85U_HISTORICAL_DECISION_REPLAY_V2.json", EXPECTED["u2_result_sha256"])
    handoff = _load_json(u2 / "C85U_STAGE_U2_HANDOFF.json", EXPECTED["u2_handoff_sha256"])
    _require(result.get("context_replay_registry_sha256") == _canonical_sha(context_replay_rows),
             "U2 context replay registry digest drift")
    _require(result.get("maximum_absolute_differences") == maxima and
             result.get("selected_regime_mismatches") == 0 and
             result.get("finite_Q0_action_records_replayed") == finite_records and
             result.get("method_context_rows") == compared,
             "U2 persisted replay result drift")
    _require(handoff.get("U2_result_sha256") == EXPECTED["u2_result_sha256"] and
             handoff.get("U1_manifest_sha256") == EXPECTED["u1_manifest_sha256"] and
             handoff.get("U1_handoff_sha256") == EXPECTED["u1_handoff_sha256"],
             "U2 handoff linkage drift")
    return {
        "status": "PASS",
        "contexts": len(contexts),
        "method_context_rows": compared,
        "finite_Q0_action_records": finite_records,
        "Q0_shards": len(shards),
        "maximum_absolute_differences": maxima,
        "selected_regime_mismatches": regime_mismatches,
        "result_sha256": EXPECTED["u2_result_sha256"],
        "handoff_sha256": EXPECTED["u2_handoff_sha256"],
    }


def _replay_authorization_and_lifecycle(root: Path) -> dict[str, Any]:
    lock_path = REPORT_DIR / "C85U_EXECUTION_LOCK_V2.json"
    auth_path = REPORT_DIR / "C85U_V2_PI_AUTHORIZATION_RECORD.json"
    _require(_sha256_file(lock_path) == EXPECTED["lock_sha256"], "C85U lock SHA drift")
    auth = _load_json(auth_path, EXPECTED["authorization_sha256"])
    receipt_path = Path(str(auth["consumption_ledger_path"]))
    receipt = _load_json(receipt_path, EXPECTED["consumption_sha256"])
    protected = _load_json(root / "C85U_PROTECTED_INPUT_REPLAY_V2.json",
                           EXPECTED["protected_replay_sha256"])
    context = _load_json(root / "C85U_EXECUTION_CONTEXT_V2.json")
    _require(auth.get("schema_version") == "c85u_direct_pi_authorization_record_v2" and
             auth.get("direct_statement_exact") == "授权 C85U" and
             auth.get("direct_explicit_PI_authorization") is True,
             "C85U authorization record drift")
    protected_false = (
        "C85E", "C86", "active_acquisition", "real_data",
        "new_data_or_model_zoo", "manuscript",
    )
    _require(all(auth.get(field) is False for field in protected_false),
             "C85U authorization protected field drift")
    binding = {
        "authorization_id": auth["authorization_id"],
        "authorization_file_sha256": EXPECTED["authorization_sha256"],
        "execution_lock_sha256": EXPECTED["lock_sha256"],
        "execution_lock_commit": EXPECTED["lock_commit"],
        "output_root": str(root),
    }
    for value in (receipt, protected, context):
        _require(all(value.get(key) == expected for key, expected in binding.items()
                     if key in value), "C85U authorization/attempt binding drift")
    _require(receipt.get("attempt_id") == protected.get("attempt_id") == context.get("attempt_id"),
             "C85U attempt identity drift")
    _require(context.get("receipt_sha256") == EXPECTED["consumption_sha256"] and
             context.get("protected_replay_sha256") == EXPECTED["protected_replay_sha256"],
             "C85U execution context receipt linkage drift")
    _require(protected.get("target_artifact_rows") == 1944 and
             protected.get("target_sidecar_rows") == 1944 and
             protected.get("target_artifact_total_bytes") == 48_018_748_054 and
             protected.get("evaluation_label_table_rows") == 4848,
             "C85U protected replay arithmetic drift")

    auth_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(auth_path.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT, check=True, text=True, capture_output=True,
    ).stdout.strip()
    _require(auth_commit == "f4b05c3dbed962348efe9cab56374854120a3667",
             "C85U authorization commit drift")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED["lock_commit"], auth_commit],
        cwd=REPO_ROOT,
    )
    _require(ancestry.returncode == 0, "C85U lock/authorization chronology drift")

    final = root / "final_acceptance_bundle"
    lifecycle_path = final / "C85U_LIFECYCLE.jsonl"
    _require(_sha256_file(lifecycle_path) == EXPECTED["lifecycle_sha256"],
             "C85U lifecycle SHA drift")
    lifecycle = [json.loads(line) for line in lifecycle_path.read_text(encoding="utf-8").splitlines()]
    _require(len(lifecycle) == len(LIFECYCLE_STAGES) and
             tuple(row["stage"] for row in lifecycle) == LIFECYCLE_STAGES and
             [row["sequence"] for row in lifecycle] == list(range(len(LIFECYCLE_STAGES))),
             "C85U lifecycle sequence drift")
    for row in lifecycle:
        _require(row.get("attempt_id") == receipt["attempt_id"] and
                 row.get("authorization_binding_sha256") == receipt["authorization_binding_sha256"] and
                 row.get("execution_lock_sha256") == EXPECTED["lock_sha256"] and
                 row.get("output_root") == str(root),
                 "C85U lifecycle identity drift")
    _require(lifecycle[2]["artifact_or_receipt_sha256"] == EXPECTED["consumption_sha256"] and
             lifecycle[4]["artifact_or_receipt_sha256"] == EXPECTED["protected_replay_sha256"] and
             lifecycle[6]["artifact_or_receipt_sha256"] == EXPECTED["u1_handoff_sha256"] and
             lifecycle[8]["artifact_or_receipt_sha256"] == EXPECTED["u2_handoff_sha256"] and
             lifecycle[10]["artifact_or_receipt_sha256"] == EXPECTED["acceptance_manifest_sha256"] and
             lifecycle[11]["artifact_or_receipt_sha256"] == EXPECTED["completion_sha256"],
             "C85U lifecycle artifact linkage drift")
    copied = final / "authorization_consumed.json"
    _require(copied.read_bytes() == receipt_path.read_bytes(), "copied authorization receipt drift")
    stage_receipts = []
    for stage, prerequisite, expected_sha in (
        ("U1", EXPECTED["protected_replay_sha256"], lifecycle[6]["details"]["stage_receipt_sha256"]),
        ("U2", EXPECTED["u1_handoff_sha256"], lifecycle[8]["details"]["stage_receipt_sha256"]),
    ):
        path = Path(lifecycle[6 if stage == "U1" else 8]["details"]["stage_receipt_path"])
        value = _load_json(path, expected_sha)
        _require(value.get("stage") == stage and value.get("prerequisite_sha256") == prerequisite and
                 value.get("attempt_id") == receipt["attempt_id"], "stage receipt binding drift")
        stage_receipts.append({"stage": stage, "sha256": expected_sha, "status": "PASS"})
    return {
        "status": "PASS",
        "authorization_commit": auth_commit,
        "authorization_sha256": EXPECTED["authorization_sha256"],
        "consumption_receipt_sha256": EXPECTED["consumption_sha256"],
        "protected_replay_sha256": EXPECTED["protected_replay_sha256"],
        "lock_sha256": EXPECTED["lock_sha256"],
        "attempt_id": receipt["attempt_id"],
        "lifecycle_sha256": EXPECTED["lifecycle_sha256"],
        "lifecycle_events": len(lifecycle),
        "stage_receipts": stage_receipts,
    }


def _replay_final_acceptance(root: Path) -> dict[str, Any]:
    final = root / "final_acceptance_bundle"
    manifest_path = final / "C85U_RESULT_ARTIFACT_MANIFEST.json"
    manifest = _load_json(manifest_path, EXPECTED["acceptance_manifest_sha256"])
    completion = _load_json(final / "C85U_COMPLETION_RECEIPT.json", EXPECTED["completion_sha256"])
    result = _load_json(final / "C85U_EXECUTION_RESULT.json", EXPECTED["result_sha256"])
    _require(manifest.get("schema_version") == "c85u_result_artifact_manifest_v2" and
             completion.get("schema_version") == "c85u_completion_receipt_v2" and
             result.get("schema_version") == "c85u_execution_result_v2",
             "final acceptance schema drift")
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list) and manifest.get("artifact_count") == len(rows),
             "final acceptance artifact count drift")
    names: set[str] = set()
    for row in rows:
        path = final / str(row["path"])
        _require(path.is_file() and path.stat().st_size == int(row["size_bytes"]) and
                 _sha256_file(path) == row["sha256"] and row["path"] not in names,
                 "final acceptance artifact identity drift")
        names.add(str(row["path"]))
    expected_names = {
        "C85U_EXECUTION_RESULT.json", "U1_ACCEPTANCE_IDENTITY.json",
        "U2_ACCEPTANCE_IDENTITY.json", "authorization_consumed.json",
        "preflight_completed.json", "protected_input_replay_receipt.json",
    }
    _require(names == expected_names, "final acceptance artifact coverage drift")
    _require(result.get("gate") == "C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED" and
             result.get("contexts") == EXPECTED_CONTEXTS and
             result.get("candidate_utility_rows") == EXPECTED_CANDIDATE_ROWS and
             result.get("historical_method_context_rows_replayed") == EXPECTED_METHOD_ROWS and
             result.get("finite_Q0_action_records_replayed") == EXPECTED_FINITE_Q0_RECORDS,
             "final acceptance result arithmetic drift")
    _require(all(int(value) == 0 for value in result["protected_counters"].values()),
             "final acceptance protected counter nonzero")
    _require(completion.get("manifest_sha256") == EXPECTED["acceptance_manifest_sha256"] and
             completion.get("result_sha256") == EXPECTED["result_sha256"] and
             completion.get("U1_manifest_sha256") == EXPECTED["u1_manifest_sha256"] and
             completion.get("U2_result_sha256") == EXPECTED["u2_result_sha256"],
             "completion receipt linkage drift")
    staging = list(root.parent.glob(f".{root.name}.staging-*")) + list(root.glob(".*.staging-*"))
    _require(not staging, "residual C85U staging root detected")
    return {
        "status": "PASS",
        "result_sha256": EXPECTED["result_sha256"],
        "acceptance_manifest_sha256": EXPECTED["acceptance_manifest_sha256"],
        "completion_receipt_sha256": EXPECTED["completion_sha256"],
        "lifecycle_sha256": EXPECTED["lifecycle_sha256"],
        "manifest_artifacts": len(rows),
        "protected_counters_nonzero": 0,
        "residual_staging_roots": 0,
    }


def replay_c85u_acceptance(root: str | Path = C85U_ROOT) -> dict[str, Any]:
    """Perform the complete private replay and return a sanitized certificate."""
    base = Path(root).resolve()
    _require(base == C85U_ROOT.resolve(), "unbound C85U root")
    authorization = _replay_authorization_and_lifecycle(base)
    u1, contexts = _replay_u1(base)
    u2 = _replay_u2(base, contexts)
    acceptance = _replay_final_acceptance(base)
    return {
        "schema_version": "c85ep2_c85u_input_acceptance_certificate_v1",
        "status": "PASS_C85U_INPUT_ACCEPTED_FOR_C85E_LOCK_READINESS",
        "epistemic_role": "POST_C84S_EXPLORATORY_DERIVED_INPUT",
        "source_root": str(base),
        "authorization_and_lifecycle": authorization,
        "U1_utility_field_replay": u1,
        "U2_historical_endpoint_replay": u2,
        "final_acceptance_replay": acceptance,
        "public_output_excludes_scientific_values": True,
        "C84_primary_gate": "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous",
        "C84_label_frontier": "C84-L4",
        "theorem_statuses_changed": False,
        "C85E_executed": False,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    _require(bool(values), f"refusing empty evidence table: {path}")
    fields = list(values[0])
    _require(all(list(row) == fields for row in values), f"table field-order drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    return _sha256_file(path)


def write_acceptance_evidence(
    certificate: Mapping[str, Any], *, table_dir: Path, certificate_path: Path,
) -> dict[str, str]:
    """Write only sanitized identity/count/hash/pass-fail acceptance evidence."""
    auth = certificate["authorization_and_lifecycle"]
    u1 = certificate["U1_utility_field_replay"]
    u2 = certificate["U2_historical_endpoint_replay"]
    final = certificate["final_acceptance_replay"]
    outputs = {
        "c85u_authorization_lifecycle_replay.csv": [
            {"component": "lock", "identity_sha256": auth["lock_sha256"], "count": 1, "status": "PASS"},
            {"component": "authorization", "identity_sha256": auth["authorization_sha256"], "count": 1, "status": "PASS"},
            {"component": "consumption_receipt", "identity_sha256": auth["consumption_receipt_sha256"], "count": 1, "status": "PASS"},
            {"component": "protected_replay", "identity_sha256": auth["protected_replay_sha256"], "count": 1, "status": "PASS"},
            {"component": "lifecycle", "identity_sha256": auth["lifecycle_sha256"], "count": auth["lifecycle_events"], "status": "PASS"},
            *({"component": f"stage_receipt_{row['stage']}", "identity_sha256": row["sha256"], "count": 1, "status": row["status"]} for row in auth["stage_receipts"]),
        ],
        "c85u_u1_artifact_replay.csv": [{
            "status": u1["status"], "contexts": u1["contexts"],
            "candidate_rows": u1["candidate_rows"], "context_artifacts": u1["context_artifacts"],
            "complete_tree_bytes": u1["complete_tree_bytes"],
            "context_artifact_bytes": u1["context_artifact_bytes"],
            "candidate_index_sha256": u1["candidate_index_sha256"],
            "v2_manifest_sha256": u1["v2_manifest_sha256"],
            "maximum_midrank_replay_error": u1["maximum_midrank_replay_error"],
            "maximum_utility_replay_error": u1["maximum_utility_replay_error"],
            "maximum_regret_replay_error": u1["maximum_regret_replay_error"],
            "maximum_rank_replay_error": u1["maximum_rank_replay_error"],
            "forbidden_payload_fields": u1["forbidden_payload_fields"],
        }],
        "c85u_u2_endpoint_replay.csv": [{
            "status": u2["status"], "contexts": u2["contexts"],
            "method_context_rows": u2["method_context_rows"],
            "finite_Q0_action_records": u2["finite_Q0_action_records"],
            "Q0_shards": u2["Q0_shards"],
            "selected_utility_max_abs": u2["maximum_absolute_differences"]["selected_utility"],
            "standardized_regret_max_abs": u2["maximum_absolute_differences"]["standardized_regret"],
            "top1_max_abs": u2["maximum_absolute_differences"]["top1"],
            "top5_max_abs": u2["maximum_absolute_differences"]["top5"],
            "top10_max_abs": u2["maximum_absolute_differences"]["top10"],
            "selected_regime_mismatches": u2["selected_regime_mismatches"],
            "result_sha256": u2["result_sha256"], "handoff_sha256": u2["handoff_sha256"],
        }],
        "c85u_acceptance_bundle_replay.csv": [{
            "status": final["status"], "result_sha256": final["result_sha256"],
            "acceptance_manifest_sha256": final["acceptance_manifest_sha256"],
            "completion_receipt_sha256": final["completion_receipt_sha256"],
            "lifecycle_sha256": final["lifecycle_sha256"],
            "manifest_artifacts": final["manifest_artifacts"],
            "protected_counters_nonzero": final["protected_counters_nonzero"],
            "residual_staging_roots": final["residual_staging_roots"],
        }],
    }
    hashes = {name: _write_csv(table_dir / name, rows) for name, rows in outputs.items()}
    certificate_path.write_bytes(_canonical_bytes(dict(certificate)) + b"\n")
    certificate_sha = _sha256_file(certificate_path)
    sidecar = certificate_path.with_suffix(".sha256")
    sidecar.write_text(f"{certificate_sha}  {certificate_path.name}\n", encoding="ascii")
    hashes[certificate_path.name] = certificate_sha
    hashes[sidecar.name] = _sha256_file(sidecar)
    return hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay-accepted-c85u", nargs="?")
    parser.add_argument("--source-root", type=Path, default=C85U_ROOT)
    parser.add_argument("--table-dir", type=Path, default=REPORT_DIR / "c85ep2_tables")
    parser.add_argument(
        "--certificate", type=Path,
        default=REPORT_DIR / "C85EP2_C85U_INPUT_ACCEPTANCE_CERTIFICATE.json",
    )
    args = parser.parse_args(argv)
    certificate = replay_c85u_acceptance(args.source_root)
    write_acceptance_evidence(
        certificate, table_dir=args.table_dir, certificate_path=args.certificate,
    )
    print(certificate["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "C85EP2ReplayError", "C85U_ROOT", "replay_c85u_acceptance",
    "write_acceptance_evidence",
]
