"""C85U Stage-U2 replay of frozen C84S decision endpoints.

This module never receives an evaluation-label path or a target-artifact path.
It derives endpoints only from the frozen U1 utility field and immutable Stage-B
action records, then compares them with the historical method-context table.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from oaci.multidataset import c84s_evaluation as evaluation
from oaci.multidataset.c84s_common import require, sha256_file
from oaci.multidataset.c84sr1_common import (
    FIXED_METHODS,
    Q0_BUDGET_CODES,
    Q0_CHAINS,
    SCORE_METHODS,
)
from oaci.multidataset.c84sr3_common import expected_methods, finite_budgets
from oaci.multidataset.c84sr3_q0_store import read_context_shard

from .c85u_persistence import load_context_artifact
from .c85u_result_manifest import validate_utility_manifest


SELECTION_MANIFEST_SHA256 = "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4"
RESULT_MANIFEST_SHA256 = "516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5"
REPLAY_SCHEMA = "c85u_historical_decision_replay_v1"
REPLAY_FIELDS = (
    "selected_utility", "standardized_regret", "top1", "top5", "top10",
)


def _scalar(payload: Mapping[str, np.ndarray], field: str) -> Any:
    value = np.asarray(payload[field])
    require(value.shape == (), f"C85U U2 scalar shape drift: {field}")
    return value.item()


def _identity(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "dataset": str(_scalar(payload, "dataset")),
        "target_subject_id": str(_scalar(payload, "target_subject_id")),
        "panel": str(_scalar(payload, "panel")),
        "training_seed": int(_scalar(payload, "training_seed")),
        "level": int(_scalar(payload, "level")),
    }


def _fixed_endpoint(
    utility: np.ndarray, regimes: Sequence[str], selected: int,
) -> dict[str, Any]:
    best = int(np.lexsort((np.arange(81), -utility))[0])
    hit = float(int(selected) == best)
    return {
        "standardized_regret": evaluation.standardized_regret(utility, selected),
        "selected_utility": float(utility[selected]),
        "top1": hit,
        "top5": hit,
        "top10": hit,
        "selected_regime": str(regimes[selected]),
    }


def _q0_batch_endpoint(
    orders: np.ndarray, utility: np.ndarray, regimes: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(orders, dtype=np.int16)
    require(values.ndim == 2 and values.shape[1] == 81,
            "C85U U2 Q0 order matrix shape drift")
    require(all(set(row.tolist()) == set(range(81)) for row in values),
            "C85U U2 Q0 order is not a permutation")
    best = int(np.lexsort((np.arange(81), -utility))[0])
    selected = values[:, 0]
    selected_utility = utility[selected]
    spread = float(np.max(utility) - np.min(utility))
    regrets = (
        np.zeros(len(values), dtype=float)
        if spread <= 1e-15
        else (np.max(utility) - selected_utility) / spread
    )
    regime_values = np.asarray(regimes, dtype=str)[selected]
    return {
        "standardized_regret": float(np.mean(regrets)),
        "selected_utility": float(np.mean(selected_utility)),
        "top1": float(np.mean(np.any(values[:, :1] == best, axis=1))),
        "top5": float(np.mean(np.any(values[:, :5] == best, axis=1))),
        "top10": float(np.mean(np.any(values[:, :10] == best, axis=1))),
        "selected_regime": str(regime_values[0])
        if len(values) == 1 else "STOCHASTIC_Q0",
    }


def replay_context_endpoints(
    *,
    payload: Mapping[str, np.ndarray],
    score_orders: Mapping[str, np.ndarray],
    fixed_selected_indices: Mapping[str, int],
    q0_payload: Mapping[str, np.ndarray],
    q0_chains: int = Q0_CHAINS,
) -> dict[str, dict[str, Any]]:
    """Reconstruct only the six decision fields registered for U2."""
    utility = np.asarray(payload["composite_utility"], dtype=float)
    candidate_ids = np.asarray(payload["candidate_id"], dtype=str)
    regimes = tuple(map(str, np.asarray(payload["regime"], dtype=str).tolist()))
    dataset = str(_scalar(payload, "dataset"))
    require(utility.shape == candidate_ids.shape == (81,) and len(regimes) == 81,
            "C85U U2 utility identity shape drift")
    require(set(score_orders) == set(SCORE_METHODS), "C85U U2 score-method set drift")
    require(set(fixed_selected_indices) == set(FIXED_METHODS),
            "C85U U2 fixed-method set drift")

    endpoints: dict[str, dict[str, Any]] = {
        "B0": evaluation.evaluate_uniform_random(utility),
        "B5": evaluation.evaluate_oracle(utility, regimes),
    }
    for method, selected in fixed_selected_indices.items():
        require(0 <= int(selected) < 81, "C85U U2 fixed candidate range drift")
        endpoints[method] = _fixed_endpoint(utility, regimes, int(selected))
    for method, order in score_orders.items():
        endpoints[method] = evaluation.evaluate_order(order, utility, regimes)

    finite_codes = np.asarray(q0_payload["finite_budget_code"], dtype=np.uint8)
    finite_orders = np.asarray(q0_payload["finite_candidate_order"], dtype=np.uint8)
    for budget in finite_budgets(dataset):
        orders = finite_orders[finite_codes == Q0_BUDGET_CODES[budget]]
        require(len(orders) == q0_chains,
                f"C85U U2 Q0 chain coverage drift: {dataset}/B{budget}")
        endpoints[f"Q0_B{budget}"] = _q0_batch_endpoint(orders, utility, regimes)
    full_orders = np.asarray(q0_payload["FULL_candidate_order"], dtype=np.uint8)
    require(full_orders.shape == (1, 81), "C85U U2 Q0 FULL coverage drift")
    endpoints["Q0_FULL"] = _q0_batch_endpoint(full_orders, utility, regimes)
    require(set(endpoints) == set(expected_methods(dataset)),
            "C85U U2 method coverage drift")
    return endpoints


def compare_context_endpoints(
    reconstructed: Mapping[str, Mapping[str, Any]],
    historical: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    require(set(reconstructed) == set(historical),
            "C85U U2 historical method set drift")
    maxima = {field: 0.0 for field in REPLAY_FIELDS}
    for method in reconstructed:
        observed = reconstructed[method]
        expected = historical[method]
        for field in REPLAY_FIELDS:
            difference = abs(float(observed[field]) - float(expected[field]))
            maxima[field] = max(maxima[field], difference)
            require(difference <= tolerance,
                    f"C85U U2 endpoint replay mismatch: {method}/{field}")
        require(str(observed["selected_regime"]) == str(expected["selected_regime"]),
                f"C85U U2 selected regime mismatch: {method}")
    return maxima


def _load_json_exact(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(path.is_file() and sha256_file(path) == expected_sha256,
            f"C85U U2 manifest identity drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "C85U U2 manifest object malformed")
    return value


def _artifact_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = manifest["artifacts"]
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def _verify_stage_b(selection_root: Path) -> dict[str, Any]:
    manifest_path = selection_root / "C84S_SELECTION_FREEZE_MANIFEST_V3.json"
    manifest = _load_json_exact(manifest_path, SELECTION_MANIFEST_SHA256)
    require(manifest["contexts"] == 944 and manifest["Q0_records"] == 8_750_000,
            "C85U U2 selection-freeze arithmetic drift")
    require(manifest["evaluation_label_descriptor_received"] is False,
            "C85U U2 selection freeze received evaluation descriptor")
    artifact_by_name = {
        Path(str(row["path"])).name: row for row in _artifact_rows(manifest)
    }
    for name in (
        "candidate_ranks.csv", "fixed_default_selections.csv",
        "q0_selection_shard_index.csv",
    ):
        require(name in artifact_by_name, f"C85U U2 Stage-B action object absent: {name}")
        row = artifact_by_name[name]
        path = selection_root / str(row["path"])
        require(path.is_file() and sha256_file(path) == str(row["sha256"]),
                f"C85U U2 Stage-B action identity drift: {name}")
    return manifest


def _utility_contexts(
    utility_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    manifest = json.loads(
        (utility_root / "C85U_CANDIDATE_UTILITY_MANIFEST.json").read_text(encoding="utf-8")
    )
    identities: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for artifact in manifest["context_artifacts"]:
        payload, _ = load_context_artifact(
            utility_root / str(artifact["path"]), expected_sha256=str(artifact["sha256"]),
        )
        context_id = str(_scalar(payload, "context_id"))
        require(context_id not in payloads, "C85U U2 duplicate utility context")
        payloads[context_id] = payload
        identities[context_id] = {
            **_identity(payload),
            "candidate_ids": tuple(map(str, payload["candidate_id"].tolist())),
        }
    return identities, payloads


def _load_actions(
    selection_root: Path, contexts: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, int]],
    dict[str, dict[str, str]],
]:
    score_orders = {
        context_id: {
            method: np.full(81, -1, dtype=np.int16) for method in SCORE_METHODS
        }
        for context_id in contexts
    }
    rank_rows = 0
    with (selection_root / "candidate_ranks.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            context_id = str(row["context_id"])
            method = str(row["method_id"])
            candidate_index = int(row["candidate_index"])
            rank = int(row["rank"])
            require(context_id in contexts and method in SCORE_METHODS,
                    "C85U U2 score-order identity drift")
            require(0 <= candidate_index < 81 and 1 <= rank <= 81,
                    "C85U U2 score-order range drift")
            require(str(row["candidate_id"]) == contexts[context_id]["candidate_ids"][candidate_index],
                    "C85U U2 score-order candidate drift")
            order = score_orders[context_id][method]
            require(order[rank - 1] == -1, "C85U U2 duplicate score rank")
            order[rank - 1] = candidate_index
            rank_rows += 1
    require(rank_rows == len(contexts) * len(SCORE_METHODS) * 81,
            "C85U U2 rank-row coverage drift")
    require(all(np.array_equal(np.sort(order), np.arange(81))
                for methods in score_orders.values() for order in methods.values()),
            "C85U U2 score order is incomplete")

    fixed = {context_id: {} for context_id in contexts}
    fixed_rows = 0
    with (selection_root / "fixed_default_selections.csv").open(
        newline="", encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            context_id = str(row["context_id"])
            method = str(row["method_id"])
            index = int(row["selected_candidate_index"])
            require(context_id in contexts and method in FIXED_METHODS,
                    "C85U U2 fixed action identity drift")
            require(method not in fixed[context_id] and 0 <= index < 81,
                    "C85U U2 duplicate or out-of-range fixed action")
            require(str(row["selected_candidate_id"]) == contexts[context_id]["candidate_ids"][index],
                    "C85U U2 fixed candidate identity drift")
            fixed[context_id][method] = index
            fixed_rows += 1
    require(fixed_rows == len(contexts) * len(FIXED_METHODS) and
            all(set(row) == set(FIXED_METHODS) for row in fixed.values()),
            "C85U U2 fixed-action coverage drift")

    shards: dict[str, dict[str, str]] = {}
    with (selection_root / "q0_selection_shard_index.csv").open(
        newline="", encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            context_id = str(row["context_id"])
            require(context_id in contexts and context_id not in shards,
                    "C85U U2 Q0 shard-index context drift")
            shards[context_id] = dict(row)
    require(set(shards) == set(contexts), "C85U U2 Q0 shard coverage drift")
    return score_orders, fixed, shards


def _historical_table_identity(
    result_manifest_path: Path, historical_table_path: Path,
) -> tuple[dict[str, Any], str]:
    manifest = _load_json_exact(result_manifest_path, RESULT_MANIFEST_SHA256)
    matches = [
        row for row in _artifact_rows(manifest)
        if Path(str(row["path"])).name == "method_context_decisions.csv"
    ]
    require(len(matches) == 1, "C85U U2 historical method table manifest linkage drift")
    row = matches[0]
    require(historical_table_path.is_file() and
            sha256_file(historical_table_path) == str(row["sha256"]),
            "C85U U2 historical method table SHA drift")
    return dict(row), sha256_file(historical_table_path)


def _load_historical_rows(
    path: Path,
) -> dict[tuple[str, str, str, int, int, str], dict[str, str]]:
    result: dict[tuple[str, str, str, int, int, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row["dataset"]), str(row["target_subject_id"]), str(row["panel"]),
                int(row["training_seed"]), int(row["level"]), str(row["method_id"]),
            )
            require(key not in result, "C85U U2 duplicate historical method row")
            result[key] = dict(row)
    require(len(result) == 18_432, "C85U U2 historical method-context coverage drift")
    return result


def _write_json_fsync(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def run_historical_decision_replay(
    *,
    utility_root: str | Path,
    selection_root: str | Path,
    result_manifest_path: str | Path,
    historical_table_path: str | Path,
    final_root: str | Path,
    q0_chains: int = Q0_CHAINS,
    expected_contexts: int = 944,
    expected_rows: int = 18_432,
) -> dict[str, Any]:
    """Run U2 without any label or target-array input."""
    require(q0_chains == Q0_CHAINS, "C85U U2 real Q0 chain reduction forbidden")
    utility_base = Path(utility_root)
    utility_replay = validate_utility_manifest(
        utility_base, expected_contexts=expected_contexts,
        expected_candidate_rows=expected_contexts * 81,
    )
    selection_base = Path(selection_root)
    _verify_stage_b(selection_base)
    contexts, payloads = _utility_contexts(utility_base)
    require(len(contexts) == expected_contexts, "C85U U2 utility context coverage drift")
    score_orders, fixed, shards = _load_actions(selection_base, contexts)
    historical_identity, historical_sha = _historical_table_identity(
        Path(result_manifest_path), Path(historical_table_path),
    )
    historical = _load_historical_rows(Path(historical_table_path))

    maxima = {field: 0.0 for field in REPLAY_FIELDS}
    compared = 0
    q0_shards = 0
    finite_chain_records = 0
    context_digest_rows: list[dict[str, Any]] = []
    for context_id in sorted(contexts):
        payload = payloads[context_id]
        shard = shards[context_id]
        q0_payload, shard_replay = read_context_shard(
            selection_base / str(shard["path"]),
            expected_sha256=str(shard["sha256"]), chains=q0_chains,
        )
        require(shard_replay["context_id"] == context_id,
                "C85U U2 Q0 shard/context mismatch")
        endpoints = replay_context_endpoints(
            payload=payload, score_orders=score_orders[context_id],
            fixed_selected_indices=fixed[context_id], q0_payload=q0_payload,
            q0_chains=q0_chains,
        )
        identity = contexts[context_id]
        expected = {
            method: historical[(
                str(identity["dataset"]), str(identity["target_subject_id"]),
                str(identity["panel"]), int(identity["training_seed"]),
                int(identity["level"]), method,
            )]
            for method in endpoints
        }
        context_maxima = compare_context_endpoints(endpoints, expected)
        for field, value in context_maxima.items():
            maxima[field] = max(maxima[field], value)
        compared += len(endpoints)
        q0_shards += 1
        finite_chain_records += len(q0_payload["finite_budget_code"])
        context_digest_rows.append({
            "context_id": context_id,
            "method_count": len(endpoints),
            "utility_vector_sha256": str(_scalar(payload, "utility_vector_sha256")),
            "q0_shard_sha256": str(shard["sha256"]),
        })
    require(compared == expected_rows, "C85U U2 reconstructed method row count drift")
    require(compared == len(historical), "C85U U2 unmatched historical rows")
    if expected_contexts == 944:
        require(finite_chain_records == 8_749_056,
                "C85U U2 finite Q0 record count drift")

    final = Path(final_root)
    require(not final.exists(), "C85U U2 final replay root already exists")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        result = {
            "schema_version": REPLAY_SCHEMA,
            "status": "PASS_HISTORICAL_DECISION_ENDPOINTS_EXACTLY_REPLAYED",
            "utility_manifest_sha256": utility_replay["manifest_sha256"],
            "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
            "historical_result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "historical_method_context_table": {
                "path": str(historical_table_path),
                "sha256": historical_sha,
                "manifest_identity": historical_identity,
            },
            "contexts": len(contexts),
            "method_context_rows": compared,
            "Q0_shards": q0_shards,
            "finite_Q0_chain_records_replayed": finite_chain_records,
            "endpoint_tolerance": 1e-12,
            "maximum_absolute_differences": maxima,
            "selected_regime_mismatches": 0,
            "context_replay_registry_sha256": hashlib.sha256(
                json.dumps(context_digest_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "forbidden_analysis": {
                "Q1": 0, "Q2": 0, "max_T": 0, "LOTO": 0,
                "label_frontier": 0, "taxonomy": 0, "new_pvalues": 0,
            },
            "protected_input_access": {
                "evaluation_label_rows": 0,
                "target_logit_arrays": 0,
                "construction_label_rows": 0,
            },
            "acceptance_for_C85E": True,
        }
        result_path = staging / "C85U_HISTORICAL_DECISION_REPLAY.json"
        result_sha = _write_json_fsync(result_path, result)
        with (staging / "C85U_HISTORICAL_DECISION_REPLAY.sha256").open("xb") as handle:
            handle.write(f"{result_sha}  {result_path.name}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        replayed = json.loads(result_path.read_text(encoding="utf-8"))
        require(replayed == result, "C85U U2 persisted replay result drift")
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, final)
        return {**result, "result_sha256": result_sha, "root": str(final)}
    except BaseException:
        raise


__all__ = [
    "REPLAY_FIELDS", "REPLAY_SCHEMA", "compare_context_endpoints",
    "replay_context_endpoints", "run_historical_decision_replay",
]
