"""Sharded non-object Q0 selection storage and exact coverage replay."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats

from . import c84s_q0_budget as q0
from .c84s_common import require, sha256_file
from .c84sr1_common import (
    CANDIDATES, Q0_BUDGET_CODES, Q0_CHAINS, context_id, finite_budgets,
)


SCHEMA_VERSION = "c84sr1_q0_context_shard_v1"
SHARD_INDEX_FIELDS = (
    "context_id", "dataset", "target_subject_id", "panel", "training_seed",
    "level", "path", "sha256", "bytes", "candidate_count",
    "finite_records", "FULL_records", "total_records",
)
BudgetProvider = Callable[[str], tuple[int, ...]]
_PLAN_CACHE: dict[tuple[str, str, str, int, tuple[int, ...]], dict[str, Any]] = {}


def _digest_bytes(hex_digest: str) -> np.ndarray:
    value = bytes.fromhex(str(hex_digest))
    require(len(value) == 32, "Q0 digest length drift")
    return np.frombuffer(value, dtype=np.uint8).copy()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def expected_finite_records(
    dataset: str, chains: int = Q0_CHAINS, *,
    budget_provider: BudgetProvider = finite_budgets,
) -> int:
    return len(budget_provider(dataset)) * int(chains)


def validate_payload(
    payload: Mapping[str, np.ndarray], *, chains: int = Q0_CHAINS,
    budget_provider: BudgetProvider = finite_budgets,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    required = {
        "schema_version", "dataset", "target_subject_id", "panel",
        "training_seed", "level", "context_id", "candidate_ids",
        "finite_chain", "finite_chain_seed", "finite_budget_code",
        "finite_sample_size", "finite_sample_digest", "finite_selected_index",
        "finite_candidate_order", "finite_candidate_score_digest",
        "finite_construction_metric_digest", "FULL_sample_size",
        "FULL_sample_digest", "FULL_selected_index", "FULL_candidate_order",
        "FULL_candidate_score_digest", "FULL_construction_metric_digest",
    }
    require(set(payload) == required, "Q0 shard field-set drift")

    scalar_text = lambda key: str(np.asarray(payload[key]).item().decode("ascii")
                                  if np.asarray(payload[key]).dtype.kind == "S"
                                  else np.asarray(payload[key]).item())
    require(scalar_text("schema_version") == schema_version, "Q0 shard schema drift")
    dataset = scalar_text("dataset")
    identity = {
        "dataset": dataset,
        "target_subject_id": scalar_text("target_subject_id"),
        "panel": scalar_text("panel"),
        "training_seed": int(np.asarray(payload["training_seed"]).item()),
        "level": int(np.asarray(payload["level"]).item()),
    }
    require(scalar_text("context_id") == context_id(identity), "Q0 shard context identity drift")
    candidate_ids = np.asarray(payload["candidate_ids"])
    require(candidate_ids.dtype.kind == "S" and candidate_ids.shape == (CANDIDATES,),
            "Q0 candidate-ID table drift")
    decoded = [value.decode("ascii") for value in candidate_ids.tolist()]
    require(len(set(decoded)) == CANDIDATES, "Q0 candidate IDs are not unique")

    budgets = budget_provider(dataset)
    count = expected_finite_records(dataset, chains, budget_provider=budget_provider)
    one_dimensional = {
        "finite_chain": np.uint16,
        "finite_chain_seed": np.uint64,
        "finite_budget_code": np.uint8,
        "finite_sample_size": np.uint16,
        "finite_selected_index": np.uint8,
    }
    for name, dtype in one_dimensional.items():
        value = np.asarray(payload[name])
        require(value.dtype == dtype and value.shape == (count,), f"Q0 shard array drift: {name}")
    for name in ("finite_sample_digest", "finite_candidate_score_digest", "finite_construction_metric_digest"):
        value = np.asarray(payload[name])
        require(value.dtype == np.uint8 and value.shape == (count, 32), f"Q0 digest array drift: {name}")
    orders = np.asarray(payload["finite_candidate_order"])
    require(orders.dtype == np.uint8 and orders.shape == (count, CANDIDATES),
            "Q0 candidate-order array drift")
    require(np.all(np.sort(orders, axis=1) == np.arange(CANDIDATES, dtype=np.uint8)),
            "Q0 finite candidate order is not a permutation")
    require(np.all(np.asarray(payload["finite_selected_index"]) == orders[:, 0]),
            "Q0 selected index differs from order head")
    for budget in budgets:
        mask = np.asarray(payload["finite_budget_code"]) == Q0_BUDGET_CODES[budget]
        observed = np.sort(np.asarray(payload["finite_chain"])[mask])
        require(np.array_equal(observed, np.arange(chains, dtype=np.uint16)),
                f"Q0 chain coverage drift: {dataset}/B{budget}")
        require(np.all(np.asarray(payload["finite_sample_size"])[mask] == 2 * budget),
                f"Q0 sample-size drift: B{budget}")
    require(set(np.unique(np.asarray(payload["finite_budget_code"])).tolist()) ==
            {Q0_BUDGET_CODES[value] for value in budgets},
            "Q0 finite budget coverage drift")

    require(np.asarray(payload["FULL_sample_size"]).dtype == np.uint16 and
            np.asarray(payload["FULL_sample_size"]).shape == (1,), "Q0 FULL sample-size drift")
    for name in ("FULL_sample_digest", "FULL_candidate_score_digest", "FULL_construction_metric_digest"):
        value = np.asarray(payload[name])
        require(value.dtype == np.uint8 and value.shape == (1, 32), f"Q0 FULL digest drift: {name}")
    full_order = np.asarray(payload["FULL_candidate_order"])
    require(full_order.dtype == np.uint8 and full_order.shape == (1, CANDIDATES),
            "Q0 FULL order drift")
    require(np.array_equal(np.sort(full_order[0]), np.arange(CANDIDATES, dtype=np.uint8)),
            "Q0 FULL order is not a permutation")
    full_selected = np.asarray(payload["FULL_selected_index"])
    require(full_selected.dtype == np.uint8 and full_selected.shape == (1,) and
            full_selected[0] == full_order[0, 0], "Q0 FULL selected index drift")
    return {
        **identity, "context_id": context_id(identity),
        "candidate_count": CANDIDATES, "finite_records": count,
        "FULL_records": 1, "total_records": count + 1,
    }


def write_context_shard(
    path: str | Path, payload: Mapping[str, np.ndarray], *, chains: int = Q0_CHAINS,
    budget_provider: BudgetProvider = finite_budgets,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    path = Path(path)
    require(not path.exists(), "Q0 shard already exists")
    identity = validate_payload(
        payload, chains=chains, budget_provider=budget_provider,
        schema_version=schema_version,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as archive:
        replay = {name: archive[name] for name in archive.files}
    require(set(replay) == set(payload), "persisted Q0 shard fields drift")
    for name, value in payload.items():
        require(np.array_equal(replay[name], value), f"persisted Q0 shard array drift: {name}")
    validate_payload(
        replay, chains=chains, budget_provider=budget_provider,
        schema_version=schema_version,
    )
    return {
        **identity, "path": str(path), "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def read_context_shard(
    path: str | Path, *, expected_sha256: str | None = None, chains: int = Q0_CHAINS,
    budget_provider: BudgetProvider = finite_budgets,
    schema_version: str = SCHEMA_VERSION,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    require(path.is_file(), "Q0 shard absent")
    if expected_sha256 is not None:
        require(sha256_file(path) == expected_sha256, "Q0 shard SHA drift")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    identity = validate_payload(
        payload, chains=chains, budget_provider=budget_provider,
        schema_version=schema_version,
    )
    return payload, identity


def build_context_payload(
    *,
    identity: Mapping[str, Any],
    candidate_ids: Sequence[str],
    target_logits: np.ndarray,
    target_trial_ids: Sequence[str],
    construction_rows: Sequence[Mapping[str, Any]],
    chains: int = Q0_CHAINS,
    budget_provider: BudgetProvider = finite_budgets,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, np.ndarray]:
    """Compute one Q0 shard with exact streams and batched candidate metrics."""
    dataset = str(identity["dataset"])
    target = str(identity["target_subject_id"])
    candidate_ids = tuple(map(str, candidate_ids))
    require(len(candidate_ids) == CANDIDATES and len(set(candidate_ids)) == CANDIDATES,
            "Q0 candidate-ID coverage drift")
    target_trial_ids = np.asarray(target_trial_ids, dtype=str)
    target_logits = np.asarray(target_logits, dtype=float)
    require(target_logits.shape == (CANDIDATES, len(target_trial_ids), 2),
            "Q0 target logits/trial shape drift")
    require(len(set(target_trial_ids.tolist())) == len(target_trial_ids), "target trial IDs are not unique")
    target_index = {value: index for index, value in enumerate(target_trial_ids.tolist())}
    rows = [row for row in construction_rows
            if str(row["dataset"]) == dataset and str(row["target_subject_id"]) == target]
    require(rows, "Q0 construction rows absent for target")
    construction_ids = np.asarray([str(row["target_trial_id"]) for row in rows], dtype=str)
    require(set(construction_ids.tolist()).issubset(target_index), "construction IDs are outside target artifact")
    labels = np.asarray([int(row["canonical_class_label"]) for row in rows], dtype=int)
    indices = np.asarray([target_index[value] for value in construction_ids], dtype=int)
    construction_logits = target_logits[:, indices]
    budgets = budget_provider(dataset)
    plan_identity = hashlib.sha256(
        ("\n".join(construction_ids.tolist()) + "\n" +
         "".join(map(str, labels.tolist()))).encode("utf-8")
    ).hexdigest()
    plan_key = (dataset, target, plan_identity, int(chains), tuple(budgets))
    if plan_key not in _PLAN_CACHE:
        canonical = {class_id: np.where(labels == class_id)[0] for class_id in (0, 1)}
        require(set(np.unique(labels).tolist()) == {0, 1}, "Q0 construction class mapping drift")
        sample_indices = {
            budget: np.empty((chains, 2 * budget), dtype=np.uint16)
            for budget in budgets
        }
        chain_seed = np.empty(chains, dtype=np.uint64)
        sample_digest = {
            budget: np.empty((chains, 32), dtype=np.uint8)
            for budget in budgets
        }
        for chain in range(chains):
            seed = q0.stream_seed(dataset, target, chain)
            chain_seed[chain] = seed
            rng = np.random.Generator(np.random.PCG64(seed))
            orders = {class_id: rng.permutation(canonical[class_id]) for class_id in (0, 1)}
            for budget in budgets:
                require(all(len(orders[class_id]) >= budget for class_id in (0, 1)),
                        f"Q0 budget {budget} is infeasible")
                selected = np.concatenate((orders[0][:budget], orders[1][:budget]))
                sample_indices[budget][chain] = selected
                sample_digest[budget][chain] = _digest_bytes(
                    q0.sample_digest(construction_ids[selected])
                )
        full_indices = np.concatenate((canonical[0], canonical[1])).astype(np.uint16)
        _PLAN_CACHE[plan_key] = {
            "chain_seed": chain_seed, "sample_indices": sample_indices,
            "sample_digest": sample_digest, "full_indices": full_indices,
            "full_digest": _digest_bytes(q0.sample_digest(construction_ids[full_indices])),
        }
    plan = _PLAN_CACHE[plan_key]
    finite_count = len(budgets) * chains
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(schema_version.encode("ascii"), dtype="S64"),
        "dataset": np.asarray(dataset.encode("ascii"), dtype="S32"),
        "target_subject_id": np.asarray(target.encode("ascii"), dtype="S32"),
        "panel": np.asarray(str(identity["panel"]).encode("ascii"), dtype="S4"),
        "training_seed": np.asarray(int(identity["training_seed"]), dtype=np.uint8),
        "level": np.asarray(int(identity["level"]), dtype=np.uint8),
        "context_id": np.asarray(context_id(identity).encode("ascii"), dtype="S32"),
        "candidate_ids": np.asarray([value.encode("ascii") for value in candidate_ids], dtype="S64"),
        "finite_chain": np.empty(finite_count, dtype=np.uint16),
        "finite_chain_seed": np.empty(finite_count, dtype=np.uint64),
        "finite_budget_code": np.empty(finite_count, dtype=np.uint8),
        "finite_sample_size": np.empty(finite_count, dtype=np.uint16),
        "finite_sample_digest": np.empty((finite_count, 32), dtype=np.uint8),
        "finite_selected_index": np.empty(finite_count, dtype=np.uint8),
        "finite_candidate_order": np.empty((finite_count, CANDIDATES), dtype=np.uint8),
        "finite_candidate_score_digest": np.empty((finite_count, 32), dtype=np.uint8),
        "finite_construction_metric_digest": np.empty((finite_count, 32), dtype=np.uint8),
    }
    def batched_scores(selected_indices: np.ndarray, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        selected_indices = np.asarray(selected_indices, dtype=int)
        count = len(selected_indices)
        all_scores = np.empty((count, CANDIDATES), dtype=float)
        all_metrics = np.empty((count, CANDIDATES, 3), dtype=float)
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            index = selected_indices[start:stop]
            # Advanced indexing yields candidate x batch x sample x class.
            logits = np.transpose(construction_logits[:, index, :], (1, 0, 2, 3))
            batch_labels = labels[index]
            shifted = logits - np.max(logits, axis=3, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities, axis=3, keepdims=True)
            prediction = np.argmax(probabilities, axis=3)
            class0 = batch_labels == 0
            class1 = batch_labels == 1
            bacc = 0.5 * (
                np.sum((prediction == 0) & class0[:, None, :], axis=2) /
                np.sum(class0, axis=1)[:, None]
                + np.sum((prediction == 1) & class1[:, None, :], axis=2) /
                np.sum(class1, axis=1)[:, None]
            )
            label_index = np.broadcast_to(batch_labels[:, None, :, None],
                                          (*probabilities.shape[:3], 1))
            true_probability = np.take_along_axis(probabilities, label_index, axis=3)[..., 0]
            nll = -np.mean(np.log(np.clip(true_probability, 1e-12, 1.0)), axis=2)
            confidence = np.max(probabilities, axis=3)
            correctness = prediction == batch_labels[:, None, :]
            ece = np.zeros_like(nll)
            edges = np.linspace(0.0, 1.0, 16)
            for bin_index in range(15):
                right = confidence <= edges[bin_index + 1] if bin_index == 14 else confidence < edges[bin_index + 1]
                mask = (confidence >= edges[bin_index]) & right
                bin_count = np.sum(mask, axis=2)
                accuracy = np.divide(
                    np.sum(correctness & mask, axis=2), bin_count,
                    out=np.zeros_like(nll), where=bin_count > 0,
                )
                mean_confidence = np.divide(
                    np.sum(confidence * mask, axis=2), bin_count,
                    out=np.zeros_like(nll), where=bin_count > 0,
                )
                ece += (bin_count / confidence.shape[2]) * np.abs(accuracy - mean_confidence)
            metrics = np.stack((bacc, nll, ece), axis=2)
            oriented = np.stack((
                (stats.rankdata(metrics[:, :, 0], method="average", axis=1) - 1.0) / 80.0,
                (stats.rankdata(-metrics[:, :, 1], method="average", axis=1) - 1.0) / 80.0,
                (stats.rankdata(-metrics[:, :, 2], method="average", axis=1) - 1.0) / 80.0,
            ), axis=2)
            all_scores[start:stop] = np.mean(oriented, axis=2)
            all_metrics[start:stop] = metrics
        return all_scores, all_metrics

    position = 0
    for budget in budgets:
        selected_matrix = np.asarray(plan["sample_indices"][budget], dtype=int)
        scores, metrics = batched_scores(selected_matrix)
        orders = np.argsort(-scores, axis=1, kind="stable").astype(np.uint8)
        block = slice(position, position + chains)
        payload["finite_chain"][block] = np.arange(chains, dtype=np.uint16)
        payload["finite_chain_seed"][block] = plan["chain_seed"]
        payload["finite_budget_code"][block] = Q0_BUDGET_CODES[budget]
        payload["finite_sample_size"][block] = 2 * budget
        payload["finite_sample_digest"][block] = plan["sample_digest"][budget]
        payload["finite_selected_index"][block] = orders[:, 0]
        payload["finite_candidate_order"][block] = orders
        for offset in range(chains):
            payload["finite_candidate_score_digest"][position + offset] = _digest_bytes(
                _array_digest(np.asarray(scores[offset], dtype="<f8"))
            )
            payload["finite_construction_metric_digest"][position + offset] = _digest_bytes(
                _array_digest(np.asarray(metrics[offset], dtype="<f8"))
            )
        position += chains
    require(position == finite_count, "Q0 finite record arithmetic drift")
    full_selected = np.asarray(plan["full_indices"], dtype=int)
    full_scores, full_metrics = q0.candidate_scores(construction_logits[:, full_selected], labels[full_selected])
    full_order = q0.descending_order(full_scores).astype(np.uint8)
    payload.update({
        "FULL_sample_size": np.asarray([len(full_selected)], dtype=np.uint16),
        "FULL_sample_digest": np.asarray([plan["full_digest"]], dtype=np.uint8),
        "FULL_selected_index": np.asarray([full_order[0]], dtype=np.uint8),
        "FULL_candidate_order": np.asarray([full_order], dtype=np.uint8),
        "FULL_candidate_score_digest": np.asarray([_digest_bytes(_array_digest(np.asarray(full_scores, dtype="<f8")))], dtype=np.uint8),
        "FULL_construction_metric_digest": np.asarray([_digest_bytes(_array_digest(np.asarray(full_metrics, dtype="<f8")))], dtype=np.uint8),
    })
    validate_payload(
        payload, chains=chains, budget_provider=budget_provider,
        schema_version=schema_version,
    )
    return payload


def synthetic_payload(
    identity: Mapping[str, Any], candidate_ids: Sequence[str], *, chains: int = Q0_CHAINS,
    budget_provider: BudgetProvider = finite_budgets,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, np.ndarray]:
    """Production-schema fixture used by full-scale storage calibration."""
    dataset = str(identity["dataset"])
    budgets = budget_provider(dataset)
    count = len(budgets) * chains
    chain = np.tile(np.arange(chains, dtype=np.uint16), len(budgets))
    budget_code = np.repeat(np.asarray([Q0_BUDGET_CODES[value] for value in budgets], dtype=np.uint8), chains)
    base = np.arange(CANDIDATES, dtype=np.uint8)
    orders = np.tile(base, (count, 1))
    # Rotate deterministically while preserving a complete permutation.
    offsets = (chain.astype(np.uint32) + budget_code.astype(np.uint32)) % CANDIDATES
    orders = (orders + offsets[:, None]).astype(np.uint8) % CANDIDATES
    digest = np.zeros((count, 32), dtype=np.uint8)
    digest[:, 0] = budget_code
    digest[:, 1] = (chain & 0xFF).astype(np.uint8)
    digest[:, 2] = (chain >> 8).astype(np.uint8)
    payload = {
        "schema_version": np.asarray(schema_version.encode("ascii"), dtype="S64"),
        "dataset": np.asarray(dataset.encode("ascii"), dtype="S32"),
        "target_subject_id": np.asarray(str(identity["target_subject_id"]).encode("ascii"), dtype="S32"),
        "panel": np.asarray(str(identity["panel"]).encode("ascii"), dtype="S4"),
        "training_seed": np.asarray(int(identity["training_seed"]), dtype=np.uint8),
        "level": np.asarray(int(identity["level"]), dtype=np.uint8),
        "context_id": np.asarray(context_id(identity).encode("ascii"), dtype="S32"),
        "candidate_ids": np.asarray([str(value).encode("ascii") for value in candidate_ids], dtype="S64"),
        "finite_chain": chain,
        "finite_chain_seed": chain.astype(np.uint64) + np.uint64(84),
        "finite_budget_code": budget_code,
        "finite_sample_size": (2 * budget_code.astype(np.uint16)),
        "finite_sample_digest": digest.copy(),
        "finite_selected_index": orders[:, 0].copy(),
        "finite_candidate_order": orders,
        "finite_candidate_score_digest": digest.copy(),
        "finite_construction_metric_digest": digest.copy(),
        "FULL_sample_size": np.asarray([32], dtype=np.uint16),
        "FULL_sample_digest": np.zeros((1, 32), dtype=np.uint8),
        "FULL_selected_index": np.asarray([0], dtype=np.uint8),
        "FULL_candidate_order": np.arange(CANDIDATES, dtype=np.uint8)[None, :],
        "FULL_candidate_score_digest": np.zeros((1, 32), dtype=np.uint8),
        "FULL_construction_metric_digest": np.zeros((1, 32), dtype=np.uint8),
    }
    validate_payload(
        payload, chains=chains, budget_provider=budget_provider,
        schema_version=schema_version,
    )
    return payload
