"""Locked passive Q0 construction-label sampling and selection for C84S."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .c84s_common import C84SContractError, canonical_sha256, digest_low64, require


CANDIDATES = 81
PRIMARY_BUDGETS: tuple[int | str, ...] = (1, 2, 4, 8, "FULL")
SECONDARY_BUDGETS: tuple[int, ...] = (16, 32)
CHAINS = 2048


def stream_seed(dataset: str, target_subject: str | int, chain: int) -> int:
    require(chain >= 0, "Q0 chain index is negative")
    return digest_low64(f"C84_Q0_V1|{dataset}|{target_subject}|{chain}")


def nested_trial_samples(
    trial_ids: Sequence[str],
    labels: np.ndarray,
    *,
    dataset: str,
    target_subject: str | int,
    chain: int,
    budgets: Sequence[int | str] = PRIMARY_BUDGETS,
) -> dict[int | str, np.ndarray]:
    trial_id = np.asarray(trial_ids, dtype=str)
    labels = np.asarray(labels, dtype=int)
    require(trial_id.ndim == labels.ndim == 1 and len(trial_id) == len(labels), "Q0 trial/label shape drift")
    require(len(set(trial_id.tolist())) == len(trial_id), "Q0 construction trial IDs are not unique")
    require(set(np.unique(labels)) == {0, 1}, "Q0 class mapping drift")
    requested = tuple(budgets)
    require(requested and requested[-1] == "FULL", "Q0 budget grid must terminate in FULL")
    finite = [int(value) for value in requested if value != "FULL"]
    require(finite == sorted(set(finite)), "Q0 finite budgets are not strictly increasing")
    rng = np.random.Generator(np.random.PCG64(stream_seed(dataset, target_subject, chain)))
    orders = {class_id: rng.permutation(np.where(labels == class_id)[0]) for class_id in (0, 1)}
    output: dict[int | str, np.ndarray] = {}
    for budget in requested:
        if budget == "FULL":
            selected = np.concatenate([orders[0], orders[1]])
        else:
            count = int(budget)
            require(all(len(orders[c]) >= count for c in (0, 1)), f"Q0 budget {count} is infeasible")
            selected = np.concatenate([orders[0][:count], orders[1][:count]])
        output[budget] = trial_id[selected]
    for left, right in zip(requested, requested[1:]):
        require(set(output[left]).issubset(set(output[right])), "Q0 nested sample contract failed")
    return output


def sample_digest(trial_ids: Sequence[str]) -> str:
    return canonical_sha256([str(value) for value in trial_ids])


def midrank_percentile(values: np.ndarray) -> np.ndarray:
    from scipy import stats
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.all(np.isfinite(values)), "midrank input drift")
    if len(values) <= 1:
        return np.ones_like(values)
    return (stats.rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def endpoint_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    require(logits.ndim == 2 and logits.shape[1] == 2 and logits.shape[0] == len(labels), "Q0 endpoint shape drift")
    require(set(np.unique(labels)) == {0, 1}, "Q0 endpoint lacks a class")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    prediction = np.argmax(probabilities, axis=1)
    bacc = float(np.mean([np.mean(prediction[labels == c] == c) for c in (0, 1)]))
    nll = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))))
    confidence = np.max(probabilities, axis=1)
    correctness = (prediction == labels).astype(float)
    edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for index in range(15):
        right = confidence <= edges[index + 1] if index == 14 else confidence < edges[index + 1]
        mask = (confidence >= edges[index]) & right
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correctness[mask])) - float(np.mean(confidence[mask])))
    return {"bAcc": bacc, "NLL": nll, "ECE": float(ece)}


def candidate_scores(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=float)
    require(logits.ndim == 3 and logits.shape[0] == CANDIDATES and logits.shape[2] == 2, "Q0 candidate logits shape drift")
    metrics = np.asarray([
        list(endpoint_metrics(logits[index], labels).values())
        for index in range(CANDIDATES)
    ])
    oriented = np.column_stack((
        midrank_percentile(metrics[:, 0]),
        midrank_percentile(-metrics[:, 1]),
        midrank_percentile(-metrics[:, 2]),
    ))
    return np.mean(oriented, axis=1), metrics


def descending_order(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    require(scores.shape == (CANDIDATES,) and np.all(np.isfinite(scores)), "Q0 score vector drift")
    return np.lexsort((np.arange(CANDIDATES), -scores))


def select_chain(
    logits: np.ndarray,
    construction_trial_ids: Sequence[str],
    construction_labels: np.ndarray,
    *,
    dataset: str,
    target_subject: str | int,
    chain: int,
    budgets: Sequence[int | str] = PRIMARY_BUDGETS,
) -> list[dict[str, Any]]:
    logits = np.asarray(logits, dtype=float)
    trial_id = np.asarray(construction_trial_ids, dtype=str)
    labels = np.asarray(construction_labels, dtype=int)
    require(logits.shape == (CANDIDATES, len(trial_id), 2), "Q0 context logits shape drift")
    index = {value: position for position, value in enumerate(trial_id.tolist())}
    samples = nested_trial_samples(
        trial_id, labels, dataset=dataset, target_subject=target_subject,
        chain=chain, budgets=budgets,
    )
    rows: list[dict[str, Any]] = []
    full_cache: tuple[np.ndarray, np.ndarray] | None = None
    for budget in budgets:
        selected_ids = samples[budget]
        selected_index = np.asarray([index[value] for value in selected_ids], dtype=int)
        if budget == "FULL" and full_cache is not None:
            scores, metrics = full_cache
        else:
            scores, metrics = candidate_scores(logits[:, selected_index], labels[selected_index])
            if budget == "FULL":
                full_cache = scores, metrics
        order = descending_order(scores)
        rows.append({
            "dataset": dataset,
            "target_subject_id": str(target_subject),
            "chain": int(chain),
            "chain_seed": int(stream_seed(dataset, target_subject, chain)),
            "budget": str(budget),
            "sample_trial_id_sha256": sample_digest(selected_ids),
            "sample_size": len(selected_ids),
            "selected_candidate_index": int(order[0]),
            "top5_candidate_indices": order[:5].astype(int).tolist(),
            "top10_candidate_indices": order[:10].astype(int).tolist(),
            "candidate_score_vector_sha256": hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest(),
            "construction_metrics_sha256": hashlib.sha256(np.asarray(metrics, dtype="<f8").tobytes()).hexdigest(),
        })
    return rows

