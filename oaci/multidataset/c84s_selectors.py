"""Frozen C84 zero-label and strict-source selector formulas.

The formulas are the C82 registry formulas generalized only over the frozen
class dimension K=2. This module has no target-label-view dependency.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize, special, stats

from .c84s_common import C84SContractError, require


CANDIDATES = 81
PRIMARY_ZERO_METHODS = ("U5", "U7", "U11", "U13", "U14", "U15")
SELECTION_METHODS = ("B1", "B2", "B3", "B4O", "B4S", "S1") + PRIMARY_ZERO_METHODS


def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=float)
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def balanced_accuracy(
    prediction: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray | None = None,
    *,
    n_classes: int | None = None,
) -> float:
    prediction = np.asarray(prediction, dtype=int)
    labels = np.asarray(labels, dtype=int)
    require(prediction.shape == labels.shape and prediction.ndim == 1, "balanced-accuracy shape drift")
    if domains is None:
        domains = np.zeros(len(labels), dtype=int)
    domains = np.asarray(domains)
    require(domains.shape == labels.shape, "source-domain shape drift")
    classes = int(np.max(labels)) + 1 if n_classes is None else int(n_classes)
    require(set(np.unique(labels)) == set(range(classes)), "source class mapping drift")
    domain_scores = []
    for domain in np.unique(domains):
        recalls = []
        for class_id in range(classes):
            mask = (domains == domain) & (labels == class_id)
            require(np.any(mask), "source domain/class cell is absent")
            recalls.append(float(np.mean(prediction[mask] == class_id)))
        domain_scores.append(float(np.mean(recalls)))
    return float(np.mean(domain_scores))


def _domain_mean(values: np.ndarray, domains: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    domains = np.asarray(domains)
    return float(np.mean([np.mean(values[domains == domain]) for domain in np.unique(domains)]))


def source_summary(
    probabilities: np.ndarray, labels: np.ndarray, domains: np.ndarray,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    domains = np.asarray(domains)
    require(probabilities.ndim == 2 and probabilities.shape[0] == len(labels), "source probability shape drift")
    require(probabilities.shape[1] == 2, "C84 source probabilities must use K=2")
    prediction = np.argmax(probabilities, axis=1)
    true_probability = probabilities[np.arange(len(labels)), labels]
    return {
        "bacc": balanced_accuracy(prediction, labels, domains, n_classes=2),
        "negative_nll": -_domain_mean(-np.log(np.clip(true_probability, 1e-12, 1.0)), domains),
        "mean_msp": _domain_mean(np.max(probabilities, axis=1), domains),
    }


def atc_threshold(confidence: np.ndarray, error_rate: float) -> float:
    ordered = np.sort(np.asarray(confidence, dtype=float), kind="mergesort")
    below = int(np.rint(np.clip(error_rate, 0.0, 1.0) * len(ordered)))
    if below <= 0:
        return -math.inf
    if below >= len(ordered):
        return math.inf
    return float((ordered[below - 1] + ordered[below]) / 2.0)


def score_atc(
    source_probabilities: np.ndarray, source_labels: np.ndarray,
    source_domains: np.ndarray, target_probabilities: np.ndarray,
) -> float:
    summary = source_summary(source_probabilities, source_labels, source_domains)
    threshold = atc_threshold(np.max(source_probabilities, axis=1), 1.0 - summary["bacc"])
    return float(np.mean(np.max(target_probabilities, axis=1) > threshold))


def score_nuclear_norm(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    require(probabilities.ndim == 2 and probabilities.shape[1] == 2, "NuclearNorm input shape drift")
    n, k = probabilities.shape
    return float(np.linalg.norm(probabilities, ord="nuc") / math.sqrt(min(n, k) * n))


def score_mano(logits: np.ndarray, p: int = 4) -> float:
    logits = np.asarray(logits, dtype=float)
    require(logits.ndim == 2 and logits.shape[1] == 2 and p == 4, "MaNo fixed input/p drift")
    uniform = np.full_like(logits, 1.0 / logits.shape[1])
    log_probability = logits - special.logsumexp(logits, axis=1, keepdims=True)
    delta = float(-np.mean(np.sum(uniform * log_probability, axis=1)))
    if delta > 5.0:
        matrix = softmax(logits)
    else:
        matrix = logits + 1.0 + 0.5 * logits ** 2
        matrix -= np.min(matrix, axis=1, keepdims=True)
        denominator = np.sum(matrix, axis=1, keepdims=True)
        matrix = np.divide(
            matrix, denominator, out=np.full_like(matrix, 1.0 / matrix.shape[1]),
            where=denominator > 0,
        )
    return float(np.linalg.norm(matrix.reshape(-1), ord=p) / (matrix.size ** (1.0 / p)))


def largest_remainder_counts(prior: np.ndarray, n: int) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    require(prior.ndim == 1 and len(prior) == 2, "C84 COT prior shape drift")
    require(np.all(prior >= 0) and np.isclose(np.sum(prior), 1.0), "COTT prior is invalid")
    exact = prior * n
    counts = np.floor(exact).astype(int)
    remaining = n - int(np.sum(counts))
    order = np.lexsort((np.arange(len(prior)), -(exact - counts)))
    counts[order[:remaining]] += 1
    return counts


def cot_matched_costs(probabilities: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Exact capacitated assignment using repeated class slots."""
    probabilities = np.asarray(probabilities, dtype=float)
    require(probabilities.ndim == 2 and probabilities.shape[1] == 2, "COTT target shape drift")
    n = probabilities.shape[0]
    counts = largest_remainder_counts(prior, n)
    slots = np.repeat(np.arange(2), counts)
    require(len(slots) == n, "COTT class-slot arithmetic drift")
    costs = 1.0 - probabilities[:, slots]
    row, column = optimize.linear_sum_assignment(costs)
    assigned_class = slots[column[np.argsort(row)]]
    return 1.0 - probabilities[np.arange(n), assigned_class]


def score_cott(
    source_probabilities: np.ndarray, source_labels: np.ndarray,
    source_domains: np.ndarray, target_probabilities: np.ndarray,
    prior: np.ndarray,
) -> float:
    summary = source_summary(source_probabilities, source_labels, source_domains)
    source_cost = 1.0 - source_probabilities[np.arange(len(source_labels)), np.asarray(source_labels, dtype=int)]
    threshold = atc_threshold(-source_cost, 1.0 - summary["bacc"])
    target_cost = cot_matched_costs(target_probabilities, prior)
    return float(np.mean(-target_cost > threshold))


def score_snd(probabilities: np.ndarray, temperature: float = 0.05) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    require(probabilities.ndim == 2 and probabilities.shape[1] == 2, "SND input shape drift")
    require(temperature == 0.05, "SND temperature drift")
    norm = np.linalg.norm(probabilities, axis=1, keepdims=True)
    features = probabilities / np.clip(norm, 1e-12, None)
    similarity = features @ features.T / temperature
    np.fill_diagonal(similarity, -np.inf)
    maximum = np.max(similarity, axis=1, keepdims=True)
    weights = np.exp(similarity - maximum)
    np.fill_diagonal(weights, 0.0)
    denominator = np.sum(weights, axis=1, keepdims=True)
    require(np.all(denominator > 0), "SND neighborhood denominator is zero")
    weights /= denominator
    return float(np.mean(entropy(weights, axis=1)))


def agreement_matrix(prediction: np.ndarray, n_classes: int = 2) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=int)
    require(prediction.ndim == 2 and np.all((prediction >= 0) & (prediction < n_classes)), "ALine prediction shape/value drift")
    one_hot = np.eye(n_classes, dtype=float)[prediction]
    return sum(
        one_hot[:, :, class_id] @ one_hot[:, :, class_id].T
        for class_id in range(n_classes)
    ) / prediction.shape[1]


def score_aline(
    source_prediction: np.ndarray, target_prediction: np.ndarray,
    source_bacc: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    source_agreement = agreement_matrix(source_prediction)
    target_agreement = agreement_matrix(target_prediction)
    upper = np.triu_indices(source_prediction.shape[0], 1)
    n_source, n_target = source_prediction.shape[1], target_prediction.shape[1]
    source_rate = (source_agreement[upper] * n_source + 0.5) / (n_source + 1.0)
    target_rate = (target_agreement[upper] * n_target + 0.5) / (n_target + 1.0)
    x, y = stats.norm.ppf(source_rate), stats.norm.ppf(target_rate)
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    bacc_rate = (np.asarray(source_bacc, dtype=float) * n_source + 0.5) / (n_source + 1.0)
    score = stats.norm.cdf(slope * stats.norm.ppf(bacc_rate) + intercept)
    residual = y - (slope * x + intercept)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return score, {
        "slope": float(slope), "intercept": float(intercept),
        "pair_count": int(len(x)),
        "pair_R2": float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else 0.0,
    }


def descending_order(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    require(scores.shape == (CANDIDATES,) and np.all(np.isfinite(scores) | np.isneginf(scores)), "candidate score shape/value drift")
    return np.lexsort((np.arange(CANDIDATES), -scores))


def validate_canonical_candidate_metadata(
    regimes: Sequence[str], trajectory_orders: Sequence[int], candidate_ids: Sequence[str] | None = None,
) -> None:
    regimes_array = np.asarray(regimes)
    order_array = np.asarray(trajectory_orders, dtype=int)
    require(regimes_array.shape == order_array.shape == (CANDIDATES,), "candidate metadata shape drift")
    require(regimes_array.tolist() == ["ERM"] + ["OACI"] * 40 + ["SRC"] * 40,
            "candidate regime canonical order drift")
    require(order_array.tolist() == [0] + list(range(1, 41)) + list(range(1, 41)),
            "candidate trajectory canonical order drift")
    if candidate_ids is not None:
        require(len(candidate_ids) == CANDIDATES and len(set(map(str, candidate_ids))) == CANDIDATES,
                "candidate ID coverage/uniqueness drift")


def score_context(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    source_domains: np.ndarray,
    target_logits: np.ndarray,
    regimes: Sequence[str],
    trajectory_orders: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    source_probability = np.asarray(source_probabilities, dtype=float)
    target_logits = np.asarray(target_logits, dtype=float)
    require(source_probability.ndim == 3 and source_probability.shape[0] == CANDIDATES, "source candidate shape drift")
    require(target_logits.ndim == 3 and target_logits.shape[0] == CANDIDATES, "target candidate shape drift")
    require(source_probability.shape[2] == target_logits.shape[2] == 2, "C84 selector class dimension drift")
    regimes_array = np.asarray(regimes)
    order_array = np.asarray(trajectory_orders, dtype=int)
    validate_canonical_candidate_metadata(regimes_array, order_array)
    target_probability = softmax(target_logits)
    summaries = [source_summary(source_probability[i], source_labels, source_domains) for i in range(CANDIDATES)]
    source_bacc = np.asarray([row["bacc"] for row in summaries])
    prior = np.bincount(np.asarray(source_labels, dtype=int), minlength=2).astype(float)
    prior /= np.sum(prior)
    scores: dict[str, np.ndarray] = {}
    fixed = {
        "B1": np.where(regimes_array == "ERM")[0],
        "B2": np.where((regimes_array == "OACI") & (order_array == 40))[0],
        "B3": np.where((regimes_array == "SRC") & (order_array == 40))[0],
        "B4O": np.where((regimes_array == "OACI") & (order_array == 20))[0],
        "B4S": np.where((regimes_array == "SRC") & (order_array == 20))[0],
    }
    for method, matches in fixed.items():
        require(len(matches) == 1, f"fixed control identity drift: {method}")
        value = np.full(CANDIDATES, -np.inf)
        value[int(matches[0])] = 1.0
        scores[method] = value
    scores["S1"] = source_bacc
    scores["U5"] = np.asarray([score_nuclear_norm(p) for p in target_probability])
    scores["U7"] = np.asarray([
        score_atc(source_probability[i], source_labels, source_domains, target_probability[i])
        for i in range(CANDIDATES)
    ])
    scores["U11"] = np.asarray([score_mano(value) for value in target_logits])
    scores["U13"] = np.asarray([
        score_cott(source_probability[i], source_labels, source_domains, target_probability[i], prior)
        for i in range(CANDIDATES)
    ])
    scores["U14"] = np.asarray([score_snd(value) for value in target_probability])
    scores["U15"], aline = score_aline(
        np.argmax(source_probability, axis=2), np.argmax(target_probability, axis=2), source_bacc,
    )
    require(set(scores) == set(SELECTION_METHODS), "frozen selector execution set drift")
    require(all(value.shape == (CANDIDATES,) for value in scores.values()), "selector score vector shape drift")
    return scores, aline


def load_frozen_context_artifacts(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    target_subject_id: str | int,
) -> dict[str, Any]:
    """Future Stage B adapter for one already-hashed 81-candidate context."""
    require(len(descriptors) == CANDIDATES, "frozen context requires 81 descriptors")
    ordered = sorted(
        descriptors,
        key=lambda row: (
            0 if row["regime"] == "ERM" else 1 if row["regime"] == "OACI" else 2,
            int(row["trajectory_order"]), str(row["unit_id"]),
        ),
    )
    regimes = [str(row["regime"]) for row in ordered]
    trajectory_orders = [int(row["trajectory_order"]) for row in ordered]
    candidate_ids = [str(row["unit_id"]) for row in ordered]
    validate_canonical_candidate_metadata(regimes, trajectory_orders, candidate_ids)
    source_probabilities, target_logits = [], []
    source_labels = source_domains = source_trial_ids = target_trial_ids = None
    for descriptor in ordered:
        with np.load(descriptor["source_audit"]["path"], allow_pickle=False) as archive:
            require(str(archive["unit_id"].item()) == str(descriptor["unit_id"]), "source artifact unit drift")
            candidate_source_labels = np.asarray(archive["source_class_label"], dtype=int)
            candidate_source_domains = np.asarray(archive["source_domain_id"])
            candidate_source_ids = np.asarray(archive["source_trial_id"], dtype=str)
            if source_labels is None:
                source_labels, source_domains, source_trial_ids = (
                    candidate_source_labels, candidate_source_domains, candidate_source_ids,
                )
            else:
                require(np.array_equal(candidate_source_labels, source_labels), "source labels differ across candidates")
                require(np.array_equal(candidate_source_domains, source_domains), "source domains differ across candidates")
                require(np.array_equal(candidate_source_ids, source_trial_ids), "source trial order differs across candidates")
            source_probabilities.append(np.asarray(archive["probabilities"], dtype=float))
        with np.load(descriptor["complete_target_unlabeled"]["path"], allow_pickle=False) as archive:
            require(str(archive["unit_id"].item()) == str(descriptor["unit_id"]), "target artifact unit drift")
            subject = np.asarray(archive["target_subject_id"]).astype(str)
            mask = subject == str(target_subject_id)
            require(np.any(mask), "target subject absent from candidate artifact")
            candidate_target_ids = np.asarray(archive["target_trial_id"], dtype=str)[mask]
            if target_trial_ids is None:
                target_trial_ids = candidate_target_ids
            else:
                require(np.array_equal(candidate_target_ids, target_trial_ids), "target trial order differs across candidates")
            target_logits.append(np.asarray(archive["logits"], dtype=float)[mask])
    return {
        "candidate_ids": candidate_ids, "regimes": regimes,
        "trajectory_orders": trajectory_orders,
        "source_probabilities": np.stack(source_probabilities),
        "source_labels": np.asarray(source_labels),
        "source_domains": np.asarray(source_domains),
        "source_trial_ids": np.asarray(source_trial_ids),
        "target_logits": np.stack(target_logits),
        "target_trial_ids": np.asarray(target_trial_ids),
    }
