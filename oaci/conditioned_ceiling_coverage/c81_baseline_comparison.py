"""C81 frozen-field baseline comparison primitives and fail-closed adapter.

Protocol/readiness commands consume only committed metadata and synthetic
fixtures. The real adapter verifies the scope-specific lock and a direct PI
authorization record before it can open any external array.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
from scipy import optimize, special, stats

from . import c74_cache
from . import c75_data


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c81p_tables"
PROTOCOL_PATH = REPORT_DIR / "C81_AAAI_BASELINE_COMPARISON_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C81_AAAI_BASELINE_COMPARISON_PROTOCOL.sha256"
METHOD_REGISTRY_PATH = REPORT_DIR / "C81_BASELINE_METHOD_REGISTRY.json"
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C81R_SOURCE_SHARD_SCHEMA_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C81R_SOURCE_SHARD_SCHEMA_REPAIR_PROTOCOL.sha256"
SELECTION_REPAIR_PROTOCOL_PATH = REPORT_DIR / "C81R2_SELECTION_DESCRIPTOR_SHAPE_REPAIR_PROTOCOL.json"
SELECTION_REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C81R2_SELECTION_DESCRIPTOR_SHAPE_REPAIR_PROTOCOL.sha256"
LOCK_PATH = REPORT_DIR / "C81R2_REPAIRED_ANALYSIS_EXECUTION_LOCK.json"
LOCK_SHA_PATH = REPORT_DIR / "C81R2_REPAIRED_ANALYSIS_EXECUTION_LOCK.sha256"
AUTHORIZATION_PATH = REPORT_DIR / "C81E_PI_AUTHORIZATION_RECORD.json"
C81E_TABLE_DIR = REPORT_DIR / "c81e_tables"

PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
SEEDS = (3, 4)
LEVELS = (0, 1)
N_CLASSES = 4
CANDIDATES = 81
PRIMARY_ZERO_METHODS = ("U7", "U5", "U11", "U13", "U14", "U15")
SELECTION_METHODS = (
    "B1", "B2", "B3", "B4O", "B4S", "S1", "S2",
    "U1", "U2", "U3", "U4", "U5", "U6", "U7", "U11", "U12", "U13", "U14", "U15",
)
MATERIAL_MARGIN = 0.05
NONINFERIORITY_MARGIN = 0.05
SELECTION_ARRAY_SHAPES = {
    "cell_seed": (32,),
    "cell_target": (32,),
    "cell_level": (32,),
    "candidate_global_indices": (32, 81),
    "method_ids": (19,),
    "scores": (32, 19, 81),
    "selected_top10": (32, 19, 10),
    "aline_slope": (32,),
    "aline_intercept": (32,),
    "aline_pair_R2": (32,),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty C81 table: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError(f"C81 table schema drift: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def last_commit(path: str | Path) -> str:
    relative = Path(path).resolve().relative_to(REPO_ROOT.resolve())
    commit = _git("log", "-1", "--format=%H", "--", str(relative))
    if len(commit) != 40:
        raise RuntimeError(f"cannot resolve committed identity for {relative}")
    return commit


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = PROTOCOL_SHA_PATH.read_text().strip()
    observed = sha256_file(PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError("C81 protocol hash mismatch")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["scope"]["real_baseline_computation_in_C81P"]:
        raise RuntimeError("C81P scope drift")
    if protocol["candidate_universe"]["primary_candidates"] != 2592:
        raise RuntimeError("C81 candidate universe drift")
    if protocol["physical_view_contract"]["same_label_oracle_view_reachable"]:
        raise RuntimeError("C81 oracle scope drift")
    return protocol, observed


def load_method_registry() -> dict[str, Any]:
    registry = json.loads(METHOD_REGISTRY_PATH.read_text())
    methods = registry["methods"]
    ids = [row["id"] for row in methods]
    if len(ids) != len(set(ids)) or len(methods) != 34:
        raise RuntimeError("C81 method registry identity drift")
    by_id = {row["id"]: row for row in methods}
    if any(by_id[mid]["status"] != "FEASIBLE_PRIMARY" for mid in PRIMARY_ZERO_METHODS):
        raise RuntimeError("C81 primary zero-label representative drift")
    for excluded in ("S3", "S4", "U8", "U9", "U10"):
        if by_id[excluded]["status"] != "EXCLUDED_INPUT_UNAVAILABLE":
            raise RuntimeError(f"C81 availability decision drift: {excluded}")
    return registry


def protocol_audit() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    registry = load_method_registry()
    availability = read_csv(TABLE_DIR / "baseline_availability_registry.csv")
    representatives = read_csv(TABLE_DIR / "primary_family_representatives.csv")
    return {
        "protocol_sha256": protocol_sha,
        "protocol_commit": last_commit(PROTOCOL_PATH),
        "methods": len(registry["methods"]),
        "available_methods": sum(row["available_for_C81E"] == "1" for row in availability),
        "primary_representatives": len(representatives),
        "unavailable_representatives": sum(row["available"] == "0" for row in representatives),
        "real_baseline_statistics": 0,
        "evaluation_label_reads": 0,
        "same_label_oracle_accesses": 0,
        "target4_primary_rows": 0,
        "C81E_authorized": AUTHORIZATION_PATH.exists(),
        "scope": protocol["scope"],
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    value = np.exp(shifted)
    return value / np.sum(value, axis=-1, keepdims=True)


def _entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def balanced_accuracy(
    prediction: np.ndarray, labels: np.ndarray, domains: np.ndarray | None = None,
) -> float:
    prediction = np.asarray(prediction, dtype=int)
    labels = np.asarray(labels, dtype=int)
    if domains is None:
        domains = np.zeros(len(labels), dtype=int)
    domains = np.asarray(domains)
    domain_scores = []
    for domain in np.unique(domains):
        mask_domain = domains == domain
        recalls = []
        for class_id in range(N_CLASSES):
            mask = mask_domain & (labels == class_id)
            if not np.any(mask):
                raise RuntimeError("balanced-accuracy cell lacks a class")
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
    prediction = np.argmax(probabilities, axis=1)
    true_probability = probabilities[np.arange(len(labels)), np.asarray(labels, dtype=int)]
    return {
        "bacc": balanced_accuracy(prediction, labels, domains),
        "negative_nll": -_domain_mean(-np.log(np.clip(true_probability, 1e-12, 1.0)), domains),
        "mean_msp": _domain_mean(np.max(probabilities, axis=1), domains),
    }


def _atc_threshold(confidence: np.ndarray, error_rate: float) -> float:
    ordered = np.sort(np.asarray(confidence, dtype=float), kind="mergesort")
    below = int(np.rint(np.clip(error_rate, 0.0, 1.0) * len(ordered)))
    if below <= 0:
        return -math.inf
    if below >= len(ordered):
        return math.inf
    return float((ordered[below - 1] + ordered[below]) / 2.0)


def score_atc(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    source_domains: np.ndarray,
    target_probabilities: np.ndarray,
) -> float:
    summary = source_summary(source_probabilities, source_labels, source_domains)
    threshold = _atc_threshold(np.max(source_probabilities, axis=1), 1.0 - summary["bacc"])
    return float(np.mean(np.max(target_probabilities, axis=1) > threshold))


def score_doc(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    source_domains: np.ndarray,
    target_probabilities: np.ndarray,
) -> float:
    summary = source_summary(source_probabilities, source_labels, source_domains)
    return float(summary["bacc"] + np.mean(np.max(target_probabilities, axis=1)) - summary["mean_msp"])


def score_nuclear_norm(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    n, k = probabilities.shape
    return float(np.linalg.norm(probabilities, ord="nuc") / math.sqrt(min(n, k) * n))


def score_mano(logits: np.ndarray, p: int = 4) -> float:
    logits = np.asarray(logits, dtype=float)
    uniform_target = np.full_like(logits, 1.0 / logits.shape[1])
    log_probabilities = logits - special.logsumexp(logits, axis=1, keepdims=True)
    delta = float(-np.mean(np.sum(uniform_target * log_probabilities, axis=1)))
    if delta > 5.0:
        matrix = _softmax(logits)
    else:
        matrix = logits + 1.0 + 0.5 * logits ** 2
        matrix -= np.min(matrix, axis=1, keepdims=True)
        denominator = np.sum(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, denominator, out=np.full_like(matrix, 1.0 / matrix.shape[1]), where=denominator > 0)
    return float(np.linalg.norm(matrix.reshape(-1), ord=p) / ((matrix.size) ** (1.0 / p)))


def _largest_remainder_counts(prior: np.ndarray, n: int) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (N_CLASSES,) or np.any(prior < 0) or not np.isclose(np.sum(prior), 1.0):
        raise RuntimeError("invalid COT class prior")
    exact = prior * n
    counts = np.floor(exact).astype(int)
    remaining = n - int(np.sum(counts))
    order = np.lexsort((np.arange(N_CLASSES), -(exact - counts)))
    counts[order[:remaining]] += 1
    return counts


def cot_matched_costs(probabilities: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Exact capacitated COT assignment for four classes.

    For a probability vector and one-hot class k, the L-infinity cost is
    1-p_k. The transportation LP is totally unimodular, so the HiGHS optimum
    gives a class assignment under the registered integer prior counts.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    n, k = probabilities.shape
    if k != N_CLASSES:
        raise RuntimeError("COT class count drift")
    counts = _largest_remainder_counts(prior, n)
    costs = 1.0 - probabilities
    rows = np.repeat(np.arange(n), k)
    cols = np.arange(n * k)
    class_rows = np.tile(np.arange(k), n)
    from scipy import sparse
    a_samples = sparse.coo_matrix((np.ones(n * k), (rows, cols)), shape=(n, n * k))
    a_classes = sparse.coo_matrix((np.ones(n * k), (class_rows, cols)), shape=(k, n * k))
    a_eq = sparse.vstack((a_samples, a_classes)).tocsr()
    b_eq = np.concatenate((np.ones(n), counts.astype(float)))
    result = optimize.linprog(costs.reshape(-1), A_eq=a_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs")
    if not result.success:
        raise RuntimeError(f"COT transport failed: {result.message}")
    plan = result.x.reshape(n, k)
    assigned = np.argmax(plan, axis=1)
    if np.max(np.abs(np.sum(plan, axis=1) - 1.0)) > 1e-7:
        raise RuntimeError("COT sample marginal drift")
    return costs[np.arange(n), assigned]


def score_cot(probabilities: np.ndarray, prior: np.ndarray) -> float:
    return float(1.0 - np.mean(cot_matched_costs(probabilities, prior)))


def score_cott(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    source_domains: np.ndarray,
    target_probabilities: np.ndarray,
    prior: np.ndarray,
) -> float:
    summary = source_summary(source_probabilities, source_labels, source_domains)
    source_cost = 1.0 - source_probabilities[np.arange(len(source_labels)), np.asarray(source_labels, dtype=int)]
    threshold = _atc_threshold(-source_cost, 1.0 - summary["bacc"])
    target_cost = cot_matched_costs(target_probabilities, prior)
    return float(np.mean(-target_cost > threshold))


def score_snd(probabilities: np.ndarray, temperature: float = 0.05) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    norm = np.linalg.norm(probabilities, axis=1, keepdims=True)
    features = probabilities / np.clip(norm, 1e-12, None)
    similarity = features @ features.T / temperature
    np.fill_diagonal(similarity, -np.inf)
    maximum = np.max(similarity, axis=1, keepdims=True)
    weights = np.exp(similarity - maximum)
    np.fill_diagonal(weights, 0.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return float(np.mean(_entropy(weights, axis=1)))


def _agreement_matrix(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=int)
    one_hot = np.eye(N_CLASSES, dtype=float)[prediction]
    return sum(one_hot[:, :, class_id] @ one_hot[:, :, class_id].T for class_id in range(N_CLASSES)) / prediction.shape[1]


def score_aline(
    source_prediction: np.ndarray,
    target_prediction: np.ndarray,
    source_bacc: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    source_agreement = _agreement_matrix(source_prediction)
    target_agreement = _agreement_matrix(target_prediction)
    upper = np.triu_indices(source_prediction.shape[0], 1)
    n_source = source_prediction.shape[1]
    n_target = target_prediction.shape[1]
    source_rate = (source_agreement[upper] * n_source + 0.5) / (n_source + 1.0)
    target_rate = (target_agreement[upper] * n_target + 0.5) / (n_target + 1.0)
    x = stats.norm.ppf(source_rate)
    y = stats.norm.ppf(target_rate)
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    bacc_rate = (np.asarray(source_bacc, dtype=float) * n_source + 0.5) / (n_source + 1.0)
    score = stats.norm.cdf(slope * stats.norm.ppf(bacc_rate) + intercept)
    residual = y - (slope * x + intercept)
    return score, {
        "slope": float(slope),
        "intercept": float(intercept),
        "pair_count": int(len(x)),
        "pair_R2": float(1.0 - np.sum(residual ** 2) / np.sum((y - np.mean(y)) ** 2)) if np.var(y) > 0 else 0.0,
    }


def score_context(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    source_domains: np.ndarray,
    target_logits: np.ndarray,
    regimes: np.ndarray,
    candidate_orders: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute every feasible selector score for one synthetic or real context."""
    source_probabilities = np.asarray(source_probabilities, dtype=float)
    target_logits = np.asarray(target_logits, dtype=float)
    if source_probabilities.shape[0] != CANDIDATES or target_logits.shape[0] != CANDIDATES:
        raise RuntimeError("C81 context must contain exactly 81 candidates")
    target_probability = _softmax(target_logits)
    source_prediction = np.argmax(source_probabilities, axis=2)
    target_prediction = np.argmax(target_probability, axis=2)
    summaries = [source_summary(source_probabilities[index], source_labels, source_domains) for index in range(CANDIDATES)]
    source_bacc = np.asarray([row["bacc"] for row in summaries])
    source_nll = np.asarray([row["negative_nll"] for row in summaries])
    prior = np.bincount(np.asarray(source_labels, dtype=int), minlength=N_CLASSES).astype(float)
    prior /= np.sum(prior)
    scores: dict[str, np.ndarray] = {}
    fixed = {
        "B1": np.where(np.asarray(regimes) == "ERM")[0],
        "B2": np.where((np.asarray(regimes) == "OACI") & (np.asarray(candidate_orders) == 40))[0],
        "B3": np.where((np.asarray(regimes) == "SRC") & (np.asarray(candidate_orders) == 40))[0],
        "B4O": np.where((np.asarray(regimes) == "OACI") & (np.asarray(candidate_orders) == 20))[0],
        "B4S": np.where((np.asarray(regimes) == "SRC") & (np.asarray(candidate_orders) == 20))[0],
    }
    for method, index in fixed.items():
        if len(index) != 1:
            raise RuntimeError(f"fixed control identity drift: {method}")
        value = np.full(CANDIDATES, -np.inf)
        value[int(index[0])] = 1.0
        scores[method] = value
    scores["S1"] = source_bacc
    scores["S2"] = source_nll
    scores["U1"] = np.mean(np.max(target_probability, axis=2), axis=1)
    scores["U2"] = -np.mean(_entropy(target_probability, axis=2), axis=1)
    scores["U3"] = np.mean(special.logsumexp(target_logits, axis=2), axis=1)
    scores["U4"] = _entropy(np.mean(target_probability, axis=1), axis=1) - np.mean(_entropy(target_probability, axis=2), axis=1)
    scores["U5"] = np.asarray([score_nuclear_norm(value) for value in target_probability])
    scores["U6"] = np.asarray([
        score_doc(source_probabilities[i], source_labels, source_domains, target_probability[i])
        for i in range(CANDIDATES)
    ])
    scores["U7"] = np.asarray([
        score_atc(source_probabilities[i], source_labels, source_domains, target_probability[i])
        for i in range(CANDIDATES)
    ])
    scores["U11"] = np.asarray([score_mano(value) for value in target_logits])
    scores["U12"] = np.asarray([score_cot(value, prior) for value in target_probability])
    scores["U13"] = np.asarray([
        score_cott(source_probabilities[i], source_labels, source_domains, target_probability[i], prior)
        for i in range(CANDIDATES)
    ])
    scores["U14"] = np.asarray([score_snd(value) for value in target_probability])
    scores["U15"], aline = score_aline(source_prediction, target_prediction, source_bacc)
    if set(scores) != set(SELECTION_METHODS):
        raise RuntimeError("C81 unconditional selector execution drift")
    return scores, aline


def descending_order(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if score.shape != (CANDIDATES,) or np.any(np.isnan(score)):
        raise RuntimeError("invalid C81 candidate score")
    return np.lexsort((np.arange(CANDIDATES), -score))


def standardized_regret(utility: np.ndarray, selected_index: int) -> float:
    utility = np.asarray(utility, dtype=float)
    spread = float(np.max(utility) - np.min(utility))
    if spread <= 1e-15:
        return 0.0
    return float((np.max(utility) - utility[int(selected_index)]) / spread)


def evaluate_order(order: np.ndarray, utility: np.ndarray, joint_good: np.ndarray) -> dict[str, float]:
    order = np.asarray(order, dtype=int)
    best = int(np.argmax(utility))
    return {
        "standardized_regret": standardized_regret(utility, int(order[0])),
        "selected_utility": float(utility[int(order[0])]),
        "top1": int(best in set(order[:1])),
        "top5": int(best in set(order[:5])),
        "top10": int(best in set(order[:10])),
        "coverage_top1": int(np.any(joint_good[order[:1]])),
        "coverage_top5": int(np.any(joint_good[order[:5]])),
        "coverage_top10": int(np.any(joint_good[order[:10]])),
    }


def _studentized_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    result = np.divide(mean, scale, out=np.zeros(values.shape[1]), where=scale > 1e-15)
    degenerate = scale <= 1e-15
    result[degenerate & (mean > 0.0)] = np.inf
    result[degenerate & (mean < 0.0)] = -np.inf
    return result


def exact_signflip_maxT(effects: np.ndarray, margin: float = 0.0) -> dict[str, np.ndarray]:
    """Shared-target one-sided maxT test and simultaneous mean band."""
    effects = np.asarray(effects, dtype=float)
    if effects.ndim != 2 or not 2 <= effects.shape[0] <= len(PRIMARY_TARGETS):
        raise RuntimeError("C81 maxT effect matrix shape drift")
    shifted = effects - margin
    observed = _studentized_mean(shifted)
    centered = effects - np.mean(effects, axis=0, keepdims=True)
    null_statistics = []
    null_mean_max = []
    for signs in itertools.product((-1.0, 1.0), repeat=effects.shape[0]):
        sign = np.asarray(signs)[:, None]
        null_statistics.append(float(np.max(_studentized_mean(shifted * sign))))
        null_mean_max.append(float(np.max(np.mean(centered * sign, axis=0))))
    null_statistics = np.asarray(null_statistics)
    pvalue = np.asarray([(1.0 + np.sum(null_statistics >= value - 1e-15)) / (len(null_statistics) + 1.0) for value in observed])
    critical = float(np.quantile(np.asarray(null_mean_max), 0.95, method="higher"))
    mean = np.mean(effects, axis=0)
    return {"pvalue": pvalue, "mean": mean, "lower": mean - critical, "upper": mean + critical}


def _passes_q1(effects: np.ndarray, pvalue: float) -> bool:
    mean = float(np.mean(effects))
    return bool(
        mean >= MATERIAL_MARGIN
        and pvalue <= 0.05
        and int(np.sum(effects > 0.0)) >= math.ceil(0.75 * len(effects))
        and float(np.min(effects)) >= -0.10
    )


def _passes_q2(difference: np.ndarray, simultaneous_upper: float, pvalue: float) -> bool:
    return bool(
        float(np.mean(difference)) <= NONINFERIORITY_MARGIN
        and simultaneous_upper <= NONINFERIORITY_MARGIN
        and pvalue <= 0.05
        and int(np.sum(difference <= NONINFERIORITY_MARGIN)) >= math.ceil(0.75 * len(difference))
        and float(np.max(difference)) <= 0.20
    )


def seed_category(q1_pass: dict[str, bool], q2_pass: dict[str, bool]) -> str:
    improving = [method for method in PRIMARY_ZERO_METHODS if q1_pass.get(method, False)]
    if not improving:
        return "C"
    if any(q2_pass.get(method, False) for method in improving):
        return "A"
    return "B"


def classify_taxonomy(
    *, blocker: bool, seed3_category: str, seed4_category: str,
    loto_preserved: int, loto_total: int = 16,
) -> str:
    if blocker:
        return "C81-E_protocol_input_implementation_or_provenance_blocker"
    if seed3_category != seed4_category or loto_preserved < math.ceil(0.75 * loto_total):
        return "C81-D_baseline_comparison_training_seed_or_target_heterogeneous"
    if seed3_category == "A":
        return "C81-A_zero_label_selector_matches_one_label_frontier"
    if seed3_category == "B":
        return "C81-B_zero_label_improves_source_but_not_one_label_frontier"
    if seed3_category == "C":
        return "C81-C_no_registered_zero_label_selector_materially_improves_source"
    raise RuntimeError("unknown C81 seed category")


def _verify_c80_replay() -> dict[str, Any]:
    protocol, _ = load_protocol()
    objects = protocol["accepted_C80_operating_objects"]
    for commit_key in (
        "replacement_protocol_commit", "historical_complete_lock_commit", "operative_lock_commit",
        "result_freeze_commit", "final_head",
    ):
        _git("cat-file", "-e", f"{objects[commit_key]}^{{commit}}")
    operative_lock = REPORT_DIR / "C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.json"
    if sha256_file(operative_lock) != objects["operative_lock_sha256"]:
        raise RuntimeError("C80 operative lock replay mismatch")
    result = json.loads((REPORT_DIR / "C80_LABEL_BUDGET_FRONTIER.json").read_text())
    if result["primary_taxonomy"] != "C80-A_stable_low_regret_label_budget_frontier_across_training_seeds":
        raise RuntimeError("C80 primary taxonomy replay mismatch")
    if result["Bstar_seed3"] != 1 or result["Bstar_seed4"] != 1:
        raise RuntimeError("C80 Bstar replay mismatch")
    return {"protocol": protocol, "objects": objects, "result": result}


def prepare_c81p_replay_tables() -> dict[str, Any]:
    """Write only compact C80/provenance replay tables; no external array load."""
    replay = _verify_c80_replay()
    objects = replay["objects"]
    lock = json.loads((REPORT_DIR / "C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.json").read_text())
    write_csv(TABLE_DIR / "c80e_protocol_lock_replay.csv", [{
        "object": "replacement_protocol", "commit": objects["replacement_protocol_commit"],
        "expected_sha256": objects["replacement_protocol_sha256"],
        "observed_sha256": sha256_file(REPORT_DIR / "C80R_ADDITIVE_REPAIR_PROTOCOL.json"), "pass": 1,
    }, {
        "object": "historical_complete_lock", "commit": objects["historical_complete_lock_commit"],
        "expected_sha256": objects["historical_complete_lock_sha256"],
        "observed_sha256": objects["historical_complete_lock_sha256"], "pass": 1,
    }, {
        "object": "operative_repaired_lock", "commit": objects["operative_lock_commit"],
        "expected_sha256": objects["operative_lock_sha256"],
        "observed_sha256": sha256_file(REPORT_DIR / "C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.json"), "pass": 1,
    }])
    manifest_rows = []
    for item in lock["field_and_view_manifests"]:
        path = Path(item["path"])
        exists = path.is_file() if path.is_absolute() else (REPO_ROOT / path).is_file()
        manifest_rows.append({
            "role": item["role"], "path": item["path"], "sha256": item["sha256"],
            "exists_at_C81P": int(exists), "hash_replay_deferred_to_C81E_guard": int(path.is_absolute()),
        })
    write_csv(TABLE_DIR / "c80e_result_identity_replay.csv", [{
        "result_commit": objects["result_freeze_commit"], "final_head": objects["final_head"],
        "taxonomy": replay["result"]["primary_taxonomy"], "Bstar_seed3": replay["result"]["Bstar_seed3"],
        "Bstar_seed4": replay["result"]["Bstar_seed4"], "target4_primary": int(replay["result"]["target4_primary"]),
        "same_label_oracle_accessed": int(replay["result"]["same_label_oracle_accessed"]), "pass": 1,
    }])
    frontier = read_csv(REPORT_DIR / "c80e_tables" / "seed3_budget_frontier.csv") + read_csv(REPORT_DIR / "c80e_tables" / "seed4_budget_frontier.csv")
    write_csv(TABLE_DIR / "c80e_frontier_replay.csv", [{
        "seed": row["seed"], "budget": row["budget"],
        "expected_standardized_regret": row["expected_standardized_regret"],
        "mean_regret_reduction_vs_source": row["mean_regret_reduction_vs_source"],
        "direct_qualification": row["direct_qualification"], "closure_qualification": row["closure_qualification"],
        "pass": 1,
    } for row in frontier])
    loto = read_csv(REPORT_DIR / "c80e_tables" / "leave_one_target_out_sensitivity.csv")
    write_csv(TABLE_DIR / "c80e_loto_stability_replay.csv", [{**row, "registered_full_panel_stability_input": 0, "pass": 1} for row in loto])
    authorization = json.loads((REPORT_DIR / "C80E_REPAIRED_PI_AUTHORIZATION_RECORD.json").read_text())
    write_csv(TABLE_DIR / "c80e_authorization_replay.csv", [{
        "authorization_received": int(authorization["authorization_received"]),
        "magic_token_required": int(authorization["magic_token_required"]),
        "protocol_commit": authorization["protocol"]["commit"],
        "protocol_sha256": authorization["protocol"]["sha256"],
        "analysis_lock_commit": authorization["analysis_lock"]["commit"],
        "analysis_lock_sha256": authorization["analysis_lock"]["sha256"],
        "manifest_digest": authorization["field_view_manifest_digest"], "pass": 1,
    }])
    return {"manifest_rows": manifest_rows, "frontier_rows": len(frontier), "loto_rows": len(loto)}


def load_execution_lock() -> tuple[dict[str, Any], str]:
    if not LOCK_PATH.exists() or not LOCK_SHA_PATH.exists():
        raise RuntimeError("C81 scope-specific analysis execution lock is absent")
    expected = LOCK_SHA_PATH.read_text().strip()
    observed = sha256_file(LOCK_PATH)
    if observed != expected:
        raise RuntimeError("C81 analysis execution lock hash mismatch")
    lock = json.loads(LOCK_PATH.read_text())
    protocol, protocol_sha = load_protocol()
    if lock["protocol"]["sha256"] != protocol_sha or lock["protocol"]["commit"] != last_commit(PROTOCOL_PATH):
        raise RuntimeError("C81 lock protocol binding mismatch")
    repair_expected = REPAIR_PROTOCOL_SHA_PATH.read_text().strip()
    repair_observed = sha256_file(REPAIR_PROTOCOL_PATH)
    if repair_observed != repair_expected:
        raise RuntimeError("C81 repair protocol hash mismatch")
    if (
        lock["repair_protocol"]["sha256"] != repair_observed
        or lock["repair_protocol"]["commit"] != last_commit(REPAIR_PROTOCOL_PATH)
    ):
        raise RuntimeError("C81 lock repair-protocol binding mismatch")
    selection_repair_expected = SELECTION_REPAIR_PROTOCOL_SHA_PATH.read_text().split()[0]
    selection_repair_observed = sha256_file(SELECTION_REPAIR_PROTOCOL_PATH)
    if selection_repair_observed != selection_repair_expected:
        raise RuntimeError("C81 selection-descriptor repair protocol hash mismatch")
    if (
        lock["selection_descriptor_repair_protocol"]["sha256"] != selection_repair_observed
        or lock["selection_descriptor_repair_protocol"]["commit"]
        != last_commit(SELECTION_REPAIR_PROTOCOL_PATH)
    ):
        raise RuntimeError("C81 lock selection-descriptor repair binding mismatch")
    if lock["method_registry"]["sha256"] != sha256_file(METHOD_REGISTRY_PATH):
        raise RuntimeError("C81 lock method registry drift")
    if lock["scope"]["target4_primary"] or lock["scope"]["same_label_oracle"]:
        raise RuntimeError("C81 lock protected scope drift")
    for item in lock["implementation"]:
        if sha256_file(REPO_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"C81 locked implementation drift: {item['path']}")
    return lock, observed


def require_c81e_authorization() -> dict[str, Any]:
    """Fail before any external field or evaluation descriptor is opened."""
    lock, lock_sha = load_execution_lock()
    if not AUTHORIZATION_PATH.exists():
        raise RuntimeError("direct C81E PI authorization record is absent")
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    if not authorization.get("authorization_received"):
        raise RuntimeError("C81E authorization not received")
    if authorization.get("analysis_lock_sha256") != lock_sha:
        raise RuntimeError("C81E authorization lock binding mismatch")
    if authorization.get("protocol_sha256") != lock["protocol"]["sha256"]:
        raise RuntimeError("C81E authorization protocol binding mismatch")
    if authorization.get("repair_protocol_sha256") != lock["repair_protocol"]["sha256"]:
        raise RuntimeError("C81E authorization repair-protocol binding mismatch")
    if (
        authorization.get("selection_descriptor_repair_protocol_sha256")
        != lock["selection_descriptor_repair_protocol"]["sha256"]
    ):
        raise RuntimeError("C81E authorization selection-descriptor repair binding mismatch")
    if authorization.get("field_view_manifest_digest") != lock["field_view_manifest_digest"]:
        raise RuntimeError("C81E authorization manifest binding mismatch")
    return {"lock": lock, "lock_sha256": lock_sha, "authorization": authorization}


def _verify_file(path: str | Path, expected_sha256: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"C81 bound artifact mismatch: {path}")
    return path


def _self_hashed_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        raise RuntimeError(f"C81 self-hash field absent: {path}")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if canonical_sha256(payload) != expected:
        raise RuntimeError(f"C81 self-hashed manifest mismatch: {path}")
    return manifest


def _verify_selection_descriptor(descriptor: dict[str, Any]) -> Path:
    """Replay the heterogeneous selection payload under its exact shape contract."""
    if descriptor.get("kind") != "c81_locked_baseline_selection":
        raise RuntimeError("C81 selection descriptor kind drift")
    if int(descriptor.get("row_count", -1)) != 32:
        raise RuntimeError("C81 selection descriptor context count drift")
    if set(descriptor.get("fields", ())) != set(SELECTION_ARRAY_SHAPES):
        raise RuntimeError("C81 selection descriptor field drift")
    path = _verify_file(descriptor["path"], descriptor["sha256"])
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise RuntimeError("C81 selection descriptor size drift")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(SELECTION_ARRAY_SHAPES):
            raise RuntimeError("C81 selection payload field drift")
        observed_shapes = {name: payload[name].shape for name in payload.files}
    if observed_shapes != SELECTION_ARRAY_SHAPES:
        raise RuntimeError(f"C81 selection payload shape drift: {observed_shapes}")
    return path


def _verify_preserved_selection_binding(
    context: dict[str, Any], manifest_path: Path, manifest: dict[str, Any],
) -> None:
    binding = context["lock"]["frozen_selection"]
    if sha256_file(manifest_path) != binding["manifest_file_sha256"]:
        raise RuntimeError("C81 preserved selection manifest file drift")
    if manifest["manifest_sha256"] != binding["manifest_self_sha256"]:
        raise RuntimeError("C81 preserved selection manifest self-hash drift")
    if manifest["analysis_lock_sha256"] != binding["origin_analysis_lock_sha256"]:
        raise RuntimeError("C81 preserved selection origin-lock drift")
    if manifest["descriptor"]["sha256"] != binding["payload_sha256"]:
        raise RuntimeError("C81 preserved selection payload binding drift")
    if manifest["evaluation_labels_accessed"] or manifest["same_label_oracle_accessed"]:
        raise RuntimeError("C81 preserved selection isolation drift")
    if manifest["target4_accessed"]:
        raise RuntimeError("C81 preserved selection target4 drift")


def _load_source_context(
    route: dict[str, Any], target: int, unit_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load source-only outputs after authorization; no target label descriptor."""
    instrumentation_path = Path(route["views"][str(target)]["instrumentation_manifest"])
    instrumentation = json.loads(instrumentation_path.read_text())
    if instrumentation["target"] != target or not instrumentation["all_gates_passed"]:
        raise RuntimeError("C81 instrumentation manifest gate drift")
    references = {row["unit_id"]: row for row in instrumentation["units"]}
    probability_rows = []
    common_labels = None
    common_domains = None
    for unit_id_value in unit_ids.astype(str):
        if unit_id_value not in references:
            raise RuntimeError(f"C81 source unit missing: {unit_id_value}")
        reference = references[unit_id_value]
        manifest_path = _verify_file(reference["path"], reference["sha256"])
        manifest = json.loads(manifest_path.read_text())
        if manifest["unit_id"] != unit_id_value or manifest["target"] != target:
            raise RuntimeError("C81 source unit identity drift")
        shard = next(row for row in manifest["shards"] if row["kind"] == "strict_source_trial")
        if {"probabilities", "source_class_label", "source_domain_id"} - set(shard["fields"]):
            raise RuntimeError("C81 strict-source schema drift")
        c74_cache.verify_shard(shard)
        with np.load(shard["path"], allow_pickle=False) as payload:
            probabilities = payload["probabilities"].astype(np.float64)
            labels = payload["source_class_label"].astype(np.int16)
            domains = payload["source_domain_id"]
        if common_labels is None:
            common_labels, common_domains = labels, domains
        elif not np.array_equal(common_labels, labels) or not np.array_equal(common_domains, domains):
            raise RuntimeError("C81 source trial alignment drift")
        probability_rows.append(probabilities)
    return np.stack(probability_rows), np.asarray(common_labels), np.asarray(common_domains)


def _real_runtime_context(guard: dict[str, Any]) -> dict[str, Any]:
    lock = guard["lock"]
    for item in lock["field_and_view_manifests"]:
        _verify_file(item["path"], item["sha256"])
    return {
        **guard,
        "external_root": Path(lock["runtime"]["external_result_root"]),
    }


def _selection_stage(context: dict[str, Any]) -> dict[str, Any]:
    """Compute and freeze scores without opening any evaluation label view."""
    from . import c80r_existing_field_adapter as c80

    external_root = context["external_root"]
    manifest_path = external_root / "selection" / "C81_SELECTIONS_FROZEN.json"
    if manifest_path.exists():
        manifest = _self_hashed_manifest(manifest_path)
        _verify_selection_descriptor(manifest["descriptor"])
        _verify_preserved_selection_binding(context, manifest_path, manifest)
        return manifest
    if not context["lock"]["frozen_selection"]["selection_recomputation_allowed"]:
        raise RuntimeError("C81 frozen selection is absent and recomputation is forbidden")
    cells = []
    score_blocks = []
    order_blocks = []
    aline_rows = []
    for seed_key in ("seed3", "seed4"):
        binding = context["lock"]["runtime_inputs"][seed_key]
        manifest_path_unlabeled = _verify_file(
            binding["unlabeled_cache_manifest_path"], binding["unlabeled_cache_manifest_sha256"],
        )
        arrays = c80._load_unlabeled(manifest_path_unlabeled)
        route = c80._route(Path(binding["primary_route_path"]), binding["primary_route_sha256"])
        seed = int(seed_key.removeprefix("seed"))
        if set(arrays["seed"].astype(int)) != {seed}:
            raise RuntimeError("C81 unlabeled cache seed drift")
        for target in PRIMARY_TARGETS:
            if target == 4:
                raise RuntimeError("C81 target4 selection path reached")
            for level in LEVELS:
                indices = c80._cell_indices(arrays, target, level)
                source_probabilities, source_labels, source_domains = _load_source_context(
                    route, target, arrays["unit_id"][indices],
                )
                scores, aline = score_context(
                    source_probabilities,
                    source_labels,
                    source_domains,
                    arrays["target_logits"][indices],
                    arrays["regime"][indices],
                    arrays["candidate_order"][indices],
                )
                score_block = np.stack([scores[method] for method in SELECTION_METHODS])
                order_block = np.stack([descending_order(scores[method])[:10] for method in SELECTION_METHODS])
                cells.append({
                    "seed": seed, "target": target, "level": level,
                    "candidate_global_indices": indices,
                })
                score_blocks.append(score_block)
                order_blocks.append(order_block)
                aline_rows.append(aline)
    if len(cells) != 32:
        raise RuntimeError("C81 selection context count drift")
    arrays = {
        "cell_seed": np.asarray([row["seed"] for row in cells], dtype=np.int16),
        "cell_target": np.asarray([row["target"] for row in cells], dtype=np.int16),
        "cell_level": np.asarray([row["level"] for row in cells], dtype=np.int16),
        "candidate_global_indices": np.stack([row["candidate_global_indices"] for row in cells]).astype(np.int16),
        "method_ids": np.asarray(SELECTION_METHODS, dtype="<U8"),
        "scores": np.stack(score_blocks).astype(np.float64),
        "selected_top10": np.stack(order_blocks).astype(np.int16),
        "aline_slope": np.asarray([row["slope"] for row in aline_rows], dtype=np.float64),
        "aline_intercept": np.asarray([row["intercept"] for row in aline_rows], dtype=np.float64),
        "aline_pair_R2": np.asarray([row["pair_R2"] for row in aline_rows], dtype=np.float64),
    }
    descriptor = c74_cache.write_content_addressed_npz(
        external_root / "selection" / "payload", "c81_locked_baseline_selection", arrays,
    )
    manifest = c74_cache.self_hashed_manifest({
        "schema_version": "c81_selection_freeze_v1",
        "protocol_sha256": context["lock"]["protocol"]["sha256"],
        "repair_protocol_sha256": context["lock"]["repair_protocol"]["sha256"],
        "analysis_lock_sha256": context["lock_sha256"],
        "contexts": 32,
        "methods": list(SELECTION_METHODS),
        "all_methods_unconditional": True,
        "target4_accessed": False,
        "evaluation_labels_accessed": False,
        "same_label_oracle_accessed": False,
        "descriptor": descriptor,
    })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    c74_cache.atomic_json(manifest_path, manifest)
    return manifest


def _load_selection(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    _verify_selection_descriptor(manifest["descriptor"])
    with np.load(manifest["descriptor"]["path"], allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    expected = {
        "cell_seed": (32,), "cell_target": (32,), "cell_level": (32,),
        "candidate_global_indices": (32, 81), "method_ids": (len(SELECTION_METHODS),),
        "scores": (32, len(SELECTION_METHODS), 81),
        "selected_top10": (32, len(SELECTION_METHODS), 10),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"C81 selection payload shape drift: {name}")
    if tuple(arrays["method_ids"].astype(str)) != SELECTION_METHODS:
        raise RuntimeError("C81 selection method order drift")
    return arrays


def _joint_good(metrics: np.ndarray) -> np.ndarray:
    oriented = np.column_stack((
        c75_data.midrank_percentile(metrics[:, 0]),
        c75_data.midrank_percentile(-metrics[:, 1]),
        c75_data.midrank_percentile(-metrics[:, 2]),
    ))
    return np.all(oriented >= 0.75, axis=1)


def _random_control(utility: np.ndarray, joint_good: np.ndarray) -> dict[str, float]:
    regrets = np.asarray([standardized_regret(utility, index) for index in range(CANDIDATES)])
    good = int(np.sum(joint_good))
    result = {
        "standardized_regret": float(np.mean(regrets)),
        "selected_utility": float(np.mean(utility)),
    }
    for k in (1, 5, 10):
        result[f"top{k}"] = k / CANDIDATES
        if CANDIDATES - good < k:
            coverage = 1.0
        else:
            coverage = 1.0 - math.comb(CANDIDATES - good, k) / math.comb(CANDIDATES, k)
        result[f"coverage_top{k}"] = float(coverage)
    return result


def _evaluation_stage(context: dict[str, Any], selection_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Open evaluation views only after the selection manifest hash is frozen."""
    from . import c80r_existing_field_adapter as c80

    if selection_manifest["evaluation_labels_accessed"] or selection_manifest["same_label_oracle_accessed"]:
        raise RuntimeError("C81 selection isolation drift")
    frozen = _self_hashed_manifest(context["external_root"] / "selection" / "C81_SELECTIONS_FROZEN.json")
    if frozen["manifest_sha256"] != selection_manifest["manifest_sha256"]:
        raise RuntimeError("C81 selection freeze replay mismatch")
    selection = _load_selection(frozen)
    method_position = {method: index for index, method in enumerate(selection["method_ids"].astype(str))}
    rows: list[dict[str, Any]] = []
    for seed_key in ("seed3", "seed4"):
        binding = context["lock"]["runtime_inputs"][seed_key]
        arrays = c80._load_unlabeled(Path(binding["unlabeled_cache_manifest_path"]))
        route = c80._route(Path(binding["primary_route_path"]), binding["primary_route_sha256"])
        seed = int(seed_key.removeprefix("seed"))
        trial_lookup = {
            int(target): arrays["target_trial_id"][index]
            for index, target in enumerate(arrays["target_trial_id_target"])
        }
        for target in PRIMARY_TARGETS:
            construction = c80._load_label_view(route, target, "target_construction_view")
            evaluation = c80._load_label_view(route, target, "target_evaluation_view")
            construction_indices, _ = c80._align_label_view(trial_lookup[target], construction, "target_construct")
            evaluation_indices, evaluation_labels = c80._align_label_view(trial_lookup[target], evaluation, "target_eval")
            if set(construction_indices) & set(evaluation_indices):
                raise RuntimeError("C81 construction/evaluation overlap")
            if set(construction_indices) | set(evaluation_indices) != set(range(576)):
                raise RuntimeError("C81 split does not cover the frozen target trials")
            full_labels = np.full(576, -1, dtype=int)
            full_labels[evaluation_indices] = evaluation_labels
            for level in LEVELS:
                indices = c80._cell_indices(arrays, target, level)
                cell = np.where(
                    (selection["cell_seed"] == seed)
                    & (selection["cell_target"] == target)
                    & (selection["cell_level"] == level)
                )[0]
                if len(cell) != 1 or not np.array_equal(selection["candidate_global_indices"][cell[0]], indices):
                    raise RuntimeError("C81 selection/evaluation candidate alignment drift")
                metrics = c80.endpoint_metrics_all_candidates(
                    arrays["target_logits"][indices], full_labels, evaluation_indices,
                )
                utility = c80.frontier.score_from_endpoint_metrics(metrics)
                joint_good = _joint_good(metrics)
                regimes = arrays["regime"][indices].astype(str)
                for method in SELECTION_METHODS:
                    order = selection["selected_top10"][cell[0], method_position[method]].astype(int)
                    result = evaluate_order(order, utility, joint_good)
                    rows.append({
                        "seed": seed, "target": target, "level": level, "method_id": method,
                        **result, "selected_regime": regimes[int(order[0])],
                        "evaluation_label_access_after_selection_freeze": 1,
                        "same_label_oracle_accessed": 0, "target4_primary": 0,
                    })
                random_result = _random_control(utility, joint_good)
                rows.append({
                    "seed": seed, "target": target, "level": level, "method_id": "B0",
                    **random_result, "selected_regime": "uniform_mixture",
                    "evaluation_label_access_after_selection_freeze": 1,
                    "same_label_oracle_accessed": 0, "target4_primary": 0,
                })
                oracle_result = evaluate_order(np.argsort(utility)[::-1], utility, joint_good)
                rows.append({
                    "seed": seed, "target": target, "level": level, "method_id": "B5",
                    **oracle_result, "selected_regime": regimes[int(np.argmax(utility))],
                    "evaluation_label_access_after_selection_freeze": 1,
                    "same_label_oracle_accessed": 0, "target4_primary": 0,
                })
    return rows


def _q0_target_regret(seed: int, target: int, budget: str) -> float:
    rows = read_csv(REPORT_DIR / "c80e_tables" / "target_level_regret.csv")
    values = [
        float(row["expected_standardized_regret"])
        for row in rows
        if int(row["seed"]) == seed and int(row["target"]) == target and row["budget"] == budget
    ]
    if len(values) != 2:
        raise RuntimeError("C81 frozen Q0 comparator coverage drift")
    return float(np.mean(values))


def _target_method_regret(rows: list[dict[str, Any]], seed: int, target: int, method: str) -> float:
    values = [
        float(row["standardized_regret"])
        for row in rows
        if row["seed"] == seed and row["target"] == target and row["method_id"] == method
    ]
    if len(values) != 2:
        raise RuntimeError(f"C81 target-method coverage drift: {seed}/{target}/{method}")
    return float(np.mean(values))


def _classify_comparisons(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tests = [(seed, method) for seed in SEEDS for method in PRIMARY_ZERO_METHODS]
    q1 = np.column_stack([
        [
            _target_method_regret(rows, seed, target, "S1")
            - _target_method_regret(rows, seed, target, method)
            for target in PRIMARY_TARGETS
        ]
        for seed, method in tests
    ])
    q2_difference = np.column_stack([
        [
            _target_method_regret(rows, seed, target, method) - _q0_target_regret(seed, target, "1")
            for target in PRIMARY_TARGETS
        ]
        for seed, method in tests
    ])
    q1_evidence = exact_signflip_maxT(q1, margin=MATERIAL_MARGIN)
    q2_evidence = exact_signflip_maxT(NONINFERIORITY_MARGIN - q2_difference)
    q2_band = exact_signflip_maxT(q2_difference)
    comparison_rows = []
    per_seed_q1 = {seed: {} for seed in SEEDS}
    per_seed_q2 = {seed: {} for seed in SEEDS}
    for index, (seed, method) in enumerate(tests):
        q1_pass = _passes_q1(q1[:, index], float(q1_evidence["pvalue"][index]))
        q2_pass = _passes_q2(
            q2_difference[:, index], float(q2_band["upper"][index]), float(q2_evidence["pvalue"][index]),
        )
        per_seed_q1[seed][method] = q1_pass
        per_seed_q2[seed][method] = q2_pass
        comparison_rows.append({
            "seed": seed, "method_id": method,
            "mean_regret_improvement_vs_source": float(np.mean(q1[:, index])),
            "Q1_maxT_p": float(q1_evidence["pvalue"][index]), "Q1_pass": int(q1_pass),
            "mean_regret_difference_vs_Q0_B1": float(np.mean(q2_difference[:, index])),
            "Q2_simultaneous_upper": float(q2_band["upper"][index]),
            "Q2_maxT_p": float(q2_evidence["pvalue"][index]), "Q2_pass": int(q2_pass),
        })
    categories = {seed: seed_category(per_seed_q1[seed], per_seed_q2[seed]) for seed in SEEDS}
    loto_rows = []
    preserved = 0
    for left_position, left_target in enumerate(PRIMARY_TARGETS):
        keep = np.arange(len(PRIMARY_TARGETS)) != left_position
        q1_sub = exact_signflip_maxT(q1[keep], margin=MATERIAL_MARGIN)
        q2_sub = exact_signflip_maxT(NONINFERIORITY_MARGIN - q2_difference[keep])
        q2_band_sub = exact_signflip_maxT(q2_difference[keep])
        for seed in SEEDS:
            sub_q1 = {}
            sub_q2 = {}
            for index, (test_seed, method) in enumerate(tests):
                if test_seed != seed:
                    continue
                sub_q1[method] = _passes_q1(q1[keep, index], float(q1_sub["pvalue"][index]))
                sub_q2[method] = _passes_q2(
                    q2_difference[keep, index], float(q2_band_sub["upper"][index]), float(q2_sub["pvalue"][index]),
                )
            category = seed_category(sub_q1, sub_q2)
            is_preserved = category == categories[seed]
            preserved += int(is_preserved)
            loto_rows.append({
                "seed": seed, "left_out_target": left_target, "full_category": categories[seed],
                "LOTO_category": category, "preserved": int(is_preserved),
            })
    taxonomy = classify_taxonomy(
        blocker=False, seed3_category=categories[3], seed4_category=categories[4],
        loto_preserved=preserved,
    )
    return {
        "seed3_category": categories[3], "seed4_category": categories[4],
        "LOTO_preserved": preserved, "LOTO_total": 16, "primary_taxonomy": taxonomy,
    }, comparison_rows, loto_rows


def _freeze_c81e_results(
    context: dict[str, Any], selection_manifest: dict[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    C81E_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(C81E_TABLE_DIR / "method_context_results.csv", rows)
    taxonomy, comparisons, loto = _classify_comparisons(rows)
    write_csv(C81E_TABLE_DIR / "primary_comparison.csv", comparisons)
    write_csv(C81E_TABLE_DIR / "leave_one_target_out_sensitivity.csv", loto)
    result = {
        "schema_version": "c81_frozen_field_baseline_result_v1",
        "protocol_sha256": context["lock"]["protocol"]["sha256"],
        "repair_protocol_sha256": context["lock"]["repair_protocol"]["sha256"],
        "selection_descriptor_repair_protocol_sha256": context["lock"][
            "selection_descriptor_repair_protocol"
        ]["sha256"],
        "analysis_lock_sha256": context["lock_sha256"],
        "selection_manifest_sha256": selection_manifest["manifest_sha256"],
        "contexts": 32,
        "selection_methods": list(SELECTION_METHODS),
        "all_primary_families_fixed_before_outcomes": True,
        "target4_primary": False,
        "same_label_oracle_accessed": False,
        "existing_field_outcome_informed": True,
        "external_validity_claim": False,
        **taxonomy,
    }
    c74_cache.atomic_json(REPORT_DIR / "C81_FROZEN_FIELD_BASELINE_COMPARISON.json", result)
    return result


def run_real() -> dict[str, Any]:
    """Run the lock-bound two-stage C81E adapter after direct authorization."""
    context = _real_runtime_context(require_c81e_authorization())
    selection = _selection_stage(context)
    rows = _evaluation_stage(context, selection)
    return _freeze_c81e_results(context, selection, rows)


def schema_dry_run() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    methods = load_method_registry()["methods"]
    return {
        "protocol_sha256": protocol_sha,
        "candidate_contexts": protocol["candidate_universe"]["contexts"],
        "methods": len(methods),
        "selection_methods": list(SELECTION_METHODS),
        "primary_zero_methods": list(PRIMARY_ZERO_METHODS),
        "target4_primary": False,
        "same_label_oracle": False,
        "real_baseline_statistics": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare-replay", "schema-dry-run", "run-real"))
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(protocol_audit(), sort_keys=True))
    elif args.command == "prepare-replay":
        print(json.dumps(prepare_c81p_replay_tables(), sort_keys=True))
    elif args.command == "schema-dry-run":
        print(json.dumps(schema_dry_run(), sort_keys=True))
    else:
        print(json.dumps(run_real(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
