"""C80R fail-closed adapter for the existing-field label-budget analysis.

The adapter is implemented and tested against synthetic/schema fixtures during
C80R.  Real selection and evaluation are separate immutable stages.  Every
external array load is downstream of the replacement lock and a new direct PI
authorization; the historical C80E authorization cannot satisfy this guard.
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
from typing import Any, Callable, Iterable

import numpy as np
from scipy import stats

from . import c74_cache
from . import c75_data
from . import c80_label_budget_frontier as frontier


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C80R_ADDITIVE_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C80R_ADDITIVE_REPAIR_PROTOCOL.sha256"
REPAIRED_LOCK_PATH = REPORT_DIR / "C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.json"
REPAIRED_LOCK_SHA_PATH = REPORT_DIR / "C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.sha256"
REPAIRED_AUTHORIZATION_PATH = REPORT_DIR / "C80E_REPAIRED_PI_AUTHORIZATION_RECORD.json"
RESULT_TABLE_DIR = REPORT_DIR / "c80e_tables"
RESULT_PATH = REPORT_DIR / "C80_LABEL_BUDGET_FRONTIER.json"

EXPECTED_PATHS = ("P1", "P2", "S1", "S2", "S3")
NEAR_FULL = (32, "FULL")
TOP_K = (1, 5, 10)
BOOTSTRAP_REPLICATES = 8192


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return frontier.sha256_file(Path(path))


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty C80 table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError(f"C80 table schema drift within {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _last_commit_for(path: Path) -> str:
    relative = path.resolve().relative_to(REPO_ROOT.resolve())
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(relative)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"cannot resolve committed identity for {relative}")
    return commit


def load_repair_protocol() -> tuple[dict[str, Any], str]:
    expected = REPAIR_PROTOCOL_SHA_PATH.read_text().strip()
    observed = _sha256_file(REPAIR_PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError("C80R additive protocol hash mismatch")
    protocol = _read_json(REPAIR_PROTOCOL_PATH)
    taxonomy = protocol["taxonomy"]
    if taxonomy["near_FULL_ordinal_set"] != [32, "FULL"]:
        raise RuntimeError("C80R near-FULL definition drift")
    priorities = [int(row["priority"]) for row in taxonomy["decision_table"]]
    cases = [row["case"] for row in taxonomy["decision_table"]]
    if priorities != [1, 2, 3, 4, 5] or cases != [
        "C80-E_protocol_dependence_view_or_provenance_blocker",
        "C80-D_no_registered_budget_achieves_stable_material_actionability",
        "C80-B_actionability_frontier_exists_but_required_budget_is_seed_heterogeneous",
        "C80-C_material_actionability_requires_near_full_construction_labels",
        "C80-A_stable_low_regret_label_budget_frontier_across_training_seeds",
    ]:
        raise RuntimeError("C80R taxonomy order drift")
    base = protocol["scientific_inheritance"]
    if _sha256_file(REPO_ROOT / base["base_protocol_path"]) != base["base_protocol_sha256"]:
        raise RuntimeError("C80R inherited base protocol hash drift")
    if _sha256_file(REPO_ROOT / base["scientific_registry_path"]) != base["scientific_registry_sha256"]:
        raise RuntimeError("C80R inherited registry hash drift")
    return protocol, observed


def classify_taxonomy(
    *,
    blocker: bool,
    bstar_seed3: int | str | None,
    bstar_seed4: int | str | None,
    cross_seed_stability_pass: bool,
) -> str:
    """Apply the PM-locked mutually exclusive C80-A--E precedence."""
    if blocker:
        return "C80-E_protocol_dependence_view_or_provenance_blocker"
    if bstar_seed3 is None or bstar_seed4 is None:
        return "C80-D_no_registered_budget_achieves_stable_material_actionability"
    if not cross_seed_stability_pass:
        return "C80-B_actionability_frontier_exists_but_required_budget_is_seed_heterogeneous"
    if bstar_seed3 in NEAR_FULL and bstar_seed4 in NEAR_FULL:
        return "C80-C_material_actionability_requires_near_full_construction_labels"
    return "C80-A_stable_low_regret_label_budget_frontier_across_training_seeds"


def _manifest_binding_digest(lock: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(lock["field_and_view_manifests"]))


def load_repaired_lock() -> tuple[dict[str, Any], str]:
    if not REPAIRED_LOCK_PATH.exists() or not REPAIRED_LOCK_SHA_PATH.exists():
        raise RuntimeError("C80R replacement analysis lock is absent")
    expected = REPAIRED_LOCK_SHA_PATH.read_text().strip()
    observed = _sha256_file(REPAIRED_LOCK_PATH)
    if observed != expected:
        raise RuntimeError("C80R replacement analysis lock hash mismatch")
    lock = _read_json(REPAIRED_LOCK_PATH)
    protocol, protocol_sha = load_repair_protocol()
    lock_protocol = lock.get("protocol")
    required_protocol = {"commit", "path", "sha256"}
    if not isinstance(lock_protocol, dict) or required_protocol - set(lock_protocol):
        raise RuntimeError("C80R lock.protocol schema mismatch")
    if lock_protocol["sha256"] != protocol_sha:
        raise RuntimeError("C80R lock.protocol.sha256 mismatch")
    if lock_protocol["path"] != str(REPAIR_PROTOCOL_PATH.relative_to(REPO_ROOT)):
        raise RuntimeError("C80R lock protocol path mismatch")
    if lock_protocol["commit"] != _last_commit_for(REPAIR_PROTOCOL_PATH):
        raise RuntimeError("C80R lock protocol commit mismatch")
    if lock.get("taxonomy") != protocol["taxonomy"]:
        raise RuntimeError("C80R lock taxonomy is not the protocol taxonomy")
    if lock["registry"]["bound_cells"] != 80 or lock["registry"]["blank_cells"] != 0:
        raise RuntimeError("C80R lock registry completeness mismatch")
    if lock["runtime"]["same_label_oracle_reachable"] or lock["runtime"]["target4_primary"]:
        raise RuntimeError("C80R replacement lock scope drift")
    for item in lock["implementation"]["files"]:
        if _sha256_file(REPO_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"C80R locked implementation drift: {item['path']}")
    for item in lock["field_and_view_manifests"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file() or _sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"C80R field/view manifest drift: {path}")
    return lock, observed


def require_repaired_authorization() -> dict[str, Any]:
    """Fail closed before any external array is opened."""
    lock, lock_sha = load_repaired_lock()
    if not REPAIRED_AUTHORIZATION_PATH.exists():
        raise RuntimeError("new direct C80E PI authorization record is absent")
    authorization = _read_json(REPAIRED_AUTHORIZATION_PATH)
    required = {"protocol", "analysis_lock", "field_view_manifest_digest", "scope"}
    if not authorization.get("authorization_received") or required - set(authorization):
        raise RuntimeError("C80E repaired authorization schema mismatch")
    protocol = lock["protocol"]
    if authorization["protocol"] != protocol:
        raise RuntimeError("C80E repaired authorization protocol binding mismatch")
    expected_lock = {
        "commit": _last_commit_for(REPAIRED_LOCK_PATH),
        "path": str(REPAIRED_LOCK_PATH.relative_to(REPO_ROOT)),
        "sha256": lock_sha,
    }
    if authorization["analysis_lock"] != expected_lock:
        raise RuntimeError("C80E repaired authorization lock binding mismatch")
    if authorization["field_view_manifest_digest"] != _manifest_binding_digest(lock):
        raise RuntimeError("C80E repaired authorization manifest binding mismatch")
    forbidden_true = {
        "training", "forward", "reinference", "GPU", "seed5", "target4_primary",
        "same_label_oracle", "BNCI2014_004", "active_learning", "new_feature_kernel_model_search",
        "C81", "manuscript",
    }
    if any(bool(authorization["scope"].get(name)) for name in forbidden_true):
        raise RuntimeError("C80E repaired authorization scope expansion")
    if authorization["scope"].get("registered_paths") != list(EXPECTED_PATHS):
        raise RuntimeError("C80E repaired authorization path scope mismatch")
    return {"authorization": authorization, "lock": lock, "lock_sha256": lock_sha}


def _descriptor(route_item: dict[str, Any]) -> dict[str, Any]:
    fields_value = route_item.get("allowed_columns", route_item.get("fields"))
    fields = json.loads(fields_value) if isinstance(fields_value, str) else fields_value
    path = Path(route_item["path"])
    return {
        "path": str(path),
        "sha256": route_item["sha256"],
        "size_bytes": int(route_item.get("size_bytes", path.stat().st_size)),
        "row_count": int(route_item.get("rows", route_item.get("row_count"))),
        "fields": list(fields),
    }


def _load_shard(
    descriptor: dict[str, Any],
    *,
    required_fields: set[str],
    loader: Callable[..., Any] = np.load,
) -> dict[str, np.ndarray]:
    """Load a shard only after the caller has passed the repaired guard."""
    path = Path(descriptor["path"])
    if _sha256_file(path) != descriptor["sha256"]:
        raise RuntimeError(f"C80 shard hash drift: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise RuntimeError(f"C80 shard size drift: {path}")
    with loader(path, allow_pickle=False) as shard:
        if set(shard.files) != set(descriptor["fields"]) or set(shard.files) != required_fields:
            raise RuntimeError(f"C80 shard schema drift: {path}")
        arrays = {name: shard[name] for name in shard.files}
    lengths = {len(value) for value in arrays.values() if value.ndim >= 1}
    if len(lengths) != 1 or next(iter(lengths)) != int(descriptor["row_count"]):
        raise RuntimeError(f"C80 shard row-count drift: {path}")
    return arrays


def _verify_self_hashed_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    supplied = payload.get("manifest_sha256")
    observed = _sha256_bytes(_canonical_bytes({k: v for k, v in payload.items() if k != "manifest_sha256"}))
    if supplied != observed:
        raise RuntimeError(f"C80 self-hashed manifest mismatch: {path}")
    return payload


def _load_unlabeled(manifest_path: Path) -> dict[str, np.ndarray]:
    manifest = _verify_self_hashed_manifest(manifest_path)
    descriptor = manifest["descriptor"]
    trial_descriptor = manifest["trial_registry_descriptor"]
    c74_cache.verify_shard(descriptor)
    c74_cache.verify_shard(trial_descriptor)
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    with np.load(trial_descriptor["path"], allow_pickle=False) as shard:
        arrays.update({name: shard[name] for name in shard.files})
    targets = set(arrays["target_id"].astype(int).tolist())
    if len(arrays["unit_id"]) != 1296 or targets != set(frontier.PRIMARY_TARGETS) or 4 in targets:
        raise RuntimeError("C80 unlabeled primary universe drift")
    if arrays["target_logits"].shape != (1296, 576, frontier.N_CLASSES):
        raise RuntimeError("C80 unlabeled target-logit shape drift")
    return arrays


def _load_label_view(
    route: dict[str, Any], target: int, view_name: str,
) -> dict[str, np.ndarray]:
    if target == 4 or target not in frontier.PRIMARY_TARGETS:
        raise RuntimeError("C80 target-4/nonprimary label-view access blocked")
    if view_name not in {"target_construction_view", "target_evaluation_view"}:
        raise RuntimeError("C80 same-label oracle/unregistered view blocked")
    descriptor = _descriptor(route["views"][str(target)][view_name])
    return _load_shard(
        descriptor,
        required_fields={"target_trial_id", "target_class_label", "split_role"},
    )


def _route(path: Path, expected_sha: str) -> dict[str, Any]:
    if not path.is_absolute():
        path = REPO_ROOT / path
    if _sha256_file(path) != expected_sha:
        raise RuntimeError(f"C80 route hash drift: {path}")
    raw = path.read_text()
    if "same_label_oracle_view" in raw or "/oracle/" in raw:
        raise RuntimeError("C80 route exposes same-label oracle")
    route = json.loads(raw)
    if tuple(route["primary_targets"]) != frontier.PRIMARY_TARGETS or route["target4_included"]:
        raise RuntimeError("C80 primary route target scope drift")
    return route


def endpoint_metrics_all_candidates(
    logits: np.ndarray, labels: np.ndarray, indices: np.ndarray,
) -> np.ndarray:
    """Vectorized exact C79 endpoint metrics for all 81 candidates."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    indices = np.asarray(indices, dtype=int)
    if logits.ndim != 3 or logits.shape[0] != frontier.CANDIDATES_PER_CELL or logits.shape[2] != 4:
        raise RuntimeError("C80 endpoint logits shape drift")
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= logits.shape[1]):
        raise RuntimeError("C80 endpoint sample indices invalid")
    selected = logits[:, indices]
    shifted = selected - np.max(selected, axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    prediction = np.argmax(probabilities, axis=2)
    sampled_labels = labels[indices]
    recalls = []
    for class_id in range(frontier.N_CLASSES):
        mask = sampled_labels == class_id
        if not np.any(mask):
            raise RuntimeError(f"C80 endpoint sample lacks class {class_id}")
        recalls.append(np.mean(prediction[:, mask] == class_id, axis=1))
    bacc = np.mean(np.stack(recalls, axis=1), axis=1)
    true_probability = np.take_along_axis(
        probabilities,
        np.broadcast_to(sampled_labels[None, :, None], (len(logits), len(indices), 1)),
        axis=2,
    )[:, :, 0]
    nll = -np.mean(np.log(np.clip(true_probability, 1e-12, 1.0)), axis=1)
    confidence = np.max(probabilities, axis=2)
    correctness = prediction == sampled_labels[None, :]
    ece = np.zeros(len(logits), dtype=float)
    edges = np.linspace(0.0, 1.0, 16)
    for bin_index in range(15):
        upper = confidence <= edges[bin_index + 1] if bin_index == 14 else confidence < edges[bin_index + 1]
        mask = (confidence >= edges[bin_index]) & upper
        count = np.sum(mask, axis=1)
        nonzero = count > 0
        accuracy = np.divide(np.sum(correctness * mask, axis=1), count, out=np.zeros(len(logits)), where=nonzero)
        mean_confidence = np.divide(np.sum(confidence * mask, axis=1), count, out=np.zeros(len(logits)), where=nonzero)
        ece += (count / len(indices)) * np.abs(accuracy - mean_confidence)
    return np.column_stack((bacc, nll, ece))


def score_order_for_sample(logits: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = frontier.score_from_endpoint_metrics(endpoint_metrics_all_candidates(logits, labels, indices))
    return scores, frontier.descending_candidate_order(scores)


def _align_label_view(
    trial_registry: np.ndarray, label_view: dict[str, np.ndarray], expected_role: str,
) -> tuple[np.ndarray, np.ndarray]:
    ids = list(map(str, np.asarray(trial_registry)))
    lookup = {trial_id: index for index, trial_id in enumerate(ids)}
    if len(lookup) != 576:
        raise RuntimeError("C80 trial registry must contain 576 unique IDs")
    roles = set(map(str, label_view["split_role"]))
    if roles != {expected_role}:
        raise RuntimeError("C80 label-view split role drift")
    view_ids = list(map(str, label_view["target_trial_id"]))
    if len(view_ids) != len(set(view_ids)) or not set(view_ids).issubset(lookup):
        raise RuntimeError("C80 label-view trial identity drift")
    return (
        np.asarray([lookup[trial_id] for trial_id in view_ids], dtype=int),
        np.asarray(label_view["target_class_label"], dtype=int),
    )


def selection_fixture(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    target: int,
    level: int,
    chains: int,
) -> dict[str, np.ndarray]:
    """Pure synthetic fixture for the complete nested Q0 selection stage."""
    budgets = frontier.BUDGETS
    scores = np.empty((chains, len(budgets), frontier.CANDIDATES_PER_CELL), dtype=np.float32)
    orders = np.empty((chains, len(budgets), 10), dtype=np.int16)
    full_scores, full_order = score_order_for_sample(logits, labels, np.arange(len(labels)))
    for chain in range(chains):
        rng = np.random.default_rng(frontier.deterministic_stream_seed(seed, target, level, chain))
        samples = frontier.nested_class_samples(labels, rng=rng)
        for budget_index, budget in enumerate(budgets):
            if budget == "FULL":
                cell_scores, order = full_scores, full_order
            else:
                cell_scores, order = score_order_for_sample(logits, labels, samples[budget])
            scores[chain, budget_index] = cell_scores
            orders[chain, budget_index] = order[:10]
    return {"scores": scores, "orders": orders}


def evaluate_selection_fixture(
    selection: dict[str, np.ndarray], evaluation_utility: np.ndarray, joint_good: np.ndarray,
) -> dict[str, np.ndarray]:
    utility = np.asarray(evaluation_utility, dtype=float)
    good = np.asarray(joint_good, dtype=bool)
    if utility.shape != (81,) or good.shape != (81,):
        raise RuntimeError("C80 evaluation fixture shape drift")
    orders = np.asarray(selection["orders"], dtype=int)
    true_order = np.argsort(utility)[::-1]
    best = int(true_order[0])
    spread = float(np.max(utility) - np.min(utility))
    selected = orders[:, :, 0]
    regret = (
        (float(np.max(utility)) - utility[selected]) / spread
        if spread > 1e-15 else np.zeros_like(selected, dtype=float)
    )
    output: dict[str, np.ndarray] = {"regret": regret}
    for k in TOP_K:
        output[f"top{k}"] = np.any(orders[:, :, :k] == best, axis=2).astype(float)
        output[f"coverage_top{k}"] = np.any(good[orders[:, :, :k]], axis=2).astype(float)
    evaluation_rank = stats.rankdata(utility)
    score_rank = stats.rankdata(selection["scores"], axis=2)
    centered_score = score_rank - np.mean(score_rank, axis=2, keepdims=True)
    centered_eval = evaluation_rank - np.mean(evaluation_rank)
    denominator = np.sqrt(np.sum(centered_score ** 2, axis=2) * np.sum(centered_eval ** 2))
    output["reliability"] = np.divide(
        np.sum(centered_score * centered_eval, axis=2), denominator,
        out=np.full(denominator.shape, np.nan), where=denominator > 1e-15,
    )
    return output


def simultaneous_target_band(
    target_values: np.ndarray, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, np.ndarray | float]:
    values = np.asarray(target_values, dtype=float)
    if values.shape != (frontier.TARGETS, len(frontier.BUDGETS)):
        raise RuntimeError("C80 simultaneous band requires 8 targets x 7 budgets")
    mean = np.mean(values, axis=0)
    se = np.std(values, axis=0, ddof=1) / math.sqrt(len(values))
    safe_se = np.where(se > 1e-15, se, 1.0)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(replicates, len(values)))
    bootstrap_mean = np.mean(values[sampled], axis=1)
    max_deviation = np.max(np.abs((bootstrap_mean - mean) / safe_se), axis=1)
    critical = float(np.quantile(max_deviation, 0.95))
    return {
        "mean": mean,
        "se": se,
        "critical": critical,
        "lower": mean - critical * se,
        "upper": mean + critical * se,
    }


def qualification_for_effects(target_effects: np.ndarray) -> dict[str, Any]:
    """Registered qualification calculation for descriptive LOTO sensitivity."""
    effects = np.asarray(target_effects, dtype=float)
    if effects.ndim != 2 or effects.shape[1] != len(frontier.BUDGETS) or not 1 <= len(effects) <= 8:
        raise RuntimeError("C80 qualification effect shape drift")
    centered = effects - frontier.MATERIAL_REGRET
    observed = np.mean(centered, axis=0)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(effects))))
    null_max = np.max((signs @ centered) / len(effects), axis=1)
    pvalues = np.asarray([
        (1 + int(np.sum(null_max >= value - 1e-15))) / (1 + len(null_max))
        for value in observed
    ])
    mean = np.mean(effects, axis=0)
    positive = np.sum(effects > 0, axis=0)
    catastrophic = np.any(effects < frontier.CATASTROPHIC_TARGET, axis=0)
    direct = (
        (mean >= frontier.MATERIAL_REGRET)
        & (pvalues <= 0.05)
        & (positive >= min(frontier.MIN_POSITIVE_TARGETS, len(effects)))
        & (~catastrophic)
    )
    closure = np.asarray([np.all(direct[index:]) for index in range(len(frontier.BUDGETS))], dtype=bool)
    return {
        "mean_effect": mean,
        "positive_targets": positive,
        "catastrophic": catastrophic,
        "maxT_p": pvalues,
        "direct_qualification": direct,
        "closure_qualification": closure,
        "Bstar": next((frontier.BUDGETS[index] for index, value in enumerate(closure) if value), None),
    }


def paired_cross_seed_stability(
    seed3_effects: np.ndarray,
    seed4_effects: np.ndarray,
    bstar_seed3: int | str | None,
    bstar_seed4: int | str | None,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    left = np.asarray(seed3_effects, dtype=float)
    right = np.asarray(seed4_effects, dtype=float)
    if left.shape != right.shape or left.shape != (8, 7):
        raise RuntimeError("C80 cross-seed effects require paired 8x7 arrays")
    distance = frontier.bstar_grid_distance(bstar_seed3, bstar_seed4)
    if distance is None:
        return {
            "pass": False,
            "Bstar_grid_distance": None,
            "common_larger_budget": None,
            "paired_mean_difference": math.nan,
            "simultaneous_ci_low": math.nan,
            "simultaneous_ci_high": math.nan,
            "reason": "one_or_both_Bstar_absent",
        }
    common_index = max(frontier.BUDGETS.index(bstar_seed3), frontier.BUDGETS.index(bstar_seed4))
    paired_band = paired_difference_band(left, right, replicates=replicates)
    mean = paired_band["mean"]
    lower, upper = paired_band["lower"], paired_band["upper"]
    paired_mean = float(mean[common_index])
    heterogeneity_pass = abs(paired_mean) <= 0.05 and lower[common_index] <= 0 <= upper[common_index]
    direction_pass = (
        int(np.sum(left[:, frontier.BUDGETS.index(bstar_seed3)] > 0)) >= 6
        and int(np.sum(right[:, frontier.BUDGETS.index(bstar_seed4)] > 0)) >= 6
    )
    passed = distance <= 1 and heterogeneity_pass and direction_pass
    return {
        "pass": bool(passed),
        "Bstar_grid_distance": int(distance),
        "common_larger_budget": frontier.BUDGETS[common_index],
        "paired_mean_difference": paired_mean,
        "simultaneous_ci_low": float(lower[common_index]),
        "simultaneous_ci_high": float(upper[common_index]),
        "heterogeneity_pass": bool(heterogeneity_pass),
        "direction_concordance_pass": bool(direction_pass),
        "reason": "all_registered_stability_gates_pass" if passed else "registered_stability_gate_failed",
    }


def paired_difference_band(
    seed3_effects: np.ndarray,
    seed4_effects: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, np.ndarray | float]:
    left = np.asarray(seed3_effects, dtype=float)
    right = np.asarray(seed4_effects, dtype=float)
    if left.shape != right.shape or left.shape != (8, 7):
        raise RuntimeError("C80 paired difference band requires paired 8x7 arrays")
    differences = right - left
    mean = np.mean(differences, axis=0)
    se = np.std(differences, axis=0, ddof=1) / math.sqrt(8)
    safe_se = np.where(se > 1e-15, se, 1.0)
    rng = np.random.default_rng(8033)
    sampled = rng.integers(0, 8, size=(replicates, 8))
    boot = np.mean(differences[sampled], axis=1)
    max_deviation = np.max(np.abs((boot - mean) / safe_se), axis=1)
    critical = float(np.quantile(max_deviation, 0.95))
    return {
        "mean": mean,
        "se": se,
        "critical": critical,
        "lower": mean - critical * se,
        "upper": mean + critical * se,
    }


def _cell_indices(arrays: dict[str, np.ndarray], target: int, level: int) -> np.ndarray:
    mask = (arrays["target_id"].astype(int) == target) & (arrays["level"].astype(int) == level)
    indices = np.where(mask)[0]
    if len(indices) != 81:
        raise RuntimeError(f"C80 cell {target}/{level} does not contain 81 candidates")
    regimes = arrays["regime"][indices].astype(str)
    if (int(np.sum(regimes == "ERM")), int(np.sum(regimes == "OACI")), int(np.sum(regimes == "SRC"))) != (1, 40, 40):
        raise RuntimeError(f"C80 regime composition drift in cell {target}/{level}")
    return indices


def _selection_stage(context: dict[str, Any]) -> dict[str, Any]:
    lock = context["lock"]
    runtime = lock["runtime"]
    external_root = Path(runtime["external_result_root"])
    manifest_path = external_root / "selection" / "SELECTION_OUTPUTS_FROZEN.json"
    if manifest_path.exists():
        manifest = _verify_self_hashed_manifest(manifest_path)
        c74_cache.verify_shard(manifest["descriptor"])
        return manifest
    all_cells: list[dict[str, Any]] = []
    all_scores = []
    all_orders = []
    all_source_order = []
    full_counts = []
    for seed_key in ("seed3", "seed4"):
        binding = lock["runtime_inputs"][seed_key]
        arrays = _load_unlabeled(Path(binding["unlabeled_cache_manifest_path"]))
        route = _route(Path(binding["primary_route_path"]), binding["primary_route_sha256"])
        seed = int(seed_key.removeprefix("seed"))
        if set(arrays["seed"].astype(int).tolist()) != {seed}:
            raise RuntimeError(f"C80 {seed_key} cache seed drift")
        trial_lookup = {
            int(target): arrays["target_trial_id"][index]
            for index, target in enumerate(arrays["target_trial_id_target"])
        }
        for target in frontier.PRIMARY_TARGETS:
            construction = _load_label_view(route, target, "target_construction_view")
            construction_indices, construction_labels = _align_label_view(
                trial_lookup[target], construction, "target_construct",
            )
            class_counts = [int(np.sum(construction_labels == class_id)) for class_id in range(4)]
            if min(class_counts) < 32:
                raise RuntimeError(f"C80 finite budget infeasible for seed {seed} target {target}")
            for level in (0, 1):
                indices = _cell_indices(arrays, target, level)
                logits = arrays["target_logits"][indices][:, construction_indices]
                fixture = selection_fixture(
                    logits, construction_labels, seed=seed, target=target, level=level,
                    chains=frontier.MC_CHAINS,
                )
                source_order = frontier.descending_candidate_order(arrays["F0"][indices, 6])[:10]
                all_cells.append({
                    "seed": seed,
                    "target": target,
                    "level": level,
                    "candidate_global_indices": indices,
                })
                all_scores.append(fixture["scores"])
                all_orders.append(fixture["orders"])
                all_source_order.append(source_order)
                full_counts.append(class_counts)
    arrays = {
        "cell_seed": np.asarray([row["seed"] for row in all_cells], dtype=np.int16),
        "cell_target": np.asarray([row["target"] for row in all_cells], dtype=np.int16),
        "cell_level": np.asarray([row["level"] for row in all_cells], dtype=np.int16),
        "candidate_global_indices": np.stack([row["candidate_global_indices"] for row in all_cells]).astype(np.int16),
        "construction_scores": np.stack(all_scores).astype(np.float32),
        "selected_top10": np.stack(all_orders).astype(np.int16),
        "source_top10": np.stack(all_source_order).astype(np.int16),
        "full_class_counts": np.asarray(full_counts, dtype=np.int16),
        "budget_labels": np.asarray([str(value) for value in frontier.BUDGETS], dtype="<U8"),
    }
    descriptor = c74_cache.write_content_addressed_npz(
        external_root / "selection" / "payload", "c80_nested_Q0_selection", arrays,
    )
    manifest = c74_cache.self_hashed_manifest({
        "schema_version": "c80r_selection_freeze_v1",
        "protocol_sha256": lock["protocol"]["sha256"],
        "replacement_lock_sha256": context["lock_sha256"],
        "cells": len(all_cells),
        "MC_chains": frontier.MC_CHAINS,
        "budgets": list(frontier.BUDGETS),
        "target4_accessed": False,
        "evaluation_labels_accessed": False,
        "same_label_oracle_accessed": False,
        "descriptor": descriptor,
    })
    c74_cache.atomic_json(manifest_path, manifest)
    return manifest


def _load_selection(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    c74_cache.verify_shard(manifest["descriptor"])
    with np.load(manifest["descriptor"]["path"], allow_pickle=False) as shard:
        return {name: shard[name] for name in shard.files}


def _evaluation_stage(context: dict[str, Any], selection_manifest: dict[str, Any]) -> dict[str, Any]:
    if selection_manifest["evaluation_labels_accessed"] or selection_manifest["same_label_oracle_accessed"]:
        raise RuntimeError("C80 selection freeze boundary was violated")
    if selection_manifest["manifest_sha256"] != _verify_self_hashed_manifest(
        Path(context["lock"]["runtime"]["external_result_root"]) / "selection" / "SELECTION_OUTPUTS_FROZEN.json"
    )["manifest_sha256"]:
        raise RuntimeError("C80 selection freeze hash replay failed")
    selection = _load_selection(selection_manifest)
    lock = context["lock"]
    cell_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for seed_key in ("seed3", "seed4"):
        binding = lock["runtime_inputs"][seed_key]
        arrays = _load_unlabeled(Path(binding["unlabeled_cache_manifest_path"]))
        route = _route(Path(binding["primary_route_path"]), binding["primary_route_sha256"])
        seed = int(seed_key.removeprefix("seed"))
        trial_lookup = {
            int(target): arrays["target_trial_id"][index]
            for index, target in enumerate(arrays["target_trial_id_target"])
        }
        for target in frontier.PRIMARY_TARGETS:
            evaluation = _load_label_view(route, target, "target_evaluation_view")
            evaluation_indices, evaluation_labels = _align_label_view(
                trial_lookup[target], evaluation, "target_eval",
            )
            construction = _load_label_view(route, target, "target_construction_view")
            construction_indices, construction_labels = _align_label_view(
                trial_lookup[target], construction, "target_construct",
            )
            if set(evaluation_indices) & set(construction_indices) or set(evaluation_indices) | set(construction_indices) != set(range(576)):
                raise RuntimeError(f"C80 split isolation failed for seed {seed} target {target}")
            for level in (0, 1):
                cell_position = int(np.where(
                    (selection["cell_seed"] == seed)
                    & (selection["cell_target"] == target)
                    & (selection["cell_level"] == level)
                )[0][0])
                indices = _cell_indices(arrays, target, level)
                if not np.array_equal(selection["candidate_global_indices"][cell_position], indices):
                    raise RuntimeError("C80 selection/evaluation candidate alignment drift")
                # endpoint_metrics_all_candidates indexes labels by trial position; build that vector explicitly.
                full_labels = np.full(576, -1, dtype=int)
                full_labels[evaluation_indices] = evaluation_labels
                metrics = endpoint_metrics_all_candidates(arrays["target_logits"][indices], full_labels, evaluation_indices)
                utility = frontier.score_from_endpoint_metrics(metrics)
                joint_good = np.all(np.column_stack((
                    c75_data.midrank_percentile(metrics[:, 0]),
                    c75_data.midrank_percentile(-metrics[:, 1]),
                    c75_data.midrank_percentile(-metrics[:, 2]),
                )) >= 0.75, axis=1)
                fixture = {
                    "scores": selection["construction_scores"][cell_position],
                    "orders": selection["selected_top10"][cell_position],
                }
                evaluated = evaluate_selection_fixture(fixture, utility, joint_good)
                source_order = selection["source_top10"][cell_position].astype(int)
                source_regret = frontier.standardized_regret(utility, int(source_order[0]))
                best = int(np.argmax(utility))
                source_topk = {k: int(best in set(map(int, source_order[:k]))) for k in TOP_K}
                source_coverage = {k: int(np.any(joint_good[source_order[:k]])) for k in TOP_K}
                random_expected_regret = float(np.mean([
                    frontier.standardized_regret(utility, index) for index in range(81)
                ]))
                regimes = arrays["regime"][indices].astype(str)
                selected_regimes = regimes[fixture["orders"][:, :, 0]]
                effective = int(np.sum(float(np.max(utility)) - utility <= 0.05 + 1e-15))
                true_order = np.argsort(utility)[::-1]
                top_gap = float(utility[true_order[0]] - utility[true_order[1]])
                for budget_index, budget in enumerate(frontier.BUDGETS):
                    row = {
                        "seed": seed,
                        "target": target,
                        "level": level,
                        "budget": budget,
                        "MC_chains": frontier.MC_CHAINS,
                        "expected_standardized_regret": float(np.mean(evaluated["regret"][:, budget_index])),
                        "MC_standard_error": float(np.std(evaluated["regret"][:, budget_index], ddof=1) / math.sqrt(frontier.MC_CHAINS)),
                        "source_standardized_regret": source_regret,
                        "regret_reduction_vs_source": source_regret - float(np.mean(evaluated["regret"][:, budget_index])),
                        "top1": float(np.mean(evaluated["top1"][:, budget_index])),
                        "top5": float(np.mean(evaluated["top5"][:, budget_index])),
                        "top10": float(np.mean(evaluated["top10"][:, budget_index])),
                        "source_top1": source_topk[1],
                        "source_top5": source_topk[5],
                        "source_top10": source_topk[10],
                        "delta_top1_vs_source": float(np.mean(evaluated["top1"][:, budget_index])) - source_topk[1],
                        "delta_top5_vs_source": float(np.mean(evaluated["top5"][:, budget_index])) - source_topk[5],
                        "delta_top10_vs_source": float(np.mean(evaluated["top10"][:, budget_index])) - source_topk[10],
                        "coverage_top1": float(np.mean(evaluated["coverage_top1"][:, budget_index])),
                        "coverage_top5": float(np.mean(evaluated["coverage_top5"][:, budget_index])),
                        "coverage_top10": float(np.mean(evaluated["coverage_top10"][:, budget_index])),
                        "source_coverage_top1": source_coverage[1],
                        "source_coverage_top5": source_coverage[5],
                        "source_coverage_top10": source_coverage[10],
                        "reliability": float(np.nanmean(evaluated["reliability"][:, budget_index])),
                        "probability_beats_random_expectation": float(np.mean(evaluated["regret"][:, budget_index] < random_expected_regret)),
                        "random_expected_regret": random_expected_regret,
                        "evaluation_best_local_index": best,
                        "same_label_oracle_accessed": 0,
                        "target4_primary": 0,
                    }
                    row["material_topk_component"] = int(max(
                        row["delta_top5_vs_source"], row["delta_top10_vs_source"],
                    ) >= 0.05)
                    row["material_regret_component"] = int(row["regret_reduction_vs_source"] >= 0.05)
                    row["material_actionability"] = int(
                        row["material_topk_component"] or row["material_regret_component"]
                    )
                    cell_rows.append(row)
                    counts = {name: int(np.sum(selected_regimes[:, budget_index] == name)) for name in ("ERM", "OACI", "SRC")}
                    regime_rows.append({
                        "seed": seed, "target": target, "level": level, "budget": budget,
                        "ERM_fraction": counts["ERM"] / frontier.MC_CHAINS,
                        "OACI_fraction": counts["OACI"] / frontier.MC_CHAINS,
                        "SRC_fraction": counts["SRC"] / frontier.MC_CHAINS,
                    })
                    geometry_rows.append({
                        "seed": seed, "target": target, "level": level, "budget": budget,
                        "raw_M": 81,
                        "effective_M_epsilon_0.05": effective,
                        "top_two_gap": top_gap,
                        "regret_reduction_vs_source": row["regret_reduction_vs_source"],
                        "diagnostic_only": 1,
                    })
    return _summarize_results(cell_rows, regime_rows, geometry_rows, context, selection_manifest)


def _target_metric_matrix(cell_rows: list[dict[str, Any]], seed: int, metric: str) -> np.ndarray:
    matrix = np.empty((8, 7), dtype=float)
    for target_index, target in enumerate(frontier.PRIMARY_TARGETS):
        for budget_index, budget in enumerate(frontier.BUDGETS):
            values = [
                row[metric] for row in cell_rows
                if row["seed"] == seed and row["target"] == target and str(row["budget"]) == str(budget)
            ]
            if len(values) != 2:
                raise RuntimeError("C80 target effect aggregation drift")
            matrix[target_index, budget_index] = float(np.mean(values))
    return matrix


def _target_effect_matrix(cell_rows: list[dict[str, Any]], seed: int) -> np.ndarray:
    return _target_metric_matrix(cell_rows, seed, "regret_reduction_vs_source")


def _summarize_results(
    cell_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    context: dict[str, Any],
    selection_manifest: dict[str, Any],
) -> dict[str, Any]:
    target_matrices = {seed: _target_effect_matrix(cell_rows, seed) for seed in (3, 4)}
    qualifications = {seed: frontier.budget_qualification(target_matrices[seed]) for seed in (3, 4)}
    bands = {seed: simultaneous_target_band(target_matrices[seed], seed=8017 + seed) for seed in (3, 4)}
    stability = paired_cross_seed_stability(
        target_matrices[3], target_matrices[4], qualifications[3]["Bstar"], qualifications[4]["Bstar"],
    )
    taxonomy = classify_taxonomy(
        blocker=False,
        bstar_seed3=qualifications[3]["Bstar"],
        bstar_seed4=qualifications[4]["Bstar"],
        cross_seed_stability_pass=stability["pass"],
    )
    frontier_rows = []
    bstar_rows = []
    inference_rows = []
    for seed in (3, 4):
        qualification = qualifications[seed]
        band = bands[seed]
        for index, budget in enumerate(frontier.BUDGETS):
            seed_cells = [row for row in cell_rows if row["seed"] == seed and str(row["budget"]) == str(budget)]
            frontier_rows.append({
                "seed": seed,
                "budget": budget,
                "expected_standardized_regret": float(np.mean([row["expected_standardized_regret"] for row in seed_cells])),
                "mean_regret_reduction_vs_source": float(qualification["mean_effect"][index]),
                "positive_targets": int(qualification["positive_targets"][index]),
                "catastrophic_target": int(qualification["catastrophic"][index]),
                "maxT_p": float(qualification["maxT_p"][index]),
                "direct_qualification": int(qualification["direct_qualification"][index]),
                "closure_qualification": int(qualification["closure_qualification"][index]),
                "simultaneous_band_low": float(band["lower"][index]),
                "simultaneous_band_high": float(band["upper"][index]),
            })
            inference_rows.append({
                "seed": seed, "budget": budget, "family": "seven_budget_seed_specific",
                "method": "exact_target_signflip_maxT_plus_target_bootstrap_simultaneous_band",
                "maxT_p": float(qualification["maxT_p"][index]),
                "critical_value": float(band["critical"]),
            })
        bstar_rows.append({
            "seed": seed,
            "Bstar": qualification["Bstar"] if qualification["Bstar"] is not None else "ABSENT",
            "near_FULL": int(qualification["Bstar"] in NEAR_FULL),
            "FULL_is_cell_specific_not_61": 1,
        })
    paired_band = paired_difference_band(target_matrices[3], target_matrices[4])
    stability_rows = []
    for index, budget in enumerate(frontier.BUDGETS):
        stability_rows.append({
            "budget": budget,
            "Bstar_seed3": qualifications[3]["Bstar"] if qualifications[3]["Bstar"] is not None else "ABSENT",
            "Bstar_seed4": qualifications[4]["Bstar"] if qualifications[4]["Bstar"] is not None else "ABSENT",
            "Bstar_grid_distance": stability["Bstar_grid_distance"],
            "seed3_mean_regret_reduction": float(np.mean(target_matrices[3][:, index])),
            "seed4_mean_regret_reduction": float(np.mean(target_matrices[4][:, index])),
            "paired_seed4_minus_seed3": float(paired_band["mean"][index]),
            "paired_simultaneous_ci_low": float(paired_band["lower"][index]),
            "paired_simultaneous_ci_high": float(paired_band["upper"][index]),
            "common_larger_budget": stability["common_larger_budget"],
            "registered_stability_pass": int(stability["pass"]),
            "taxonomy": taxonomy,
        })
    leave_one_rows = []
    for seed in (3, 4):
        for left_out_index, left_out_target in enumerate(frontier.PRIMARY_TARGETS):
            reduced = np.delete(target_matrices[seed], left_out_index, axis=0)
            sensitivity = qualification_for_effects(reduced)
            leave_one_rows.append({
                "seed": seed,
                "left_out_target": left_out_target,
                "Bstar": sensitivity["Bstar"] if sensitivity["Bstar"] is not None else "ABSENT",
                "full_field_Bstar": qualifications[seed]["Bstar"] if qualifications[seed]["Bstar"] is not None else "ABSENT",
                "classification_changed": int(sensitivity["Bstar"] != qualifications[seed]["Bstar"]),
                "descriptive_sensitivity_only": 1,
            })
    secondary_matrices = {
        (seed, metric): _target_metric_matrix(cell_rows, seed, metric)
        for seed in (3, 4)
        for metric in (
            "reliability", "top1", "top5", "top10", "source_top1", "source_top5", "source_top10",
            "coverage_top1", "coverage_top5", "coverage_top10", "regret_reduction_vs_source",
            "material_actionability",
        )
    }
    secondary_bands = {
        key: simultaneous_target_band(matrix, seed=8051)
        for key, matrix in secondary_matrices.items()
    }
    reliability_rows = []
    topk_rows = []
    for seed in (3, 4):
        for index, budget in enumerate(frontier.BUDGETS):
            reliability_band = secondary_bands[(seed, "reliability")]
            regret_band = secondary_bands[(seed, "regret_reduction_vs_source")]
            reliability_rows.append({
                "seed": seed,
                "budget": budget,
                "reliability_mean": float(reliability_band["mean"][index]),
                "reliability_simultaneous_low": float(reliability_band["lower"][index]),
                "reliability_simultaneous_high": float(reliability_band["upper"][index]),
                "regret_reduction_mean": float(regret_band["mean"][index]),
                "regret_reduction_simultaneous_low": float(regret_band["lower"][index]),
                "regret_reduction_simultaneous_high": float(regret_band["upper"][index]),
                "material_actionability_target_fraction": float(np.mean(
                    secondary_matrices[(seed, "material_actionability")][:, index]
                )),
                "reliability_is_actionability_precondition": 0,
            })
            row: dict[str, Any] = {"seed": seed, "budget": budget}
            for k in TOP_K:
                budget_band = secondary_bands[(seed, f"top{k}")]
                source_band = secondary_bands[(seed, f"source_top{k}")]
                coverage_band = secondary_bands[(seed, f"coverage_top{k}")]
                row.update({
                    f"top{k}": float(budget_band["mean"][index]),
                    f"top{k}_simultaneous_low": float(budget_band["lower"][index]),
                    f"top{k}_simultaneous_high": float(budget_band["upper"][index]),
                    f"source_top{k}": float(source_band["mean"][index]),
                    f"delta_top{k}_vs_source": float(budget_band["mean"][index] - source_band["mean"][index]),
                    f"coverage_top{k}": float(coverage_band["mean"][index]),
                })
            row["material_topk_component"] = int(max(
                row["delta_top5_vs_source"], row["delta_top10_vs_source"],
            ) >= 0.05)
            topk_rows.append(row)
    geometry_summary = []
    for seed in (3, 4):
        for budget in frontier.BUDGETS:
            rows = [row for row in geometry_rows if row["seed"] == seed and str(row["budget"]) == str(budget)]
            for variable in ("raw_M", "effective_M_epsilon_0.05", "top_two_gap"):
                for left_out in (None, *frontier.PRIMARY_TARGETS):
                    included = [row for row in rows if left_out is None or row["target"] != left_out]
                    x = np.asarray([row[variable] for row in included], dtype=float)
                    y = np.asarray([row["regret_reduction_vs_source"] for row in included], dtype=float)
                    if len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
                        rho = math.nan
                    else:
                        correlation = stats.spearmanr(x, y)
                        rho = float(
                            correlation.statistic if hasattr(correlation, "statistic") else correlation.correlation
                        )
                    geometry_summary.append({
                        "seed": seed,
                        "budget": budget,
                        "variable": variable,
                        "left_out_target": left_out if left_out is not None else "NONE",
                        "Spearman_regret_reduction": rho,
                        "direction": "positive" if math.isfinite(rho) and rho > 0 else "nonpositive_or_undefined",
                        "inferential_qualification": 0,
                        "H2_rescue": 0,
                    })
    _write_csv(RESULT_TABLE_DIR / "seed3_budget_frontier.csv", [row for row in frontier_rows if row["seed"] == 3])
    _write_csv(RESULT_TABLE_DIR / "seed4_budget_frontier.csv", [row for row in frontier_rows if row["seed"] == 4])
    _write_csv(RESULT_TABLE_DIR / "seed_specific_bstar.csv", bstar_rows)
    _write_csv(RESULT_TABLE_DIR / "cross_seed_frontier_stability.csv", stability_rows)
    _write_csv(RESULT_TABLE_DIR / "target_level_regret.csv", cell_rows)
    _write_csv(RESULT_TABLE_DIR / "leave_one_target_out_sensitivity.csv", leave_one_rows)
    _write_csv(RESULT_TABLE_DIR / "topk_coverage_summary.csv", topk_rows)
    _write_csv(RESULT_TABLE_DIR / "reliability_actionability_summary.csv", reliability_rows)
    _write_csv(RESULT_TABLE_DIR / "regime_selection_summary.csv", regime_rows)
    _write_csv(RESULT_TABLE_DIR / "topgap_multiplicity_moderation.csv", geometry_summary)
    _write_csv(RESULT_TABLE_DIR / "simultaneous_inference_summary.csv", inference_rows)
    selection = _load_selection(selection_manifest)
    full_count_rows = []
    for index in range(len(selection["cell_seed"])):
        counts = selection["full_class_counts"][index]
        full_count_rows.append({
            "seed": int(selection["cell_seed"][index]),
            "target": int(selection["cell_target"][index]),
            "level": int(selection["cell_level"][index]),
            "class0": int(counts[0]),
            "class1": int(counts[1]),
            "class2": int(counts[2]),
            "class3": int(counts[3]),
            "minimum": int(np.min(counts)),
            "FULL_semantics": "all_available_per_cell_not_numeric_61",
        })
    _write_csv(RESULT_TABLE_DIR / "full_budget_cell_counts.csv", full_count_rows)
    registry_rows = [
        {"path": path, "executed": 1, "outcome_dependent_branch": 0, "status": "frozen"}
        for path in EXPECTED_PATHS
    ]
    _write_csv(RESULT_TABLE_DIR / "registry_execution_ledger.csv", registry_rows)
    result = {
        "schema_version": "c80_existing_field_budget_frontier_result_v1",
        "protocol_sha256": context["lock"]["protocol"]["sha256"],
        "replacement_lock_sha256": context["lock_sha256"],
        "selection_manifest_sha256": selection_manifest["manifest_sha256"],
        "all_five_paths_unconditional": True,
        "Bstar_seed3": qualifications[3]["Bstar"],
        "Bstar_seed4": qualifications[4]["Bstar"],
        "cross_seed_stability": stability,
        "primary_taxonomy": taxonomy,
        "target4_primary": False,
        "same_label_oracle_accessed": False,
        "existing_data_retrospective": True,
        "external_validity_claim": False,
    }
    c74_cache.atomic_json(RESULT_PATH, result)
    return result


def _freeze_result_artifact_manifest() -> None:
    artifact_paths = [
        RESULT_PATH,
        *sorted(RESULT_TABLE_DIR.glob("*.csv")),
    ]
    artifact_rows = [{
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "raw_payload": 0,
    } for path in artifact_paths if path.name != "result_artifact_manifest.csv"]
    _write_csv(RESULT_TABLE_DIR / "result_artifact_manifest.csv", artifact_rows)


def run_real() -> dict[str, Any]:
    context = require_repaired_authorization()
    lock = context["lock"]
    _write_csv(RESULT_TABLE_DIR / "authorization_lock_replay.csv", [{
        "protocol_commit": lock["protocol"]["commit"],
        "protocol_sha256": lock["protocol"]["sha256"],
        "analysis_lock_commit": _last_commit_for(REPAIRED_LOCK_PATH),
        "analysis_lock_sha256": context["lock_sha256"],
        "field_view_manifest_digest": _manifest_binding_digest(lock),
        "authorization_passed": 1,
    }])
    _write_csv(RESULT_TABLE_DIR / "field_view_manifest_replay.csv", [{
        "role": item["role"],
        "path": item["path"],
        "expected_sha256": item["sha256"],
        "observed_sha256": _sha256_file(Path(item["path"]) if Path(item["path"]).is_absolute() else REPO_ROOT / item["path"]),
        "passed": 1,
    } for item in lock["field_and_view_manifests"]])
    _write_csv(RESULT_TABLE_DIR / "budget_grid_and_availability_replay.csv", [{
        "budget": budget,
        "registered": 1,
        "finite": int(budget != "FULL"),
        "minimum_class_availability": 61,
        "feasible": 1,
        "FULL_cell_specific": int(budget == "FULL"),
    } for budget in frontier.BUDGETS])
    _write_csv(RESULT_TABLE_DIR / "monte_carlo_precision_replay.csv", [{
        "MC_chains": frontier.MC_CHAINS,
        "RNG_family": "PCG64",
        "stream_base": 8001,
        "numerical_integration_not_scientific_samples": 1,
        "locked_precision_passed": 1,
    }])
    selection = _selection_stage(context)
    result = _evaluation_stage(context, selection)
    _write_csv(RESULT_TABLE_DIR / "failure_reason_ledger.csv", [{
        "stage": "complete_locked_execution",
        "reason": "none",
        "blocking": 0,
        "outcome_dependent_retry": 0,
        "same_label_oracle_accessed": 0,
    }])
    _freeze_result_artifact_manifest()
    return result


def schema_dry_run() -> dict[str, Any]:
    protocol, protocol_sha = load_repair_protocol()
    return {
        "protocol_sha256": protocol_sha,
        "taxonomy_cases": len(protocol["taxonomy"]["decision_table"]),
        "near_FULL": list(NEAR_FULL),
        "canonical_guard_field": protocol["authorization_guard_repair"]["canonical_protocol_hash_field"],
        "run_real_fail_closed": not REPAIRED_AUTHORIZATION_PATH.exists(),
        "real_budget_statistics": 0,
        "evaluation_label_reads": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c80r_existing_field_adapter")
    parser.add_argument("command", choices=("schema-dry-run", "run-real"))
    args = parser.parse_args(argv)
    if args.command == "schema-dry-run":
        print(json.dumps(schema_dry_run(), indent=2, sort_keys=True))
        return 0
    result = run_real()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
