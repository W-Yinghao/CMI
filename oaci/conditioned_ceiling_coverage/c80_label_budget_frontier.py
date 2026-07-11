"""C80P protocol audit and pure label-budget frontier primitives.

This module deliberately has no real-data loader. C80P validates the protocol
and synthetic fixtures only; a future authorized C80E adapter must provide
already isolated arrays after replaying the execution lock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import c75_data


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c80p_tables"
PROTOCOL_PATH = REPORT_DIR / "C80_LABEL_BUDGET_FRONTIER_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C80_LABEL_BUDGET_FRONTIER_PROTOCOL.sha256"
LOCK_PATH = REPORT_DIR / "C80P_ANALYSIS_EXECUTION_LOCK.json"
AUTHORIZATION_PATH = REPORT_DIR / "C80E_PI_AUTHORIZATION_RECORD.json"

PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
BUDGETS: tuple[int | str, ...] = (1, 2, 4, 8, 16, 32, "FULL")
REQUESTED_BUDGETS: tuple[int | str, ...] = (1, 2, 4, 8, 16, 32, 64, "FULL")
N_CLASSES = 4
CANDIDATES_PER_CELL = 81
MC_CHAINS = 2048
TARGETS = 8
MATERIAL_REGRET = 0.05
MIN_POSITIVE_TARGETS = 6
CATASTROPHIC_TARGET = -0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty C80P table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = PROTOCOL_SHA_PATH.read_text().strip()
    observed = sha256_file(PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError(f"C80 protocol hash mismatch: {observed} != {expected}")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    return protocol, observed


def registry_audit() -> dict[str, int]:
    rows = read_csv(TABLE_DIR / "scientific_registry.csv")
    if len(rows) != 5:
        raise RuntimeError("C80 registry must contain five unconditional paths")
    categories = [name for name in rows[0] if name != "path_id"]
    if len(categories) != 16:
        raise RuntimeError("C80 registry must bind sixteen categories per path")
    blank = sum(not row[name].strip() for row in rows for name in categories)
    bound = sum(bool(row[name].strip()) for row in rows for name in categories)
    paths = [row["path_id"] for row in rows]
    if paths != ["P1", "P2", "S1", "S2", "S3"] or blank:
        raise RuntimeError("C80 registry path order or completeness drift")
    return {"paths": len(rows), "categories": len(categories), "bound_cells": bound, "blank_cells": blank}


def protocol_audit() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    if protocol["budget_design"]["locked_primary_grid"] != list(BUDGETS):
        raise RuntimeError("C80 locked budget grid drift")
    if protocol["monte_carlo"]["selected_chain_count"] != MC_CHAINS:
        raise RuntimeError("C80 Monte Carlo chain count drift")
    if protocol["no_outcome_boundary"]["real_seed3_seed4_budget_statistics_before_protocol"] != 0:
        raise RuntimeError("C80 protocol timing boundary failed")
    if protocol["information_boundary"]["same_label_oracle_reachable"]:
        raise RuntimeError("C80 same-label oracle became reachable")
    hashes = []
    for value in protocol["accepted_inputs"].values():
        if isinstance(value, dict) and "path" in value:
            path = REPO_ROOT / value["path"]
            observed = sha256_file(path)
            hashes.append({"path": value["path"], "expected": value["sha256"], "observed": observed})
            if observed != value["sha256"]:
                raise RuntimeError(f"C80 accepted-input hash drift: {value['path']}")
    registry = registry_audit()
    return {
        "protocol_sha256": protocol_sha,
        "accepted_input_hashes": len(hashes),
        "registry": registry,
        "real_budget_statistics": 0,
        "same_label_oracle_accessed": False,
        "C80E_authorized": AUTHORIZATION_PATH.exists(),
    }


def deterministic_stream_seed(seed: int, target: int, level: int, chain: int, *, base: int = 8001) -> int:
    key = f"C80|{seed}|{target}|{level}|{chain}".encode("ascii")
    low32 = int.from_bytes(hashlib.sha256(key).digest()[:4], byteorder="big", signed=False)
    return low32 ^ base


def nested_class_samples(
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    budgets: Iterable[int | str] = BUDGETS,
) -> dict[int | str, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    classes = sorted(set(labels.tolist()))
    if classes != list(range(N_CLASSES)):
        raise RuntimeError("C80 Q0 requires exactly classes 0,1,2,3")
    order = {class_id: rng.permutation(np.where(labels == class_id)[0]) for class_id in classes}
    output: dict[int | str, np.ndarray] = {}
    for budget in budgets:
        if budget == "FULL":
            count = min(len(order[class_id]) for class_id in classes)
            # FULL means every available row, so concatenate complete class permutations.
            selected = np.concatenate([order[class_id] for class_id in classes])
        else:
            count = int(budget)
            if any(len(order[class_id]) < count for class_id in classes):
                raise RuntimeError(f"C80 infeasible finite budget {budget}")
            selected = np.concatenate([order[class_id][:count] for class_id in classes])
        output[budget] = np.asarray(selected, dtype=int)
    finite = [budget for budget in budgets if budget != "FULL"]
    for left, right in zip(finite, finite[1:]):
        if not set(output[left]).issubset(set(output[right])):
            raise RuntimeError("C80 nested Q0 sampling contract failed")
    if finite and not set(output[finite[-1]]).issubset(set(output["FULL"])):
        raise RuntimeError("C80 finite budget is not nested in FULL")
    return output


def score_from_endpoint_metrics(metrics: np.ndarray) -> np.ndarray:
    metrics = np.asarray(metrics, dtype=float)
    if metrics.shape != (CANDIDATES_PER_CELL, 3):
        raise RuntimeError("C80 selector requires 81 candidates x [bAcc,NLL,ECE]")
    return np.mean(np.column_stack((
        c75_data.midrank_percentile(metrics[:, 0]),
        c75_data.midrank_percentile(-metrics[:, 1]),
        c75_data.midrank_percentile(-metrics[:, 2]),
    )), axis=1)


def score_candidates_from_logits(logits: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=int)
    indices = np.asarray(indices, dtype=int)
    if logits.ndim != 3 or logits.shape[0] != CANDIDATES_PER_CELL or logits.shape[2] != N_CLASSES:
        raise RuntimeError("C80 logits must have shape [81,trials,4]")
    metrics = []
    for candidate in range(CANDIDATES_PER_CELL):
        endpoint = c75_data.endpoint_metrics(logits[candidate, indices], labels[indices])
        metrics.append([endpoint["bAcc"], endpoint["NLL"], endpoint["ECE"]])
    return score_from_endpoint_metrics(np.asarray(metrics))


def descending_candidate_order(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (CANDIDATES_PER_CELL,):
        raise RuntimeError("C80 candidate score shape drift")
    return np.argsort(scores)[::-1]


def standardized_regret(evaluation_utility: np.ndarray, selected_index: int) -> float:
    utility = np.asarray(evaluation_utility, dtype=float)
    if utility.shape != (CANDIDATES_PER_CELL,):
        raise RuntimeError("C80 evaluation utility shape drift")
    spread = float(np.max(utility) - np.min(utility))
    if spread <= 1e-15:
        return 0.0
    return (float(np.max(utility)) - float(utility[int(selected_index)])) / spread


def exact_maxT_pvalues(target_effects: np.ndarray, *, null_margin: float = MATERIAL_REGRET) -> np.ndarray:
    """One-sided exact target-sign max-stat p-values across registered budgets."""
    effects = np.asarray(target_effects, dtype=float)
    if effects.shape != (TARGETS, len(BUDGETS)):
        raise RuntimeError("C80 max-T requires 8 targets x 7 budgets")
    centered = effects - float(null_margin)
    observed = np.mean(centered, axis=0)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=TARGETS)))
    null_max_array = np.max((signs @ centered) / TARGETS, axis=1)
    return np.asarray([
        (1 + int(np.sum(null_max_array >= value - 1e-15))) / (1 + len(null_max_array))
        for value in observed
    ])


def budget_qualification(target_regret_reduction: np.ndarray) -> dict[str, Any]:
    effects = np.asarray(target_regret_reduction, dtype=float)
    if effects.shape != (TARGETS, len(BUDGETS)):
        raise RuntimeError("C80 qualification requires 8 targets x 7 budgets")
    pvalues = exact_maxT_pvalues(effects)
    means = np.mean(effects, axis=0)
    positive = np.sum(effects > 0, axis=0)
    catastrophic = np.any(effects < CATASTROPHIC_TARGET, axis=0)
    direct = (
        (means >= MATERIAL_REGRET)
        & (pvalues <= 0.05)
        & (positive >= MIN_POSITIVE_TARGETS)
        & (~catastrophic)
    )
    closure = np.asarray([bool(np.all(direct[index:])) for index in range(len(BUDGETS))])
    bstar = next((BUDGETS[index] for index, value in enumerate(closure) if value), None)
    return {
        "mean_effect": means,
        "positive_targets": positive,
        "catastrophic": catastrophic,
        "maxT_p": pvalues,
        "direct_qualification": direct,
        "closure_qualification": closure,
        "Bstar": bstar,
    }


def bstar_grid_distance(left: int | str | None, right: int | str | None) -> int | None:
    if left is None or right is None:
        return None
    return abs(BUDGETS.index(left) - BUDGETS.index(right))


def assert_c80e_authorized() -> dict[str, Any]:
    if not LOCK_PATH.exists():
        raise RuntimeError("C80 analysis execution lock is absent")
    if not AUTHORIZATION_PATH.exists():
        raise RuntimeError("C80E direct PI authorization record is absent")
    lock = json.loads(LOCK_PATH.read_text())
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    protocol_sha = PROTOCOL_SHA_PATH.read_text().strip()
    if lock.get("protocol_sha256") != protocol_sha or authorization.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("C80E authorization/lock binding mismatch")
    return authorization


def binding_contract() -> dict[str, Any]:
    protocol, protocol_sha = load_protocol()
    return {
        "protocol_sha256": protocol_sha,
        "budgets": list(BUDGETS),
        "MC_chains": MC_CHAINS,
        "primary_targets": list(PRIMARY_TARGETS),
        "target4_primary": False,
        "same_label_oracle": False,
        "real_budget_analysis_started": False,
        "C80E_authorized": AUTHORIZATION_PATH.exists(),
        "success_gate": protocol["C80P_success_gate"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-protocol", "show-binding", "run-real"))
    args = parser.parse_args(argv)
    if args.command == "audit-protocol":
        print(json.dumps(protocol_audit(), indent=2, sort_keys=True))
        return 0
    if args.command == "show-binding":
        print(json.dumps(binding_contract(), indent=2, sort_keys=True))
        return 0
    assert_c80e_authorized()
    raise RuntimeError("C80E real-data adapter is intentionally unavailable in C80P")


if __name__ == "__main__":
    raise SystemExit(main())
