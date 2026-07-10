"""Registered C77 multi-regime association/transport/actionability benchmark."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
from scipy import stats

from . import c77_protocol


TABLE_DIR = c77_protocol.TABLE_DIR
SHARD_DIR = TABLE_DIR / "synthetic_shards"


def grid(protocol: dict | None = None) -> list[dict]:
    protocol = protocol or json.loads(c77_protocol.PROTOCOL_PATH.read_text())
    locked = protocol["synthetic_contract"]["grid"]
    keys = (
        "candidate_count", "effective_multiplicity", "top_gap",
        "association_strength", "transport_heterogeneity", "label_budget",
    )
    rows = []
    for values in itertools.product(*(locked[key] for key in keys)):
        row = dict(zip(keys, values))
        if row["effective_multiplicity"] <= row["candidate_count"]:
            row["cell_id"] = "syn_" + c77_protocol.payload_sha256(row)[:16]
            rows.append(row)
    return rows


def _compress_top(values: np.ndarray, effective_multiplicity: int, top_gap: float) -> np.ndarray:
    order = np.argsort(values)[::-1]
    result = values.copy()
    k = min(int(effective_multiplicity), len(values))
    top = float(values[order[0]])
    for rank, index in enumerate(order[:k]):
        result[index] = top - float(top_gap) * rank
    if k < len(values):
        boundary = top - float(top_gap) * max(k, 1)
        remainder = order[k:]
        result[remainder] = np.minimum(result[remainder], boundary - 0.02 * np.arange(1, len(remainder) + 1))
    return result


def _fit(train_X: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    ridge = 1e-6 * np.eye(train_X.shape[1])
    return np.linalg.solve(train_X.T @ train_X + ridge, train_X.T @ train_y)


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if denominator <= 0 else 1.0 - float(np.sum((y - pred) ** 2)) / denominator


def _simulate_once(cell: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    targets, regimes, trajectories = 9, 3, 2
    M = int(cell["candidate_count"])
    groups = []
    local_excess_r2 = []
    label_sigma = 0.9 / math.sqrt(float(cell["label_budget"]))
    for target in range(targets):
        target_shift = rng.normal()
        for regime in range(regimes):
            regime_shift = rng.normal()
            for trajectory in range(trajectories):
                source = rng.normal(size=M)
                coordinate = rng.normal(size=M)
                nonlinear = coordinate * coordinate - 1.0
                beta = float(cell["association_strength"]) * (
                    1.0 + float(cell["transport_heterogeneity"]) * (
                        0.45 * target_shift + 0.35 * regime_shift + 0.20 * rng.normal()
                    )
                )
                raw = 0.35 * source + beta * nonlinear + rng.normal(scale=0.70, size=M)
                utility = _compress_top(raw, int(cell["effective_multiplicity"]), float(cell["top_gap"]))
                measured = utility + rng.normal(scale=label_sigma, size=M)
                source -= np.mean(source)
                nonlinear -= np.mean(nonlinear)
                measured -= np.mean(measured)
                utility -= np.mean(utility)
                correlation = float(np.corrcoef(nonlinear, measured)[0, 1])
                local_excess_r2.append(correlation * correlation - 1.0 / max(M - 1, 1))
                groups.append({
                    "target": target, "regime": regime, "trajectory": trajectory,
                    "source": source, "nonlinear": nonlinear,
                    "measured": measured, "utility": utility,
                })
    local = np.asarray(local_excess_r2)
    test = stats.ttest_1samp(local, 0.0, alternative="greater")
    local_p = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0

    def evaluate_holdout(field: str, values: range) -> tuple[list[float], list[float], list[float]]:
        increments, top1_deltas, regret_reductions = [], [], []
        for heldout in values:
            train = [item for item in groups if item[field] != heldout]
            test_groups = [item for item in groups if item[field] == heldout]
            train_source = np.concatenate([item["source"] for item in train])
            train_nonlinear = np.concatenate([item["nonlinear"] for item in train])
            train_y = np.concatenate([item["measured"] for item in train])
            base_coef = _fit(train_source[:, None], train_y)
            full_coef = _fit(np.column_stack((train_source, train_nonlinear)), train_y)
            base_y, full_y, true_y = [], [], []
            fold_top_delta, fold_regret = [], []
            for item in test_groups:
                base = item["source"][:, None] @ base_coef
                full = np.column_stack((item["source"], item["nonlinear"])) @ full_coef
                truth = item["utility"]
                base_y.append(base); full_y.append(full); true_y.append(truth)
                best = int(np.argmax(truth))
                base_choice, full_choice = int(np.argmax(base)), int(np.argmax(full))
                fold_top_delta.append(float(full_choice == best) - float(base_choice == best))
                fold_regret.append(float(truth[full_choice] - truth[base_choice]))
            base_y = np.concatenate(base_y); full_y = np.concatenate(full_y); true_y = np.concatenate(true_y)
            increments.append(_r2(true_y, full_y) - _r2(true_y, base_y))
            top1_deltas.append(float(np.mean(fold_top_delta)))
            regret_reductions.append(float(np.mean(fold_regret)))
        return increments, top1_deltas, regret_reductions

    increments, top1_deltas, regret_reductions = evaluate_holdout("regime", range(regimes))
    trajectory_increments, _, _ = evaluate_holdout("trajectory", range(trajectories))
    incremental = float(np.nanmean(increments))
    trajectory_incremental = float(np.nanmean(trajectory_increments))
    regret = float(np.mean(regret_reductions))
    top1 = float(np.mean(top1_deltas))
    transport = bool(
        incremental >= 0.02 and trajectory_incremental >= 0.02
        and all(value > 0 for value in increments)
        and all(value > 0 for value in trajectory_increments)
    )
    actionable = bool(transport and regret >= 0.02 and top1 > 0)
    return {
        "association_detected": int(local_p < 0.05),
        "mean_local_excess_r2": float(np.mean(local)),
        "local_p": local_p,
        "incremental_R2": incremental,
        "positive_regimes": int(sum(value > 0 for value in increments)),
        "trajectory_incremental_R2": trajectory_incremental,
        "positive_trajectories": int(sum(value > 0 for value in trajectory_increments)),
        "transport_qualified": int(transport),
        "top1_increment": top1,
        "regret_reduction": regret,
        "actionability_qualified": int(actionable),
    }


def simulate_cell(cell: dict, replicates: int, base_seed: int) -> dict:
    outcomes = [
        _simulate_once(cell, base_seed + 1000003 * replicate + int(cell["cell_id"][-8:], 16))
        for replicate in range(int(replicates))
    ]
    result = dict(cell)
    result.update({
        "replicates": int(replicates),
        "association_detection_rate": float(np.mean([row["association_detected"] for row in outcomes])),
        "mean_local_excess_r2": float(np.mean([row["mean_local_excess_r2"] for row in outcomes])),
        "median_incremental_R2": float(np.median([row["incremental_R2"] for row in outcomes])),
        "median_trajectory_incremental_R2": float(np.median([row["trajectory_incremental_R2"] for row in outcomes])),
        "transport_qualification_rate": float(np.mean([row["transport_qualified"] for row in outcomes])),
        "mean_positive_regimes": float(np.mean([row["positive_regimes"] for row in outcomes])),
        "mean_positive_trajectories": float(np.mean([row["positive_trajectories"] for row in outcomes])),
        "mean_top1_increment": float(np.mean([row["top1_increment"] for row in outcomes])),
        "mean_regret_reduction": float(np.mean([row["regret_reduction"] for row in outcomes])),
        "actionability_qualification_rate": float(np.mean([row["actionability_qualified"] for row in outcomes])),
        "association_detection_count": int(sum(row["association_detected"] for row in outcomes)),
        "transport_qualification_count": int(sum(row["transport_qualified"] for row in outcomes)),
        "actionability_qualification_count": int(sum(row["actionability_qualified"] for row in outcomes)),
    })
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run_shard(shard: int, shards: int, workers: int | None = None) -> Path:
    protocol = json.loads(c77_protocol.PROTOCOL_PATH.read_text())
    if c77_protocol.sha256(c77_protocol.PROTOCOL_PATH) != c77_protocol.PROTOCOL_SHA_PATH.read_text().strip():
        raise RuntimeError("C77 synthetic run requires intact locked protocol")
    if int(shards) != int(protocol["synthetic_contract"]["shards"]):
        raise RuntimeError("C77 synthetic shard count differs from protocol")
    cells = [cell for index, cell in enumerate(grid(protocol)) if index % shards == shard]
    workers = workers or max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 48))
    rows = Parallel(n_jobs=workers, backend="loky", verbose=5)(
        delayed(simulate_cell)(cell, int(protocol["synthetic_contract"]["replicates"]), int(protocol["synthetic_contract"]["seed"]))
        for cell in cells
    )
    path = SHARD_DIR / f"shard_{shard:02d}_of_{shards:02d}.csv"
    _write_csv(path, rows)
    return path


def merge_shards() -> tuple[list[dict], list[dict]]:
    protocol = json.loads(c77_protocol.PROTOCOL_PATH.read_text())
    shards = int(protocol["synthetic_contract"]["shards"])
    rows: list[dict] = []
    for shard in range(shards):
        path = SHARD_DIR / f"shard_{shard:02d}_of_{shards:02d}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with open(path, newline="") as stream:
            rows.extend(csv.DictReader(stream))
    expected = grid(protocol)
    if len(rows) != len(expected) or {row["cell_id"] for row in rows} != {row["cell_id"] for row in expected}:
        raise RuntimeError("C77 synthetic shard coverage mismatch")
    rows.sort(key=lambda row: row["cell_id"])
    _write_csv(TABLE_DIR / "synthetic_transport_phase_diagram.csv", rows)

    null = [row for row in rows if float(row["association_strength"]) == 0.0]
    stable_signal = [row for row in rows if float(row["association_strength"]) == 0.5 and float(row["transport_heterogeneity"]) == 0.0]
    heterogeneous = [row for row in rows if float(row["association_strength"]) == 0.5 and float(row["transport_heterogeneity"]) == 1.5]
    high_ties = [row for row in stable_signal if int(row["effective_multiplicity"]) == 20]
    low_ties = [row for row in stable_signal if int(row["effective_multiplicity"]) == 2]

    def pooled_rate(selected: list[dict], count: str) -> float:
        numerator = sum(int(row[count]) for row in selected)
        denominator = sum(int(row["replicates"]) for row in selected)
        return numerator / denominator

    summary = [
        {"gate": "null_association_FPR", "observed": pooled_rate(null, "association_detection_count"), "operator": "<=", "threshold": protocol["synthetic_contract"]["FPR_upper_gate"], "passed": int(pooled_rate(null, "association_detection_count") <= protocol["synthetic_contract"]["FPR_upper_gate"]), "scope": "all registered null cells pooled; cellwise values disclosed"},
        {"gate": "stable_local_association_power", "observed": pooled_rate(stable_signal, "association_detection_count"), "operator": ">=", "threshold": protocol["synthetic_contract"]["power_gate"], "passed": int(pooled_rate(stable_signal, "association_detection_count") >= protocol["synthetic_contract"]["power_gate"]), "scope": "association_strength=0.5;transport_heterogeneity=0"},
        {"gate": "heterogeneity_reduces_transport", "observed": pooled_rate(stable_signal, "transport_qualification_count") - pooled_rate(heterogeneous, "transport_qualification_count"), "operator": ">", "threshold": 0.0, "passed": int(pooled_rate(stable_signal, "transport_qualification_count") > pooled_rate(heterogeneous, "transport_qualification_count")), "scope": "strength=0.5 stable minus heterogeneity=1.5"},
        {"gate": "effective_multiplicity_reduces_actionability", "observed": pooled_rate(low_ties, "actionability_qualification_count") - pooled_rate(high_ties, "actionability_qualification_count"), "operator": ">", "threshold": 0.0, "passed": int(pooled_rate(low_ties, "actionability_qualification_count") > pooled_rate(high_ties, "actionability_qualification_count")), "scope": "effective multiplicity 2 minus 20 under stable signal"},
        {"gate": "all_cells_completed", "observed": len(rows), "operator": "==", "threshold": len(expected), "passed": int(len(rows) == len(expected)), "scope": f"{shards} content-disjoint Slurm shards"},
    ]
    _write_csv(TABLE_DIR / "power_and_false_positive_plan.csv", summary)
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    shard_parser = sub.add_parser("shard")
    shard_parser.add_argument("--index", type=int, required=True)
    shard_parser.add_argument("--count", type=int, default=8)
    shard_parser.add_argument("--workers", type=int)
    sub.add_parser("merge")
    args = parser.parse_args(argv)
    if args.command == "shard":
        path = run_shard(args.index, args.count, args.workers)
        print(path)
    else:
        rows, summary = merge_shards()
        print(json.dumps({"cells": len(rows), "gates": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
