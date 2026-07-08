"""Aggregate review-completion off-diagonal geometry stress rows.

Reads raw W1GEOM_OFFDIAG JSONL rows and writes a machine-readable CSV under
``h2cmi/results/review_completion``. Unit is (dataset, pair, subject), with
source seeds averaged within unit before cluster bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


NB = 10000
PERTS = ["rotation", "mixing", "strong_reref", "block_mixing"]
DIAG = [
    "bacc_fixed_reference_oneshot_uniform",
    "bacc_fixed_iterative_geometry_uniform",
    "bacc_joint_geometry_uniform",
    "bacc_latent_im_diag_uniform",
    "bacc_pooled_uniform",
]
FULL_COV = ["bacc_coral_latent", "bacc_source_recolored_ea"]
ALLOPS = ["bacc_identity_uniform"] + DIAG + FULL_COV
THRESHOLDS = [0.0, -0.01, -0.02]


def _load(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for path in glob.glob(pat):
            if "probe" in os.path.basename(path):
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("panel") == "W1GEOM_OFFDIAG" and "perturbation" in row and not row.get("provenance_fail"):
                        row["_path"] = path
                        rows.append(row)
    return rows


def _cluster_boot(vals_by_cluster: dict[tuple, list[float]], seed: int = 0) -> dict:
    keys = [k for k, vals in vals_by_cluster.items() if vals]
    if len(keys) < 2:
        return dict(mean=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), n_clusters=len(keys), n_units=0)
    arrs = {k: np.asarray(vals_by_cluster[k], dtype=float) for k in keys}
    unit_vals = np.asarray([arrs[k].mean() for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    bs = np.empty(NB, dtype=float)
    for i in range(NB):
        idx = rng.integers(0, len(keys), len(keys))
        bs[i] = unit_vals[idx].mean()
    return dict(
        mean=float(unit_vals.mean()),
        ci_lo=float(np.percentile(bs, 2.5)),
        ci_hi=float(np.percentile(bs, 97.5)),
        n_clusters=len(keys),
        n_units=len(keys),
    )


def _seed_avg(rows: list[dict]) -> dict[tuple, dict[str, float]]:
    tmp = defaultdict(lambda: defaultdict(list))
    for row in rows:
        unit = (row["dataset"], row["pair"], int(row["subject"]), row["perturbation"])
        for op in ALLOPS:
            if op in row:
                tmp[unit][op].append(float(row[op]))
    return {unit: {op: float(np.mean(vals)) for op, vals in ops.items()} for unit, ops in tmp.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=["results/h2cmi/review_completion_offdiag/w1offdiag_*.jsonl"],
        help="glob(s) for raw W1GEOM_OFFDIAG JSONL files",
    )
    ap.add_argument(
        "--out",
        default="h2cmi/results/review_completion/geometry_capacity_offdiagonal_results.csv",
    )
    args = ap.parse_args()

    rows = _load(args.inputs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = [
        "family",
        "perturbation",
        "metric",
        "operator",
        "threshold",
        "mean",
        "ci_lo",
        "ci_hi",
        "n_clusters",
        "n_units",
        "bootstrap_cluster",
        "seed_policy",
        "inputs",
    ]
    by_unit = _seed_avg(rows)
    out_rows = []
    input_list = ",".join(sorted({r["_path"] for r in rows}))
    for pert in PERTS:
        units = {u: vals for u, vals in by_unit.items() if u[3] == pert}
        for op in ALLOPS:
            stat = _cluster_boot({(u[0], u[1], u[2]): [vals[op]] for u, vals in units.items() if op in vals})
            out_rows.append(
                dict(
                    family="per_operator_BA",
                    perturbation=pert,
                    metric="balanced_accuracy",
                    operator=op.replace("bacc_", ""),
                    threshold="",
                    **stat,
                )
            )

        contrast = {}
        best_diag = {}
        best_full = {}
        for u, vals in units.items():
            if all(op in vals for op in DIAG + FULL_COV):
                best_diag[u] = max(vals[op] for op in DIAG)
                best_full[u] = max(vals[op] for op in FULL_COV)
                contrast[u] = best_full[u] - best_diag[u]
        for metric, vals_by_unit in [
            ("best_full_cov_minus_best_diagonal_latent", contrast),
            ("best_diagonal_latent_BA", best_diag),
            ("best_full_cov_sensor_or_coral_BA", best_full),
        ]:
            stat = _cluster_boot({(u[0], u[1], u[2]): [v] for u, v in vals_by_unit.items()})
            out_rows.append(
                dict(
                    family="contrast",
                    perturbation=pert,
                    metric=metric,
                    operator="",
                    threshold="",
                    **stat,
                )
            )

        for op in DIAG + FULL_COV:
            for threshold in THRESHOLDS:
                vals_by_cluster = {}
                for u, vals in units.items():
                    if op in vals and "bacc_identity_uniform" in vals:
                        vals_by_cluster[(u[0], u[1], u[2])] = [float((vals[op] - vals["bacc_identity_uniform"]) < threshold)]
                stat = _cluster_boot(vals_by_cluster, seed=stable_seed(op, pert, threshold))
                out_rows.append(
                    dict(
                        family="negative_change_rate",
                        perturbation=pert,
                        metric="operator_minus_identity_lt_threshold",
                        operator=op.replace("bacc_", ""),
                        threshold=threshold,
                        **stat,
                    )
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            row["bootstrap_cluster"] = "(dataset,pair,subject)"
            row["seed_policy"] = "source seeds averaged within unit before bootstrap"
            row["inputs"] = input_list
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"[OFFDIAG] rows={len(rows)} unit_perturbations={len(by_unit)} -> {args.out}")
    return 0


def stable_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(text.encode()[:8].ljust(8, b"0"), "little") & ((1 << 32) - 1)


if __name__ == "__main__":
    raise SystemExit(main())
