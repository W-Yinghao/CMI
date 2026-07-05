"""CIGL R2b — leakage-vs-task Pareto report. Consumes the unified per-run rows (source/target bAcc + graph/
node leakage proxies + hardened stats) and computes the leakage-vs-task Pareto frontier over METHODS on the
same backbone. R2 answers "does CMI regularization have INDEPENDENT value on the task/leakage Pareto front?"
— NOT "who has the highest accuracy". A method is dominated if another achieves >= its target task AND <= its
graph leakage AND <= its node leakage, with >= one strict improvement.
"""
from __future__ import annotations
import json
import numpy as np

# the unified per-run row schema (one row per dataset x fold x target x seed x method)
ROW_SCHEMA = ("dataset", "fold", "target_subject", "seed", "method", "lambda_g", "lambda_node",
              "source_bacc", "target_bacc", "graph_kl_proxy", "node_kl_proxy", "graph_perm_p",
              "node_perm_p", "graph_fdr_q", "node_fdr_q", "multiprobe_detect_count",
              "task_retention_delta", "leakage_reduction_delta")

# lower is better for leakage; higher is better for task
_LEAKAGE = ("graph_kl_proxy", "node_kl_proxy")
_TASK = "target_bacc"


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_by_method(rows, dataset=None):
    """Mean over folds/seeds per (dataset, method). Returns {method: {task, graph_kl, node_kl, n}}."""
    agg = {}
    for r in rows:
        if dataset is not None and r.get("dataset") != dataset:
            continue
        agg.setdefault(r["method"], {"task": [], "graph": [], "node": []})
        agg[r["method"]]["task"].append(r.get(_TASK))
        agg[r["method"]]["graph"].append(r.get("graph_kl_proxy"))
        agg[r["method"]]["node"].append(r.get("node_kl_proxy"))
    return {m: {"target_bacc": _mean(v["task"]), "graph_kl": _mean(v["graph"]),
                "node_kl": _mean(v["node"]), "n": len(v["task"])} for m, v in agg.items()}


def _dominates(a, b):
    """a dominates b iff a is >= on task AND <= on BOTH leakages, with at least one strict improvement."""
    ge_task = a["target_bacc"] >= b["target_bacc"] - 1e-9
    le_g = a["graph_kl"] <= b["graph_kl"] + 1e-9
    le_n = a["node_kl"] <= b["node_kl"] + 1e-9
    strict = (a["target_bacc"] > b["target_bacc"] + 1e-9 or a["graph_kl"] < b["graph_kl"] - 1e-9
              or a["node_kl"] < b["node_kl"] - 1e-9)
    return ge_task and le_g and le_n and strict


def compute_pareto(method_points):
    """Mark each method dominated / on-frontier. method_points: {method: {target_bacc, graph_kl, node_kl}}."""
    methods = list(method_points)
    out = {}
    for m in methods:
        dominated_by = [o for o in methods if o != m and _dominates(method_points[o], method_points[m])]
        out[m] = {**method_points[m], "dominated_flag": bool(dominated_by),
                  "dominated_by": dominated_by, "on_frontier": not dominated_by}
    return out


def pareto_report(rows, datasets=None):
    """Full report: per-dataset method aggregate + Pareto flags. `rows` = list of unified row dicts (from the
    per-run JSON). Robust to missing optional fields (works on dummy/prior JSON)."""
    if datasets is None:
        datasets = sorted({r.get("dataset") for r in rows if r.get("dataset") is not None})
    report = {}
    for ds in datasets:
        pts = aggregate_by_method(rows, dataset=ds)
        report[ds] = compute_pareto(pts)
    frontier = {ds: sorted(m for m, v in report[ds].items() if v["on_frontier"]) for ds in report}
    return {"schema": list(ROW_SCHEMA), "by_dataset": report, "frontier": frontier, "n_rows": len(rows)}


def load_rows(path):
    """Load unified rows from a JSON file (list of row dicts, or {'rows': [...]})."""
    obj = json.load(open(path))
    return obj["rows"] if isinstance(obj, dict) and "rows" in obj else obj
