"""CIGL R2b — Pareto report: domination rule + consumes dummy result JSON without breaking."""
import json
import numpy as np

from cmi.eval.pareto_report import (aggregate_by_method, compute_pareto, pareto_report,
                                    load_rows, ROW_SCHEMA, _dominates)


def _rows():
    # 3 methods x 2 folds. cigl: low leakage, similar task -> frontier. dann: low task -> maybe dominated.
    # erm: high leakage -> dominated. High graph/node KL = more leakage (lower is better).
    base = []
    for f in range(2):
        base += [
            {"dataset": "2a", "fold": f, "method": "erm", "target_bacc": 0.46, "graph_kl_proxy": 1.25, "node_kl_proxy": 0.52},
            {"dataset": "2a", "fold": f, "method": "cigl_graph_node", "target_bacc": 0.46, "graph_kl_proxy": 0.66, "node_kl_proxy": 0.30},
            {"dataset": "2a", "fold": f, "method": "dann", "target_bacc": 0.40, "graph_kl_proxy": 0.70, "node_kl_proxy": 0.35},
        ]
    return base


def test_domination_rule():
    a = {"target_bacc": 0.46, "graph_kl": 0.66, "node_kl": 0.30}      # cigl
    b = {"target_bacc": 0.46, "graph_kl": 1.25, "node_kl": 0.52}      # erm (same task, more leakage)
    assert _dominates(a, b) and not _dominates(b, a)                  # cigl dominates erm
    c = {"target_bacc": 0.50, "graph_kl": 0.66, "node_kl": 0.30}      # more task, same leakage
    assert _dominates(c, a) and not _dominates(a, c)


def test_pareto_report_flags_and_frontier():
    rows = _rows()
    rep = pareto_report(rows)
    r2a = rep["by_dataset"]["2a"]
    assert r2a["erm"]["dominated_flag"] is True                       # erm dominated by cigl (same task, less leak)
    assert r2a["cigl_graph_node"]["on_frontier"] is True             # cigl on frontier
    assert "cigl_graph_node" in rep["frontier"]["2a"]
    assert rep["n_rows"] == 6 and set(ROW_SCHEMA) >= {"target_bacc", "graph_kl_proxy", "node_kl_proxy"}
    # aggregation means over folds
    agg = aggregate_by_method(rows, "2a")
    assert agg["cigl_graph_node"]["graph_kl"] == 0.66 and agg["erm"]["n"] == 2


def test_consumes_dummy_json(tmp_path):
    p = tmp_path / "dummy.json"
    json.dump({"rows": _rows()}, open(p, "w"))
    rep = pareto_report(load_rows(p))                                 # must not break on file input
    assert rep["frontier"]["2a"]
    # also robust to rows missing optional fields
    partial = [{"dataset": "x", "method": "erm", "target_bacc": 0.5, "graph_kl_proxy": 1.0, "node_kl_proxy": 0.5}]
    assert pareto_report(partial)["by_dataset"]["x"]["erm"]["on_frontier"] is True
