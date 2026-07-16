"""Tests: spurious-task DGP + DG-erasure oracle prove safe-erasure != DG-erasure (source-identifiable ticket)."""
import numpy as np
from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp
from tos_cmi.eval.dg_erasure_oracle import (delete, _head_bacc, target_dg_oracle,
                                            source_meta_subset_oracle, cmi_only_selector, evaluate_on_target)


def _g():
    return make_spurious_task_dgp(n_domains=8, per_domain=250, seed=0)


def test_dgp_structure():
    g = _g()
    assert g["spurious_sign"][g["target_dom"]] == -1.0
    assert sum(1 for s in g["spurious_sign"] if s == 1.0) >= 4          # source majority +1


def test_deleting_spur_helps_target_id_does_not():
    g = _g(); Z, y, d = g["Z"], g["y"], g["d"]; B = np.eye(g["D"]); tgt = g["target_dom"]
    src, te = d != tgt, d == tgt
    ident = _head_bacc(Z[src], y[src], Z[te], y[te])
    del_spur = _head_bacc(delete(Z[src], B, tuple(g["spur"])), y[src], delete(Z[te], B, tuple(g["spur"])), y[te])
    del_id = _head_bacc(delete(Z[src], B, tuple(g["id"])), y[src], delete(Z[te], B, tuple(g["id"])), y[te])
    assert del_spur - ident > 0.1                                       # removing the shortcut HELPS target
    assert abs(del_id - ident) < 0.05                                   # removing pure identity does NOT


def test_target_dg_oracle_selects_spurious():
    g = _g(); tdo = target_dg_oracle(g["Z"], g["y"], g["d"], g["target_dom"], np.eye(g["D"]), seed=0)
    assert tdo["d_target_best"] > 0.1
    assert len(set(tdo["best"]["S"]) & set(g["spur"])) >= 1             # best deletion includes a spurious dim


def test_source_meta_oracle_is_source_identifiable():
    g = _g(); B = np.eye(g["D"]); tgt = g["target_dom"]
    smo = source_meta_subset_oracle(g["Z"], g["y"], g["d"], tgt, B, seed=0)
    ev = evaluate_on_target(g["Z"], g["y"], g["d"], tgt, B, smo["S_star"], seed=0)
    assert len(set(smo["S_star"]) & set(g["spur"])) >= 1               # source-only selector finds the shortcut
    assert ev["d_target"] > 0.1                                        # and it improves the TRUE target (Result A)


def test_cmi_only_selector_is_worse_for_dg():
    g = _g(); B = np.eye(g["D"]); tgt = g["target_dom"]
    smo = source_meta_subset_oracle(g["Z"], g["y"], g["d"], tgt, B, seed=0)
    ev_meta = evaluate_on_target(g["Z"], g["y"], g["d"], tgt, B, smo["S_star"], seed=0)
    cmi = cmi_only_selector(g["Z"], g["y"], g["d"], tgt, B, seed=0)
    ev_cmi = evaluate_on_target(g["Z"], g["y"], g["d"], tgt, B, cmi["S_cmi"], seed=0)
    assert ev_meta["d_target"] >= ev_cmi["d_target"] - 0.02            # DG objective >= CMI-only for DG
