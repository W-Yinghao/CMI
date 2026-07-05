"""C17 source-signal identifiability audit: source-only atlas columns, diagnostic-only target labels,
within-fold permutation, top-k enrichment, rank-correlation robustness, LOTO multivariate probe + permutation
baseline + non-deployability, axis decomposition, class-boundary table, deterministic taxonomy, no target
oracle as a method. Synthetic atlas rows (build_atlas output schema)."""
from __future__ import annotations

import numpy as np

from oaci.identifiability.axis_decomposition import axis_decomposition
from oaci.identifiability.multivariate_probe import multivariate_probe
from oaci.identifiability.signal_atlas import SOURCE_SIGNALS, build_atlas, source_columns
from oaci.identifiability.target_labels import assert_diagnostic_only, is_source_column, is_target_column
from oaci.identifiability.taxonomy import case_taxonomy
from oaci.identifiability.univariate import _perm_p, _spearman, _topk_enrichment, univariate_identifiability


def _cand(h, *, is_erm, tb, tn, feasible=True, **src):
    d = {"model_hash": h, "is_erm": is_erm, "feasible": feasible, "target_worst_bacc": tb, "target_worst_nll": tn,
         "target_worst_ece": 0.1, "source_guard_worst_bacc": 0.5, "source_guard_worst_nll": 1.2,
         "source_guard_worst_ece": 0.1, "source_audit_worst_bacc": 0.5, "source_audit_worst_nll": 1.2,
         "source_audit_worst_ece": 0.1, "selection_leakage_point": 1.0, "audit_leakage_point": 0.8,
         "R_src": 0.8, "balanced_err": 0.3, "train_surrogate": 1.1, "epoch": 100}
    d.update(src)
    return d


def _folds(mk):
    return [{"seed": s, "target": t, "levels": {str(L): {"candidates": mk(s, t, L), "selected": {"ERM": "e", "OACI": "o"}}
                                                for L in (0, 1)}} for s in (0, 1, 2) for t in range(1, 4)]


def _atlas():
    def mk(s, t, L):
        # source_audit_worst_bacc perfectly ranks target bacc within each fold-level
        return [_cand("e", is_erm=True, tb=0.50, tn=1.20),
                _cand(f"a{s}{t}{L}", is_erm=False, tb=0.55, tn=1.10, source_audit_worst_bacc=0.60),
                _cand(f"b{s}{t}{L}", is_erm=False, tb=0.52, tn=1.15, source_audit_worst_bacc=0.55),
                _cand(f"c{s}{t}{L}", is_erm=False, tb=0.48, tn=1.25, source_audit_worst_bacc=0.45)]
    return build_atlas(_folds(mk))


def test_signal_atlas_uses_source_only_columns():
    rows = _atlas()
    assert source_columns(rows) == [f"src__{s}" for s in SOURCE_SIGNALS]
    assert all(is_source_column(c) for c in source_columns(rows))
    assert all(k.startswith(("src__", "tgt__")) or k in ("seed", "target", "level", "model_hash", "diagnostic_only_non_deployable") for r in rows for k in r)


def test_target_labels_marked_diagnostic_only_non_deployable():
    rows = _atlas()
    assert assert_diagnostic_only(rows) and all(r["diagnostic_only_non_deployable"] for r in rows)
    bad = dict(rows[0]); bad["diagnostic_only_non_deployable"] = False
    try:
        assert_diagnostic_only([bad])
    except ValueError:
        pass
    else:
        raise AssertionError("missing diagnostic_only flag must raise")
    assert is_target_column("tgt__target_bacc_good") and not is_target_column("src__R_src")


def test_rank_correlation_handles_constant_series():
    assert _spearman([1, 1, 1], [1, 2, 3]) is None          # constant x -> None, not a crash
    assert _spearman([1, 2, 3], [3, 2, 1]) < -0.99


def test_univariate_identifiability_permutation_is_within_target_seed_level():
    # source_audit_worst_bacc perfectly ranks target within each fold-level -> low perm p
    rows = _atlas()
    u = univariate_identifiability(rows, perm_seed=0)
    v = u["per_signal"]["source_audit_worst_bacc"]
    assert v["mean_within_fold_spearman_bacc"] is not None and v["mean_within_fold_spearman_bacc"] > 0.9
    assert v["perm_p_bacc"] is not None and v["perm_p_bacc"] < 0.05


def test_topk_enrichment_matches_manual_counts():
    rows = _atlas()
    fl = {}
    for r in rows:
        fl.setdefault((r["seed"], r["target"], r["level"]), []).append(r)
    e = _topk_enrichment(fl, "src__source_audit_worst_bacc", k=2)
    # top-2 by source_audit bacc are the a,b candidates (tb 0.55,0.52 > ERM 0.50) -> all good
    assert e["topk_good_rate"] == 1.0 and 0.0 < e["base_rate"] < 1.0


def test_multivariate_probe_uses_leave_one_target_out_and_permutation_baseline():
    m = multivariate_probe(_atlas(), perm_seed=0, n_perm=15)
    assert set(m["per_target_auc"]) <= {"1", "2", "3"}         # per held-out TARGET
    assert m["permutation_p"] is not None and m["permutation_mean_auc"] is not None
    assert m["loto_auc"] is not None


def test_multivariate_probe_does_not_emit_deployable_selector():
    m = multivariate_probe(_atlas(), n_perm=15)
    assert m["non_deployable"] is True and "selector" not in m and "chosen_model_hash" not in m


def test_axis_decomposition_separates_calibration_and_accuracy():
    u = univariate_identifiability(_atlas())
    a = axis_decomposition(u)
    assert "accuracy" in a["by_axis"] and "calibration" in a["by_axis"]
    assert "source_signals_see_calibration_more_than_accuracy" in a


def test_case_taxonomy_is_deterministic_and_no_target_oracle_method():
    u = univariate_identifiability(_atlas()); m = multivariate_probe(_atlas()); a = axis_decomposition(u)
    t1 = case_taxonomy(u, m, a); t2 = case_taxonomy(u, m, a)
    assert t1["case_label"] == t2["case_label"]
    assert t1["case_label"] in ("case_I_source_identifiable_accuracy", "case_II_calibration_identifiable_only",
                                "case_III_multivariate_weak_identifiability", "case_IV_source_unidentifiable_competence")
    # taxonomy never recommends a deployable target-oracle selector
    assert "deployable" not in t1["next_science"] or "not" in t1["next_science"].lower()


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c17-identifiability tests")


if __name__ == "__main__":
    _run_all()
