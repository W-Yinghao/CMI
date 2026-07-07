"""Project A — tests for the multi-dataset summary combiner (Step 10).

Builds fake per-dataset summary dicts (4-class and binary) and checks that the combiner refuses to
pool RAW bAcc across mixed class counts, pools chance-normalized excess instead, and preserves the
claim-boundary flags. Run:

    python -m h2cmi.tests.test_observability_multidataset_summary
"""
from __future__ import annotations

from h2cmi.observability.combine_summaries import combine


def _fake_summary(dataset, n_classes, runs):
    """One per-dataset summary dict mirroring what result_index.build_summary/validate produce,
    with chance-normalized per-run fields computed exactly as load_run would."""
    chance = 1.0 / n_classes
    ok = []
    for r in runs:
        rr = dict(r, status="ok", n_classes=n_classes, chance_bacc=round(chance, 4))
        rr["strict_dg_bacc_excess_norm"] = round((r["strict_dg_bacc"] - chance) / (1 - chance), 4)
        rr["online_tta_bacc_excess_norm"] = round((r["online_tta_bacc"] - chance) / (1 - chance), 4)
        rr["offline_tta_gain_bacc_norm"] = round(r["offline_tta_gain_bacc"] / (1 - chance), 4)
        ok.append(rr)

    def _mean(k):
        return round(sum(x[k] for x in ok) / len(ok), 4)

    gains = [x["offline_tta_gain_bacc"] for x in ok]
    agg = {
        "n_classes": n_classes, "chance_bacc": round(chance, 4),
        "n_runs": len(ok), "n_ok": len(ok), "n_skipped": 0,
        "mean_strict_dg_bacc": _mean("strict_dg_bacc"),
        "mean_strict_dg_bacc_excess_norm": _mean("strict_dg_bacc_excess_norm"),
        "mean_offline_tta_gain_bacc_norm": _mean("offline_tta_gain_bacc_norm"),
        "overall": {"offline_tta_harm_rate": round(sum(1 for g in gains if g < 0) / len(gains), 4)},
        "all_forbidden_violations_empty": True, "all_target_metrics_oracle_only": True,
        "all_target_metrics_identifiable_null": True, "all_prior_claims_compliant": True,
        "no_unknown_estimands": True, "missing_cells": [],
    }
    return {"dataset": dataset, "aggregate": agg, "validation": {"all_valid": True}, "runs": ok}


def test_combiner_rejects_overall_raw_bacc_across_mixed_class_counts():
    s4 = _fake_summary("BNCI2014_001", 4,
                       [{"strict_dg_bacc": 0.40, "online_tta_bacc": 0.38, "offline_tta_gain_bacc": -0.02}])
    s2 = _fake_summary("BNCI2014_004", 2,
                       [{"strict_dg_bacc": 0.40, "online_tta_bacc": 0.42, "offline_tta_gain_bacc": 0.01}])
    c = combine([s4, s2])
    assert c["mixed_n_classes"] is True
    assert c["raw_bacc_overall_suppressed"] is True
    assert "overall_raw_bacc" not in c
    assert "mean_strict_dg_bacc" not in c["overall_normalized"]     # no pooled raw bАcc anywhere
    # within-dataset raw bАcc is still preserved for reference
    assert c["per_dataset"]["BNCI2014_001"]["mean_strict_dg_bacc"] == 0.40


def test_combiner_uses_normalized_excess_for_overall():
    # 4-class strict 0.40 -> excess_norm (0.40-0.25)/0.75 = 0.2 ; binary 0.75 -> (0.75-0.5)/0.5 = 0.5
    s4 = _fake_summary("D4", 4, [{"strict_dg_bacc": 0.40, "online_tta_bacc": 0.40, "offline_tta_gain_bacc": 0.0}])
    s2 = _fake_summary("D2", 2, [{"strict_dg_bacc": 0.75, "online_tta_bacc": 0.75, "offline_tta_gain_bacc": 0.0}])
    c = combine([s4, s2])
    assert c["overall_normalized"]["mean_strict_dg_bacc_excess_norm"] == round((0.2 + 0.5) / 2, 4)
    assert c["overall_normalized"]["n_ok"] == 2


def test_cross_dataset_aggregate_uses_normalized_not_raw_bacc():
    # identical raw bАcc 0.60 means different things at K=4 vs K=2 -> normalized must differ
    s4 = _fake_summary("D4", 4, [{"strict_dg_bacc": 0.60, "online_tta_bacc": 0.60, "offline_tta_gain_bacc": 0.0}])
    s2 = _fake_summary("D2", 2, [{"strict_dg_bacc": 0.60, "online_tta_bacc": 0.60, "offline_tta_gain_bacc": 0.0}])
    c = combine([s4, s2])
    n4 = c["per_dataset"]["D4"]["mean_strict_dg_bacc_excess_norm"]   # (0.60-0.25)/0.75 = 0.4667
    n2 = c["per_dataset"]["D2"]["mean_strict_dg_bacc_excess_norm"]   # (0.60-0.50)/0.50 = 0.2
    assert n4 != n2 and c["raw_bacc_overall_suppressed"] is True


def test_combiner_preserves_all_claim_boundary_flags():
    s4 = _fake_summary("D4", 4, [{"strict_dg_bacc": 0.40, "online_tta_bacc": 0.40, "offline_tta_gain_bacc": -0.01}])
    s2 = _fake_summary("D2", 2, [{"strict_dg_bacc": 0.60, "online_tta_bacc": 0.60, "offline_tta_gain_bacc": -0.01}])
    c = combine([s4, s2])
    assert c["all_datasets_valid"] is True
    assert c["all_target_metrics_identifiable_null"] is True
    assert c["any_forbidden_violations"] is False and c["any_dataset_missing_cells"] is False
    for ds in ("D4", "D2"):
        pd = c["per_dataset"][ds]
        assert pd["all_target_metrics_identifiable_null"] is True
        assert pd["all_forbidden_violations_empty"] is True
    # a constituent forbidden violation must flip the aggregate
    s_bad = _fake_summary("Dbad", 2, [{"strict_dg_bacc": 0.60, "online_tta_bacc": 0.60, "offline_tta_gain_bacc": 0.0}])
    s_bad["aggregate"]["all_forbidden_violations_empty"] = False
    assert combine([s4, s_bad])["any_forbidden_violations"] is True


def test_combiner_neutralizes_all_skip_dataset():
    # a dataset with n_ok=0 (all legal skips -- e.g. invalid for the loader/paradigm) must NOT flip
    # the aggregate to a violation: its fail-closed empty-set flags are neutral, but it stays valid
    s4 = _fake_summary("D4", 4, [{"strict_dg_bacc": 0.40, "online_tta_bacc": 0.40, "offline_tta_gain_bacc": -0.01}])
    s_skip = {"dataset": "DskipAll",
              "aggregate": {"n_classes": None, "n_runs": 3, "n_ok": 0, "n_skipped": 3,
                            "all_forbidden_violations_empty": False,   # fail-closed on empty ok set
                            "all_target_metrics_oracle_only": False,
                            "all_target_metrics_identifiable_null": False,
                            "all_prior_claims_compliant": False, "no_unknown_estimands": False,
                            "missing_cells": [], "overall": {"offline_tta_harm_rate": None}},
              "validation": {"all_valid": True}, "runs": []}
    c = combine([s4, s_skip])
    assert c["n_datasets_with_ok_runs"] == 1
    assert c["any_forbidden_violations"] is False            # all-skip dataset is neutral
    assert c["all_target_metrics_identifiable_null"] is True  # only the active dataset counts
    assert c["all_datasets_valid"] is True                   # a legal all-skip grid is still valid


def _all_skip_summary(dataset):
    """A per-dataset summary for a grid where every cell is a legal skip (n_ok=0): aggregate flags
    are null + claim_boundary_status not_applicable, mirroring result_index.aggregate."""
    return {"dataset": dataset,
            "aggregate": {"n_classes": None, "n_runs": 3, "n_ok": 0, "n_skipped": 3,
                          "claim_boundary_status": "not_applicable_all_skipped",
                          "all_forbidden_violations_empty": None, "all_target_metrics_oracle_only": None,
                          "all_target_metrics_identifiable_null": None, "all_prior_claims_compliant": None,
                          "no_unknown_estimands": None, "missing_cells": [],
                          "overall": {"offline_tta_harm_rate": None}},
            "validation": {"all_valid": True}, "runs": []}


def test_combiner_marks_all_skip_dataset_not_applicable():
    s4 = _fake_summary("D4", 4, [{"strict_dg_bacc": 0.40, "online_tta_bacc": 0.40, "offline_tta_gain_bacc": -0.01}])
    c = combine([s4, _all_skip_summary("DskipAll")])
    pd = c["per_dataset"]["DskipAll"]
    assert pd["claim_boundary_status"] == "not_applicable_all_skipped"
    assert pd["all_target_metrics_identifiable_null"] is None      # null, not False
    assert c["n_datasets_all_skipped"] == 1 and c["datasets_all_skipped"] == ["DskipAll"]


def test_combiner_top_level_flags_ignore_all_skip_datasets_but_report_them():
    s4 = _fake_summary("D4", 4, [{"strict_dg_bacc": 0.40, "online_tta_bacc": 0.40, "offline_tta_gain_bacc": -0.01}])
    s2 = _fake_summary("D2", 2, [{"strict_dg_bacc": 0.60, "online_tta_bacc": 0.60, "offline_tta_gain_bacc": -0.01}])
    c = combine([s4, s2, _all_skip_summary("DskipAll")])
    assert c["all_target_metrics_identifiable_null"] is True       # only active (ok-run) datasets count
    assert c["any_forbidden_violations"] is False
    assert c["all_datasets_valid"] is True                         # a legal all-skip grid is still valid
    assert c["n_datasets_all_skipped"] == 1 and "DskipAll" in c["datasets_all_skipped"]
    assert c["n_datasets_with_ok_runs"] == 2


ALL_TESTS = [
    test_combiner_rejects_overall_raw_bacc_across_mixed_class_counts,
    test_combiner_uses_normalized_excess_for_overall,
    test_cross_dataset_aggregate_uses_normalized_not_raw_bacc,
    test_combiner_preserves_all_claim_boundary_flags,
    test_combiner_neutralizes_all_skip_dataset,
    test_combiner_marks_all_skip_dataset_not_applicable,
    test_combiner_top_level_flags_ignore_all_skip_datasets_but_report_them,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} MULTIDATASET-SUMMARY TESTS PASSED")


if __name__ == "__main__":
    run()
