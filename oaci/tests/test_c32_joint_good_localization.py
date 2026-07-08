"""C32 Joint-Good Localization / Selection-Regret Anatomy tests."""
from __future__ import annotations

import numpy as np

from oaci.endpoint_geometry import endpoint_labels
from oaci.joint_good_localization import (information_ladder, landscape, regret, report, schema, taxonomy,
                                          topk)


def _synth(n_targets=6, n_per=14, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for t in range(1, n_targets + 1):
        offset = (t - n_targets / 2) * 0.2
        erm_b, erm_n, erm_e = 0.50 + offset, 1.20 - offset, 0.20 - offset * 0.1
        selected_idx = 0
        for i in range(n_per):
            comp = i / (n_per - 1)
            bacc = erm_b + 0.08 * comp
            nll = erm_n - 0.14 * comp
            ece = erm_e - 0.04 * comp
            rows.append({
                "mode": "in_regime", "regime": "r", "seed": 0, "target": t, "level": 0,
                "model_hash": f"{t}-{i}", "score": offset + comp + rng.randn() * 0.01,
                "R_src": comp, "label": int(comp > 0.05), "order": i, "epoch": 10 + i * 5,
                "bacc": bacc, "nll": nll, "ece": ece, "erm_bacc": erm_b, "erm_nll": erm_n,
                "erm_ece": erm_e, "selected_oaci": int(i == selected_idx),
                "target_confidence_mean": comp + rng.randn() * 0.01,
                "target_confidence_std": 0.1 + 0.01 * i,
                "target_entropy_mean": 1.0 - comp,
                "target_entropy_std": 0.2 + 0.01 * rng.rand(),
                "target_margin_mean": comp,
                "target_margin_std": 0.1 + 0.01 * rng.rand(),
                "target_logit_norm_mean": 1.0 + comp,
                "target_logit_norm_std": 0.2 + 0.01 * rng.rand(),
                "target_pred_prop_c0": 0.25 + 0.05 * comp,
                "target_pred_prop_c1": 0.25 - 0.02 * comp,
                "target_pred_prop_c2": 0.25,
                "target_pred_prop_c3": 0.25 - 0.03 * comp,
            })
    return endpoint_labels.attach_labels(rows, margin=schema.IMPROVE_MARGIN)


def test_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_landscape_and_random_baseline():
    rows = _synth()
    land = landscape.joint_good_landscape(rows)
    rb = topk.random_baseline(rows, schema.TOP_KS)
    assert land["joint_good_rate"] > 0.25
    assert land["trajectory_any_joint_fraction"] == 1.0
    assert next(r for r in rb["topk"] if r["k"] == 1)["hit_rate"] > 0.25


def test_source_topk_enrichment_defined():
    rows = _synth()
    tk = topk.topk_enrichment(rows, lambda r: r["score"], schema.TOP_KS)
    assert next(r for r in tk["topk"] if r["k"] == 1)["hit_rate"] == 1.0
    assert all(r["hit_enrichment"] is not None for r in tk["topk"])


def test_selected_regret_reports_nearest_joint_without_hashes():
    rows = _synth()
    sel = regret.selected_oaci_regret(rows)
    assert sel["summary"]["selected_joint_hit_rate"] < 1.0
    assert sel["summary"]["median_nearest_order_distance"] is not None
    assert "model_hash" not in sel["per_trajectory"][0]


def test_information_ladder_runs_with_fixed_label_free_features():
    rows = _synth()
    lad = information_ladder.localization_ladder(rows, schema.TOP_KS)
    names = {m["strategy"] for m in lad["models"]}
    assert {"source_score", "target_unlabeled_loto", "target_grouped_centered_score"} <= names
    assert lad["meta"]["target_unlabeled_missing_predictions"] == 0


def test_taxonomy_deterministic():
    rows = _synth()
    land = landscape.joint_good_landscape(rows)
    rb = topk.random_baseline(rows, schema.TOP_KS)
    src = topk.topk_enrichment(rows, lambda r: r["score"], schema.TOP_KS)
    sel = regret.selected_oaci_regret(rows)
    lad = information_ladder.localization_ladder(rows, schema.TOP_KS)
    tax = taxonomy.classify(land, rb, src, sel, lad)
    assert schema.J1 in tax["cases"]
    assert tax["cases"]


def test_report_forbids_affirmative_selector_claims():
    for bad in ("a joint-good selector is ready", "deployable localization works", "selected-checkpoint artifact"):
        try:
            report._guard_forbidden(bad)
            raise AssertionError("guard failed")
        except ValueError:
            pass
    report._guard_forbidden("not a joint-good selector; no selected-checkpoint artifact; not deployable localization.")


def test_real_artifact_smoke_runs_read_only():
    res = report.run()
    p = res["primary"]
    assert res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH
    assert p["landscape"]["n_candidates"] > 0
    assert p["taxonomy"]["cases"]
    assert res["target_unlabeled"]["missing_rows"] == 0
