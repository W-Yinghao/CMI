"""C34 Continuous Local Regret / Source-Objective Direction Audit tests."""
from __future__ import annotations

from oaci.continuous_regret import (endpoint_utility, gauge_locality, local_direction, margin_free_taxonomy,
                                    report, schema, selected_pair_regret, source_objective_components)


def _synth(n_units=5, n=7):
    rows = []
    for u in range(n_units):
        for i in range(n):
            selected = int(i == 3)
            better = int(i in (4, 5))
            bacc_delta = -0.02 + 0.012 * i + (0.025 if better else 0.0)
            nll_delta = -0.05 + 0.015 * i + (0.035 if better else 0.0)
            ece_delta = -0.04 + 0.010 * i + (0.025 if better else 0.0)
            score = 0.8 - 0.04 * i - (0.08 if better else 0.0)
            rows.append({
                "seed": 0, "target": u + 1, "level": 0, "regime": "r",
                "order": i, "epoch": i * 5, "selected_oaci": selected,
                "joint_good": int(bacc_delta > 0 and (nll_delta > 0 or ece_delta > 0)),
                "bacc_delta": bacc_delta, "nll_improve": nll_delta, "ece_improve": ece_delta,
                "bacc": 0.5 + bacc_delta, "nll": 1.0 - nll_delta, "ece": 0.2 - ece_delta,
                "score": score, "R_src": 1.0 - score, "c30_source_rank": i / max(n - 1, 1),
                "robust_core_score": score - 0.1,
                "target_unlabeled_r3_score": 0.3 + 0.01 * i,
                "target_grouped_centered_score": score - 0.4,
                "target_label_oracle_score": min(bacc_delta, nll_delta, ece_delta),
                "target_margin_mean": 0.2 + 0.03 * i,
                "joint_margin": min(bacc_delta, max(nll_delta, ece_delta)),
                "feat__selection_leakage_point": score + 0.05,
                "feat__audit_leakage_point": score + 0.02,
                "feat__source_guard_nll": 1.0 - score,
                "feat__source_audit_nll": 1.1 - score,
                "feat__source_guard_ece": 0.25 - 0.01 * i,
                "feat__source_audit_ece": 0.30 - 0.01 * i,
                "feat__source_audit_confidence": score,
            })
    endpoint_utility.attach_endpoint_utilities(rows)
    return rows


def test_config_hash_locked():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_endpoint_utility_registry_has_vectors_and_scalars():
    rows = _synth()
    reg = endpoint_utility.endpoint_registry(rows)
    assert reg
    assert {"target_bacc_delta", "target_nll_delta", "target_ece_delta"}.issubset(reg[0])
    assert "continuous_joint_min_margin" in reg[0]
    assert "endpoint_vector_norm_regret" in reg[0]


def test_selected_pair_regret_finds_continuous_better():
    res = selected_pair_regret.selected_pair_regret(_synth())
    assert res["summary"]["n_selected_continuous_better_pairs"] == 5
    assert res["summary"]["real_endpoint_regret_fraction"] is not None
    assert any(p["comparison"] == "nearest_continuous_better" for p in res["pairs"])


def test_local_direction_has_random_baseline():
    res = local_direction.local_source_direction(_synth())
    assert res["pair_rows"]
    assert res["random_rows"]
    assert res["summary"]["random_pairwise_auc"] == 0.5
    assert any(r["strategy"] == "target_unlabeled_r3_score" for r in res["random_aggregate"])


def test_component_conflict_records_unavailable_source_bacc():
    res = source_objective_components.source_objective_conflict(_synth())
    row = next(r for r in res["aggregate"] if r["component"] == "source_audit_worst_bacc")
    assert row["available"] is False
    assert res["summary"]["source_score_wrong_direction_fraction"] is not None


def test_taxonomy_and_gauge_are_deterministic():
    rows = _synth()
    selected = selected_pair_regret.selected_pair_regret(rows)
    direction = local_direction.local_source_direction(rows)
    components = source_objective_components.source_objective_conflict(rows)
    gauge = gauge_locality.gauge_locality(selected, direction)
    tax = margin_free_taxonomy.classify(endpoint_utility.summarize_registry(rows), selected, direction,
                                        components, gauge)
    assert tax["cases"]
    assert schema.M2 in tax["cases"]


def test_report_vector_precedes_scalar_and_guard_blocks_overclaim():
    rows = _synth()
    selected = selected_pair_regret.selected_pair_regret(rows)
    direction = local_direction.local_source_direction(rows)
    components = source_objective_components.source_objective_conflict(rows)
    gauge = gauge_locality.gauge_locality(selected, direction)
    res = {"config_hash": schema.LOCKED_C19_CONFIG_HASH, "selected_pairs": selected,
           "source_direction": direction, "source_objective_components": components,
           "gauge_locality": gauge,
           "binary_vs_continuous_boundary": {"summary": {"status_counts": {"real_endpoint_regret": 5}}},
           "taxonomy": margin_free_taxonomy.classify(endpoint_utility.summarize_registry(rows), selected,
                                                     direction, components, gauge)}
    md = report.render_md(res)
    assert md.index("Endpoint vectors first") < md.index("Fixed scalar summaries")
    try:
        report._guard_forbidden("continuous-regret selector works")
        raise AssertionError("guard failed")
    except ValueError:
        pass
    report._guard_forbidden("not a continuous-regret selector; no OACI rescue is claimed.")


def test_real_artifact_smoke_runs_read_only():
    res = report.run()
    assert res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH
    assert res["taxonomy"]["cases"]
    assert res["endpoint_registry"]
    assert res["source_direction"]["random_rows"]
