"""C33 Local Trajectory Boundary / Checkpoint Neighborhood Audit tests."""
from __future__ import annotations

import numpy as np

from oaci.local_boundary import (artifact_loader, local_gradients, local_information_ladder, plateau_audit,
                                 report, schema, selected_pair_audit, taxonomy, trajectory_boundary)


def _synth(n_units=6, n=12):
    rows = []
    for u in range(n_units):
        for i in range(n):
            # Alternating local boundary with a bad selected point adjacent to good candidates.
            good = int(i in (4, 6, 7, 10))
            score = 0.5 + 0.01 * i + (0.005 if good else 0.0)
            rows.append({
                "seed": 0, "target": u + 1, "level": 0, "regime": "r", "mode": "in_regime",
                "model_hash": f"{u}-{i}", "order": i, "epoch": i * 5, "selected_oaci": int(i == 5),
                "joint_good": good, "accuracy_good": good, "calibration_good": good, "pareto_good": good,
                "score": score, "R_src": score, "bacc": 0.5 + 0.02 * good, "nll": 1.0 - 0.03 * good,
                "ece": 0.2 - 0.01 * good, "bacc_delta": 0.02 * good, "nll_improve": 0.03 * good,
                "ece_improve": 0.01 * good, "target_confidence_mean": score, "target_confidence_std": 0.1,
                "target_entropy_mean": 1 - score, "target_entropy_std": 0.1, "target_margin_mean": score,
                "target_margin_std": 0.1, "target_logit_norm_mean": score, "target_logit_norm_std": 0.1,
                "target_pred_prop_c0": 0.25, "target_pred_prop_c1": 0.25, "target_pred_prop_c2": 0.25,
                "target_pred_prop_c3": 0.25, "feat__source_guard_nll": score,
                "feat__source_guard_ece": score, "feat__source_audit_nll": score,
            })
    artifact_loader.attach_local_scores(rows, schema.PRIMARY_MARGIN)
    return rows


def test_config_hash_locked():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_boundary_geometry_and_neighborhoods():
    b = trajectory_boundary.boundary_geometry(_synth())
    assert b["summary"]["n_units"] == 6
    assert b["summary"]["mean_transition_rate"] > 0
    assert b["summary"]["pm1_contains_joint_fraction"] == 1.0


def test_selected_pair_audit_classifies_local_pairs():
    p = selected_pair_audit.selected_pair_audit(_synth())
    assert p["summary"]["n_pairs"] == 6
    assert p["summary"]["median_order_delta"] == 1.0
    assert p["summary"]["source_flat_fraction"] is not None


def test_gradient_and_plateau_audits():
    rows = _synth()
    g = local_gradients.local_gradient_alignment(rows)
    pl = plateau_audit.plateau_audit(rows)
    assert g["summary"]["n_transition_pairs"] > 0
    assert pl["summary"]["mean_plateau_size"] >= 1.0


def test_local_information_ladder_has_random_baseline():
    ladder = local_information_ladder.local_random_and_ladder(_synth())
    assert ladder["random_rows"]
    assert any(r["strategy"] == "target_unlabeled_r3_score" for r in ladder["aggregate"])
    assert all("local_random_top1_hit_rate" in r for r in ladder["aggregate"])


def test_taxonomy_deterministic():
    rows = _synth()
    b = trajectory_boundary.boundary_geometry(rows)
    p = selected_pair_audit.selected_pair_audit(rows)
    g = local_gradients.local_gradient_alignment(rows)
    pl = plateau_audit.plateau_audit(rows)
    lad = local_information_ladder.local_random_and_ladder(rows)
    tax = taxonomy.classify(b, p, g, pl, lad)
    assert tax["cases"]
    assert schema.B1 in tax["cases"]


def test_report_forbidden_guard():
    for bad in ("local-boundary selector works", "target-unlabeled DG success", "target-grouped oracle as method"):
        try:
            report._guard_forbidden(bad)
            raise AssertionError("guard failed")
        except ValueError:
            pass
    report._guard_forbidden("not a local-boundary selector; no target-unlabeled DG success is claimed.")


def test_real_artifact_smoke_runs_read_only():
    res = report.run()
    assert res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH
    assert res["primary"]["taxonomy"]["cases"]
    assert res["primary"]["boundary"]["summary"]["n_units"] > 0
