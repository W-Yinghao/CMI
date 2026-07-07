"""C24 Calibration Information Ladder / Identifiability Boundary Audit. Frozen C19 config locked; R3/R4 (target-
unlabeled) are NOT proxied from method-final logits and NOT finalized read-only (REQUIRES_REINFERENCE); the R3
feature builder is label-free (rejects target-label tokens) and deterministic; the source-only witnesses fire
when source summaries collide on divergent offsets; the few-label diagnostic recovers when the offset is a
scalar; oracle rungs recover; the report forbids DG-success/selector language. Synthetic score rows only."""
from __future__ import annotations

import numpy as np

from oaci.competence_probe import schema as c19
from oaci.information_ladder import (artifact_loader, few_label_calibration, oracle_ladder, report, rung_registry,
                                     schema, source_witness, target_unlabeled_features, taxonomy)


def _rows(mode="in_regime", regimes=("S0_full_support", "S2_rare_cells", "S3_nonestimable_cells"),
          n_targets=9, n_per=16, offset=0.8, signal=1.0, offset_in_features=True, seed=0):
    rng = np.random.RandomState(seed); rows = []
    for regime in regimes:
        for t in range(1, n_targets + 1):
            toff = (t - n_targets / 2) * offset
            for k in range(n_per):
                good = k % 2 == 0
                sc = toff + (signal if good else -signal) * 0.5 + rng.randn() * 0.3
                r = {"mode": mode, "regime": regime, "seed": 0, "target": t, "level": 0,
                     "model_hash": f"{regime}{t:02d}{k:03d}", "score": sc, "label": 1 if good else 0,
                     "epoch": 10 + k * 5, "order": k, "R_src": 0.6 + 0.02 * t, "train_surrogate": 1.1}
                fbase = (toff if offset_in_features else 0.0)
                for f in c19.ROBUST_CORE_FEATURES:
                    r["feat__" + f] = fbase + (0.3 if good else -0.3) + rng.randn() * 0.1
                rows.append(r)
    return rows


def test_c19_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_ladder_marks_r3r4_requires_reinference_readonly():
    av = {"per_candidate_target_unlabeled_ready": False, "r3r4_status": schema.STATUS_REQUIRES_REINFERENCE,
          "method_final_note": "x", "method_final_target_audit_count": 0}
    lad = {r["rung"]: r for r in rung_registry.ladder(av)}
    assert lad["R3"]["status"] == schema.STATUS_REQUIRES_REINFERENCE
    assert lad["R4"]["status"] == schema.STATUS_REQUIRES_REINFERENCE
    assert lad["R1"]["status"] == schema.STATUS_OK and lad["R6"]["status"] == schema.STATUS_OK
    assert rung_registry.reinference_rungs() == ("R3", "R4")


def test_r3r4_never_proxied_returns_blocked_without_sidecar():
    av = {"per_candidate_target_unlabeled_ready": False, "method_final_note": "wrong population"}
    out = target_unlabeled_features.build_target_unlabeled_gauge(_rows(), av, "in_regime", sidecar=None)
    assert out["status"] == schema.STATUS_REQUIRES_REINFERENCE and out["gauge_table"] is None


def test_target_unlabeled_features_are_label_free():
    names = target_unlabeled_features.target_unlabeled_feature_names()
    target_unlabeled_features.assert_no_target_labels(names)     # must not raise (word-boundary aware: 'y' in 'entropy' is fine)
    for tok in ("label", "bacc", "nll", "ece", "target_center", "target_rank", "score"):
        assert not any(tok in n.lower() for n in names)
    # 'y' must not appear as a standalone underscore-delimited token
    assert not any("y" in n.lower().split("_") for n in names)


def test_target_unlabeled_feature_builder_rejects_forbidden_token():
    try:
        target_unlabeled_features.assert_no_target_labels(["target_entropy_mean", "target_bacc_good"])
        raise AssertionError("expected ValueError for forbidden target-label token")
    except ValueError:
        pass


def test_label_free_geometry_never_reads_labels_and_is_deterministic():
    rng = np.random.RandomState(1); logits = rng.randn(200, 4)
    a = target_unlabeled_features.label_free_confidence_geometry(logits)
    b = target_unlabeled_features.label_free_confidence_geometry(logits.copy())
    assert a == b                                               # deterministic
    assert set(target_unlabeled_features.target_unlabeled_feature_names()) == set(a)
    props = sum(v for k, v in a.items() if k.startswith("target_pred_prop_c"))
    assert abs(props - 1.0) < 1e-9                              # predicted-class proportions sum to 1


def test_witnesses_show_source_nonidentifying_when_offset_not_in_features():
    # offset NOT in source features -> source distance does not predict offset distance -> non-identifying
    w = source_witness.witness_audit(_rows(offset_in_features=False, seed=2), "in_regime")
    assert not w.get("insufficient_units")
    assert w["source_predicts_offset"] is False and w["source_nonidentifying"] is True
    assert w["n_strong_witnesses"] > 0


def test_witnesses_show_source_predicts_offset_when_offset_in_features():
    # offset IS in source features -> source distance predicts offset distance -> identifying (not non-id)
    w = source_witness.witness_audit(_rows(offset_in_features=True, seed=2), "in_regime")
    assert w["source_predicts_offset"] is True and w["source_nonidentifying"] is False


def test_few_label_recovers_scalar_offset():
    few = few_label_calibration.few_label_curve(_rows(), "in_regime")
    assert few["target_centered_oracle"] > few["raw_pooled"]     # offset breaks pooling; centering recovers
    # a few labels per class should close a large fraction of the oracle gap
    assert few["few_labels_recover"] is True
    k0 = next(c for c in few["curve"] if c["k_per_class"] == 0)
    assert k0["gap_closed"] is not None


def test_oracle_ladder_recovers_over_raw():
    from oaci.score_gauge import gauge_feature_registry as gfr, offset_model
    rows = _rows(); gt = gfr.build_gauge_table(rows, "in_regime"); fit = offset_model.fit_offsets(gt)
    o = oracle_ladder.oracle_ladder(rows, "in_regime", fit["offset_hat_loto"])
    assert o["target_centered_oracle"] > o["raw_pooled"] and o["oracle_gap_over_raw"] > 0


def test_taxonomy_not_final_while_r3r4_pending():
    witnesses = {"source_nonidentifying": True, "n_strong_witnesses": 10, "mantel_corr_source_offset": 0.05, "mantel_perm_p": 0.6}
    few = {"few_labels_recover": True, "curve": [{"k_per_class": 2, "gap_closed": 0.6}], "max_gap_closed": 0.6}
    oracle = {"oracle_gap_over_raw": 0.1}
    identity = {"source_features_identity_separable": True}
    t = taxonomy.gauge_taxonomy(witnesses, few, oracle, identity, r1_gap=-0.8, r3r4=None)
    assert t["final"] is False
    assert schema.I2 in t["unresolved_pending_reinference"] and schema.I3 in t["unresolved_pending_reinference"]
    assert schema.I1 in t["established_readonly"] and schema.I4 in t["established_readonly"]


def test_taxonomy_finalizes_and_gates_identity_leakage_when_r3r4_supplied():
    witnesses = {"source_nonidentifying": True, "n_strong_witnesses": 10}
    few = {"few_labels_recover": False, "curve": [{"k_per_class": 8, "gap_closed": 0.1}], "max_gap_closed": 0.1}
    oracle = {"oracle_gap_over_raw": 0.1}
    identity = {"source_features_identity_separable": True}
    # apparent unlabeled recovery but does NOT generalize LOTO + identity-laden -> I7, not I2
    r3r4 = {"status": schema.STATUS_OK, "gap_closed": 0.5, "auc_improve": 0.04, "loto_generalizes": False}
    t = taxonomy.gauge_taxonomy(witnesses, few, oracle, identity, r1_gap=-0.8, r3r4=r3r4)
    assert t["final"] is True and t["primary_provisional"] == schema.I7


def test_no_selector_gate_and_forbidden_guard():
    # build a minimal res to exercise the gate
    rows = _rows()
    from oaci.score_gauge import gauge_feature_registry as gfr, offset_model, identity_leakage_audit as ila, risk_family
    gt = gfr.build_gauge_table(rows, "in_regime"); fit = offset_model.fit_offsets(gt)
    res = {"config_hash": schema.LOCKED_C19_CONFIG_HASH,
           "identity_leakage": ila.identity_leakage_audit(rows, "in_regime"),
           "target_unlabeled_r3r4": {"status": schema.STATUS_REQUIRES_REINFERENCE},
           "taxonomy": {"final": False}, "diagnostic_only_non_deployable": True}
    gate = {g["check"]: g["passed"] for g in report._no_selector_gate(res)}
    assert all(gate.values())
    # forbidden-claim guard: affirmative DG-success fires, negated passes
    try:
        report._guard_forbidden("this shows dg success on the target"); raise AssertionError("guard failed to fire")
    except ValueError:
        pass
    report._guard_forbidden("R3/R4 are not DG success and not a target-unlabeled selector.")


def test_availability_probe_is_readonly_and_reports_method_final_only(tmp_path=None):
    av = artifact_loader.target_unlabeled_availability()
    assert av["r3r4_status"] in (schema.STATUS_REQUIRES_REINFERENCE, schema.STATUS_OK)
    assert "method-final" in av["method_final_note"].lower()


def test_reinfer_structural_gates_pass():
    from oaci.information_ladder import target_reinfer
    st = target_reinfer.structural_gates()
    assert st["G5_features_label_free"] and st["G6_no_target_endpoint_metric_in_features"]
    assert st["G7_no_selected_checkpoint_artifact"] and st["G8_labels_only_for_validation_not_features"]


def test_reinfer_sidecar_is_label_free(tmp_path):
    import json
    from oaci.information_ladder import target_reinfer
    geom = target_unlabeled_features.label_free_confidence_geometry(np.random.RandomState(0).randn(50, 4))
    res = [{"per_candidate": [{"seed": 0, "target": 4, "level": 0, "model_hash": "abc",
                              "epoch": 10, "target_unlabeled": geom}]}]
    out = str(tmp_path / "sidecar.json")
    n = target_reinfer.write_sidecar(res, out)
    d = json.load(open(out))
    assert n == 1 and d["config_hash"] == schema.LOCKED_C19_CONFIG_HASH
    # structurally impossible for R3/R4 to touch labels: no y / label / logits / endpoint keys anywhere
    blob = json.dumps(d).lower()
    for forbidden in ('"y"', '"label"', 'logits', 'bacc', 'worst', 'target_nll'):
        assert forbidden not in blob


def test_reinfer_merge_fold_partials(tmp_path):
    import json
    from oaci.information_ladder import target_reinfer
    fold_dir = tmp_path / "folds"; fold_dir.mkdir()
    geom = target_unlabeled_features.label_free_confidence_geometry(np.random.RandomState(1).randn(40, 4))
    for s, t in [(0, 4), (1, 7)]:
        json.dump({"per_candidate": [{"seed": s, "target": t, "level": 0, "model_hash": f"h{s}{t}",
                                     "target_unlabeled": geom}]},
                  open(fold_dir / f"seed-{s}-target-{t:03d}.json", "w"))
    out = str(tmp_path / "merged.json")
    n = target_reinfer.merge_fold_partials(str(fold_dir), out)
    assert n == 2 and len(json.load(open(out))["per_candidate"]) == 2
