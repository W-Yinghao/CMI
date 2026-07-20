"""C23 Target-Free Score Calibration / Gauge Audit. Frozen C19 config locked; the target-identity-leakage HARD
GATE runs before any positive claim; the primary gauge is target-anonymous (no target id / subject / score /
label / target-wise transform); the offset model uses fixed ridge L2=1.0 with NO grid search, evaluated leave-
one-target-out; no selector / selected-checkpoint artifact; deterministic G1-G6 taxonomy; report forbids
selector/detector/deployable language. Synthetic score rows only (no real data, no re-scoring)."""
from __future__ import annotations

import numpy as np

from oaci.competence_probe import schema as c19
from oaci.score_gauge import (ceiling_ladder, gauge_feature_registry, identity_leakage_audit, offset_model, report,
                              risk_family, schema, taxonomy)


def _rows(mode="in_regime", regimes=("S0_full_support", "S2_rare_cells", "S3_nonestimable_cells"),
          n_targets=9, n_per=14, offset=0.8, signal=1.0, offset_in_features=True, seed=0):
    """Synthetic C22-shaped rows: per-target score OFFSET (breaks pooling) + within-target ranking signal. When
    offset_in_features, the per-target offset is ALSO visible in the source features (so a target-free gauge can
    recover it AND source features are target-identity-separable)."""
    rng = np.random.RandomState(seed); rows = []
    for regime in regimes:
        for t in range(1, n_targets + 1):
            toff = (t - n_targets / 2) * offset
            for k in range(n_per):
                good = k % 2 == 0
                sc = toff + (signal if good else -signal) * 0.5 + rng.randn() * 0.3
                r = {"mode": mode, "regime": regime, "seed": 0, "target": t, "level": 0,
                     "model_hash": f"{regime}{t}{k}", "score": sc, "label": 1 if good else 0,
                     "epoch": 10 + k * 5, "order": k, "R_src": 0.6 + 0.02 * t, "train_surrogate": 1.1}
                fbase = (toff if offset_in_features else 0.0)
                for f in c19.ROBUST_CORE_FEATURES:
                    r["feat__" + f] = fbase + (0.3 if good else -0.3) + rng.randn() * 0.1
                rows.append(r)
    return rows


def test_c19_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert schema.frozen_config_hash() == "664007686afb520f"


def test_primary_gauge_excludes_target_identity():
    names = gauge_feature_registry.gauge_feature_names()
    gauge_feature_registry.assert_target_anonymous(names)          # must not raise
    for tok in ("target", "subject", "score", "label", "regime", "seed", "domain_id", "target_rank"):
        assert not any(tok in n.lower() for n in names)
    assert len(names) == 2 * len(c19.ROBUST_CORE_FEATURES) + len(schema.GAUGE_EXTRA)


def test_gauge_registry_rejects_forbidden_token():
    try:
        gauge_feature_registry.assert_target_anonymous(["source_guard_nll__mean", "target_center"])
        raise AssertionError("expected ValueError for forbidden target-wise token")
    except ValueError:
        pass


def test_gauge_table_offset_matches_per_target_mean_score():
    rows = _rows()
    gt = gauge_feature_registry.build_gauge_table(rows, "in_regime")
    assert len(gt) == 9
    for t, d in gt.items():
        want = float(np.mean([r["score"] for r in rows if r["mode"] == "in_regime" and r["target"] == t]))
        assert abs(d["offset"] - want) < 1e-9
        assert "score" not in d["gauge"] and "target" not in d["gauge"]


def test_identity_leakage_audit_runs_and_flags_separable_features():
    # offset in features -> targets ARE separable from source features (well above 1/9 chance)
    idn = identity_leakage_audit.identity_leakage_audit(_rows(offset_in_features=True), "in_regime")
    assert idn["target_id_accuracy_from_source_features"] is not None
    assert idn["chance"] == schema.IDENTITY_LEAKAGE_CHANCE
    assert idn["target_id_accuracy_from_source_features"] > idn["chance"]
    assert idn["source_features_identity_separable"] is True


def test_identity_leakage_low_when_features_carry_no_offset():
    # no offset in features -> source features do NOT identify the target
    idn = identity_leakage_audit.identity_leakage_audit(_rows(offset_in_features=False, seed=3), "in_regime")
    assert idn["target_id_accuracy_from_source_features"] < schema.IDENTITY_LEAKAGE_CEILING


def test_offset_model_fixed_regularization_no_grid_search():
    import inspect
    src = inspect.getsource(offset_model)
    assert "GridSearch" not in src and "grid" not in src.lower().replace("no grid", "")
    assert schema.RIDGE_L2 == 1.0 and schema.MODEL == "ridge"
    gt = gauge_feature_registry.build_gauge_table(_rows(), "in_regime")
    fit = offset_model.fit_offsets(gt)
    assert set(fit["offset_hat_loto"]) == set(fit["targets"]) and fit["n_targets"] == 9


def test_offset_loto_recovers_when_offset_in_features():
    # when the per-target offset is visible in source features, LOTO offset prediction should be well above chance
    gt = gauge_feature_registry.build_gauge_table(_rows(offset_in_features=True), "in_regime")
    fit = offset_model.fit_offsets(gt)
    assert fit["loto_r2"] is not None and fit["loto_r2"] > 0.3


def test_offset_loto_fails_when_offset_not_in_features():
    # offset NOT in features -> target-free LOTO cannot recover it (source-unobservable offset)
    gt = gauge_feature_registry.build_gauge_table(_rows(offset_in_features=False, seed=5), "in_regime")
    fit = offset_model.fit_offsets(gt)
    assert (fit["loto_r2"] or -1) < 0.3


def test_ceiling_ladder_oracle_beats_raw_with_offset():
    rows = _rows()
    gt = gauge_feature_registry.build_gauge_table(rows, "in_regime")
    fit = offset_model.fit_offsets(gt)
    lad = ceiling_ladder.ceiling_ladder(rows, "in_regime", fit["offset_hat_loto"])
    # oracle target-centering removes the offset -> pooled AUC rises above raw
    assert lad["target_centered_oracle"] > lad["raw_pooled"]
    assert lad["within_target_ceiling"] is not None


def test_taxonomy_g3_when_identity_laden_and_loto_fails():
    ladder = {"raw_pooled": 0.5, "source_gauge_loto": 0.55, "regime_centered": 0.5,
              "target_centered_oracle": 0.8, "target_rank_oracle": 0.8, "within_target_ceiling": 0.7,
              "gap_closed_source_gauge": 0.10, "auc_improve_source_gauge": 0.05}
    fit = {"loto_r2": -0.5}
    identity = {"source_features_identity_separable": True}
    diag = {"loto_beats_permutation": False}
    risk = {"gap_closed": 0.0}
    t = taxonomy.gauge_taxonomy(ladder, fit, identity, diag, risk, epoch_residual=False)
    assert t["primary_case"] == schema.G3 and t["diagnostic_only_non_deployable"]


def test_taxonomy_g5_when_offset_source_unobservable():
    ladder = {"raw_pooled": 0.5, "source_gauge_loto": 0.50, "regime_centered": 0.5,
              "target_centered_oracle": 0.8, "target_rank_oracle": 0.8, "within_target_ceiling": 0.7,
              "gap_closed_source_gauge": 0.0, "auc_improve_source_gauge": 0.0}
    fit = {"loto_r2": -0.3}
    t = taxonomy.gauge_taxonomy(ladder, fit, {"source_features_identity_separable": False},
                               {"loto_beats_permutation": False}, {"gap_closed": 0.0}, epoch_residual=False)
    assert t["primary_case"] == schema.G5


def test_no_selected_checkpoint_artifact_emitted():
    rows = _rows()
    gt = gauge_feature_registry.build_gauge_table(rows, "in_regime")
    fit = offset_model.fit_offsets(gt)
    idn = identity_leakage_audit.identity_leakage_audit(rows, "in_regime")
    lad = ceiling_ladder.ceiling_ladder(rows, "in_regime", fit["offset_hat_loto"])
    rf = risk_family.risk_family_gauge(gt, rows, "in_regime")
    res = {"config_hash": schema.LOCKED_C19_CONFIG_HASH, "gauge_feature_names": gauge_feature_registry.gauge_feature_names(),
           "identity_leakage": idn, "offset_fit": fit, "ceiling_ladder": lad, "risk_family": rf,
           "diagnostic_only_non_deployable": True}
    gate = {g["check"]: g["passed"] for g in report._no_selector_gate(res)}
    assert all(gate.values())
    # the offset model returns target-LEVEL offsets, never a per-candidate selected checkpoint
    assert set(fit["offset_hat_loto"]) == set(range(1, 10))


def test_report_forbids_selector_detector_language():
    bad = "This is a deployable selector that rescues OACI."
    try:
        report._guard_forbidden(bad)
        raise AssertionError("forbidden-claim guard failed to fire")
    except ValueError:
        pass
    # a legitimate diagnostic sentence passes
    report._guard_forbidden("Diagnostic-only: the per-target offset is not identifiable from source summaries.")
