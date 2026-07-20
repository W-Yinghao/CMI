"""C28 Source-Target Logit-Factor Homology. Frozen C19 locked; source/target factor definitions IDENTICAL
(gate #5); homology metrics align when the source factor encodes the offset and misalign when independent;
source-factor offset prediction + residual decomposition run; taxonomy is deterministic (H6 when source predicts
the offset, H7 when it does not); the report forbids selector/rescue language. Synthetic logits only."""
from __future__ import annotations

import numpy as np

from oaci.source_target_homology import (artifact_loader, error_geometry, factor_registry, homology_metrics,
                                        offset_prediction, report, residual_decomposition, schema, taxonomy)


def _synth(source_aligned=False, n_targets=9, n_per=8, n_samp=40, seed=0):
    rng = np.random.RandomState(seed); score_rows = []; cands = []
    for t in range(1, n_targets + 1):
        off = (t - n_targets / 2) / (n_targets / 2)
        for k in range(n_per):
            good = k % 2 == 0; sc = off + (0.5 if good else -0.5) + rng.randn() * 0.15; mh = f"{t:02d}{k:03d}"
            score_rows.append({"mode": "in_regime", "regime": "S0", "seed": 0, "target": t, "level": 0,
                               "model_hash": mh, "score": sc, "label": 1 if good else 0})
            Lt = rng.randn(n_samp, 4) * 0.5; Lt[:, 0] += off * 2.0
            Ls = rng.randn(n_samp, 4) * 0.5
            if source_aligned:
                Ls[:, 0] += off * 2.0
            cand = {"seed": 0, "target": t, "level": 0, "model_hash": mh, "is_erm": False, "feasible": True,
                    "tgt_feats": factor_registry.candidate_features(Lt)}
            for role in schema.SOURCE_ROLES:
                cand[f"src_{role}_feats"] = factor_registry.candidate_features(Ls)
                pred = Ls.argmax(1); ysrc = rng.randint(0, 4, n_samp)
                cand[f"src_{role}_recall"] = [float((pred[ysrc == c] == c).mean()) if np.any(ysrc == c) else 0.0 for c in range(4)]
                cand[f"src_{role}_domain_std"] = 0.1
            cands.append(cand)
    raw, oracle, _, _ = artifact_loader.raw_oracle(score_rows, "in_regime")
    labels = {(0, t): rng.randint(0, 4, n_samp) for t in range(1, n_targets + 1)}
    return cands, score_rows, raw, oracle, labels


def test_config_and_identical_definition():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    factor_registry.assert_identical_definition()           # gate #5 (must not raise)
    assert schema.CARRIER_NAMES == tuple(f"conf_c{k}" for k in range(4))


def test_homology_aligned_vs_misaligned():
    ca, sr, raw, oracle, _ = _synth(source_aligned=True)
    h = homology_metrics.homology(ca, "source_guard")
    assert h["cosine_mean"] > 0.3                            # source encodes offset -> aligns with target
    cm, sr2, _, _, _ = _synth(source_aligned=False, seed=5)
    hm = homology_metrics.homology(cm, "source_guard")
    assert hm["cosine_mean"] < h["cosine_mean"]             # independent source -> weaker alignment


def test_offset_prediction_reference_and_source():
    ca, sr, raw, oracle, _ = _synth(source_aligned=True)
    off = offset_prediction.offset_prediction(ca, sr, "in_regime", raw, oracle)
    assert off["target_carrier_gap"] is not None            # target carrier reference computed
    assert "source_carrier__source_guard" in off["per_gauge"]
    # when source encodes the offset, the source gauge should recover something
    assert off["per_gauge"]["source_carrier__source_guard"]["gap_closed"] is not None


def test_source_unobservable_when_misaligned():
    ca, sr, raw, oracle, _ = _synth(source_aligned=False, seed=7)
    off = offset_prediction.offset_prediction(ca, sr, "in_regime", raw, oracle)
    # independent source factor should NOT robustly recover the target offset
    assert off["source_predicts_offset"] is False


def test_residual_decomposition_runs():
    ca, sr, raw, oracle, _ = _synth(source_aligned=False, seed=3)
    rd = residual_decomposition.residual_decomposition(ca, sr, "in_regime", raw, oracle, "source_guard")
    for k in ("full_target_carrier", "source_explained", "target_residual"):
        assert rd[k]["gap_closed"] is not None
    assert "residual_carries_offset" in rd


def test_error_geometry_runs():
    ca, sr, raw, oracle, labels = _synth()
    e = error_geometry.error_geometry(ca, "source_guard", labels)
    assert "source_factor_vs_source_recall" in e and "tracks_source_error_only" in e


def test_taxonomy_h6_when_source_predicts_else_h7():
    hom = {"aligned": True, "misaligned": False}
    offp = {"source_predicts_offset": True}
    errg = {"tracks_source_error_only": False}
    rd = {"residual_carries_offset": False}
    t = taxonomy.gauge_taxonomy(hom, offp, errg, rd)
    assert t["primary_case"] == schema.H6 and schema.H3 in t["established"]
    t2 = taxonomy.gauge_taxonomy({"aligned": False, "misaligned": True}, {"source_predicts_offset": False},
                                {"source_factor_vs_source_recall": 0.66},
                                {"residual_carries_offset": False, "residual_over_source_explained": True})
    assert t2["primary_case"] == schema.H7
    assert schema.H2 in t2["established"] and schema.H4 in t2["established"] and schema.H5 in t2["established"]


def test_report_forbids_selector_language():
    for bad in ("the source factor is a selector now", "source-only detector achieved"):
        try:
            report._guard_forbidden(bad); raise AssertionError("guard failed to fire")
        except ValueError:
            pass
    report._guard_forbidden("the source factor is NOT a selector; diagnostic-only.")
