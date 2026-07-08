"""C27 Confidence-Occupancy Logit Geometry Counterfactual. Frozen C19 config locked; per-candidate logit
features (occupancy sums to 1, class-conditioned confidence, class bias) computed from logits only; logit
transforms behave (temperature preserves occupancy, logit-norm normalizes); counterfactual + sufficiency
scaffolds run and flag destroyers; taxonomy is deterministic (L1-L7); the report forbids deployable/selector
language. Synthetic logits only (no real data)."""
from __future__ import annotations

import numpy as np

from oaci.logit_geometry import (artifact_loader, class_conditioned_confidence, factor_registry, label_alignment,
                                 logit_counterfactuals, report, schema, sufficiency_necessity, taxonomy)


def _synth(n_targets=9, n_per=8, n_samp=48, seed=0):
    rng = np.random.RandomState(seed); score_rows = []; cands = []
    for t in range(1, n_targets + 1):
        off = (t - n_targets / 2) / (n_targets / 2)
        for k in range(n_per):
            good = k % 2 == 0
            sc = off + (0.5 if good else -0.5) + rng.randn() * 0.15
            mh = f"{t:02d}{k:03d}"
            score_rows.append({"mode": "in_regime", "regime": "S0", "seed": 0, "target": t, "level": 0,
                               "model_hash": mh, "score": sc, "label": 1 if good else 0})
            L = rng.randn(n_samp, 4) * 0.5
            L[:, 0] += off * 2.0                              # class-0 logit encodes the offset (occupancy/conf track it)
            cands.append({"seed": 0, "target": t, "level": 0, "model_hash": mh, "L": L, "splits": {}})
    for c in cands:
        c["feats"] = factor_registry.candidate_features(c["L"])
    raw, oracle, _, _ = artifact_loader.raw_oracle(score_rows, "in_regime")
    return cands, score_rows, raw, oracle


def test_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_candidate_features_cover_all_families_and_occupancy_sums_to_one():
    rng = np.random.RandomState(1); L = rng.randn(50, 4)
    f = factor_registry.candidate_features(L)
    for name in factor_registry.all_feature_names():
        assert name in f
    assert abs(sum(f[f"occ_c{k}"] for k in range(4)) - 1.0) < 1e-9


def test_logit_transforms_behave():
    rng = np.random.RandomState(2); L = rng.randn(60, 4)
    # temperature preserves argmax (occupancy)
    assert np.array_equal(logit_counterfactuals._temperature(L).argmax(1), L.argmax(1))
    # logit-norm normalization -> unit rows
    Ln = logit_counterfactuals._logit_norm_normalize(L)
    assert np.allclose(np.linalg.norm(Ln, axis=1), 1.0, atol=1e-6)
    # class-bias centering -> per-class sample-mean ~ 0
    Lc = logit_counterfactuals._class_bias_center(L)
    assert np.allclose(Lc.mean(0), 0.0, atol=1e-6)


def test_full_r3_recovers_on_synth():
    cands, score_rows, raw, oracle = _synth()
    r = artifact_loader.recover(cands, score_rows, "in_regime", raw, oracle, factor_registry.select("occupancy", "global_confidence"))
    assert r["gap_closed"] is not None and r["gap_closed"] > 0.2   # class-0 logit encodes offset -> full-R3 recovers


def test_counterfactuals_run_and_flag_structure():
    cands, score_rows, raw, oracle = _synth()
    cf = logit_counterfactuals.counterfactuals(cands, score_rows, "in_regime", raw, oracle)
    assert set(cf["per_intervention"]) == set(schema.INTERVENTIONS)
    assert cf["per_intervention"]["raw"].get("destroys_recovery") is False   # raw never "destroys"
    assert isinstance(cf["destroyers"], list)


def test_sufficiency_necessity_reports_gap_and_identity():
    cands, score_rows, raw, oracle = _synth()
    suff = sufficiency_necessity.sufficiency_necessity(cands, score_rows, "in_regime", raw, oracle)
    combos = {s["combo"] for s in suff}
    assert "occupancy+global_confidence" in combos
    for s in suff:
        assert "gap_closed" in s and "target_id_accuracy" in s


def test_label_alignment_uses_quarantined_labels():
    cands, score_rows, raw, oracle = _synth()
    rng = np.random.RandomState(3)
    labels = {(0, t): rng.randint(0, 4, cands[0]["L"].shape[0]) for t in range(1, 10)}
    la = label_alignment.label_alignment(cands, labels, destroyers=["temperature"])
    assert "raw" in la["predmix_recall_corr_by_intervention"]
    assert "offset_and_alignment_coupled" in la


def test_taxonomy_deterministic_cases():
    # L1: class-conditioned confidence explains
    cc = {"class_conditioned_confidence_explains": True}
    cf = {"destroyers": []}
    full = {"survives_permutation": True}
    t = taxonomy.gauge_taxonomy(cc, cf, full, full_id_acc=0.7, label={"offset_and_alignment_coupled": False})
    assert t["primary_case"] == schema.L1 and schema.L5 in t["established"]   # id-entangled disclosed
    # L4: shuffle destroys -> sample-level coupling
    t2 = taxonomy.gauge_taxonomy({"class_conditioned_confidence_explains": False},
                                {"destroyers": ["confidence_shuffle"]}, {"survives_permutation": True},
                                full_id_acc=0.7, label={"offset_and_alignment_coupled": True})
    assert t2["primary_case"] == schema.L4 and schema.L6 in t2["established"]
    # L5: full recovery not permutation-robust
    t3 = taxonomy.gauge_taxonomy({"class_conditioned_confidence_explains": False}, {"destroyers": []},
                                {"survives_permutation": False}, full_id_acc=0.7, label={"offset_and_alignment_coupled": False})
    assert t3["primary_case"] == schema.L5


def test_report_forbids_deployable_and_selector_language():
    for bad in ("this is a deployable calibration", "a logit-geometry selector"):
        try:
            report._guard_forbidden(bad); raise AssertionError("guard failed to fire")
        except ValueError:
            pass
    report._guard_forbidden("NOT a deployable calibration; NOT a selector; identity-entangled and diagnostic-only.")
