"""C25 Target-Unlabeled Gauge Mechanism + Grouping Boundary. Frozen C19 config locked; R3 families are FROZEN
and partition the 12 features (no feature selection); Shapley satisfies efficiency (sums to the full gap); the
carrying family is credited; the R4 random-dim control detects the small-N/high-dim-noise collapse; the
identity audit does not fire when the recovery survives the permutation control; grouping is a separate problem
class; the report forbids 'grouping is source-only' / 'oracle as method' language. Synthetic gauges only."""
from __future__ import annotations

import numpy as np

from oaci.score_gauge import gauge_feature_registry as gfr
from oaci.score_gauge.ceiling_ladder import _pooled_auc
from oaci.unlabeled_gauge import (family_registry, grouping_boundary, identity_signature, r3_family_decomposition,
                                  r4_interference, report, schema, taxonomy)


def _synth(n_targets=9, n_per=14, signal_family="confidence_entropy", noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    names = list(schema.ALL_R3_FEATURES)
    rows = []
    tmp = {}
    for t in range(1, n_targets + 1):
        off = (t - n_targets / 2) * 0.6
        for k in range(n_per):
            good = k % 2 == 0
            sc = off + (0.5 if good else -0.5) + rng.randn() * 0.2
            rows.append({"mode": "in_regime", "regime": "S0_full_support", "seed": 0, "target": t, "level": 0,
                         "model_hash": f"{t:02d}{k:03d}", "score": sc, "label": 1 if good else 0})
        tmp[t] = off
    mr = rows
    tmean = {t: float(np.mean([c["score"] for c in mr if c["target"] == t])) for t in tmp}
    gauge = {}
    for t in tmp:
        gv = {}
        for f in names:
            gv[f] = (tmean[t] + rng.randn() * noise) if f in schema.FAMILIES[signal_family] else rng.randn() * noise
        gauge[t] = {"gauge": gv, "offset": tmean[t]}
    raw = _pooled_auc(mr); oracle = _pooled_auc(mr, subtract=lambda r: tmean[r["target"]])
    return rows, gauge, names, raw, oracle, tmean


def test_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_families_partition_the_12_r3_features():
    family_registry.assert_partition(schema.ALL_R3_FEATURES)   # must not raise
    assert len(schema.ALL_R3_FEATURES) == 12 and len(schema.FAMILIES) == 3
    try:
        family_registry.assert_partition(list(schema.ALL_R3_FEATURES)[:-1])
        raise AssertionError("expected partition failure")
    except ValueError:
        pass


def test_shapley_efficiency_and_credits_carrying_family():
    rows, gt, names, raw, oracle, _ = _synth(signal_family="confidence_entropy")
    sh = r3_family_decomposition.shapley(rows, gt, "in_regime", raw, oracle)
    # Shapley efficiency: sum of family values == full-coalition gap
    assert abs(sum(sh["shapley"].values()) - sh["full_gap"]) < 1e-9
    # the family that encodes the offset is the dominant contributor
    assert sh["dominant_family"] == "confidence_entropy"
    assert sh["shapley"]["confidence_entropy"] > sh["shapley"]["margin_logitnorm"]
    assert sh["shapley"]["confidence_entropy"] > sh["shapley"]["pred_class_prop"]


def test_family_only_recovers_for_signal_family():
    rows, gt, names, raw, oracle, _ = _synth(signal_family="pred_class_prop")
    fo = {f["family"]: f for f in r3_family_decomposition.family_only(rows, gt, "in_regime", raw, oracle)}
    assert fo["pred_class_prop"]["gap_closed"] > fo["confidence_entropy"]["gap_closed"]


def test_r4_random_dim_control_detects_small_n_noise_collapse():
    rows, r3_gt, r3_names, raw, oracle, tmean = _synth(signal_family="confidence_entropy")
    # source gauge = pure noise over the 34 source feature names
    rng = np.random.RandomState(1); snames = gfr.gauge_feature_names()
    source_gt = {t: {"gauge": {n: float(rng.randn()) for n in snames}, "offset": tmean[t]} for t in r3_gt}
    r4 = r4_interference.interference_audit(rows, source_gt, snames, r3_gt, r3_names, "in_regime", raw, oracle)
    assert r4["r3_gap"] > r4["r4_gap"]                          # adding source collapses
    assert r4["random_dims_also_collapse"] is True             # random noise reproduces it
    assert r4["mechanism"] == "small_N_high_dim_noise"


def test_identity_signature_not_dominant_when_recovery_survives():
    joined = [{"target": t, **{f: float(t) + 0.01 * i for i, f in enumerate(schema.ALL_R3_FEATURES)}}
              for t in range(1, 10) for _ in range(6)]
    fam_only = [{"family": f, "gap_closed": (0.5 if f == "confidence_entropy" else 0.1), "survives_permutation": True}
                for f in schema.FAMILIES]
    idn = identity_signature.identity_signature_audit(joined, fam_only, full_survives_permutation=True, source_id_accuracy=0.54)
    assert idn["identity_signature_dominates"] is False        # survives permutation -> not identity-dominated
    assert idn["recovery_survives_loto_permutation"] is True


def test_grouping_boundary_is_separate_problem_class():
    gb = grouping_boundary.grouping_boundary(r1_gap=-0.8, r3_gap=0.49, r6_gap=1.0, r5_refine_gap=1.4, within_ceiling=0.66)
    assert gb["grouping_is_separate_problem_class"] is True
    assert abs(gb["grouping_value_over_marginal"] - (1.0 - 0.49)) < 1e-9
    rungs = {r["rung"]: r for r in gb["ladder"]}
    # source-only uses no target inputs/grouping; grouped-zero-label uses grouping + held-out scores, no labels
    assert rungs["source_only_DG"]["target_grouping"] is False
    assert rungs["target_grouped_transductive_zero_label"]["target_grouping"] is True
    assert rungs["target_grouped_transductive_zero_label"]["target_labels"] is False
    assert rungs["target_grouped_transductive_zero_label"]["uses_held_out_target_scores"] is True


def test_taxonomy_deterministic_and_flags_secondaries():
    shap = {"single_family_dominates": True, "dominant_family": "confidence_entropy", "dominant_share": 0.7}
    identity = {"identity_signature_dominates": False}
    r4 = {"r3_gap": 0.49, "r4_gap": -0.48, "mechanism": "small_N_high_dim_noise"}
    grouping = {"grouping_is_separate_problem_class": True}
    t = taxonomy.gauge_taxonomy(shap, identity, r4, grouping)
    assert t["primary_case"] == schema.U1
    assert schema.U6 in t["established"] and schema.U7 in t["established"]


def test_taxonomy_u5_when_distributed():
    shap = {"single_family_dominates": False, "dominant_family": "margin_logitnorm", "dominant_share": 0.4}
    t = taxonomy.gauge_taxonomy(shap, {"identity_signature_dominates": False},
                               {"r3_gap": 0.49, "r4_gap": -0.4, "mechanism": "small_N_high_dim_noise"},
                               {"grouping_is_separate_problem_class": True})
    assert t["primary_case"] == schema.U5


def test_report_forbids_grouping_as_source_only_language():
    try:
        report._guard_forbidden("this shows target grouping is source-only after all")
        raise AssertionError("guard failed to fire")
    except ValueError:
        pass
    report._guard_forbidden("target grouping is NOT source-only; it is a separate problem class.")
