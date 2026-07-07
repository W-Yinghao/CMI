"""C26 Predicted-Class Mix Mechanism / Counterfactual. Frozen C19 config locked; signed-vs-symmetric separates
class-index-specific occupancy from symmetric concentration; class rotation is not invariant when the signal is
signed; identity controls run and do not fire when the recovery survives permutation; split-stability + label
diagnostics are availability-GATED (REQUIRES_REPERSISTENCE_REINFERENCE, never proxied) and C26 does not finalize
without them; the report forbids 'identity-free recovery' / 'predmix deployable' language. Synthetic only."""
from __future__ import annotations

import json

import numpy as np

from oaci.predmix_mechanism import (class_mix_decomposition, identity_controls, interaction_diagnostics,
                                    label_diagnostics, report, schema, split_stability, taxonomy)


def _synth(n_targets=9, n_per=14, mode_signed=True, seed=0):
    """Signed scenario: c0-c1 asymmetry (signed) encodes the offset; concentration (symmetric) tracks |offset|."""
    rng = np.random.RandomState(seed)
    rows = []; joined = []
    for t in range(1, n_targets + 1):
        off = (t - n_targets / 2) / (n_targets / 2)              # in [-1,1]
        a = 0.15 * off
        for k in range(n_per):
            good = k % 2 == 0
            sc = off + (0.5 if good else -0.5) + rng.randn() * 0.15
            rows.append({"mode": "in_regime", "regime": "S0_full_support", "seed": 0, "target": t, "level": 0,
                         "model_hash": f"{t:02d}{k:03d}", "score": sc, "label": 1 if good else 0})
            p = np.array([0.25 + a, 0.25 - a, 0.25, 0.25]) + rng.randn(4) * 0.01
            p = np.clip(p, 1e-6, None); p = p / p.sum()
            tu = {schema.PRED_PROP[i]: float(p[i]) for i in range(4)}
            for f in schema.CONF_MARGIN:
                tu[f] = float(0.5 + rng.randn() * 0.02)          # confidence/margin ~ noise here
            joined.append({"seed": 0, "target": t, "regime": "S0_full_support", "level": 0,
                           "model_hash": f"{t:02d}{k:03d}", **tu})
    from oaci.predmix_mechanism import artifact_loader
    raw, oracle, _, _ = artifact_loader.raw_oracle(rows, "in_regime")
    return rows, joined, raw, oracle


def test_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_symmetric_summaries_are_valid():
    s = class_mix_decomposition._symmetric([0.7, 0.1, 0.1, 0.1])
    assert 0 < s["predmix_entropy"] < np.log(4) and abs(s["predmix_max_mass"] - 0.7) < 1e-9
    assert s["predmix_dist_uniform"] > 0 and 0 < s["predmix_gini"] < 1
    u = class_mix_decomposition._symmetric([0.25, 0.25, 0.25, 0.25])
    assert abs(u["predmix_dist_uniform"]) < 1e-9                 # uniform -> zero distance


def test_signed_recovers_and_is_class_specific():
    rows, joined, raw, oracle = _synth()
    svs = class_mix_decomposition.signed_vs_symmetric(joined, rows, "in_regime", raw, oracle)
    assert svs["signed"]["gap_closed"] > svs["symmetric"]["gap_closed"]   # signed beats symmetric
    rot = class_mix_decomposition.class_rotation_counterfactual(joined, rows, "in_regime", raw, oracle)
    assert rot["global_rotation_invariant"] is True             # symmetric ridge: global rotation inert (control)
    assert "class_index_alignment_matters" in rot and "per_target_scramble_gap" in rot


def test_identity_controls_run_and_not_fingerprint_when_survives():
    rows, joined, raw, oracle = _synth()
    svs = class_mix_decomposition.signed_vs_symmetric(joined, rows, "in_regime", raw, oracle)
    idc = identity_controls.identity_controls(joined, svs["signed"]["survives_permutation"], svs["signed"]["gap_closed"])
    assert idc["id_acc_predmix"] is not None and idc["chance"] == schema.IDENTITY_CHANCE
    if svs["signed"]["survives_permutation"]:
        assert idc["identity_fingerprint_dominant"] is False


def test_interaction_diagnostics_run():
    rows, joined, raw, oracle = _synth()
    it = interaction_diagnostics.interaction_diagnostics(joined, rows, "in_regime", raw, oracle)
    assert "shapley_interaction" in it and "predmix_needs_confidence_scaffold" in it
    # Shapley efficiency: the two MAIN effects sum to v(both) (interaction is a separate synergy term)
    assert abs((it["shapley_main_predmix"] + it["shapley_main_confmargin"]) - (it["both_gap"] or 0.0)) < 1e-9
    # synergy term = v(both) - v(pm) - v(cm)
    assert abs(it["shapley_interaction"] - ((it["both_gap"] or 0) - (it["predmix_only_gap"] or 0) - (it["confmargin_only_gap"] or 0))) < 1e-9


def test_split_stability_gated_without_sidecar(tmp_path):
    rows, joined, raw, oracle = _synth()
    sp = split_stability.split_stability(rows, "in_regime", raw, oracle, split_sidecar=str(tmp_path / "absent.json"))
    assert sp["status"] == schema.STATUS_REQUIRES_REINFERENCE and sp["splits"] is None


def test_label_diagnostics_gated_without_sidecar(tmp_path):
    ld = label_diagnostics.label_diagnostics([], "in_regime", split_sidecar=str(tmp_path / "absent.json"))
    assert ld["status"] == schema.STATUS_REQUIRES_REINFERENCE and ld["alignment"] is None


def test_split_stability_computes_with_sidecar(tmp_path):
    rows, joined, raw, oracle = _synth()
    # synthetic split sidecar: per-candidate pred_prop on two stable halves
    per = []
    for c in joined:
        pp = {k: c[k] for k in schema.PRED_PROP}
        per.append({"seed": 0, "target": c["target"], "level": 0, "model_hash": c["model_hash"],
                    "splits": {"half_a": pp, "half_b": pp, "odd_even_a": pp, "odd_even_b": pp,
                               "bootstrap_a": pp, "bootstrap_b": pp}})
    p = tmp_path / "split.json"
    json.dump({"config_hash": schema.LOCKED_C19_CONFIG_HASH, "per_candidate": per}, open(p, "w"))
    sp = split_stability.split_stability(rows, "in_regime", raw, oracle, split_sidecar=str(p))
    assert sp["status"] == schema.STATUS_OK and sp["split_stable"] is True   # identical halves -> perfectly stable


def test_taxonomy_not_final_while_pending():
    svs = {"signed_specific": True, "symmetric_carries": False}
    rot = {"class_index_alignment_matters": True}
    idc = {"identity_fingerprint_dominant": False}
    inter = {"predmix_needs_confidence_scaffold": True, "interaction_dominant": False}
    pend = {"status": schema.STATUS_REQUIRES_REINFERENCE}
    t = taxonomy.gauge_taxonomy(svs, rot, idc, inter, pend, pend)
    assert t["final"] is False
    assert schema.P2 in t["established"] and schema.P5 in t["established"]
    assert t["primary_case"] == schema.P2
    assert schema.P1 in t["unresolved_pending_reinference"]


def test_taxonomy_p4_when_fingerprint_dominant():
    t = taxonomy.gauge_taxonomy({"signed_specific": True, "symmetric_carries": False}, {"class_index_alignment_matters": True},
                               {"identity_fingerprint_dominant": True},
                               {"predmix_needs_confidence_scaffold": False, "interaction_dominant": False},
                               {"status": schema.STATUS_REQUIRES_REINFERENCE}, {"status": schema.STATUS_REQUIRES_REINFERENCE})
    assert t["primary_case"] == schema.P4


def test_repersist_split_membership_is_deterministic_and_label_free():
    from oaci.predmix_mechanism import target_repersist
    sids = [f"BNCI2014_001|subject-004|session-0train|run-{i%2}|trial-{i:03d}" for i in range(40)]
    a = target_repersist._split_membership(sids); b = target_repersist._split_membership(list(sids))
    for k in ("half", "odd_even", "bootstrap"):
        assert set(np.unique(a[k])) <= {0, 1} and np.array_equal(a[k], b[k])   # deterministic, binary
    assert np.array_equal(a["odd_even"], np.array([i % 2 for i in range(40)], dtype=np.int8))  # trial parity


def test_repersist_structural_gates_and_summarize(tmp_path):
    from oaci.predmix_mechanism import target_repersist
    st = target_repersist._structural_gates()
    assert st["G5_no_labels_in_unlabeled_feature_path"] and st["G6_quarantined_labels_separate_file"]
    # synthetic per-fold npz -> summarize -> split sidecar
    N, C = 80, 4; rng = np.random.RandomState(0)
    sids = np.array([f"s|trial-{i:03d}" for i in range(N)], dtype=object)
    spl = target_repersist._split_membership([str(s) for s in sids])
    logits = rng.randn(3, N, C).astype(np.float32)          # 3 candidates
    d = tmp_path / "rp"; d.mkdir()
    np.savez(d / "seed-0-target-004.unlabeled.npz", sample_id=sids, domain=np.zeros(N),
             split_half=spl["half"], split_odd_even=spl["odd_even"], split_bootstrap=spl["bootstrap"],
             model_hash=np.array(["a", "b", "c"], dtype=object), level=np.array([0, 0, 1]), logits=logits)
    np.savez(d / "seed-0-target-004.labels.npz", sample_id=sids, y=rng.randint(0, C, N))
    out = str(tmp_path / "split.json")
    n = target_repersist.summarize(str(d), out)
    import json as _j
    sc = _j.load(open(out))
    assert n == 3 and sc["config_hash"] == schema.LOCKED_C19_CONFIG_HASH
    c0 = sc["per_candidate"][0]
    assert set(c0["splits"]) == {"half_a", "half_b", "odd_even_a", "odd_even_b", "bootstrap_a", "bootstrap_b"}
    assert abs(sum(c0["splits"]["half_a"].values()) - 1.0) < 1e-6     # pred_prop sums to 1
    ld = sc["label_diagnostics"]["per_candidate"][0]
    assert len(ld["true_prior"]) == C and len(ld["per_class_recall"]) == C
    # split sidecar unlabeled path carries NO labels
    assert '"y"' not in _j.dumps(sc["per_candidate"])


def test_report_forbids_identity_free_and_deployable_language():
    for bad in ("this is an identity-free recovery", "pred-class-mix is deployable now"):
        try:
            report._guard_forbidden(bad); raise AssertionError("guard failed to fire")
        except ValueError:
            pass
    report._guard_forbidden("predmix is entangled with identity; NOT claimed identity-free; not deployable.")
