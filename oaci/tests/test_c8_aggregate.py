"""C8 multi-seed aggregation logic (K1 counts + descriptive stats + multiplicity-corrected sweep summary +
real multi-seed K2) and the NARROW approved-provenance relaxation. Synthetic fold records. Standalone."""
from __future__ import annotations

from oaci.confirmatory.c8_aggregate import (_K1_NPERM, _K1_SPLIT, _K1_STAT, aggregate_c8, render_c8_report_md)

_DET = "leakage_reduction_detected"
_STOP = "stop_no_detectable_heldout_leakage_reduction"


def _k1(status, *, delta=0.01, p_lower=0.30, probe="PROBE"):
    return {"k1_status": status, "observed_delta": delta, "p_lower": p_lower, "p_two_sided": 2 * p_lower,
            "alpha": 0.05, "n_permutations": _K1_NPERM, "statistic": _K1_STAT, "split_role": _K1_SPLIT,
            "permutation_plan_hash": "ph", "audit_support_hash": "as", "audit_population_hash": "ap",
            "probe_config_hash": probe, "null_content_hash": "nh", "observed_delta_content_hash": "oh"}


def _k2cfg():
    return {"min_seeds": 3, "level_policy": "both_levels",
            "endpoints": ("worst_domain_bacc", "worst_domain_nll")}


def _fold(seed, target, *, deep=True, tfit=True, fam="FAM", prov="PROV", k1=_STOP, delta=0.01, p_lower=0.30,
          probe="PROBE", art_schema="oaci-artifact-v1", dec_schema="oaci-artifact-v1",
          eb=0.40, ob=0.42, en=1.20, on=1.10):
    levels = {L: {"k1": _k1(k1, delta=delta, p_lower=p_lower, probe=probe), "k2cfg": _k2cfg(),
                  "worst": {"ERM": {"bacc": eb, "nll": en}, "OACI": {"bacc": ob, "nll": on}}} for L in (0, 1)}
    return {"seed": seed, "target": target, "deep_verification_ok": deep, "target_fit_empty": tfit,
            "protocol_family": fam, "provenance_hash": prov, "context_hash": f"c{seed}{target}",
            "artifact_schema_version": art_schema, "decision_schema_version": dec_schema,
            "artifact_scientific_hash": "sci", "artifact_pure_science_hash": "pure", "levels": levels}


def _all(**kw):
    return [_fold(s, t, **kw) for s in (0, 1, 2) for t in range(1, 10)]


def _split_prov(folds, k, p2):
    """Give the first k folds a different provenance_hash p2."""
    out = [dict(f) for f in folds]
    for i in range(k):
        out[i] = dict(out[i], provenance_hash=p2)
    return out


def test_c8_aggregate_requires_27_folds_one_family():
    r = aggregate_c8(_all(), seeds=[0, 1, 2])
    assert r["n_folds"] == 27 and r["seeds"] == [0, 1, 2] and r["targets"] == list(range(1, 10))
    assert r["protocol_family"] == "FAM"
    for bad in (_all()[:-1],
                [dict(_fold(0, 1), deep_verification_ok=False)] + _all()[1:],
                [dict(_fold(0, 1), target_fit_empty=False)] + _all()[1:]):
        try:
            aggregate_c8(bad, seeds=[0, 1, 2])
        except ValueError:
            continue
        raise AssertionError("bad fold set must be rejected")


def test_c8_one_provenance_group_accepted():
    r = aggregate_c8(_all(prov="P0"), seeds=[0, 1, 2])
    assert r["provenance_transition"]["accepted"] and r["provenance_transition"]["n_groups"] == 1


def test_c8_two_provenance_groups_accepted_under_invariant_match():
    # execution-only split: 11 folds at P1, 16 at P0, everything else identical -> ACCEPTED + recorded
    r = aggregate_c8(_split_prov(_all(prov="P0"), 11, "P1"), seeds=[0, 1, 2],
                     transition_commits=["7931091", "a1a09b8"])
    tr = r["provenance_transition"]
    assert tr["accepted"] and tr["n_groups"] == 2 and sorted(tr["provenance_hashes"]) == ["P0", "P1"]
    assert tr["affected_folds"] == {"P0": 16, "P1": 11} and tr["commits"] == ["7931091", "a1a09b8"]


def test_c8_three_provenance_groups_rejected():
    folds = _split_prov(_all(prov="P0"), 11, "P1")
    folds[20] = dict(folds[20], provenance_hash="P2")               # a third group -> not execution-only
    try:
        aggregate_c8(folds, seeds=[0, 1, 2])
    except ValueError:
        return
    raise AssertionError(">2 provenance groups must be rejected")


def test_c8_two_groups_rejected_when_science_config_differs():
    # 2 provenance groups BUT a differing probe_config_hash (a real science change) -> rejected
    folds = _split_prov(_all(prov="P0"), 11, "P1")
    for i in range(11):
        for L in (0, 1):
            folds[i]["levels"][L]["k1"] = dict(folds[i]["levels"][L]["k1"], probe_config_hash="OTHER")
    try:
        aggregate_c8(folds, seeds=[0, 1, 2])
    except ValueError:
        return
    raise AssertionError("differing probe_config_hash across a 2-group split must be rejected")


def test_c8_two_groups_rejected_when_schema_differs():
    folds = _split_prov(_all(prov="P0"), 11, "P1")
    for i in range(11):
        folds[i] = dict(folds[i], artifact_schema_version="v2")
    try:
        aggregate_c8(folds, seeds=[0, 1, 2])
    except ValueError:
        return
    raise AssertionError("differing artifact schema across a 2-group split must be rejected")


def test_c8_approved_allowlist_enforced():
    folds = _split_prov(_all(prov="P0"), 11, "P1")
    aggregate_c8(folds, seeds=[0, 1, 2], approved_provenance=["P0", "P1"])   # ok
    try:
        aggregate_c8(folds, seeds=[0, 1, 2], approved_provenance=["P0", "PX"])
    except ValueError:
        return
    raise AssertionError("observed provenance must equal the explicit allowlist")


def test_c8_k1_counts_and_stats_per_level():
    folds = _all(k1=_STOP)
    folds[0]["levels"][0]["k1"]["k1_status"] = _DET
    r = aggregate_c8(folds, seeds=[0, 1, 2])
    assert r["k1_counts"]["0"]["leakage_reduction_detected"] == 1 and r["k1_counts"]["0"]["n"] == 27
    assert r["k1_counts"]["1"]["stop_no_detectable_heldout_leakage_reduction"] == 27
    assert len(r["k1_per_fold"]) == 54
    assert r["k1_stats"]["0"]["mean"] is not None and r["k1_overall"]["n_tests"] == 54


def test_c8_aggregate_is_canonical_json_serializable():
    """Regression: the whole aggregate must serialize via canonical_json (no int mapping keys). This is the
    main() write path a logic/render test does not exercise."""
    from oaci.artifacts.canonical_json import canonical_json_bytes
    r = aggregate_c8(_all(), seeds=[0, 1, 2], transition_commits=["7931091", "a1a09b8"])
    b = canonical_json_bytes(r)                                    # raises TypeError on any int/non-str key
    assert b and isinstance(b, (bytes, bytearray)) and b'"k1_counts"' in b


def test_c8_k1_sweep_stop_when_no_bh_survivor():
    # a few scattered small p_lower but none survives BH over 54 -> sweep STOP
    folds = _all(k1=_STOP, p_lower=0.30)
    for i, (s, t, L, p) in enumerate([(0, 1, 0, 0.005), (0, 2, 1, 0.03), (1, 1, 0, 0.031)]):
        by = {(f["seed"], f["target"]): f for f in folds}
        by[(s, t)]["levels"][L]["k1"] = dict(by[(s, t)]["levels"][L]["k1"], k1_status=_DET, p_lower=p)
    r = aggregate_c8(folds, seeds=[0, 1, 2])
    assert r["k1_overall"]["k1_sweep_status"] == _STOP
    assert r["k1_overall"]["multiplicity"]["n_bh_survive"] == 0


def test_c8_k1_sweep_detected_when_strong_reproducible():
    folds = _all(k1=_STOP, p_lower=0.30)
    by = {(f["seed"], f["target"]): f for f in folds}
    for s in (0, 1, 2):
        for L in (0, 1):
            by[(s, 1)]["levels"][L]["k1"] = dict(by[(s, 1)]["levels"][L]["k1"], k1_status=_DET, p_lower=1e-5)
    r = aggregate_c8(folds, seeds=[0, 1, 2])
    assert r["k1_overall"]["k1_sweep_status"] == _DET
    assert r["k1_overall"]["multiplicity"]["n_bh_survive"] >= 1


def test_c8_k2_reproducible_gain_across_seeds():
    r = aggregate_c8(_all(eb=0.40, ob=0.42, en=1.20, on=1.10), seeds=[0, 1, 2])
    assert r["k2"]["k2_status"] == "reproducible_gain" and r["k2"]["n_seeds"] == 3
    assert len(r["k2_units"]) == 6 and r["k2_agg"]["worst_domain_bacc"]["n_improved"] == 6


def test_c8_k2_stops_when_no_reproducible_gain():
    r = aggregate_c8(_all(eb=0.45, ob=0.40, en=1.10, on=1.25), seeds=[0, 1, 2])
    assert r["k2"]["k2_status"] == "stop_no_reproducible_gain"
    assert r["k2_agg"]["worst_domain_bacc"]["n_harmed"] == 6


def test_c8_worst_target_uses_min_bacc_max_nll():
    folds = _all(ob=0.50, on=1.00)
    folds[0]["levels"][0]["worst"]["OACI"] = {"bacc": 0.20, "nll": 2.00}
    r = aggregate_c8(folds, seeds=[0, 1, 2])
    u = next(u for u in r["k2_units"] if u["seed"] == 0 and u["level"] == 0)
    assert u["deltas"]["worst_domain_bacc"] < 0 and u["deltas"]["worst_domain_nll"] > 0


def test_c8_report_md_has_all_sections():
    md = render_c8_report_md(aggregate_c8(_all(), seeds=[0, 1, 2], transition_commits=["7931091", "a1a09b8"]))
    for s in ("## run", "## verification", "## K1", "## K2", "## provenance transition",
              "## decision hierarchy", "minimum-seed", "multiplicity control", "K1 sweep status",
              "required_min_seeds"):
        assert s in md, f"missing section: {s}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} c8-aggregate tests")


if __name__ == "__main__":
    _run_all()
