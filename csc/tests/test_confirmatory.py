"""
CSC confirmatory-runner tests (frozen freeze-candidate). These check the harness mechanics ONLY and
run a TINY DEV smoke with a NON-unseen base_seed and small G — they never touch the unseen base_seed
(900000) and never call the --execute confirmatory run. The runner is frozen scaffolding pending a
separate run authorization.
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

from csc.run_confirmatory import (
    load_tag, frozen_cfg, power_min_fired, run_point, evaluate_point, _point_from, TAG_PATH,
)
from csc.protocol import _cp_bound
from csc.run_envelope import EnvelopePoint


# 1 ---- power threshold is computed from the REALIZED N_valid (not hard-coded 37/59) ---------------
def test_power_min_fired_is_realized_N():
    # the reviewer's table: 59->37, 60->37, 66->41, 72->44 (all at bar 0.50)
    assert power_min_fired(59, 0.50) == 37
    assert power_min_fired(60, 0.50) == 37
    assert power_min_fired(66, 0.50) == 41
    assert power_min_fired(72, 0.50) == 44
    # and the returned k is the SMALLEST with CP_lower >= bar, k-1 falls short
    for n in (59, 66, 72):
        k = power_min_fired(n, 0.50)
        assert _cp_bound(k, n, side="lower") >= 0.50 and _cp_bound(k - 1, n, side="lower") < 0.50
    print("OK power_min_fired computed from realized N_valid: 59->37, 60->37, 66->41, 72->44")


# 2 ---- the frozen tag matches the running method + the reviewer-bound choices --------------------
def test_tag_matches_frozen_method_and_bound_choices():
    tag = load_tag()
    cfg = frozen_cfg()
    assert cfg.hash() == tag["expected_manifest_hash"], "frozen manifest must match the tag"
    assert tag["K"] == 1 and tag["claim_type"] == "pointwise"
    assert [p["name"] for p in tag["core_points"]] == ["P_baseline"]
    assert tag["base_seed"] == 900_000 and tag["G"] == 66 and tag["N_valid_min"] == 59
    assert tag["source_invalid_cap"] == 0.10 and tag["max_forbidden_failures"] == 0
    assert tag["power_bar"] == 0.50
    # P_baseline overrides == EnvelopePoint defaults; P_strong is secondary-only with effect 20
    assert _point_from(tag["core_points"][0]) == EnvelopePoint()
    assert _point_from(tag["secondary_descriptive_points"][0]) == EnvelopePoint(concept_effect_size=20.0)
    assert all("secondary" in p["role"] for p in tag["secondary_descriptive_points"])
    print("OK frozen tag: manifest match, K=1 P_baseline pointwise, G=66/N>=59/cap0.10/maxfail0/bar0.50")


# 3 ---- evaluate_point gating: INCONCLUSIVE / FAIL-on-forbidden / PASS need both, conjunction -----
def test_evaluate_point_gating():
    tag = load_tag()

    def rec(status="VALID", fired=True, forbidden_kind=None):
        states = {k: "UNIDENTIFIABLE" for k in __import__("csc.run_envelope", fromlist=["KINDS"]).KINDS}
        states["boundary_coupled"] = "CONCEPT_SUSPECT" if fired else "UNIDENTIFIABLE"
        if forbidden_kind:                              # force a forbidden state on a must-abstain kind
            states[forbidden_kind] = "COVARIATE_COMPATIBLE"
        return dict(states=states, vis_fail_reason="FIRED" if fired else "residual_T_not_sig",
                    source_status=status, concept_evidenced=True, attribution_unreliable=False)

    # too few valid -> INCONCLUSIVE
    few = [rec() for _ in range(40)]
    assert evaluate_point(few, tag)["verdict"] == "INCONCLUSIVE"
    # >10% source-invalid -> INCONCLUSIVE even with enough total
    capbust = [rec() for _ in range(59)] + [rec(status="INVALID_SUPPORT") for _ in range(7)]
    assert evaluate_point(capbust, tag)["verdict"] == "INCONCLUSIVE"
    # 59 valid all fired, 0 forbidden -> PASS (both endpoints)
    good = [rec() for _ in range(59)]
    g = evaluate_point(good, tag)
    assert g["verdict"] == "PASS" and g["forbidden"] == 0 and g["fired"] == 59
    # any forbidden -> FAIL even with full power (max_forbidden_failures=0)
    bad = [rec() for _ in range(58)] + [rec(forbidden_kind="clean")]
    assert evaluate_point(bad, tag)["verdict"] == "FAIL"
    # enough valid + 0 forbidden but power below threshold (only 36/59 fired) -> FAIL (non-vacuity)
    weak = [rec(fired=True) for _ in range(36)] + [rec(fired=False) for _ in range(23)]
    w = evaluate_point(weak, tag)
    assert w["verdict"] == "FAIL" and w["forbidden"] == 0 and w["fired"] == 36 and w["min_fired_for_pass"] == 37
    print("OK gating: INCONCLUSIVE (few/cap), PASS (both), FAIL (any forbidden), FAIL (power<thr, non-vacuity)")


# 4 ---- power both-ways: source-invalid exclusion cannot inflate the headline -----------------------
def test_power_reported_both_ways():
    tag = load_tag()
    from csc.run_envelope import KINDS

    def rec(status, fired):
        states = {k: "UNIDENTIFIABLE" for k in KINDS}
        states["boundary_coupled"] = "CONCEPT_SUSPECT" if fired else "UNIDENTIFIABLE"
        return dict(states=states, vis_fail_reason="FIRED" if fired else "residual_T_not_sig",
                    source_status=status, concept_evidenced=True, attribution_unreliable=False)

    # 60 valid (40 fired) + 6 source-invalid (cannot fire). cap = 6/66 = 0.0909 <= 0.10 -> evaluable
    recs = [rec("VALID", True) for _ in range(40)] + [rec("VALID", False) for _ in range(20)] \
        + [rec("INVALID_SUPPORT", False) for _ in range(6)]
    e = evaluate_point(recs, tag)
    assert e["n_valid"] == 60 and e["source_invalid"] == 6
    # conditional = 40/60 = 0.667 ; unconditional = 40/66 = 0.606 -> conditional is higher
    assert abs(e["power_conditional"] - 40 / 60) < 1e-9
    assert abs(e["power_unconditional"] - 40 / 66) < 1e-9
    assert e["power_conditional"] > e["power_unconditional"], "exclusion must be visible (both reported)"
    print("OK power reported both conditional (40/60) and unconditional (40/66): exclusion not hidden")


# 5 ---- TINY DEV smoke: run_point/evaluate_point execute end-to-end on a NON-unseen seed ----------
def test_dev_smoke_not_unseen():
    from csc.protocol import ProtocolConfig
    tag = load_tag()
    # LIGHT cfg (mechanics smoke only -- the frozen regime is irrelevant to "does the runner execute";
    # frozen_cfg() heaviness is exercised by test 2's manifest-match, not here).
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=40, target_n_boot=30, tau_n_pseudotargets=40, oracle_boot=10)
    cfg.validate()
    # DELIBERATELY not the unseen base_seed (900000) and tiny G -- this is a mechanics smoke, NOT the
    # confirmatory run.
    assert tag["base_seed"] == 900_000
    recs = run_point(EnvelopePoint(), cfg, G=3, base_seed=111, n_jobs=1)
    assert len(recs) == 3 and all("states" in r and "source_status" in r for r in recs)
    out = evaluate_point(recs, tag)
    assert out["verdict"] == "INCONCLUSIVE"             # G=3 < N_valid_min -> must be INCONCLUSIVE
    assert 0.0 <= out["forbidden_cp_upper"] <= 1.0
    print(f"OK DEV smoke (base_seed 111, G=3): runs end-to-end -> INCONCLUSIVE (G<59), never the unseen set")


if __name__ == "__main__":
    test_power_min_fired_is_realized_N()
    test_tag_matches_frozen_method_and_bound_choices()
    test_evaluate_point_gating()
    test_power_reported_both_ways()
    test_dev_smoke_not_unseen()
    print("\nall CSC confirmatory-runner tests passed")
