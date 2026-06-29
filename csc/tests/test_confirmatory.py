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


# 4 ---- unconditional guard: source-invalid exclusion CANNOT inflate the headline (reviewer fix) ---
def test_unconditional_power_guard():
    tag = load_tag()
    from csc.run_envelope import KINDS

    def rec(status, fired):
        states = {k: "UNIDENTIFIABLE" for k in KINDS}
        states["boundary_coupled"] = "CONCEPT_SUSPECT" if fired else "UNIDENTIFIABLE"
        return dict(states=states, vis_fail_reason="FIRED" if fired else "residual_T_not_sig",
                    source_status=status, concept_evidenced=True, attribution_unreliable=False)

    # the reviewer's example: 37 fired / 60 valid / 66 generated (6 source-invalid; cap 6/66=0.091 ok).
    # conditional 37/60 CLEARS the 0.50 bar (min_fired over 60 = 37), but unconditional 37/66 does NOT
    # (min_fired over 66 = 41). With the max() rule the headline must FAIL.
    recs = [rec("VALID", True) for _ in range(37)] + [rec("VALID", False) for _ in range(23)] \
        + [rec("INVALID_SUPPORT", False) for _ in range(6)]
    e = evaluate_point(recs, tag)
    assert e["n_valid"] == 60 and e["source_invalid"] == 6 and e["G"] == 66 and e["fired"] == 37
    assert e["min_fired_conditional"] == 37 and e["min_fired_unconditional"] == 41
    assert e["min_fired_for_pass"] == 41, "headline threshold must be the max (unconditional) = 41"
    assert e["power_conditional_cp_lower"] >= 0.50, "conditional alone would have passed"
    assert e["power_unconditional_cp_lower"] < 0.50, "unconditional is below the bar"
    assert e["power_pass"] is False and e["verdict"] == "FAIL", "exclusion must NOT lift the headline"
    # both powers reported; conditional > unconditional (exclusion visible, never hidden)
    assert abs(e["power_conditional"] - 37 / 60) < 1e-9 and abs(e["power_unconditional"] - 37 / 66) < 1e-9
    assert e["power_conditional"] > e["power_unconditional"]
    print("OK unconditional guard: 37/60 valid passes conditional but FAILS (uncond needs 41/66); not hidden")


# 4a2 ---- frozen-code guard fails closed unless HEAD == frozen tag commit AND tree is clean --------
def test_frozen_code_ref_guard():
    from csc.run_confirmatory import _code_ref_ok
    H = "a" * 40
    # exact match + clean -> ok
    assert _code_ref_ok(H, H, dirty=False) is True
    # mismatch, missing expected, or dirty -> fail closed
    for head, exp, dirty in [(H, "b" * 40, False),   # HEAD != frozen tag
                             (H, "", False),          # tag did not resolve
                             (H, H, True)]:            # dirty tree
        try:
            _code_ref_ok(head, exp, dirty)
            raise AssertionError(f"guard should fail closed for {(head[:4], exp[:4], dirty)}")
        except SystemExit:
            pass
    # the tag declares the expected_code_ref
    assert load_tag()["expected_code_ref"] == "refs/tags/csc-confirmatory-v1"
    print("OK frozen-code guard: HEAD==tag & clean -> ok; mismatch/unresolved/dirty -> fail closed")


# 4a3 ---- stale/foreign artifact is refused (freshness guard) ------------------------------------
def test_stale_artifact_refused():
    import json, tempfile, copy
    from csc.run_confirmatory import verify_fresh_payload, TGT_SEED_BASE
    tag = load_tag()
    JOB, COMMIT = "999999", "c" * 40

    def fresh():
        return dict(slurm_job_id=JOB, manifest_hash=tag["expected_manifest_hash"],
                    base_seed=tag["base_seed"],
                    code_provenance=dict(git_head=COMMIT, expected_code_commit=COMMIT, git_status_clean=True),
                    seed_derivation=dict(
                        source_seed_range=[tag["base_seed"], tag["base_seed"] + tag["G"] - 1],
                        target_seed_range=[TGT_SEED_BASE + tag["base_seed"],
                                           TGT_SEED_BASE + tag["base_seed"] + tag["G"] - 1]))

    def write(p):
        path = tempfile.mktemp(suffix=".json"); json.dump(p, open(path, "w")); return path

    # a genuinely fresh payload of THIS job at the frozen commit passes
    verify_fresh_payload(write(fresh()), JOB, COMMIT, tag)
    # every staleness/foreignness signal is refused (infrastructure SystemExit)
    mutators = {
        "old job (stale)": lambda p: p.update(slurm_job_id="111111"),
        "foreign commit": lambda p: p["code_provenance"].update(git_head="d" * 40),
        "dirty tree": lambda p: p["code_provenance"].update(git_status_clean=False),
        "wrong manifest": lambda p: p.update(manifest_hash="deadbeef"),
        "wrong base_seed": lambda p: p.update(base_seed=600000),
        "wrong target seeds": lambda p: p["seed_derivation"].update(target_seed_range=[0, 65]),
    }
    for label, mut in mutators.items():
        p = fresh(); mut(p)
        try:
            verify_fresh_payload(write(p), JOB, COMMIT, tag)
            raise AssertionError(f"freshness guard should reject: {label}")
        except SystemExit:
            pass
    print("OK stale/foreign artifact refused (old job, foreign commit, dirty, manifest, seed mismatches)")


# 4b ---- seed derivation is recorded explicitly (source vs target streams, disjoint) --------------
def test_seed_streams_recorded():
    from csc.run_confirmatory import seed_streams
    tag = load_tag()
    ss = seed_streams(tag)
    assert ss["source_seed_range"] == [900_000, 900_065]
    assert ss["target_seed_range"] == [1_800_000, 1_800_065]
    assert ss["target_seed_base"] == 900_000
    assert ss["source_target_seed_streams_disjoint"] is True
    print("OK seed derivation recorded: sources 900000..900065 -> targets 1800000..1800065 (disjoint)")


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
    test_frozen_code_ref_guard()
    test_stale_artifact_refused()
    test_unconditional_power_guard()
    test_seed_streams_recorded()
    test_dev_smoke_not_unseen()
    print("\nall CSC confirmatory-runner tests passed")
