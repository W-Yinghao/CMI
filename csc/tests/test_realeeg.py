"""Fail-closed tests for the CSC real-EEG DRY-RUN freeze package (CSC-realEEG-P1).

Verifies the package is frozen, self-consistent, and fail-closed; that --execute is refused; and that
robustness-only / descriptive artifacts cannot affect PASS/FAIL. Runnable via pytest or directly:
  python csc/tests/test_realeeg.py
"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from csc.mininfo import run_realeeg_validation as R
from csc.mininfo import build_lee2019_b3_cache as B

MONTAGE16 = ["FC3", "FC1", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
             "CP3", "CP1", "CPz", "CP2", "CP4"]
_CACHE_PRESENT = os.path.exists(json.load(open(R.CACHE_MANIFEST))["provenance"]["cache_path"])


def _dry_with_temp(mutate):
    """Copy the 4 manifests, mutate(dict), write to a temp dir, point the runner at them, run dry_run()."""
    keys = [("cache", R.CACHE_MANIFEST), ("bank", R.BANK_MANIFEST),
            ("routeA", R.ROUTEA_MANIFEST), ("b3", R.B3_MANIFEST)]
    mans = {k: json.load(open(p)) for k, p in keys}
    mutate(mans)
    orig = (R.CACHE_MANIFEST, R.BANK_MANIFEST, R.ROUTEA_MANIFEST, R.B3_MANIFEST)
    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for k, man in mans.items():
            p = os.path.join(td, f"{k}.json"); json.dump(man, open(p, "w")); paths[k] = p
        R.CACHE_MANIFEST, R.BANK_MANIFEST, R.ROUTEA_MANIFEST, R.B3_MANIFEST = (
            paths["cache"], paths["bank"], paths["routeA"], paths["b3"])
        try:
            return R.dry_run()
        finally:
            (R.CACHE_MANIFEST, R.BANK_MANIFEST, R.ROUTEA_MANIFEST, R.B3_MANIFEST) = orig


# ---- self-consistency ----
def test_manifests_load():
    for p in (R.CACHE_MANIFEST, R.BANK_MANIFEST, R.ROUTEA_MANIFEST, R.B3_MANIFEST):
        json.load(open(p))


def test_cache_montage_16_exact_everywhere():
    cache = json.load(open(R.CACHE_MANIFEST))
    assert cache["channels"] == MONTAGE16
    assert B.MONTAGE == MONTAGE16 and len(B.MONTAGE) == 16
    assert cache["normalize"] is None


def test_dry_run_passes_on_real_package():
    assert R.dry_run() is True


def test_execute_fails_closed_without_tag():
    # no csc-realeeg-v1 tag exists -> guarded execute must fail closed (exit 2)
    assert R.execute() == 2


def test_smoke_uses_non_real_seed():
    import csc.mininfo.realeeg_engine as E
    assert E.REAL_BASE_SEED == 20_000_000
    # smoke default seed must not be the real base seed
    import inspect
    src = inspect.getsource(E.smoke)
    assert "REAL_BASE_SEED" in src and "seed != REAL_BASE_SEED" in src


# ---- Route A label_unit adaptation for MI (reviewer-required) ----
def _toy_cohort():
    import csc.mininfo.realeeg_engine as E
    import numpy as np
    coh = E._toy_cache(np.random.default_rng(111))
    return E, E.build_cohort("NULL_cov", coh, np.random.default_rng(7))


def test_routeA_MI_label_unit_subject_fails_closed():
    import csc.protocol as P
    E, (Z, Y, D, G) = _toy_cohort()
    r = E.certify_A(Z, Y, D, G, seed=111, cfg=P.ProtocolConfig(label_unit="subject"))
    assert str(r["state"]).startswith("REFUSED_label_unit_must_be_trial")


def test_routeA_MI_label_unit_trial_runs():
    E, (Z, Y, D, G) = _toy_cohort()
    r = E.certify_A(Z, Y, D, G, seed=111, cfg=E.frozen_A_cfg())
    assert not str(r["state"]).startswith(("REFUSED", "ENGINE_ERROR"))  # a real Certificate


def test_routeA_cluster_unit_is_subject_not_trial():
    import csc.mininfo.realeeg_engine as E
    cfg = E.frozen_A_cfg()
    assert cfg.label_unit == "trial" and cfg.analysis_unit == "subject"


def test_routeA_manifest_declares_trial_label_unit():
    rc = json.load(open(R.ROUTEA_MANIFEST))["route_A_config"]
    assert rc["label_unit"] == "trial"
    assert rc["analysis_unit"] == "subject" and rc["cluster_unit"] == "biological_subject"
    assert rc["synthetic_A"]["label_unit"] == "subject"


def test_verdict_abstain_null_is_inconclusive_not_pass():
    import csc.mininfo.realeeg_engine as E
    rec = [dict(condition="NULL_cov", gating=True, B3=dict(state="NEED_MORE_LABELS", confirmed=False))
           for _ in range(20)]
    r = E._b3_rates(rec, "NULL_cov", 2000, 1, 0.20)
    assert r["status"] == "INCONCLUSIVE" and r["n_valid"] == 0   # V1: abstain must NOT pad the type-I denominator


def test_verdict_denominator_excludes_abstain_invalid():
    import csc.mininfo.realeeg_engine as E
    states = ["NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT"] * 10 + ["NEED_MORE_LABELS"] * 10
    rec = [dict(condition="NULL_cov", gating=True, B3=dict(state=s, confirmed=False)) for s in states]
    r = E._b3_rates(rec, "NULL_cov", 2000, 1, 0.20)
    assert r["n_valid"] == 10 and abs(r["invalid_frac"] - 0.5) < 1e-9 and r["status"] == "INCONCLUSIVE"


def test_engine_hash_pinned_and_mismatch_fails_closed():
    ep = json.load(open(R.BANK_MANIFEST))["engine_provenance"]
    assert len(ep["engine_sha256"]) == 64 and ep["engine_file"].endswith("realeeg_engine.py")
    assert _dry_with_temp(lambda m: m["bank"]["engine_provenance"].__setitem__("engine_sha256", "0" * 64)) is False


def test_routeA_returns_bare_state_not_full_repr():
    E, (Z, Y, D, G) = _toy_cohort()
    r = E.certify_A(Z, Y, D, G, seed=111, cfg=E.frozen_A_cfg())
    assert not str(r["state"]).startswith("Certificate(")   # A1: bare state string, not the namedtuple repr


def test_prereg_states_subject_vs_trial_label_units():
    prereg = os.path.join(os.path.dirname(R.__file__), "..", "..", "notes",
                          "CSC_REALEEG_VALIDATION_PREREG_DRAFT.md")
    txt = open(prereg).read().lower()
    assert "label_unit" in txt and "trial" in txt and "subject" in txt and "motor imagery" in txt


# ---- fail-closed behaviors ----
def test_missing_channel_fails_closed(monkeypatch=None):
    """features_for_run must raise FAIL_CLOSED when a montage channel is absent."""
    import numpy as np
    orig = B.parse_run
    def fake(path):
        chan = [c for c in MONTAGE16 if c != "Cz"] + ["Fp1", "Oz"]   # Cz missing
        x = np.random.RandomState(0).randn(4000, len(chan))
        return x, 1000, chan, np.array([100, 1200]), np.array([1, 2]), "fake"
    B.parse_run = fake
    try:
        raised = False
        try:
            B.features_for_run("dummy")
        except RuntimeError as e:
            raised = "FAIL_CLOSED" in str(e)
        assert raised, "missing montage channel must fail closed"
    finally:
        B.parse_run = orig


def test_wrong_cache_hash_fails_closed():
    if not _CACHE_PRESENT:
        return  # cache absent -> hash check skipped by design (rebuildable)
    assert _dry_with_temp(lambda m: m["cache"]["provenance"].__setitem__(
        "cache_sha256", "0" * 64)) is False


def test_eligibility_mismatch_fails_closed():
    if not _CACHE_PRESENT:
        return
    assert _dry_with_temp(lambda m: m["cache"].__setitem__(
        "min_eligible_paired_subjects", 100)) is False


def test_seed_overlap_fails_closed():
    # base seed pushed INTO the B3 synthetic range -> disjointness check must fail
    assert _dry_with_temp(lambda m: m["bank"]["seed_schedule"].__setitem__(
        "realeeg_base_seed", 3000000)) is False


def test_montage_substitution_fails_closed():
    # swapping a channel (e.g. FCz back in) must break the frozen-montage check
    def mut(m):
        ch = list(m["cache"]["channels"]); ch[0] = "FCz"; m["cache"]["channels"] = ch
    assert _dry_with_temp(mut) is False


def test_method_hash_drift_fails_closed():
    assert _dry_with_temp(lambda m: m["b3"]["code_provenance"]["method_files_sha256"].__setitem__(
        "csc/mininfo/paired_calibrated.py", "0" * 64)) is False


# ---- frozen / disjoint route manifests ----
def test_routes_method_locks_pinned_and_distinct():
    rA = json.load(open(R.ROUTEA_MANIFEST))["code_provenance"]["method_files_sha256"]
    b3 = json.load(open(R.B3_MANIFEST))["code_provenance"]["method_files_sha256"]
    assert set(rA).isdisjoint(set(b3)), "Route A and B3 pin distinct method files"
    assert b3["csc/mininfo/paired_calibrated.py"].startswith("26e505ed")
    assert rA["csc/protocol.py"].startswith("9c158ea7")


def test_alpha_and_bootstraps_frozen():
    for p in (R.ROUTEA_MANIFEST, R.B3_MANIFEST):
        s = json.load(open(p))["statistics"]
        assert s["alpha_budget_per_decision_cohort"] == 0.025
        assert s["family_report_target"] == 0.05
        assert s["invalid_fraction_cap"] == 0.20
        assert s["b_cohort_bootstrap"] == 2000
    assert json.load(open(R.B3_MANIFEST))["statistics"]["b_certifier_internal_null"] == 200


# ---- robustness-only / descriptive cannot gate ----
def test_2b_and_tangent_space_are_robustness_only():
    for p in (R.ROUTEA_MANIFEST, R.B3_MANIFEST):
        ep = json.dumps(json.load(open(p))["endpoints"]).lower()
        assert "2b" in ep and "robustness-only" in ep and "not gating" in ep


def test_genuine_contrast_and_power_not_gating():
    bank = {x["name"]: x for x in json.load(open(R.BANK_MANIFEST))["conditions"]}
    assert bank["genuine_session_contrast_descriptive"]["gating"] is False
    assert bank["POS_concept"]["gating"] is False
    assert bank["POS_concept_plus_cov"]["gating"] is False
    assert bank["POS_pure_conditional"]["gating"] is False
    # gating NO_CONCEPT controls
    assert bank["NULL_cov"]["gating"] is True
    assert bank["random_label_control"]["gating"] is True


def test_gating_set_is_exactly_the_four_controls():
    gs = json.load(open(R.BANK_MANIFEST))["gating_summary"]["gating_conditions"]
    assert gs == ["NULL_cov", "NULL_label", "NULL_cov_plus_label", "random_label_control"]


def test_random_label_demotion_fails_closed():
    def mut(m):
        for cond in m["bank"]["conditions"]:
            if cond["name"] == "random_label_control":
                cond["gating"] = False
        # also demote it in the summary to simulate a full demotion attempt
        gc = m["bank"]["gating_summary"]["gating_conditions"]
        m["bank"]["gating_summary"]["gating_conditions"] = [c for c in gc if c != "random_label_control"]
    assert _dry_with_temp(mut) is False


def test_sbatch_wrapper_shape_is_valid_multiline_shell():
    import subprocess
    sb = os.path.join(os.path.dirname(R.__file__), "run_realeeg_validation.sbatch")
    lines = open(sb).read().splitlines()
    assert len(lines) > 40, "wrapper must be a real multi-line script, not a collapsed one-liner"
    assert lines[0] in ("#!/usr/bin/env bash", "#!/bin/bash")
    assert subprocess.run(["bash", "-n", sb]).returncode == 0, "bash -n must pass"
    assert any(l.startswith("#SBATCH --chdir=") for l in lines), "frozen-worktree --chdir required"
    # no UNCOMMENTED prose leaking to a command line
    for l in lines:
        assert not l.startswith(("NOT invoked", "worktree", "Runs ONLY")), f"uncommented prose leak: {l!r}"
    # freshness check must verify the new provenance fields
    txt = open(sb).read()
    assert "base_seed" in txt and "synthetic_tags_untouched" in txt and "per_cohort" in txt


def test_result_payload_provenance_fields_present():
    # the execute() payload must carry full provenance (source inspection; execute itself refuses w/o tag)
    import inspect
    src = inspect.getsource(R.execute)
    for f in ("manifest_provenance", "engine_sha256", "runner_sha256", "cache_metadata_sha256",
              "frozen_refs", "routeA_synthetic_tag", "routeB3_synthetic_tag", "synthetic_tags_untouched",
              "slurm", "seed_schedule", "genuine_contrast_descriptive_only"):
        assert f in src, f"payload missing provenance field: {f}"


def test_bootstrap_reframed_as_cohort_not_subject_clustered():
    import csc.mininfo.realeeg_engine as E
    assert hasattr(E, "cohort_bootstrap_upper") and not hasattr(E, "subject_bootstrap_upper")
    # no manifest claims a subject-clustered bound anymore
    for p in (R.BANK_MANIFEST, R.ROUTEA_MANIFEST, R.B3_MANIFEST):
        assert "subject-clustered" not in open(p).read().lower()
    # the ENGINE must not mislabel the R1 bound as a subject-clustered BOOTSTRAP (line 132's Route-A
    # 'subject-clustered inference' is a different, correct usage and is allowed)
    eng = open(E.__file__).read().lower()
    assert "subject-clustered bootstrap" not in eng and "subject_bootstrap_upper" not in eng
    assert "cohort_bootstrap_upper" in eng
    assert "b_cohort_bootstrap" in json.load(open(R.BANK_MANIFEST))["run_spec"]


def test_no_validation_result_artifact_exists():
    # the dry-run package must not ship a real validation result
    for p in [os.path.join(os.path.dirname(R.__file__), "..", "results", "realeeg_validation_result.json")]:
        assert not os.path.exists(p)


# ---- hardening from red-team (trap-control demotion, absent-cache builder hash, frozen fields, seed range) ----
def test_trap_controls_demotion_fails_closed():
    def mut(m):
        for cond in m["bank"]["conditions"]:
            if cond["name"] in ("NULL_label", "NULL_cov_plus_label"):
                cond["gating"] = False
    assert _dry_with_temp(mut) is False


def test_builder_hash_on_absent_cache_fails_closed():
    # force the cache-absent branch (nonexistent cache path) + corrupt builder hash -> must fail closed
    def mut(m):
        m["cache"]["provenance"]["cache_path"] = "/nonexistent/LEE2019_B3.npz"
        m["cache"]["provenance"]["builder_sha256"] = "0" * 64
    assert _dry_with_temp(mut) is False


def test_frozen_scalar_fields_checked():
    assert _dry_with_temp(lambda m: m["cache"].__setitem__("run", "EEG_MI_test")) is False
    assert _dry_with_temp(lambda m: m["cache"].__setitem__(
        "label_map", {"left_hand": 1, "right_hand": 0})) is False


def test_seed_disjointness_uses_ranges_not_hardcoded_literal():
    # with a forbidden range extending past base, disjointness must FAIL (guards the old walrus fail-open)
    orig = R.FORBIDDEN_SEED_RANGES
    R.FORBIDDEN_SEED_RANGES = orig + [(19000000, 21000000)]   # base 20000000 now inside a forbidden range
    try:
        assert R.dry_run() is False
    finally:
        R.FORBIDDEN_SEED_RANGES = orig


def test_gating_flags_present_and_false():
    for p in (R.ROUTEA_MANIFEST, R.B3_MANIFEST):
        gf = json.load(open(p))["gating_flags"]
        assert gf["R2_power_is_gating"] is False
        assert gf["R5_2b_is_gating"] is False
        assert gf["genuine_contrast_is_gating"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    npass = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); npass += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"[{npass}/{len(fns)} tests pass]")
    sys.exit(0 if npass == len(fns) else 1)
