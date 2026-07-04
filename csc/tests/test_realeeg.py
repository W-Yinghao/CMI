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


def test_execute_is_refused_exit2():
    assert R.refuse_execute() == 2


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
        assert s["b_subject_bootstrap"] == 2000
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
    # NULL_cov is the gating type-I null
    assert bank["NULL_cov"]["gating"] is True


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
