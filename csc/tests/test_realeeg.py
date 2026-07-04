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


def test_runner_docstring_not_stale():
    # the runner has a GUARDED, PARALLEL execute() at v2 -> its docstring must not claim dry-run-only / refuses-all
    doc = R.__doc__ or ""
    for bad in ("DRY-RUN ONLY", "structurally REFUSED", "implements NO path that runs injections"):
        assert bad not in doc, f"stale runner docstring claim contradicts the guarded execute(): {bad!r}"
    assert "guarded" in doc and "execute" in doc
    assert "csc-realeeg-v2" in doc, "v2 docstring must name the v2 tag"
    assert "performance-only" in doc.lower() or "parallel" in doc.lower()


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
    # v2: BLAS oversubscription pin (incl. NUMEXPR), v2 tag, and completeness (801 / n_tasks) in freshness check
    assert "NUMEXPR_NUM_THREADS" in txt, "v2 wrapper must pin NUMEXPR threads"
    assert "csc-realeeg-v2" in txt, "v2 wrapper must target the v2 tag"
    assert "801" in txt and "n_tasks" in txt, "v2 freshness check must assert task completeness"


def test_result_payload_provenance_fields_present():
    # the execute() payload must carry full provenance (source inspection; execute itself refuses w/o tag)
    import inspect
    src = inspect.getsource(R.execute)
    for f in ("manifest_provenance", "engine_sha256", "runner_sha256", "cache_metadata_sha256",
              "frozen_refs", "routeA_synthetic_tag", "routeB3_synthetic_tag", "synthetic_tags_untouched",
              "slurm", "seed_schedule", "genuine_contrast_descriptive_only",
              # v2 execution provenance + fail-closed handling
              "execution", "task_table_sha256", "performance_only_change", "n_tasks_expected",
              "InfraError", "blas_threads"):
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
    # the dry-run package must not ship a real validation result (v1 or v2 FINAL)
    rd = os.path.join(os.path.dirname(R.__file__), "..", "results")
    for n in ("realeeg_validation_result.json", "realeeg_validation_v2.final.json"):
        assert not os.path.exists(os.path.join(rd, n)), f"unexpected result artifact: {n}"


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


# ================================================================ v2: performance-only parallel execution
def _mini_bank():
    """Tiny bank (2 conditions x 2 cohorts) + toy cache + fast params for serial<->parallel identity tests.
    TEST seed base (NOT REAL_BASE_SEED) + tiny certifier params (identity is param-independent)."""
    import numpy as np
    import csc.mininfo.realeeg_engine as E
    import csc.protocol as P
    cache = E._toy_cache(np.random.default_rng(20240705), n_subj=8, n_trials=32)
    TEST_SEED_BASE = 555_000                    # disjoint from REAL_BASE_SEED (20_000_000)
    ml = dict(min_confirm_pairs=3, pair_integrity_min=0.95, min_epochs=6, rank=3, C=0.5,
              n_folds=3, n_boot=8, alpha_family=0.05, n_decision_budgets=2)
    cfg = P.ProtocolConfig(n_boot=8, n_dir_boot=16, target_n_boot=16, tau_n_pseudotargets=32,
                           label_unit="trial", analysis_unit="subject")
    mini = {"run_spec": {"cohorts_per_condition": 2, "subjects_per_cohort": 5,
                         "b_cohort_bootstrap": 100, "invalid_fraction_cap": 0.20},
            "seed_schedule": {"realeeg_base_seed": TEST_SEED_BASE, "condition_stride": 1000},
            "conditions": [
                {"name": "NULL_cov", "condition_index": 0, "gating": True,
                 "ground_truth": "NO_CONCEPT", "routes": ["A", "B3"]},
                {"name": "POS_concept", "condition_index": 1, "gating": False,
                 "ground_truth": "CONCEPT", "routes": ["A", "B3"]}],
            "gating_summary": {"gating_conditions": ["NULL_cov"]}}
    return E, cache, mini, ml, cfg, TEST_SEED_BASE


_SHARED = ("condition", "gating", "cohort", "seed", "ground_truth", "B3", "A")


def test_serial_parallel_identity_on_smoke_subset():
    E, cache, mini, ml, cfg, sb = _mini_bank()
    serial = E.run_validation(cache, mini, ml, cfg, sb)
    with tempfile.TemporaryDirectory() as td:
        def par(nj):
            return E.run_validation_parallel(cache, mini, ml, cfg, sb, n_jobs=nj,
                partial_path=os.path.join(td, f"p{nj}.jsonl"), checkpoint_path=os.path.join(td, f"c{nj}.json"),
                resume=False, provenance={"git_head": "t", "cache_sha256": "h"}, progress=False)
        par1, tth1 = par(1); par3, tth3 = par(3)
    # FULL-record identity: compare EVERY serial-record field (condition, seed, cohort, gating, ground_truth,
    # and the complete B3/A dicts = state + confirmed + n_sampler_failures + n_boot_invalid), not just a subset.
    # (The parallel record only ADDS task_id / condition_index for ordering.) This proves per-cohort accounting
    # -- Route A state, B3 state, invalid/abstain flags -- is byte-identical, not just the coarse verdict.
    assert set(_SHARED) == set(serial[0].keys()), "identity comparison must cover the FULL serial record schema"
    def full(rs): return {(r["condition"], r["cohort"]): {k: r.get(k) for k in _SHARED} for r in rs}
    s, a, b = full(serial), full(par1), full(par3)
    assert set(s) == set(a) == set(b)
    assert all(json.dumps(s[k], sort_keys=True) == json.dumps(a[k], sort_keys=True) for k in s), \
        "serial != parallel(n_jobs=1) on some per-cohort field"
    assert all(json.dumps(s[k], sort_keys=True) == json.dumps(b[k], sort_keys=True) for k in s), \
        "serial != parallel(n_jobs=3) on some per-cohort field"
    assert tth1 == tth3, "task_table_sha256 unstable across n_jobs"
    assert [(r["condition"], r["cohort"]) for r in par3] == [(r["condition"], r["cohort"]) for r in serial], \
        "canonical order != serial order (bootstrap array-order not preserved)"
    # verdict-level accounting (per-condition denominator contribution, boot_upper, invalid_frac, INCONCLUSIVE,
    # status, tier2/tier3) must also be byte-identical
    vs = json.dumps(E.evaluate_verdict(serial, mini), sort_keys=True, default=str)
    vp = json.dumps(E.evaluate_verdict(par3, mini), sort_keys=True, default=str)
    assert vs == vp, "verdict differs between serial and parallel"


def test_task_count_breakdown_is_honest_801():
    exe = json.load(open(R.BANK_MANIFEST))["execution_provenance"]["task_count_breakdown"]
    assert exe["n_multi_cohort_conditions"] == 8 and exe["cohorts_per_multi_cohort_condition"] == 100
    assert exe["n_genuine_descriptive_cohorts"] == 1 and exe["n_tasks_total"] == 801
    assert len(exe["multi_cohort_conditions"]) == 8 and len(exe["genuine_descriptive_conditions"]) == 1
    # honesty: NULL_real_session is a REAL-label pseudo-split, not an injected condition
    assert "NULL_real_session" in exe["multi_cohort_conditions"]
    assert "not a synonym for 'injected'" in exe["label_note"].lower() or "not injected" in exe["label_note"].lower() \
        or "not a synonym" in exe["label_note"].lower()
    # cross-check against the actual bank conditions
    bank = json.load(open(R.BANK_MANIFEST))
    multi = [c["name"] for c in bank["conditions"] if c["name"] != "genuine_session_contrast_descriptive"]
    assert exe["multi_cohort_conditions"] == multi


def test_canonical_task_table_deterministic_and_801():
    import csc.mininfo.realeeg_engine as E
    bank = json.load(open(R.BANK_MANIFEST)); base = bank["seed_schedule"]["realeeg_base_seed"]
    stride = bank["seed_schedule"]["condition_stride"]
    t1 = E.build_task_table(bank, base); t2 = E.build_task_table(bank, base)
    assert t1 == t2 and len(t1) == 801, f"expected deterministic 801, got {len(t1)}"
    assert len({t['task_id'] for t in t1}) == 801, "task_ids not unique"
    assert [(t["condition_index"], t["cohort_index"]) for t in t1] == \
           sorted((t["condition_index"], t["cohort_index"]) for t in t1), "not canonically sorted"
    assert E.task_table_sha256(t1) == E.task_table_sha256(t2), "hash unstable"
    for t in t1:
        assert t["seed"] == base + t["condition_index"] * stride + t["cohort_index"], "seed formula drift"


def test_duplicate_or_missing_task_fails_closed():
    import csc.mininfo.realeeg_engine as E
    tasks = [{"task_id": "00:X:0000", "condition_index": 0, "cohort_index": 0},
             {"task_id": "00:X:0001", "condition_index": 0, "cohort_index": 1}]
    good = [{"task_id": "00:X:0000", "condition_index": 0, "cohort": 0},
            {"task_id": "00:X:0001", "condition_index": 0, "cohort": 1}]
    assert len(E._assemble_records(tasks, good)) == 2
    for bad in (good[:1], good + [good[0]]):                 # missing, then duplicate
        try:
            E._assemble_records(tasks, bad); assert False, "assembly did not fail closed"
        except E.InfraError:
            pass


def test_worker_exception_is_infra_failure_not_silent_skip():
    import csc.mininfo.realeeg_engine as E
    import numpy as np
    tasks = [{"task_id": "00:X:0000", "condition_index": 0, "cohort_index": 0}]
    try:
        E._assemble_records(tasks, [{"task_id": "00:X:0000", "__worker_error__": "ValueError: boom"}])
        assert False, "worker error silently skipped"
    except E.InfraError as e:
        assert "error" in str(e).lower()
    # a crash OUTSIDE the certifiers (unknown condition -> build_cohort raises) surfaces as a worker error
    bad = {"task_id": "99:BAD:0000", "condition_index": 99, "cohort_index": 0, "condition": "not_a_condition",
           "seed": 1, "routes": ["B3"], "is_gating": False, "ground_truth": "x"}
    rec = E.run_one_task(bad, E._toy_cache(np.random.default_rng(1)), 3, {}, None)
    assert "__worker_error__" in rec, "outside-certifier crash must surface as worker error, not a record"


def test_resume_requires_same_task_table_and_binding():
    E, cache, mini, ml, cfg, sb = _mini_bank()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "p.jsonl"); c = os.path.join(td, "c.json")
        _, tth = E.run_validation_parallel(cache, mini, ml, cfg, sb, n_jobs=2, partial_path=p,
            checkpoint_path=c, resume=False, provenance={"git_head": "GH", "cache_sha256": "CH"}, progress=False)

        def resume_expect_fail(needle):
            try:
                E.run_validation_parallel(cache, mini, ml, cfg, sb, n_jobs=2, partial_path=p, checkpoint_path=c,
                    resume=True, provenance={"git_head": "GH", "cache_sha256": "CH"}, progress=False)
                assert False, f"resume did not fail closed on {needle}"
            except E.InfraError as e:
                assert needle in str(e).lower(), f"wrong refusal: {e}"

        ck = json.load(open(c)); ck["task_table_sha256"] = "0" * 64; json.dump(ck, open(c, "w"))
        resume_expect_fail("task_table")
        ck["task_table_sha256"] = tth; ck["git_head"] = "DIFFERENT"; json.dump(ck, open(c, "w"))
        resume_expect_fail("git_head")
        # red-team v2: a NULL / ABSENT binding field must ALSO fail closed (not silently skip the guard)
        ck["git_head"] = None; json.dump(ck, open(c, "w"))
        resume_expect_fail("git_head")
        ck.pop("git_head"); json.dump(ck, open(c, "w"))
        resume_expect_fail("git_head")


def test_partial_and_checkpoint_are_not_final_artifacts():
    import inspect, csc.mininfo.realeeg_engine as E
    src = inspect.getsource(E)
    assert "_assemble_records" in src and "missing task record" in src
    sb = open(os.path.join(os.path.dirname(R.__file__), "run_realeeg_validation.sbatch")).read()
    assert 'mv "$TMP_OUT" "$OUT"' in sb
    assert sb.index("n_tasks") < sb.index('mv "$TMP_OUT" "$OUT"'), "completeness must be checked before finalizing"


def test_nboot_cohorts_seeds_and_denominators_unchanged_from_v1():
    import csc.mininfo.realeeg_engine as E
    bank = json.load(open(R.BANK_MANIFEST)); b3 = json.load(open(R.B3_MANIFEST)); rs = bank["run_spec"]
    assert b3["statistics"]["b_certifier_internal_null"] == 200
    assert rs["cohorts_per_condition"] == 100 and rs["subjects_per_cohort"] == 30
    assert rs["b_cohort_bootstrap"] == 2000 and rs["invalid_fraction_cap"] == 0.20
    assert rs["positive_decision_budgets"] == [20, 30]
    assert bank["seed_schedule"]["realeeg_base_seed"] == 20000000
    assert E.B3_DECIDED == ("CONCEPT_CONFIRMED", "NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT")
    assert set(E.B3_ABSTAIN_INVALID) == {"NEED_MORE_LABELS", "INVALID_PAIR_STRUCTURE", "UNIDENTIFIABLE"}
    assert hasattr(E, "cohort_bootstrap_upper")


def test_execution_provenance_documents_performance_only_and_blas():
    exe = json.load(open(R.BANK_MANIFEST))["execution_provenance"]
    assert exe["mode"] == "cohort_parallel" and exe["performance_only"] is True
    assert exe["n_tasks_expected"] == 801
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert str(exe["blas_threads"][v]) == "1"
    txt = json.dumps(exe).lower()
    for k in ("b_certifier_internal_null", "cohorts_per_condition", "seed schedule",
              "denominators", "route a", "b3 method"):
        assert k in txt, f"unchanged-science list missing {k!r}"


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
