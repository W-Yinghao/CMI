"""Guards for acar/v4/develop.py (Phase-1 EXPLORATORY DEV orchestration). SYNTHETIC FIXTURES ONLY; NO real DEV cohort,
NO v3 loader, NO real DEV report, NO freeze, NO lockbox. Covers the split/provenance hardening: FOLD-LOCAL CAL→EVAL
(λ* from CAL only; EVAL never lets CAL into its denominator); COHORT-AWARE subject key (same local id in two cohorts =
two subjects); comparator contract (best_fixed + v2_replay distinct slots, G3 comparator fixed in config); score-family
registry (real_mode rejects arbitrary callables); per-config vs GLOBAL policy-frontier gap; plus the taxonomy/no-binding,
subject-macro, fallback-denominator, gap-telescoping, G6 both-diseases, lineage, permutation-invariance, fail-closed.
Run: python -m acar.v4.tests.test_develop
"""
import json
import math
import os
import shutil
import tempfile
from dataclasses import replace

import numpy as np

from acar.config import DISEASE
from acar.v4 import develop as D
from acar.v4.develop import V4OOFRecord, ScoreFamily

ACT = D.ACTIONS
A = D.A
NF = D.N_FEAT
PD_COH = DISEASE["PD"][0]
PD_COH2 = DISEASE["PD"][1]
SCZ_COH = DISEASE["SCZ"][0]


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                       # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def _feats(good):
    f = np.full((A, NF), 5.0)
    for c in (1, 2, 3):
        f[good, c] = 0.0
    return f


def _dr(good, beneficial):
    dr = np.full(A, 1.0)
    if beneficial:
        dr[good] = -1.0
    return dr


def _make_records(diseases=("PD", "SCZ"), beneficial=True, fallback_subjects=(), with_cal=True,
                  eval_dr_factor=1.0, cal_beneficial=None):
    """Cross-fit fixture: K=2 outer folds; subjects 0-3 EVAL in fold 0 (CAL in fold 1), 4-7 vice-versa. Each physical
    batch appears once as EVAL (its held-out fold) and once as CAL (the other fold)."""
    cohort = {"PD": PD_COH, "SCZ": SCZ_COH}
    cal_flag = beneficial if cal_beneficial is None else cal_beneficial
    recs = []
    for d in diseases:
        for s in range(8):
            sid = f"{d}_s{s}"
            ef = 0 if s < 4 else 1
            cf = 1 - ef
            for b in range(2 + (s % 3)):
                good = (s + b) % A
                fb = bool(sid in fallback_subjects and b == 0)
                recs.append(V4OOFRecord(d, sid, cohort[d], f"{sid}_b{b}", ef, "EVAL", fb,
                                        _dr(good, beneficial) * eval_dr_factor, _feats(good), ACT))
                if with_cal:
                    recs.append(V4OOFRecord(d, sid, cohort[d], f"{sid}_b{b}", cf, "CAL", fb,
                                            _dr(good, cal_flag), _feats(good), ACT))
    return recs


def _rep(res, d="PD", pf="safe_set", loss="mean"):
    return next(r for r in res.reports if r.disease == d and r.policy_family == pf and r.loss == loss)


# ----------------------------------------------------------------------------- verdicts + no binding language

def test_candidate_found_not_select_and_no_binding():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    assert res.verdict == D.V4_DEV_CANDIDATE_FOUND and res.run_status == D.V4_DEV_EXPLORATION_COMPLETE
    assert D.assert_no_binding_language(res)
    assert all(r.status == "EVALUATED" for r in res.reports) and not any(r.status == "SELECT" for r in res.reports)
    assert any(r.all_pass() for r in res.reports)
    assert not (set(k.lower() for k in res.manifest) & set(D._FORBIDDEN_MANIFEST_KEYS))


def test_eval_L_harm_all_populated_and_all_batch_denominator():
    # C3 additive field: every report carries a finite all-batch-denominator EVAL harm_indicator loss in [0,1], DISTINCT from
    # the conditional harm_rate. all-batch denominator >= adapted-only denominator => eval_L_harm_all <= harm_rate.
    res = D.run_dev_exploration(_make_records(beneficial=True))
    for r in res.reports:
        assert r.eval_L_harm_all is not None and math.isfinite(r.eval_L_harm_all) and 0.0 <= r.eval_L_harm_all <= 1.0
        if math.isfinite(r.harm_rate):
            assert r.eval_L_harm_all <= r.harm_rate + 1e-9       # all-batch <= conditional (adapted-only) harm


def test_no_passer_is_negative():
    res = D.run_dev_exploration(_make_records(beneficial=False))
    assert res.verdict == D.V4_DEV_NEGATIVE and not any(r.all_pass() for r in res.reports)


def test_single_disease_cannot_pass_g6():
    res = D.run_dev_exploration(_make_records(diseases=("PD",), beneficial=True))
    assert res.verdict == D.V4_DEV_NEGATIVE
    assert all(not r.g6_nonvacuous_both_diseases_pass for r in res.reports)
    assert any(r.g1_coverage_pass and r.g2_red_pass for r in res.reports)        # PD itself non-vacuous


# ----------------------------------------------------------------------------- (1) fold-local CAL->EVAL

def test_lambda_star_from_cal_not_eval():
    base = D.run_dev_exploration(_make_records(beneficial=True))
    eval_changed = D.run_dev_exploration(_make_records(beneficial=True, eval_dr_factor=3.0))
    pf0 = _rep(base).provenance["per_fold_lambda"]
    pf1 = _rep(eval_changed).provenance["per_fold_lambda"]
    assert pf0 and pf0 == pf1, "changing EVAL ΔR must NOT change λ* (it comes from CAL)"
    cal_harmful = D.run_dev_exploration(_make_records(beneficial=True, cal_beneficial=False))
    assert _rep(cal_harmful).provenance["per_fold_lambda"] != pf0   # changing CAL ΔR CAN change λ* (here: none pass)


def test_cal_records_excluded_from_eval_denominator():
    recs = _make_records(beneficial=True)
    # add a CAL-ONLY subject (never EVAL) to PD; it must not enter the EVAL subject count
    for b in range(3):
        recs.append(V4OOFRecord("PD", "PD_calonly", PD_COH, f"calonly_b{b}", 0, "CAL", False,
                                _dr(0, True), _feats(0), ACT))
    res = D.run_dev_exploration(recs)
    assert res.manifest["diseases"]["PD"]["n_eval_subjects"] == 8            # the 8 EVAL subjects, NOT 9
    assert res.manifest["diseases"]["PD"]["n_cal_subjects"] == 9             # CAL-only subject counted in CAL


# ----------------------------------------------------------------------------- (2) cohort-aware subject key

def test_cohort_aware_subject_key_not_merged():
    recs = [V4OOFRecord("PD", "sub-001", PD_COH, "b0", 0, "EVAL", False, _dr(0, True), _feats(0), ACT),
            V4OOFRecord("PD", "sub-001", PD_COH2, "b0", 0, "EVAL", False, _dr(0, True), _feats(0), ACT)]
    res = D.run_dev_exploration(recs)
    assert res.manifest["diseases"]["PD"]["n_eval_subjects"] == 2   # same local id in two cohorts ⇒ two subjects


# ----------------------------------------------------------------------------- (3) comparator contract

def test_comparator_slots_distinct_and_g3_uses_configured():
    res = D.run_dev_exploration(_make_records(beneficial=True),
                                v2_replay_red_by_disease={"PD": 0.5, "SCZ": 0.5})
    r = _rep(res)
    assert r.provenance["c0_v2_replay_red"] == 0.5
    assert r.provenance["c0_best_fixed_red"] != 0.5                 # the two comparator slots are distinct
    assert r.c0_red == r.provenance["c0_best_fixed_red"]            # default g3_comparator = best_fixed
    # g3_comparator='v2_replay' uses the v2 slot; missing values fail closed
    cfg = D.V4DevConfig(g3_comparator="v2_replay")
    res2 = D.run_dev_exploration(_make_records(beneficial=True), config=cfg,
                                 v2_replay_red_by_disease={"PD": 0.5, "SCZ": 0.5})
    assert _rep(res2).c0_red == 0.5
    _expect(ValueError, lambda: D.run_dev_exploration(_make_records(beneficial=True), config=cfg))


# ----------------------------------------------------------------------------- (4) score-family registry / real_mode

def test_real_mode_rejects_arbitrary_callable():
    custom = ScoreFamily("evil", lambda f: (f[:, :, 0], f[:, :, 0]))
    _expect(ValueError, lambda: D.run_dev_exploration(_make_records(beneficial=True), score_families=[custom],
                                                      real_mode=True))
    # names and registry objects are accepted in real_mode
    ok1 = D.run_dev_exploration(_make_records(beneficial=True), score_families=["shift_margin"], real_mode=True)
    ok2 = D.run_dev_exploration(_make_records(beneficial=True),
                                score_families=[D.SCORE_FAMILY_REGISTRY["shift_margin"]], real_mode=True)
    assert ok1.run_status == D.V4_DEV_EXPLORATION_COMPLETE and ok2.run_status == D.V4_DEV_EXPLORATION_COMPLETE
    # custom callable allowed when NOT real_mode
    assert D.run_dev_exploration(_make_records(beneficial=True), score_families=[custom]).run_status \
        == D.V4_DEV_EXPLORATION_COMPLETE


# ----------------------------------------------------------------------------- (5) per-config vs global policy gap

def test_global_policy_frontier_dominates_each_config():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    for r in res.reports:
        # global frontier ceiling ≥ each per-config ceiling ⇒ global gap ≤ per-config gap
        assert r.global_policy_family_gap <= r.per_config_policy_gap + 1e-9


# ----------------------------------------------------------------------------- frontier exactness, weighting, etc.

def test_frontier_gap_telescoping_and_info_nonneg():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    g = _rep(res).frontier_gaps
    assert abs((g["info_gap"] + g["policy_gap"] + g["calibration_gap"]) - (g["true_ceiling"] - g["calibrated_red"])) < 1e-9
    assert g["info_gap"] >= -1e-9


def test_subject_macro_weighting_used_for_c0():
    def rec(sid, bid, good):
        return V4OOFRecord("PD", sid, PD_COH, bid, 0, "EVAL", False, _dr(good, True), _feats(good), ACT)
    recs = [rec("PD_A", "PD_A_b0", 0),
            rec("PD_B", "PD_B_b0", 1), rec("PD_B", "PD_B_b1", 1), rec("PD_B", "PD_B_b2", 1)]
    res = D.run_dev_exploration(recs)
    assert abs(res.manifest["diseases"]["PD"]["c0_best_fixed_red"] - 0.0) < 1e-9   # subject-macro (NOT batch 0.5)


def test_fallback_stays_in_denominator():
    base = D.run_dev_exploration(_make_records(beneficial=True))
    fb = D.run_dev_exploration(_make_records(beneficial=True, fallback_subjects=("PD_s0", "SCZ_s0")))

    def best_cov(res):
        cs = [r.coverage for r in res.reports if r.policy_family == "safe_set" and r.loss == "mean" and r.coverage > 0]
        return max(cs) if cs else 0.0
    assert best_cov(fb) < best_cov(base) - 1e-9


def test_manifest_lineage_and_permutation_invariant():
    recs = _make_records(beneficial=True)
    a = D.run_dev_exploration(recs)
    rng = np.random.default_rng(3); perm = rng.permutation(len(recs))
    b = D.run_dev_exploration([recs[i] for i in perm])
    assert a.manifest_sha256 == b.manifest_sha256 and a.verdict == b.verdict
    lin = a.manifest["lineage"]
    assert lin["v2"] == "MEASUREMENT_ONLY" and "DEV_STOP" in lin["v3"] and lin["v4"] == "NON-BINDING"
    assert "NON-BINDING" in a.manifest["boundary"] and "NO LOCKBOX" in a.manifest["boundary"]


def test_subject_level_crossfit_invariants():
    z = np.zeros(A); zf = np.zeros((A, NF))

    def r(bid, fold, split):
        return V4OOFRecord("PD", "s1", PD_COH, bid, fold, split, False, z, zf, ACT)
    # subject EVAL in two folds (distinct batches ⇒ batch-partition passes, subject-partition must catch it)
    _expect(ValueError, lambda: D.run_dev_exploration([r("b0", 0, "EVAL"), r("b1", 1, "EVAL")]))
    # subject CAL and EVAL in the SAME fold
    _expect(ValueError, lambda: D.run_dev_exploration([r("b0", 0, "EVAL"), r("b1", 0, "CAL")]))
    # subject mixed split (CAL + FIT) within a fold
    _expect(ValueError, lambda: D.run_dev_exploration([r("b0", 0, "CAL"), r("b1", 0, "FIT")]))
    # legitimate cross-fit (EVAL fold 0, CAL fold 1) does NOT raise on the invariants
    ok = [r("b0", 0, "EVAL"), r("b1", 1, "CAL")]
    D.run_dev_exploration(ok)   # single disease ⇒ NEGATIVE, but no invariant violation


def test_real_mode_requires_explicit_families():
    _expect(ValueError, lambda: D.run_dev_exploration(_make_records(beneficial=True), real_mode=True))   # None
    assert D.run_dev_exploration(_make_records(beneficial=True), score_families=["shift_margin"],
                                 real_mode=True).run_status == D.V4_DEV_EXPLORATION_COMPLETE


def test_record_immutability_and_construction_validation():
    dr = np.array([-1.0, 1.0, 1.0]); feats = _feats(0)
    rec = V4OOFRecord("PD", "s", PD_COH, "b0", 0, "EVAL", False, dr, feats, ACT)
    dr[0] = 999.0; feats[0, 0] = 999.0                                   # mutate the ORIGINAL arrays
    assert rec.dr[0] == -1.0 and rec.features_v2[0, 0] == 5.0            # record holds an independent copy
    assert rec.dr.flags.writeable is False and rec.features_v2.flags.writeable is False
    _expect(ValueError, lambda: rec.dr.__setitem__(0, 1.0))             # read-only
    assert isinstance(rec.action_names, tuple)
    # construction-time validation
    _expect(ValueError, lambda: V4OOFRecord("PD", "s", PD_COH, "b0", 0, "EVAL", False,
                                            np.array([np.nan, 0.0, 0.0]), _feats(0), ACT))
    _expect(ValueError, lambda: V4OOFRecord("PD", "s", PD_COH, "b0", 0, "EVAL", False,
                                            np.zeros(A), np.zeros((A, NF - 1)), ACT))


def test_record_digest_permutation_invariant_and_field_sensitive():
    recs = _make_records(beneficial=True)
    a = D.run_dev_exploration(recs)
    b = D.run_dev_exploration(list(reversed(recs)))
    assert a.manifest["v4_oof_records_sha256"] == b.manifest["v4_oof_records_sha256"]
    assert isinstance(a.manifest["config_sha256"], str) and isinstance(a.manifest["score_family_registry_sha256"], str)
    assert any(p["split"] == "EVAL" and p["subject_count"] > 0 for p in a.manifest["partition"])
    # change one record's dr ⇒ digest changes
    recs2 = list(recs); r0 = recs2[0]
    recs2[0] = V4OOFRecord(r0.disease, r0.subject_id, r0.cohort_id, r0.batch_id, r0.fold, r0.split, r0.fallback,
                           r0.dr * 2.0 + 0.123, r0.features_v2, r0.action_names)
    assert D.run_dev_exploration(recs2).manifest["v4_oof_records_sha256"] != a.manifest["v4_oof_records_sha256"]
    # change one record's features ⇒ digest changes
    recs3 = list(recs); r1 = recs3[1]
    feats = np.array(r1.features_v2); feats[0, 0] += 1.0
    recs3[1] = V4OOFRecord(r1.disease, r1.subject_id, r1.cohort_id, r1.batch_id, r1.fold, r1.split, r1.fallback,
                           r1.dr, feats, r1.action_names)
    assert D.run_dev_exploration(recs3).manifest["v4_oof_records_sha256"] != a.manifest["v4_oof_records_sha256"]


def test_atomic_writer():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    base = tempfile.mkdtemp()
    try:
        outdir = os.path.join(base, "exploration_001")
        assert D.write_dev_exploration_result(res, outdir) == outdir
        assert os.path.isfile(os.path.join(outdir, "manifest.json"))
        assert os.path.isfile(os.path.join(outdir, "RESULT.json"))                   # completion sentinel
        with open(os.path.join(outdir, "manifest.json")) as f:
            txt = f.read()
        assert "NaN" not in txt and "Infinity" not in txt
        assert json.loads(txt)["manifest_sha256"] == res.manifest_sha256
        _expect(FileExistsError, lambda: D.write_dev_exploration_result(res, outdir))   # no overwrite (non-empty)
        empty = os.path.join(base, "empty"); os.mkdir(empty)
        _expect(FileExistsError, lambda: D.write_dev_exploration_result(res, empty))    # race-free: empty dir refused
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_assert_no_binding_language_rejects_illegal():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    assert D.assert_no_binding_language(res)                                            # clean result passes
    _expect(ValueError, lambda: D.assert_no_binding_language(replace(res, verdict="SELECT")))
    _expect(ValueError, lambda: D.assert_no_binding_language(replace(res, verdict="DEV_STOP")))
    _expect(ValueError, lambda: D.assert_no_binding_language(replace(res, run_status="WHATEVER")))
    bad_rep = replace(res.reports[0], status="SELECT")
    _expect(ValueError, lambda: D.assert_no_binding_language(replace(res, reports=(bad_rep,) + res.reports[1:])))
    m = dict(res.manifest); m["lockbox"] = True
    _expect(ValueError, lambda: D.assert_no_binding_language(replace(res, manifest=m)))


def test_cal_records_value_excluded_from_eval_operating_point():
    base = D.run_dev_exploration(_make_records(beneficial=True))
    recs = _make_records(beneficial=True)
    for d, coh in (("PD", PD_COH), ("SCZ", SCZ_COH)):       # add CAL-ONLY subjects (fold 0); never EVAL
        for s in range(3):
            for b in range(2):
                g = (s + b) % A
                recs.append(V4OOFRecord(d, f"{d}_co{s}", coh, f"co{s}_b{b}", 0, "CAL", False,
                                        _dr(g, True), _feats(g), ACT))
    aug = D.run_dev_exploration(recs)
    rb, ra = _rep(base), _rep(aug)
    assert abs(rb.coverage - ra.coverage) < 1e-9 and abs(rb.red - ra.red) < 1e-9       # EVAL op identical
    assert aug.manifest["diseases"]["PD"]["n_cal_subjects"] > base.manifest["diseases"]["PD"]["n_cal_subjects"]
    assert aug.manifest["diseases"]["PD"]["n_eval_subjects"] == base.manifest["diseases"]["PD"]["n_eval_subjects"]


def test_g4_requires_all_eval_folds_certified():
    recs = [r for r in _make_records(beneficial=True)
            if not (r.disease == "PD" and r.split == "CAL" and r.fold == 1)]   # PD EVAL fold 1 loses its CAL
    res = D.run_dev_exploration(recs)
    pd = [r for r in res.reports if r.disease == "PD"]
    assert pd and all(not r.g4_harm_control_pass for r in pd)                  # an uncertified EVAL fold ⇒ NOT PASS
    assert res.verdict == D.V4_DEV_NEGATIVE


def test_construction_rejects_non_float_and_control_chars():
    _expect(ValueError, lambda: V4OOFRecord("PD", "s", PD_COH, "b0", 0, "EVAL", False,
                                            np.array([1, -1, 1], dtype=np.int64), _feats(0), ACT))
    _expect(ValueError, lambda: V4OOFRecord("PD", "s", PD_COH, "b0", 0, "EVAL", False,
                                            _dr(0, True), np.zeros((A, NF), dtype=np.int64), ACT))
    _expect(ValueError, lambda: D.run_dev_exploration(
        [V4OOFRecord("PD", "a\nb", PD_COH, "b0", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)]))


def test_record_digest_action_names_injective():
    z = np.zeros(A); zf = np.zeros((A, NF))
    mk = lambda an: V4OOFRecord("PD", "s", PD_COH, "b", 0, "EVAL", False, z, zf, an)
    assert D._record_digest(mk(("a\x00b", "c", "d"))) != D._record_digest(mk(("a", "b\x00c", "d")))


def test_score_family_registry_predeclared():
    expected = {"shift_margin", "js_flip", "d_entropy_pos", "d_entropy_neg", "d_margin_neg", "flip_pos",
                "js_pos", "bures_pos", "n_eff_neg", "unc_pos"}
    assert set(D.SCORE_FAMILY_REGISTRY) == expected
    feats = np.random.default_rng(0).normal(size=(7, A, NF))
    for name, fam in D.SCORE_FAMILY_REGISTRY.items():
        harm, benefit = fam.compute(feats)
        assert harm.shape == (7, A) and benefit.shape == (7, A)
        assert np.all(np.isfinite(harm)) and np.all(np.isfinite(benefit))
    # all predeclared names resolve in real_mode and run to completion
    res = D.run_dev_exploration(_make_records(beneficial=True), score_families=sorted(expected), real_mode=True)
    assert res.run_status == D.V4_DEV_EXPLORATION_COMPLETE


def test_exact_oof_eval_coverage():
    # clean cross-fit (every subject & batch EVAL exactly once) satisfies exact coverage, in real_mode AND via flag
    assert D.run_dev_exploration(_make_records(beneficial=True), score_families=["shift_margin"],
                                 real_mode=True).run_status == D.V4_DEV_EXPLORATION_COMPLETE
    assert D.run_dev_exploration(_make_records(beneficial=True),
                                 require_exact_eval_coverage=True).run_status == D.V4_DEV_EXPLORATION_COMPLETE
    # a CAL-only subject (never EVAL) ⇒ exact coverage raises; relaxed (default) still allows it
    cal_only = _make_records(beneficial=True)
    for b in range(2):
        cal_only.append(V4OOFRecord("PD", "PD_calonly", PD_COH, f"calonly_b{b}", 1, "CAL", False,
                                    _dr(0, True), _feats(0), ACT))
    _expect(ValueError, lambda: D.run_dev_exploration(cal_only, require_exact_eval_coverage=True))
    _expect(ValueError, lambda: D.run_dev_exploration(cal_only, score_families=["shift_margin"], real_mode=True))
    assert D.run_dev_exploration(cal_only).run_status == D.V4_DEV_EXPLORATION_COMPLETE          # relaxed default
    # a batch that is never EVAL (extra CAL batch for an existing EVAL subject) ⇒ exact coverage raises
    extra = _make_records(beneficial=True)
    extra.append(V4OOFRecord("PD", "PD_s0", PD_COH, "PD_s0_calextra", 1, "CAL", False, _dr(0, True), _feats(0), ACT))
    _expect(ValueError, lambda: D.run_dev_exploration(extra, require_exact_eval_coverage=True))
    assert D.run_dev_exploration(extra).run_status == D.V4_DEV_EXPLORATION_COMPLETE             # relaxed default


def test_fail_closed_validation():
    good = _make_records(beneficial=True)
    _expect(ValueError, lambda: D.run_dev_exploration([]))
    _expect(ValueError, lambda: D.run_dev_exploration([V4OOFRecord(
        "PD", "x", PD_COH, "b0", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ("identity", "spdim", "t3a"))]))
    _expect(ValueError, lambda: D.run_dev_exploration([V4OOFRecord(
        "PD", "x", "ds999999", "b0", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)]))      # non-DEV cohort
    _expect(ValueError, lambda: D.run_dev_exploration([V4OOFRecord(
        "PD", "x", PD_COH, "b0", 0, "EVAL", False, np.array([np.nan, 0, 0]), np.zeros((A, NF)), ACT)]))
    _expect(ValueError, lambda: D.run_dev_exploration([V4OOFRecord(
        "PD", "x", PD_COH, "b0", 0, "ZZZ", False, np.zeros(A), np.zeros((A, NF)), ACT)]))            # bad split
    # exact-duplicate cell (same disease,cohort,batch,fold,split)
    dup = [V4OOFRecord("PD", "a", PD_COH, "d", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT),
           V4OOFRecord("PD", "b", PD_COH, "d", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)]
    _expect(ValueError, lambda: D.run_dev_exploration(dup))
    # same batch EVAL in two folds (OOF must partition)
    evtwice = [V4OOFRecord("PD", "a", PD_COH, "z", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT),
               V4OOFRecord("PD", "a", PD_COH, "z", 1, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)]
    _expect(ValueError, lambda: D.run_dev_exploration(evtwice))
    _expect(ValueError, lambda: D.run_dev_exploration(good, score_families=()))
    _expect(ValueError, lambda: D.run_dev_exploration(good, config=D.V4DevConfig(g3_comparator="zzz")))


def main():
    print("ACAR v4 develop (Phase-1 exploratory orchestration) guards (synthetic fixtures only):")
    for t in (test_candidate_found_not_select_and_no_binding,
              test_eval_L_harm_all_populated_and_all_batch_denominator, test_no_passer_is_negative,
              test_single_disease_cannot_pass_g6, test_lambda_star_from_cal_not_eval,
              test_cal_records_excluded_from_eval_denominator, test_cohort_aware_subject_key_not_merged,
              test_comparator_slots_distinct_and_g3_uses_configured, test_real_mode_rejects_arbitrary_callable,
              test_global_policy_frontier_dominates_each_config, test_frontier_gap_telescoping_and_info_nonneg,
              test_subject_macro_weighting_used_for_c0, test_fallback_stays_in_denominator,
              test_manifest_lineage_and_permutation_invariant, test_subject_level_crossfit_invariants,
              test_real_mode_requires_explicit_families, test_record_immutability_and_construction_validation,
              test_record_digest_permutation_invariant_and_field_sensitive, test_atomic_writer,
              test_assert_no_binding_language_rejects_illegal, test_cal_records_value_excluded_from_eval_operating_point,
              test_g4_requires_all_eval_folds_certified, test_construction_rejects_non_float_and_control_chars,
              test_record_digest_action_names_injective, test_score_family_registry_predeclared,
              test_exact_oof_eval_coverage, test_fail_closed_validation):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 DEVELOP GUARDS PASS")


if __name__ == "__main__":
    main()
