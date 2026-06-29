"""Guards for acar/v4/develop.py (Phase-1 EXPLORATORY DEV orchestration). SYNTHETIC FIXTURES ONLY; NO real DEV cohort,
NO v3 loader, NO real DEV report, NO freeze, NO lockbox. Covers the split/provenance hardening: FOLD-LOCAL CAL→EVAL
(λ* from CAL only; EVAL never lets CAL into its denominator); COHORT-AWARE subject key (same local id in two cohorts =
two subjects); comparator contract (best_fixed + v2_replay distinct slots, G3 comparator fixed in config); score-family
registry (real_mode rejects arbitrary callables); per-config vs GLOBAL policy-frontier gap; plus the taxonomy/no-binding,
subject-macro, fallback-denominator, gap-telescoping, G6 both-diseases, lineage, permutation-invariance, fail-closed.
Run: python -m acar.v4.tests.test_develop
"""
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
    for t in (test_candidate_found_not_select_and_no_binding, test_no_passer_is_negative,
              test_single_disease_cannot_pass_g6, test_lambda_star_from_cal_not_eval,
              test_cal_records_excluded_from_eval_denominator, test_cohort_aware_subject_key_not_merged,
              test_comparator_slots_distinct_and_g3_uses_configured, test_real_mode_rejects_arbitrary_callable,
              test_global_policy_frontier_dominates_each_config, test_frontier_gap_telescoping_and_info_nonneg,
              test_subject_macro_weighting_used_for_c0, test_fallback_stays_in_denominator,
              test_manifest_lineage_and_permutation_invariant, test_fail_closed_validation):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 DEVELOP GUARDS PASS")


if __name__ == "__main__":
    main()
