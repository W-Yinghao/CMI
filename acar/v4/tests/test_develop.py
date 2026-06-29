"""Guards for acar/v4/develop.py (Phase-1 EXPLORATORY DEV orchestration). SYNTHETIC FIXTURES ONLY; NO real DEV cohort,
NO v3 loader, NO real DEV report, NO freeze, NO lockbox. Proves the orchestration wires frontiers + risk_control +
hierarchy into a NON-BINDING exploratory report: never emits SELECT/DEV_STOP/binding/external language; CANDIDATE_FOUND
(not SELECT) when a config passes G0–G6; NEGATIVE when none pass; PD-only/SCZ-only cannot pass G6; subject-macro
weighting (not batch); fallback rows stay in the denominator; gap telescoping is exact; lineage recorded; manifest
digest is permutation-independent; full fail-closed contract.
Run: python -m acar.v4.tests.test_develop
"""
import numpy as np

from acar.config import DISEASE
from acar.v4 import develop as D
from acar.v4.develop import V4OOFRecord

ACT = D.ACTIONS
A = D.A
NF = D.N_FEAT
PD_COH = DISEASE["PD"][0]
SCZ_COH = DISEASE["SCZ"][0]


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                       # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def _feats_for_good(good):
    """Encode the good action with LOW harm/benefit coordinates (used by the default score families); never uses ΔR."""
    f = np.full((A, NF), 5.0)
    for c in (1, 2, 3):                                    # d_margin, flip_rate, js coordinates
        f[good, c] = 0.0
    return f


def _make_records(diseases=("PD", "SCZ"), beneficial=True, fallback_subjects=(), seed=0):
    cohort = {"PD": PD_COH, "SCZ": SCZ_COH}
    recs = []
    for d in diseases:
        for s in range(8):
            sid = f"{d}_s{s}"
            nb = 2 + (s % 3)                               # 2/3/4 batches ⇒ unequal counts
            for b in range(nb):
                good = (s + b) % A
                dr = np.full(A, 1.0)
                if beneficial:
                    dr[good] = -1.0                        # the good action reduces risk
                fb = (sid in fallback_subjects and b == 0)
                recs.append(V4OOFRecord(d, sid, cohort[d], f"{sid}_b{b}", fold=b % 5, split="EVAL",
                                        fallback=bool(fb), dr=dr, features_v2=_feats_for_good(good),
                                        action_names=ACT))
    return recs


# ----------------------------------------------------------------------------- verdicts + no binding language

def test_candidate_found_is_not_select_and_no_binding_language():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    assert res.verdict == D.V4_DEV_CANDIDATE_FOUND
    assert res.run_status == D.V4_DEV_EXPLORATION_COMPLETE
    assert D.assert_no_binding_language(res)               # status/verdict scan + no external/lockbox manifest keys
    assert all(r.status == "EVALUATED" for r in res.reports)
    assert not any(r.status == "SELECT" for r in res.reports)
    # at least one config genuinely passed all G0–G6
    assert any(r.all_pass() for r in res.reports)
    # no forbidden manifest keys
    assert not (set(k.lower() for k in res.manifest) & set(D._FORBIDDEN_MANIFEST_KEYS))


def test_no_passer_is_negative():
    res = D.run_dev_exploration(_make_records(beneficial=False))    # adaptation always harmful
    assert res.verdict == D.V4_DEV_NEGATIVE
    assert not any(r.all_pass() for r in res.reports)


def test_single_disease_cannot_pass_g6():
    res = D.run_dev_exploration(_make_records(diseases=("PD",), beneficial=True))
    assert res.verdict == D.V4_DEV_NEGATIVE                # G6 (both diseases) cannot be satisfied
    assert all(not r.g6_nonvacuous_both_diseases_pass for r in res.reports)
    # PD itself is non-vacuous (the gate fails only on the both-diseases requirement)
    assert any(r.g1_coverage_pass and r.g2_red_pass for r in res.reports)


# ----------------------------------------------------------------------------- subject-macro + fallback

def test_subject_macro_weighting_used_for_c0():
    # PD only, 2 subjects with unequal batch counts and opposite best actions ⇒ subject-macro best-fixed = 0.0 while
    # batch-weighted best-fixed would be +0.5. develop must report the subject-macro value.
    def rec(sid, bid, good):
        dr = np.full(A, 1.0); dr[good] = -1.0
        return V4OOFRecord("PD", sid, PD_COH, bid, 0, "EVAL", False, dr, _feats_for_good(good), ACT)
    recs = [rec("PD_A", "PD_A_b0", 0),
            rec("PD_B", "PD_B_b0", 1), rec("PD_B", "PD_B_b1", 1), rec("PD_B", "PD_B_b2", 1)]
    res = D.run_dev_exploration(recs)
    c0 = res.manifest["diseases"]["PD"]["c0_red"]
    assert abs(c0 - 0.0) < 1e-9, f"subject-macro best-fixed should be 0.0, got {c0}"   # NOT the batch-weighted 0.5


def test_fallback_stays_in_denominator():
    base = D.run_dev_exploration(_make_records(beneficial=True))
    fb = D.run_dev_exploration(_make_records(beneficial=True, fallback_subjects=("PD_s0", "SCZ_s0")))

    def best_cov(res, fam="safe_set", loss="mean"):
        cs = [r.coverage for r in res.reports if r.policy_family == fam and r.loss == loss and r.coverage > 0]
        return max(cs) if cs else 0.0
    assert best_cov(fb) < best_cov(base) - 1e-9            # fallback rows lower coverage ⇒ they stay in the denominator


# ----------------------------------------------------------------------------- frontier gap exactness + lineage

def test_frontier_gap_telescoping_and_info_nonneg():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    r = next(r for r in res.reports if r.all_pass())
    g = r.frontier_gaps
    tele = g["info_gap"] + g["policy_gap"] + g["calibration_gap"]
    assert abs(tele - (g["true_ceiling"] - g["calibrated_red"])) < 1e-9
    assert g["info_gap"] >= -1e-9                          # union score ceiling ≤ true ceiling
    # hierarchy summary present (B0/B1/B2 means computed via Direction-B objects)
    assert set(("b0_mean", "b1_mean", "b2_mean", "b0_minus_b1_mean")) <= set(r.hierarchy_summary)


def test_manifest_lineage_and_boundary():
    res = D.run_dev_exploration(_make_records(beneficial=True))
    lin = res.manifest["lineage"]
    assert lin["v2"] == "MEASUREMENT_ONLY" and "DEV_STOP" in lin["v3"] and lin["v4"] == "NON-BINDING"
    assert "NON-BINDING" in res.manifest["boundary"] and "NO LOCKBOX" in res.manifest["boundary"]


# ----------------------------------------------------------------------------- determinism / permutation invariance

def test_permutation_independent_manifest_and_determinism():
    recs = _make_records(beneficial=True)
    a = D.run_dev_exploration(recs)
    b = D.run_dev_exploration(list(reversed(recs)))
    rng = np.random.default_rng(3); perm = rng.permutation(len(recs))
    c = D.run_dev_exploration([recs[i] for i in perm])
    assert a.manifest_sha256 == b.manifest_sha256 == c.manifest_sha256
    assert a.verdict == b.verdict == c.verdict


# ----------------------------------------------------------------------------- fail-closed

def test_fail_closed_validation():
    good = _make_records(beneficial=True)
    _expect(ValueError, lambda: D.run_dev_exploration([]))                              # empty
    # wrong action set
    bad_act = V4OOFRecord("PD", "x", PD_COH, "x_b0", 0, "EVAL", False, np.zeros(A),
                          np.zeros((A, NF)), ("identity", "spdim", "t3a"))
    _expect(ValueError, lambda: D.run_dev_exploration([bad_act]))
    # non-DEV cohort id (no external/lockbox identifiers)
    ext = V4OOFRecord("PD", "x", "ds999999", "x_b0", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)
    _expect(ValueError, lambda: D.run_dev_exploration([ext]))
    # NaN dr
    nan = V4OOFRecord("PD", "x", PD_COH, "x_b0", 0, "EVAL", False, np.array([np.nan, 0.0, 0.0]),
                      np.zeros((A, NF)), ACT)
    _expect(ValueError, lambda: D.run_dev_exploration([nan]))
    # bad feature shape
    badf = V4OOFRecord("PD", "x", PD_COH, "x_b0", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF - 1)), ACT)
    _expect(ValueError, lambda: D.run_dev_exploration([badf]))
    # bad split / disease / empty id
    _expect(ValueError, lambda: D.run_dev_exploration(
        [V4OOFRecord("PD", "x", PD_COH, "x_b0", 0, "ZZZ", False, np.zeros(A), np.zeros((A, NF)), ACT)]))
    _expect(ValueError, lambda: D.run_dev_exploration(
        [V4OOFRecord("FLU", "x", PD_COH, "x_b0", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)]))
    _expect(ValueError, lambda: D.run_dev_exploration(
        [V4OOFRecord("PD", "", PD_COH, "x_b0", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)]))
    # duplicate (disease, cohort, batch_id)
    dup = [V4OOFRecord("PD", "a", PD_COH, "dup", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT),
           V4OOFRecord("PD", "b", PD_COH, "dup", 0, "EVAL", False, np.zeros(A), np.zeros((A, NF)), ACT)]
    _expect(ValueError, lambda: D.run_dev_exploration(dup))
    # empty score families
    _expect(ValueError, lambda: D.run_dev_exploration(good, score_families=()))


def main():
    print("ACAR v4 develop (Phase-1 exploratory orchestration) guards (synthetic fixtures only):")
    for t in (test_candidate_found_is_not_select_and_no_binding_language, test_no_passer_is_negative,
              test_single_disease_cannot_pass_g6, test_subject_macro_weighting_used_for_c0,
              test_fallback_stays_in_denominator, test_frontier_gap_telescoping_and_info_nonneg,
              test_manifest_lineage_and_boundary, test_permutation_independent_manifest_and_determinism,
              test_fail_closed_validation):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 DEVELOP GUARDS PASS")


if __name__ == "__main__":
    main()
