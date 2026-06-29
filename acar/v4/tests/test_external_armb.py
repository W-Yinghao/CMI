"""Guards for acar/v4/external_adapter.py + acar/v4/run_external_armb.py. SYNTHETIC FIXTURES ONLY; NO real cohort, NO v3
loader, NO external signal read. Proves: site-local split (NOT_EVALUABLE on too-few / missing-class; deterministic);
the criterion-A endpoint (CONFIRMED / NEGATIVE / NOT_EVALUABLE); EVAL L_harm_all separate from harm_among_adapted;
λ* from CAL only (EVAL labels do not move it); apples-to-apples v2_replay (subject-disjoint C0_FIT/C0_CAL; NOT_EVALUABLE
when empty); deterministic multi-site taxonomy (single-site flag; NOT_EVALUABLE never dropped); the CLI manifest
validation (admissible strata only; ASZED/ds007020/DEV rejected) and preflight fail-closed (output exists; missing tag).
Run: python -m acar.v4.tests.test_external_armb
"""
import copy
import json
import math
import os
import subprocess
import tempfile

import numpy as np

from acar.v4 import external_adapter as EA
from acar.v4 import run_external_armb as CLI

A = EA.A
NF = EA.N_FEAT


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                       # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def _stratum(site="zenodo14808296", disease="SCZ", n=60, harmful_frac=0.05, good_dr=-2.0, seed=0):
    """Action 0 has a low (varied) d_margin so the policy is drawn to adapt it (and the λ grid is non-degenerate). With
    good_dr=−2 it is genuinely beneficial (harmful_frac of +0.5 noise); with good_dr=+1 it is a HARMFUL TRAP (low
    d_margin but ΔR>0) — the frozen candidate should then refuse / fall below the coverage floor → NEGATIVE."""
    rng = np.random.default_rng(seed)
    subjects = {}
    for i in range(n):
        cc = f"{site}::s{i:03d}"
        elig = []
        for b in range(4):
            dr = np.full(A, 1.0)
            dr[0] = 0.5 if rng.random() < harmful_frac else good_dr
            feats = np.full((A, NF), 5.0)
            feats[0, 1] = rng.uniform(-6.0, -4.0)            # action 0: low (varied) d_margin
            elig.append((f"s{i:03d}_b{b}", dr, feats))
        subjects[cc] = {"class": "patient" if i % 2 == 0 else "hc", "eligible": elig, "fallback": []}
    return {"site": site, "disease": disease, "subjects": subjects}


def _set_dr(stratum, subjects, fn):
    st = copy.deepcopy(stratum)
    for cc in subjects:
        st["subjects"][cc]["eligible"] = [(bid, fn(np.array(dr)), feats)
                                          for (bid, dr, feats) in st["subjects"][cc]["eligible"]]
    return st


# ----------------------------------------------------------------------------- split

def test_site_local_split():
    s2c = {f"x::s{i}": ("patient" if i % 2 == 0 else "hc") for i in range(60)}
    cal, ev, stat, _ = EA.site_local_split(s2c)
    assert stat == "OK" and len(cal) >= 20 and len(ev) >= 20 and not (cal & ev)
    cal2, ev2, stat2, _ = EA.site_local_split(s2c)
    assert cal == cal2 and ev == ev2                         # deterministic
    # too few subjects
    assert EA.site_local_split({f"x::s{i}": "patient" if i % 2 else "hc" for i in range(10)})[2] == "NOT_EVALUABLE"
    # missing a class entirely
    assert EA.site_local_split({f"x::s{i}": "patient" for i in range(60)})[2] == "NOT_EVALUABLE"


# ----------------------------------------------------------------------------- endpoint (criterion A)

def test_evaluate_stratum_confirmed():
    r = EA.evaluate_stratum(_stratum(harmful_frac=0.05))
    assert r.status == "V4_EXTERNAL_CONFIRMED", (r.status, r.reason)
    assert r.L_harm_all_eval <= EA.BUDGET and r.red > 0 and r.red > r.v2_replay_red and r.coverage >= 0.15
    assert r.v2_replay_status == "OK"
    assert r.selected_lambda is not None


def test_evaluate_stratum_negative_harmful_trap():
    # action 0 looks good (low d_margin) but is harmful (ΔR=+1): the candidate cannot beat v2 / clear the floors
    r = EA.evaluate_stratum(_stratum(good_dr=1.0, harmful_frac=0.0))
    assert r.status == "V4_EXTERNAL_NEGATIVE", (r.status, r.reason)


def test_evaluate_stratum_not_evaluable_too_few():
    r = EA.evaluate_stratum(_stratum(n=10))
    assert r.status == "NOT_EVALUABLE" and "too few" in r.reason.lower()


def test_L_harm_all_vs_harm_among_adapted():
    r = EA.evaluate_stratum(_stratum(harmful_frac=0.05))
    # L_harm_all = coverage * harm_among_adapted ≤ harm_among_adapted (distinct quantities)
    assert r.L_harm_all_eval <= r.harm_among_adapted + 1e-9
    assert math.isfinite(r.L_harm_all_eval) and math.isfinite(r.harm_among_adapted)


def test_lambda_star_from_cal_not_eval():
    st = _stratum(harmful_frac=0.05)
    base = EA.evaluate_stratum(st)
    cal, ev, _, _ = EA.site_local_split({cc: c["class"] for cc, c in st["subjects"].items()})
    # perturb ONLY EVAL ΔR → λ* must NOT change (λ* comes from CAL)
    st_eval = _set_dr(st, ev, lambda dr: dr + 7.0)
    assert EA.evaluate_stratum(st_eval).selected_lambda == base.selected_lambda
    # perturb CAL ΔR (make all harmful) → λ* must change (CAL drives the calibration), or LTT no longer certifies
    st_cal = _set_dr(st, cal, lambda dr: np.full(A, 1.0))
    cal_res = EA.evaluate_stratum(st_cal)
    assert cal_res.selected_lambda is None or abs(cal_res.selected_lambda - base.selected_lambda) > 1e-9


def test_v2_replay_subject_disjoint_and_not_evaluable():
    st = _stratum()
    cal = sorted(EA.site_local_split({cc: c["class"] for cc, c in st["subjects"].items()})[0])
    c0fit, c0cal = EA._secondary_split(cal)
    assert c0fit and c0cal and not (c0fit & c0cal)            # subject-disjoint
    # empty C0_FIT → NOT_EVALUABLE
    assert EA._v2_replay_red(st, set(), c0cal, cal)[1] == "NOT_EVALUABLE"


# ----------------------------------------------------------------------------- taxonomy

def _sr(site, disease, status):
    return EA.StratumResult(site, disease, status, "", 30, 30, 0.0, 0.5, 0.2, 0.03, 0.15, 0.0, "OK")


def test_external_taxonomy_deterministic():
    ext = EA.external_taxonomy([_sr("ds007526", "PD", "V4_EXTERNAL_CONFIRMED"),
                                _sr("zenodo14808296", "SCZ", "V4_EXTERNAL_CONFIRMED")])
    assert ext.verdict == "V4_EXTERNAL_CONFIRMED"
    assert ext.per_disease["PD"]["confirmed"] and ext.per_disease["PD"]["single_site"]
    # one NEGATIVE in a disease ⇒ that disease not confirmed ⇒ overall NEGATIVE
    ext2 = EA.external_taxonomy([_sr("ds007526", "PD", "V4_EXTERNAL_NEGATIVE"),
                                 _sr("zenodo14808296", "SCZ", "V4_EXTERNAL_CONFIRMED")])
    assert ext2.verdict == "V4_EXTERNAL_NEGATIVE" and not ext2.per_disease["PD"]["confirmed"]
    # NOT_EVALUABLE stratum is listed, not dropped; a disease with no evaluable stratum is not confirmed
    ext3 = EA.external_taxonomy([_sr("ds007526", "PD", "NOT_EVALUABLE"),
                                 _sr("zenodo14808296", "SCZ", "V4_EXTERNAL_CONFIRMED")])
    assert ext3.per_disease["PD"]["n_not_evaluable"] == 1 and not ext3.per_disease["PD"]["confirmed"]
    assert len(ext3.per_disease["PD"]["strata"]) == 1


# ----------------------------------------------------------------------------- CLI manifest validation + preflight

def test_admissible_strata_constant():
    assert CLI.ADMISSIBLE_STRATA == {("zenodo14808296", "SCZ"), ("ds007526", "PD")}


def _full_st(site, disease, **over):
    st = {"site": site, "disease": disease, "dump_path": f"/{site}.npz",
          "dev_source_artifact_path": f"/{site}.dev.npz", "dataset_version": "1.0.0",
          "source_state_ref": "a" * 64, "expected_n_subjects": 77, "expected_embedding_dim": 16}
    for hf in CLI._HASH_FIELDS:
        st[hf] = "b" * 64
    st.update(over)
    return st


def _manifest(strata, commit="0" * 40):
    return {"protocol_commit": commit, "strata": strata}


def test_validate_external_manifest_exact_set_and_provenance():
    exact = [_full_st("zenodo14808296", "SCZ"), _full_st("ds007526", "PD")]
    assert len(CLI.validate_external_manifest(_manifest(exact))) == 2          # the exact admissible set passes
    # NOT a subset: a single admissible stratum is rejected (both required)
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest([_full_st("ds007526", "PD")])))
    # rejected / excluded / DEV / wrong-pair sites
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest(exact + [_full_st("14178398", "SCZ")])))
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest([_full_st("ds007020", "PD"), exact[0]])))
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest([_full_st("ds002778", "PD"), exact[0]])))
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest([_full_st("ds007526", "SCZ"), exact[0]])))
    # bad commit / missing provenance field / bad hash / bad count
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest(exact, commit="bad")))
    bad = [_full_st("zenodo14808296", "SCZ"), _full_st("ds007526", "PD")]; del bad[1]["full_dump_sha256"]
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest(bad)))
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest(
        [_full_st("zenodo14808296", "SCZ"), _full_st("ds007526", "PD", raw_pipeline_sha256="short")])))
    _expect(ValueError, lambda: CLI.validate_external_manifest(_manifest(
        [_full_st("zenodo14808296", "SCZ"), _full_st("ds007526", "PD", expected_n_subjects=0)])))


def test_preflight_fail_closed():
    root = CLI._repo_root()
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    _expect(ValueError, lambda: CLI.verify_protocol(root, "0" * 40))           # HEAD != commit
    _expect(ValueError, lambda: CLI.verify_protocol(root, head))               # tag acar-v4-protocol absent
    base = tempfile.mkdtemp()
    try:
        mpath = os.path.join(base, "m.json")
        with open(mpath, "w") as f:
            json.dump(_manifest([_full_st("ds007526", "PD")]), f)
        outdir = os.path.join(base, "out"); os.mkdir(outdir)
        _expect(FileExistsError, lambda: CLI.run(mpath, outdir))               # output exists → before git/heavy
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


def test_build_stratum_requires_frozen_artifact():
    # leakage firewall: the real path REFUSES to run without a DEV-frozen source artifact (before any heavy import)
    _expect(ValueError, lambda: EA.build_stratum_from_dump("ds007526", "PD", "/nope.npz", None))


def test_subject_class_fail_closed():
    assert EA.subject_class_from_label_values([1, 1, 1]) == "patient"
    assert EA.subject_class_from_label_values([0, 0]) == "hc"
    _expect(ValueError, lambda: EA.subject_class_from_label_values([0, 1]))    # mixed
    _expect(ValueError, lambda: EA.subject_class_from_label_values([]))        # missing
    _expect(ValueError, lambda: EA.subject_class_from_label_values([2]))       # out of range


def test_json_safe_not_evaluable():
    r = EA.evaluate_stratum(_stratum(n=10))                                    # NOT_EVALUABLE (too few)
    safe = CLI._json_safe(r.__dict__)
    s = json.dumps(safe, allow_nan=False)                                      # must NOT raise on NaN
    assert "NaN" not in s and "Infinity" not in s
    assert safe["status"] == "NOT_EVALUABLE" and safe["L_harm_all_eval"] is None and safe["coverage"] is None


def test_lambda_grid_excludes_fallback():
    # action 0's d_margin is CONSTANT over non-fallback CAL → degenerate grid → NOT_EVALUABLE. If fallback rows (zeros)
    # were (wrongly) included, the grid would span [-5, 0] and NOT be degenerate. So this proves fallback is excluded.
    subjects = {}
    for i in range(60):
        elig = [(f"s{i}_b{b}", np.array([-2.0, 1.0, 1.0]),
                 np.array([[0, -5.0] + [0] * (NF - 2), [0, 5.0] + [0] * (NF - 2), [0, 5.0] + [0] * (NF - 2)], float))
                for b in range(2)]
        subjects[f"x::s{i}"] = {"class": "patient" if i % 2 == 0 else "hc", "eligible": elig, "fallback": [f"s{i}_fb"]}
    r = EA.evaluate_stratum({"site": "zenodo14808296", "disease": "SCZ", "subjects": subjects})
    assert r.status == "NOT_EVALUABLE" and "degenerate" in r.reason.lower()


def main():
    print("ACAR v4 external Arm-B (adapter + CLI) guards (synthetic fixtures only):")
    for t in (test_site_local_split, test_evaluate_stratum_confirmed, test_evaluate_stratum_negative_harmful_trap,
              test_evaluate_stratum_not_evaluable_too_few, test_L_harm_all_vs_harm_among_adapted,
              test_lambda_star_from_cal_not_eval, test_v2_replay_subject_disjoint_and_not_evaluable,
              test_external_taxonomy_deterministic, test_admissible_strata_constant,
              test_validate_external_manifest_exact_set_and_provenance, test_preflight_fail_closed,
              test_build_stratum_requires_frozen_artifact, test_subject_class_fail_closed, test_json_safe_not_evaluable,
              test_lambda_grid_excludes_fallback):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 EXTERNAL ARM-B GUARDS PASS")


if __name__ == "__main__":
    main()
