"""Guards for acar/v4/real_adapter.py PURE record-emission core (synthetic fixtures only; NO real cohort, NO v3 loader,
NO real data read). The v3-coupled derivation (build_cohort_inputs/derive) is the same path the v3 DEV run used and is
exercised only by the real run (which fails closed via exact OOF coverage). Here we prove the split→role mapping,
fallback emission, and that emitted records form a valid EXACT-OOF cross-fit that the orchestrator accepts.
Run: python -m acar.v4.tests.test_real_adapter
"""
import numpy as np

from acar.config import DISEASE
from acar.v4 import real_adapter as RA
from acar.v4 import develop as D

PD_COH = DISEASE["PD"][0]
SCZ_COH = DISEASE["SCZ"][0]
A = D.A
NF = D.N_FEAT


def test_fold_roles_precedence_and_skip():
    asg = [{"fold": 0, "eval": {"a"}, "cal": {"b"}, "fit": {"c"}},
           {"fold": 1, "eval": {"b"}, "cal": {"a", "x"}, "fit": {"c"}}]
    fr = RA._fold_roles(asg)
    assert fr[0] == (0, {"a": "EVAL", "b": "CAL", "c": "FIT"})
    f1 = fr[1][1]
    assert f1["b"] == "EVAL" and f1["a"] == "CAL" and f1["x"] == "CAL" and f1["c"] == "FIT"
    # EVAL precedence if a subject appears in more than one set for a fold
    assert RA._fold_roles([{"fold": 0, "eval": {"a"}, "cal": {"a"}, "fit": {"a"}}])[0][1]["a"] == "EVAL"
    # a subject in NO set for a fold is omitted
    assert "z" not in RA._fold_roles([{"fold": 0, "eval": {"a"}, "cal": set(), "fit": set()}])[0][1]


def test_emit_fallback_record():
    cells = {"PD::z": {"dataset": PD_COH, "subject": "z", "eligible": [], "fallback": ["z_b0"]}}
    recs = RA._emit_records("PD", [(0, {"PD::z": "EVAL"})], cells)
    assert len(recs) == 1
    r = recs[0]
    assert r.fallback is True and r.split == "EVAL" and r.cohort_id == PD_COH and r.subject_id == "z"
    assert float(np.sum(np.abs(r.dr))) == 0.0 and float(np.sum(np.abs(r.features_v2))) == 0.0


def _synth_cells(disease, subs):
    coh = DISEASE[disease][0]
    cells = {}
    for i, s in enumerate(subs):
        elig = []
        for b in range(2):
            g = (i + b) % A
            dr = np.full(A, 1.0); dr[g] = -1.0
            feats = np.full((A, NF), 5.0)
            for c in (1, 2, 3):
                feats[g, c] = 0.0
            elig.append((f"{s}_b{b}", dr, feats))
        cells[f"{disease}::{s}"] = {"dataset": coh, "subject": s, "eligible": elig, "fallback": []}
    return cells


def test_emit_records_form_valid_exact_oof_crossfit():
    records = []
    for disease in ("PD", "SCZ"):
        subs = [f"{disease}_s{i}" for i in range(4)]
        cells = _synth_cells(disease, subs)
        e0 = {f"{disease}::{subs[i]}" for i in (0, 1)}
        e1 = {f"{disease}::{subs[i]}" for i in (2, 3)}
        asg = [{"fold": 0, "eval": e0, "cal": e1, "fit": set()},
               {"fold": 1, "eval": e1, "cal": e0, "fit": set()}]
        records += RA._emit_records(disease, RA._fold_roles(asg), cells)
    # each subject EVAL exactly once + each batch EVAL exactly once ⇒ exact OOF coverage accepted; run completes
    res = D.run_dev_exploration(records, require_exact_eval_coverage=True)
    assert res.run_status == D.V4_DEV_EXPLORATION_COMPLETE
    assert res.verdict in (D.V4_DEV_CANDIDATE_FOUND, D.V4_DEV_NEGATIVE)
    # the cross-fit emitted EVAL + CAL for every subject (EVAL once, CAL once)
    pd = res.manifest["diseases"]["PD"]
    assert pd["n_eval_subjects"] == 4 and pd["n_cal_subjects"] == 4


def main():
    print("ACAR v4 real_adapter pure-core guards (synthetic fixtures only):")
    for t in (test_fold_roles_precedence_and_skip, test_emit_fallback_record,
              test_emit_records_form_valid_exact_oof_crossfit):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 REAL-ADAPTER GUARDS PASS")


if __name__ == "__main__":
    main()
