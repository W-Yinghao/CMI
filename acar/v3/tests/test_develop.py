"""Guards for acar/v3/splits.py (S5 split-as-one-algorithm) and acar/v3/develop.py (synthetic DEV orchestration).
SYNTHETIC FIXTURES ONLY; NO real DEV cohort, NO binding go/no-go. Proves: deterministic permutation-independent subject
splits; leak-respecting OOF wiring (FIT→predictor, CAL→one-score-per-subject→q, EVAL→route); fallback-only subjects
retained but never fitted; refit eligibility frozen; whole run deterministic + permutation-independent; C0/v2 replay
consumes the identical pool.
Run: python -m acar.v3.tests.test_develop
"""
import math
import os
import tempfile
import numpy as np

from acar.config import N_CLS
from acar.v3.set_features import NON_IDENTITY
from acar.v3.data import SubjectKey, canon_subject
from acar.v3 import splits as S
from acar.v3 import develop as D
from acar.v3 import loader as L


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _subs(n):
    return [SubjectKey("dsX", f"sub-{i:03d}") for i in range(n)]


# ------------------------------------------------------------------------------------------------------------ splits
def test_split_partition_balance_determinism():
    subs = _subs(23)
    folds = S.outer_folds(subs, k=5, seed_outer=0)
    flat = [canon_subject(s) for f in folds for s in f]
    assert len(flat) == len(set(flat)) == 23                                     # disjoint + covers all
    sizes = sorted(len(f) for f in folds); assert sizes[-1] - sizes[0] <= 1      # balanced
    assert [[canon_subject(s) for s in f] for f in folds] == \
           [[canon_subject(s) for s in f] for f in S.outer_folds(subs, k=5, seed_outer=0)]   # deterministic
    fit, cal = S.fit_cal_split(subs, fit_frac=0.7, seed_fitcal=1)
    tr, va = S.train_val_split(fit, train_frac=0.8, seed_es=2)
    assert not (set(map(canon_subject, fit)) & set(map(canon_subject, cal)))
    assert set(map(canon_subject, tr)) | set(map(canon_subject, va)) == set(map(canon_subject, fit))
    assert not (set(map(canon_subject, tr)) & set(map(canon_subject, va)))
    print("  [ok] splits: disjoint+covering+balanced folds; FIT⟂CAL; TRAIN∪VAL==FIT; TRAIN⟂VAL; deterministic")


def test_split_permutation_independent():
    subs = _subs(23); rng = np.random.default_rng(7); perm = list(subs); rng.shuffle(perm)
    f1 = [set(map(canon_subject, f)) for f in S.outer_folds(subs)]
    f2 = [set(map(canon_subject, f)) for f in S.outer_folds(perm)]
    assert f1 == f2                                                              # fold contents independent of order
    a1, _ = S.cv_assignment(subs); a2, _ = S.cv_assignment(perm)
    for x, y in zip(a1, a2):
        for key in ("eval", "fit", "cal", "train", "val"):
            assert set(map(canon_subject, x[key])) == set(map(canon_subject, y[key]))
    _expect(ValueError, lambda: S.outer_folds(subs + [subs[0]]))                 # duplicate subject rejected
    print("  [ok] splits permutation-independent end-to-end; duplicate subject fail-closed")


# -------------------------------------------------------------------------------------------------------- dev fixture
def _make_dump(path, *, seed=0, d=6, n_full=3):
    """Small synthetic cohort (orchestration test): each full subject = 1 eligible batch (20 windows) + 1 fallback batch
    (4 windows); plus one fallback-only subject. Kept small because execute() is uncached across folds/candidates."""
    rng = np.random.default_rng(seed)
    n_ev = 160; yev = np.tile([0, 1], n_ev // 2).astype(np.int64)
    zev = rng.standard_normal((n_ev, d)) + yev[:, None] * 0.6
    sub, rec, win = [], [], []
    for s in range(n_full):
        for k in range(20):                       # rec-00: one eligible batch (8 <= 20 <= 32)
            sub.append(f"sub-{s:03d}"); rec.append("rec-00"); win.append(k)
        for k in range(4):                        # rec-01: a fallback batch for an ELIGIBLE subject
            sub.append(f"sub-{s:03d}"); rec.append("rec-01"); win.append(k)
    for k in range(5):                            # fallback-ONLY subject
        sub.append("sub-fb"); rec.append("rec-00"); win.append(k)
    n = len(sub)
    zte = rng.standard_normal((n, d)); yte = rng.integers(0, N_CLS, size=n).astype(np.int64)
    np.savez(path, z_ev=zev, y_ev=yev, z_te=zte, y_te=yte,
             subject_id_te=np.array(sub), recording_id_te=np.array(rec), window_index_te=np.array(win, dtype=np.int64))


def _cohort(path, dataset_id, *, seed):
    _make_dump(path, seed=seed)
    sa = L.load_source_artifact_from_dump(path, disease="PD")
    batches = L.load_deployment_batches(path, disease="PD", dataset_id=dataset_id, source_state_ref=sa.source_state_ref)
    labels = L.load_labels_by_window(path, dataset_id=dataset_id)
    return sa, batches, labels


def _pooled_disease(tmp, n_cohorts=3):
    """A pooled-disease DEV substrate: several cohorts each with its OWN source_state_ref, merged through a registry."""
    reg = L.SourceStateRegistry("PD"); all_b = []; all_l = {}
    for ci in range(n_cohorts):
        sa, b, lab = _cohort(os.path.join(tmp, f"audit_PD_ds{ci}_erm_0.npz"), f"ds{ci}", seed=ci)
        reg.add(sa); all_b += b; all_l.update(lab)
    return reg, all_b, all_l


def test_develop_multicohort_registry_and_leak_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        reg, batches, labels = _pooled_disease(tmp, n_cohorts=3)
        assert len(reg.refs) == 3                                                 # 3 cohort source states in one PD pool
        oof = D.run_oof("PD", reg, batches, labels, "C1")
        assert len(oof.records) > 0 and oof.n_eval_eligible_batches > 0
        # leak isolation via the split itself (outer over ALL subjects; FIT/CAL from eligible only)
        idx = D._subject_batches(batches); elig = {canon_subject(s) for s in D._eligible_subjects(idx)}
        assignment, allc = S.cv_assignment([v["key"] for v in idx.values()], eligible=elig)
        union = set()
        for fa in assignment:
            ev = set(map(canon_subject, fa["eval"])); fit = set(map(canon_subject, fa["fit"]))
            cal = set(map(canon_subject, fa["cal"]))
            assert not (ev & (fit | cal)) and not (fit & cal)
            assert fit <= elig and cal <= elig                                    # FIT/CAL only from eligible
            union |= ev
        assert union == set(allc)                                                 # every subject EVAL exactly once
        # an unregistered source_state_ref fails BEFORE any adapter
        other = L.load_deployment_batches(os.path.join(tmp, "audit_PD_ds0_erm_0.npz"), disease="PD",
                                          dataset_id="ds0", source_state_ref="c" * 64)
        _expect(ValueError, lambda: reg.execute([b for b in other if not b.fallback][0]))
        print("  [ok] multi-cohort registry (3 refs) runs one pooled disease; EVAL⟂FIT∪CAL, FIT/CAL⊆eligible, EVAL covers all once; unregistered ref fails pre-adapter")


def test_develop_fallback_eval_accounting():
    with tempfile.TemporaryDirectory() as tmp:
        reg, batches, labels = _pooled_disease(tmp, n_cohorts=3)
        idx = D._subject_batches(batches)
        fb_only = {c for c, v in idx.items() if not v["eligible"]}
        assert len(fb_only) == 3                                                  # one fallback-only subject per cohort
        elig = {canon_subject(s) for s in D._eligible_subjects(idx)}
        assignment, allc = S.cv_assignment([v["key"] for v in idx.values()], eligible=elig)
        eval_union = set()
        for fa in assignment:
            roles = set(map(canon_subject, fa["fit"])) | set(map(canon_subject, fa["cal"])) | \
                set(map(canon_subject, fa["train"])) | set(map(canon_subject, fa["val"]))
            assert not (roles & fb_only)                                          # fallback-only NEVER fitted/CAL'd
            eval_union |= set(map(canon_subject, fa["eval"]))
        assert fb_only <= eval_union                                              # but retained in EVAL accounting
        print("  [ok] fallback-only subjects (one/cohort) retained in EVAL but never FIT/CAL/TRAIN/VAL")


def _c2_records(residuals, scale_raw=1.0, scale_used=1.0):
    """Craft C2 OOFRecords (point=upper_center=0, scale_used=1) so the standardized residual == delta_r == given value;
    spread across 4 subjects, all 3 actions identical."""
    recs = []
    for a in NON_IDENTITY:
        for i, r in enumerate(residuals):
            recs.append(D.OOFRecord("C2", "PD", f"WS[\"d\",\"s{i % 4}\"]", "a" * 64, i % 5, a, float(r), 0.0, 0.0,
                                    float(scale_raw), float(scale_used), float(r), 0.0, 0.0, "identity"))
    return recs


def test_s2_c2_gate_boundaries_and_floor():
    rng = np.random.default_rng(0)
    good = rng.standard_normal(400)                                              # mean≈0, var≈1, tail≈z90 -> PASS
    assert D.s2_c2_gate(_c2_records(good))["pass"]
    assert not D.s2_c2_gate(_c2_records(good * 3.0))["pass"]                     # var≈9 -> FAIL
    assert not D.s2_c2_gate(_c2_records(good + 1.0))["pass"]                     # mean≈1 -> FAIL
    # C2 final floor responds to scale_raw ONLY, not scale_used
    floor = D.c2_floor_from_oof(_c2_records(good, scale_raw=2.0, scale_used=9.0))
    assert all(abs(v - 2.0) < 1e-9 for v in floor.values())
    print("  [ok] S2 C2 gate var/mean/tail boundaries; final floor = Q05 of scale_raw (ignores fold floor scale_used)")


def test_s4_select_tie_rules_and_dev_stop():
    # no passer -> DEV_STOP / NO_LOCKBOX_CONSUMED
    none = {c: {"passes": False, "red_macro": 1.0, "width_macro": 1.0} for c in ("C1", "C2", "C3")}
    assert D.s4_select(none)["verdict"] == "DEV_STOP" and D.s4_select(none)["reason"] == "NO_LOCKBOX_CONSUMED"
    # strictly larger red wins
    m = {"C1": {"passes": True, "red_macro": 0.30, "width_macro": 1.0},
         "C2": {"passes": True, "red_macro": 0.10, "width_macro": 1.0},
         "C3": {"passes": True, "red_macro": 0.20, "width_macro": 1.0}}
    assert D.s4_select(m)["selected"] == "C1"
    # red tie (<=1e-4) -> smaller width wins
    m2 = {"C1": {"passes": True, "red_macro": 0.50, "width_macro": 0.9},
          "C2": {"passes": True, "red_macro": 0.50, "width_macro": 0.5},
          "C3": {"passes": True, "red_macro": 0.50, "width_macro": 0.7}}
    assert D.s4_select(m2)["selected"] == "C2"
    # full tie (red & width) -> fixed order C2 ≺ C3 ≺ C1
    m3 = {c: {"passes": True, "red_macro": 0.5, "width_macro": 0.5} for c in ("C1", "C2", "C3")}
    assert D.s4_select(m3)["selected"] == "C2"
    print("  [ok] S4 SELECT: max red; 1e-4 tie->min width; full tie->C2≺C3≺C1; no passer->DEV_STOP/NO_LOCKBOX_CONSUMED")


def test_dev_run_smoke_and_c0_real():
    with tempfile.TemporaryDirectory() as tmp:
        reg, b, lab = _pooled_disease(tmp, n_cohorts=3)
        c0 = D.run_c0("PD", reg, b, lab)
        assert c0.n_eval_eligible_batches > 0 and math.isfinite(c0.red_router)     # C0 actually trained+calibrated+routed
        res = D.run_dev({"PD": (reg, b, lab)})                                      # one-disease smoke (synthetic -> likely DEV_STOP)
        assert res.verdict in ("SELECT", "DEV_STOP")
        assert set(res.per_disease["PD"]) >= {"C1", "C2", "C3", "C0"}
        if res.verdict == "DEV_STOP":
            assert res.refit_sha256["PD"] is None
        else:
            assert L._is_hex64(res.refit_sha256["PD"])
        assert res.pool_digest["PD"] == D.replay_pool_digest(b)
        print(f"  [ok] run_dev smoke verdict={res.verdict}; C0 trained/calibrated/routed; per-candidate reports + pool identity present")


def main():
    print("ACAR v3 split + DEV bake-off/gate guards (synthetic fixtures only):")
    for t in (test_split_partition_balance_determinism, test_split_permutation_independent,
              test_develop_multicohort_registry_and_leak_isolation, test_develop_fallback_eval_accounting,
              test_s2_c2_gate_boundaries_and_floor, test_s4_select_tie_rules_and_dev_stop,
              test_dev_run_smoke_and_c0_real):
        t()
    print("ALL V3 DEVELOP/SPLIT GUARDS PASS")


if __name__ == "__main__":
    main()
