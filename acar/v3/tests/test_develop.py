"""Guards for acar/v3/splits.py (S5 split-as-one-algorithm) and acar/v3/develop.py (synthetic DEV orchestration).
SYNTHETIC FIXTURES ONLY; NO real DEV cohort, NO binding go/no-go. Proves: deterministic permutation-independent subject
splits; leak-respecting OOF wiring (FIT→predictor, CAL→one-score-per-subject→q, EVAL→route); fallback-only subjects
retained but never fitted; refit eligibility frozen; whole run deterministic + permutation-independent; C0/v2 replay
consumes the identical pool.
Run: python -m acar.v3.tests.test_develop
"""
import os
import tempfile
import numpy as np

from acar.config import N_CLS
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
def _make_dump(path, *, seed=0, d=6, n_full=10):
    rng = np.random.default_rng(seed)
    n_ev = 240; yev = np.tile([0, 1], n_ev // 2).astype(np.int64)
    zev = rng.standard_normal((n_ev, d)) + yev[:, None] * 0.6
    sub, rec, win = [], [], []
    for s in range(n_full):
        for k in range(40):                       # rec-00: 32 + 8 -> two eligible batches
            sub.append(f"sub-{s:03d}"); rec.append("rec-00"); win.append(k)
        for k in range(4):                        # rec-01: 4 -> a fallback batch for an ELIGIBLE subject
            sub.append(f"sub-{s:03d}"); rec.append("rec-01"); win.append(k)
    for k in range(5):                            # fallback-ONLY subject
        sub.append("sub-fb"); rec.append("rec-00"); win.append(k)
    n = len(sub)
    zte = rng.standard_normal((n, d)); yte = rng.integers(0, N_CLS, size=n).astype(np.int64)
    np.savez(path, z_ev=zev, y_ev=yev, z_te=zte, y_te=yte,
             subject_id_te=np.array(sub), recording_id_te=np.array(rec), window_index_te=np.array(win, dtype=np.int64))


def _load(path):
    sa = L.load_source_artifact_from_dump(path, disease="PD")
    batches = L.load_deployment_batches(path, disease="PD", dataset_id="dsX", source_state_ref=sa.source_state_ref)
    labels = L.load_labels_by_window(path, dataset_id="dsX")
    return sa, batches, labels


def test_develop_leak_isolation_and_cal_scores():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "audit_PD_dsX_erm_0.npz"); _make_dump(p)
        sa, batches, labels = _load(p)
        res = D.run_develop("PD", sa, batches, labels, candidate="C1")
        eval_union = set()
        for f in res.folds:
            fit, cal, ev = set(f.fit_subjects), set(f.cal_subjects), set(f.eval_subjects)
            tr, va = set(f.train_subjects), set(f.val_subjects)
            assert not (ev & (fit | cal))                      # EVAL ⟂ FIT∪CAL  (no leakage into diagnostics)
            assert not (fit & cal)                             # predictor ⟂ conformal-q subjects
            assert tr | va == fit and not (tr & va)            # TRAIN∪VAL == FIT, disjoint
            assert f.n_cal_scores == len(cal)                  # EXACTLY one CAL score per (eligible) CAL subject
            eval_union |= ev
        assert eval_union == {canon_subject(s) for s in D._eligible_subjects(D._subject_batches(batches))}  # EVAL once each
        print("  [ok] OOF leak isolation (EVAL⟂FIT∪CAL, FIT⟂CAL, TRAIN∪VAL==FIT); one CAL score per CAL subject; EVAL covers all eligible once")


def test_develop_fallback_and_refit_eligibility():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "audit_PD_dsX_erm_0.npz"); _make_dump(p)
        sa, batches, labels = _load(p)
        res = D.run_develop("PD", sa, batches, labels, candidate="C1")
        assert res.n_fallback_only_subjects == 1 and res.n_eligible_subjects == 10
        fbc = canon_subject(SubjectKey("dsX", "sub-fb"))
        for f in res.folds:                                    # fallback-only subject NEVER in any split role
            for role in (f.fit_subjects, f.cal_subjects, f.eval_subjects, f.train_subjects, f.val_subjects):
                assert fbc not in role
        assert any(f.n_eval_fallback_batches > 0 for f in res.folds)            # eligible subjects' fallback batches retained in EVAL
        # refit consumed EXACTLY the eligible set (frozen inclusion, not residual-based)
        elig = D._eligible_subjects(D._subject_batches(batches))
        assert res.eligible_subject_list_sha256 == L.hash_subject_list(elig)
        assert L._is_hex64(res.refit_artifact_sha256)
        print("  [ok] fallback-only subject retained but never fitted/CAL'd; eligible subjects' fallback batches kept in EVAL; refit eligibility = frozen eligible set")


def test_develop_deterministic_and_permutation_independent():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "audit_PD_dsX_erm_0.npz"); _make_dump(p)
        sa, batches, labels = _load(p)
        r1 = D.run_develop("PD", sa, batches, labels, candidate="C1")
        r2 = D.run_develop("PD", sa, batches, labels, candidate="C1")
        assert r1.refit_artifact_sha256 == r2.refit_artifact_sha256 and r1.final_epochs == r2.final_epochs
        assert [f.q for f in r1.folds] == [f.q for f in r2.folds]
        rng = np.random.default_rng(3); perm = list(batches); rng.shuffle(perm)
        r3 = D.run_develop("PD", sa, perm, labels, candidate="C1")
        assert r3.refit_artifact_sha256 == r1.refit_artifact_sha256             # whole run permutation-independent
        assert r3.pool_digest == r1.pool_digest == D.replay_pool_digest(batches)  # C0/v2 replay consumes identical pool
        assert [f.best_epoch for f in r3.folds] == [f.best_epoch for f in r1.folds]
        print("  [ok] whole DEV run deterministic + permutation-independent (refit hash/q/epochs stable); C0/v2 replay shares the pool digest")


def main():
    print("ACAR v3 split + DEV-orchestration guards (synthetic fixtures only):")
    for t in (test_split_partition_balance_determinism, test_split_permutation_independent,
              test_develop_leak_isolation_and_cal_scores, test_develop_fallback_and_refit_eligibility,
              test_develop_deterministic_and_permutation_independent):
        t()
    print("ALL V3 DEVELOP/SPLIT GUARDS PASS")


if __name__ == "__main__":
    main()
