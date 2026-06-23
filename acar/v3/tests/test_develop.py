"""Guards for acar/v3/splits.py (S5 split-as-one-algorithm) and acar/v3/develop.py (synthetic DEV orchestration).
SYNTHETIC FIXTURES ONLY; NO real DEV cohort, NO binding go/no-go. Proves: deterministic permutation-independent subject
splits; leak-respecting OOF wiring (FIT→predictor, CAL→one-score-per-subject→q, EVAL→route); fallback-only subjects
retained but never fitted; refit eligibility frozen; whole run deterministic + permutation-independent; C0/v2 replay
consumes the identical pool.
Run: python -m acar.v3.tests.test_develop
"""
import json
import math
import os
import tempfile
import numpy as np
import torch                                    # set inter-op=1 BEFORE acar imports so the env lock verifies in-process
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

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


def _pool_with_ids(tmp, disease, dataset_ids, seed0=0):
    """A pooled disease whose cohort dataset IDs are EXACTLY `dataset_ids` (one source-state ref each). Synthetic data
    only — the IDs are strings used to exercise the binding cohort check; no real DEV value is read."""
    reg = L.SourceStateRegistry(disease); allb = []; alll = {}
    for i, ds in enumerate(dataset_ids):
        p = os.path.join(tmp, f"audit_{disease}_{ds}_erm_0.npz"); _make_dump(p, seed=seed0 + i)
        sa = L.load_source_artifact_from_dump(p, disease=disease); reg.add(sa)
        allb += L.load_deployment_batches(p, disease=disease, dataset_id=ds, source_state_ref=sa.source_state_ref)
        alll.update(L.load_labels_by_window(p, dataset_id=ds))
    return reg, allb, alll


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


def _ok_metric(**over):
    m = dict(s2_pass=True, dominance_pass=True, pd_auroc=0.70, scz_mae=0.5, c0_scz_mae=0.5,
             width_macro=0.5, c0_width_macro=1.0, coverage_macro=0.50, red_macro=0.30, c0_red_macro=0.10,
             any_q_inf=False)
    m.update(over); return m


def test_s4_eligible_each_criterion_gates():
    assert D.s4_eligible(_ok_metric())["eligible"]                                  # baseline passes
    assert not D.s4_eligible(_ok_metric(red_macro=-0.01))["eligible"]               # negative red
    assert not D.s4_eligible(_ok_metric(red_macro=0.05, c0_red_macro=0.10))["eligible"]   # red below C0
    assert not D.s4_eligible(_ok_metric(coverage_macro=0.14))["eligible"]           # coverage < 0.15
    assert not D.s4_eligible(_ok_metric(width_macro=0.71, c0_width_macro=1.0))["eligible"]  # <30% width reduction
    assert not D.s4_eligible(_ok_metric(scz_mae=0.6, c0_scz_mae=0.5))["eligible"]   # SCZ MAE worse than C0
    assert not D.s4_eligible(_ok_metric(pd_auroc=0.59))["eligible"]                 # PD center-AUROC < 0.60
    assert not D.s4_eligible(_ok_metric(any_q_inf=True))["eligible"]                # q=+inf candidate
    assert not D.s4_eligible(_ok_metric(s2_pass=False))["eligible"]
    assert not D.s4_eligible(_ok_metric(dominance_pass=False))["eligible"]
    print("  [ok] S4 eligibility: negative/below-C0 red, coverage<.15, <30% width, worse SCZ MAE, PD AUROC<.60, q=+inf, S2/dominance each block SELECT")


def test_s4_select_max_first_tie_and_dev_stop():
    none = {c: _ok_metric(eligible=False) for c in ("C1", "C2", "C3")}
    assert D.s4_select(none)["verdict"] == "DEV_STOP" and D.s4_select(none)["reason"] == "NO_LOCKBOX_CONSUMED"
    sel = lambda d: D.s4_select({c: {**v, "eligible": True} for c, v in d.items()})["selected"]
    assert sel({"C1": _ok_metric(red_macro=0.30), "C2": _ok_metric(red_macro=0.10),
                "C3": _ok_metric(red_macro=0.20)}) == "C1"                          # strictly max red
    assert sel({"C1": _ok_metric(red_macro=0.5, width_macro=0.9), "C2": _ok_metric(red_macro=0.5, width_macro=0.5),
                "C3": _ok_metric(red_macro=0.5, width_macro=0.7)}) == "C2"          # red tie -> min width
    assert sel({c: _ok_metric(red_macro=0.5, width_macro=0.5) for c in ("C1", "C2", "C3")}) == "C2"  # full tie -> order
    # NON-TRANSITIVE chain: only candidates within 1e-4 of the TRUE max are eligible to win on width
    chain = {"C2": _ok_metric(red_macro=1.00000, width_macro=0.9),
             "C3": _ok_metric(red_macro=0.99991, width_macro=0.5),       # 9e-5 below max -> in tie set
             "C1": _ok_metric(red_macro=0.99982, width_macro=0.1)}       # 1.8e-4 below max -> NOT in tie set
    assert sel(chain) == "C3"                                                       # C1's smaller width must NOT win
    print("  [ok] S4 SELECT max-first tie set (transitive vs true max); 1e-4->width; full tie->C2≺C3≺C1; none->DEV_STOP")


def test_c0_vector_is_v2_exact():
    from acar.features import paired_features, context_features, feature_vector
    with tempfile.TemporaryDirectory() as tmp:
        reg, b, lab = _pooled_disease(tmp, n_cohorts=3)
        cache = D.disease_exec_cache(reg, b, lab)
        bb = [x for x in b if not x.fallback][0]
        sa = reg.resolve(bb); exe = sa.execute(bb); state = sa._ephemeral_state()
        c = cache[D.deployment_batch_digest(bb)]
        p0 = np.asarray(exe.p0, float); z0 = np.asarray(exe.z0, float)
        for a, za, pa in exe.per_action:
            v2 = feature_vector(paired_features(p0, np.asarray(pa, float), z0, None if za is None else np.asarray(za, float)),
                                context_features(state, None if za is None else np.asarray(za, float), np.asarray(pa, float)))
            assert v2.shape == (11,) and np.array_equal(v2, c["c0feat"][a])         # bit-for-bit v2 11-D vector
    print("  [ok] v3 C0 feature == v2 feature_vector exactly (11-D, NaN->0, v2 ordering)")


def test_fallback_changes_denominators():
    with tempfile.TemporaryDirectory() as tmp:
        reg, b, lab = _pooled_disease(tmp, n_cohorts=3)
        rep_full, _ = D.develop_candidate("PD", reg, b, lab, "C1")
        no_fb = [x for x in b if not x.fallback]                                    # drop fallback batches
        rep_nofb, _ = D.develop_candidate("PD", reg, no_fb, lab, "C1")
        # denominators differ -> coverage with fallback is <= coverage without (more identity-only batches)
        assert rep_full.adaptation_coverage <= rep_nofb.adaptation_coverage + 1e-12
        c0_full = D.run_c0("PD", reg, b, lab); c0_nofb = D.run_c0("PD", reg, no_fb, lab)
        assert c0_full.n_eval_fallback_batches > 0 and c0_nofb.n_eval_fallback_batches == 0
        print("  [ok] fallback batches enter red/coverage denominators (candidate + C0); removing them changes coverage")


def test_subject_weighting_unequal_batches():
    # rare subject A (1 batch, residual 10) vs subject B (100 batches, residual 0). Subject-equal weighting gives A and
    # B 50% mass each, so the 90th pct lands at A's value (~10); a NAIVE record-weighted quantile would bury it (~0).
    recs = [D.OOFRecord("C2", "PD", 'WS["d","sA"]', "a" * 64, 0, a, 10.0, 0.0, 0.0, 1.0, 1.0, 10.0, 0.0, 0.0, "identity")
            for a in NON_IDENTITY]                                                  # subject A: residual 10 (1 batch)
    for a in NON_IDENTITY:
        for _i in range(100):
            recs.append(D.OOFRecord("C2", "PD", 'WS["d","sB"]', "b" * 64, 0, a, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, "identity"))
    g = D.s2_c2_gate(recs)
    assert all(g[a]["tail90"] > 5.0 for a in NON_IDENTITY)                          # rare subject counted equally
    assert float(np.quantile([10.0] + [0.0] * 100, 0.90)) < 1.0                     # record-weighting would bury it
    print("  [ok] S2 tails subject-equal-weighted: a rare 1-batch subject is NOT buried by a 100-batch subject")


def test_subject_macro_mean_and_q_finite():
    # width/MAE subject-macro: a rare 1-record subject counts equally with a 100-record subject
    assert abs(D._subject_macro_mean({"A": [10.0], "B": [0.0] * 100}) - 5.0) < 1e-12
    # q_finite: ANY +inf fold blocks (not just all-+inf)
    base = _ok_metric()
    assert D.s4_eligible({**base, "any_q_inf": False})["eligible"]
    assert not D.s4_eligible({**base, "any_q_inf": True})["eligible"]
    print("  [ok] subject-macro mean (unequal batch counts); q_finite blocks on ANY +inf fold")


def _cohort_input(tmp, disease, ds, seed):
    p = os.path.join(tmp, f"audit_{disease}_{ds}_erm_0.npz"); _make_dump(p, seed=seed)
    return L.build_cohort_input(p, disease=disease, dataset_id=ds)


def _seven(tmp):
    from acar.config import DISEASE
    return ([_cohort_input(tmp, "PD", ds, 100 + i) for i, ds in enumerate(DISEASE["PD"])] +
            [_cohort_input(tmp, "SCZ", ds, 200 + i) for i, ds in enumerate(DISEASE["SCZ"])])


def test_cohort_input_binding_and_swap():
    import inspect
    with tempfile.TemporaryDirectory() as tmp:
        a = _cohort_input(tmp, "PD", "ds002778", 1); b = _cohort_input(tmp, "PD", "ds003490", 2)
        a.source_artifact.verify_integrity()
        # swapping two cohorts' source artifacts is rejected AT CONSTRUCTION (batch ref != artifact ref)
        _expect(ValueError, lambda: L.CohortInput(a.dataset_id, a.disease, a.manifest, b.source_artifact,
                                                  a.batches, dict(a.labels), a.full_dump_path))
        # dropping a label (missing WindowKey coverage) is rejected
        bad_labels = {k: v for k, v in list(a.labels.items())[1:]}
        _expect(ValueError, lambda: L.CohortInput(a.dataset_id, a.disease, a.manifest, a.source_artifact,
                                                  a.batches, bad_labels, a.full_dump_path))
        # labels are IMMUTABLE post-construction (cannot desync the label hash)
        wk = next(iter(a.labels))
        def _mut():
            a.labels[wk] = 999
        _expect(TypeError, _mut)
        # raw_pipeline_sha256 must be 64-hex or None
        _expect(ValueError, lambda: L.build_cohort_input(a.full_dump_path, disease="PD", dataset_id="ds002778",
                                                         raw_pipeline_sha256="nothex"))
        # public binding/freeze APIs REQUIRE a VerifiedBindingContext (no bypass flag)
        assert "verify_env" not in inspect.signature(D.run_binding_dev).parameters
        assert list(inspect.signature(D.run_binding_dev).parameters)[0] == "context"
        assert list(inspect.signature(D.freeze_dev_run).parameters)[0] == "context"
    print("  [ok] CohortInput binds all; cohort-swap/missing-label rejected; labels immutable; binding APIs require a context")


def test_binding_seven_cohorts():
    from acar.config import DISEASE
    with tempfile.TemporaryDirectory() as tmp:
        seven = _seven(tmp)
        ctx = D._verified_context_for_tests()
        res = D.run_binding_dev(ctx, seven); assert res.verdict in ("SELECT", "DEV_STOP")   # correct 7 -> runs
        assert L._is_hex64(res.env_lock_sha256) and res.env_lock_sha256 == ctx.env_lock_sha256
        _expect(TypeError, lambda: D.run_binding_dev(seven, seven))                     # context required (no bypass)
        _expect(ValueError, lambda: D.run_binding_dev(ctx, seven[:-1]))                 # 6 cohorts -> reject
        wrong = [_cohort_input(tmp, "PD", "dsXXXXXX", 9)] + seven[1:]                   # wrong PD id -> reject
        _expect(ValueError, lambda: D.run_binding_dev(ctx, wrong))
        print(f"  [ok] binding binds the EXACT 7 cohorts {DISEASE}; context required; wrong id/count rejected; verdict={res.verdict}")


def test_json_safety():
    assert D._json_safe(float("nan")) == {"value": None, "status": "NOT_EVALUABLE"}
    assert D._json_safe(np.float64("inf")) == {"value": None, "status": "NOT_EVALUABLE"}
    assert D._json_safe(np.int64(3)) == 3 and D._json_safe(np.bool_(True)) is True
    json.dumps(D._json_safe({"x": np.float64("nan"), "y": [np.int64(1), 2.5]}), allow_nan=False)   # must not raise
    _expect(TypeError, lambda: D._json_safe(object()))                                  # no silent str-coercion
    print("  [ok] JSON safety: non-finite -> NOT_EVALUABLE; numpy scalars coerced; unknown object rejected (no default=str)")


def test_env_lock_drift():
    import acar.v3.envlock as EL
    with tempfile.TemporaryDirectory() as tmp:
        good = EL.build_env_lock(); p = os.path.join(tmp, "lock.json")
        with open(p, "w") as f:
            json.dump(good, f)
        assert L._is_hex64(EL.verify_env_lock(p))                                   # matches
        for field, val in (("torch_num_threads", 999), ("torch_interop_threads", 2)):
            bad = dict(good); bad["runtime"] = {**good["runtime"], field: val}
            bp = os.path.join(tmp, f"bad_{field}.json")
            with open(bp, "w") as f:
                json.dump(bad, f)
            _expect(ValueError, lambda p=bp: EL.verify_env_lock(p))                 # thread/inter-op drift -> fail
    print("  [ok] env-lock verify fails on torch intra-op / inter-op / threadpool runtime drift")


def test_freeze_atomic_claim():
    seven_built = {}  # cache built cohorts per tmp to avoid rebuilds
    with tempfile.TemporaryDirectory() as tmp:
        seven = _seven(tmp); ctx = D._verified_context_for_tests()
        n_run = {"n": 0}; orig = D.run_binding_dev
        D.run_binding_dev = lambda *a, **k: (n_run.__setitem__("n", n_run["n"] + 1), orig(*a, **k))[1]
        try:
            # a stale <out>.tmp (or a concurrent second runner) -> FileExistsError with ZERO DEV compute
            outdir = os.path.join(tmp, "o"); os.mkdir(outdir + ".tmp")
            _expect(FileExistsError, lambda: D.freeze_dev_run(ctx, seven, outdir)); assert n_run["n"] == 0
            # unwritable output parent -> fail before any DEV compute
            ro = os.path.join(tmp, "ro"); os.mkdir(ro); os.chmod(ro, 0o500)
            try:
                _expect(PermissionError, lambda: D.freeze_dev_run(ctx, seven, os.path.join(ro, "x")))
                assert n_run["n"] == 0
            finally:
                os.chmod(ro, 0o700)
        finally:
            D.run_binding_dev = orig
    print("  [ok] freeze atomic-claim: stale temp / concurrent / unwritable-parent fail with ZERO adapter/train/metric calls")


def test_frozen_runner_forced_select():
    """Force SELECT (monkeypatch s4_eligible) to deterministically exercise the artifact-save path: atomic write,
    refit/execute exactly once, saved==run artifact, verify_integrity on reload, file/manifest SHA, EVAL counts, tamper."""
    import acar.v3.loader as LM
    with tempfile.TemporaryDirectory() as tmp:
        seven = _seven(tmp)
        ctx = D._verified_context_for_tests(binding_command="TESTCMD")
        n_exec = {"n": 0}; n_refit = {"n": 0}
        orig_exec = LM.SourceStateArtifact.execute; orig_refit = D.refit_candidate_fixed_epochs; orig_elig = D.s4_eligible
        def counted_exec(self, batch):
            n_exec["n"] += 1; return orig_exec(self, batch)
        def counted_refit(*a, **k):
            n_refit["n"] += 1; return orig_refit(*a, **k)
        LM.SourceStateArtifact.execute = counted_exec
        D.refit_candidate_fixed_epochs = counted_refit
        D.s4_eligible = lambda m, **k: {"criteria": {"forced": True}, "eligible": True}    # force SELECT
        outdir = os.path.join(tmp, "out")
        try:
            n_elig = sum(1 for ci in seven for b in ci.batches if not b.fallback)
            res, man = D.freeze_dev_run(ctx, seven, outdir)
            assert res.verdict == "SELECT" and man["verdict"] == "SELECT"
            assert n_refit["n"] == 2                                                # refit EXACTLY once per disease
            assert n_exec["n"] == n_elig                                            # each eligible batch executed ONCE
        finally:
            LM.SourceStateArtifact.execute = orig_exec; D.refit_candidate_fixed_epochs = orig_refit
            D.s4_eligible = orig_elig
        assert os.path.exists(os.path.join(outdir, "manifest.json")) and not os.path.exists(outdir + ".tmp")  # atomic
        with open(os.path.join(outdir, "manifest.json")) as f:
            disk = json.load(f)
        assert L._is_hex64(disk["manifest_sha256"]) and disk["env_lock_sha256"] == res.env_lock_sha256
        assert set(disk["cohorts"]) == {"PD", "SCZ"} and len(disk["cohorts"]["PD"]) == 3
        assert disk["protocol"]["input_manifest_sha256"] == ctx.input_manifest_sha256
        assert disk["protocol"]["binding_command"] == "TESTCMD" and disk["protocol"]["clean_status_ok"] is True
        assert "raw_pipeline_sha256" in disk["cohorts"]["PD"][0] and "dataset_version" in disk["cohorts"]["PD"][0]
        # S9 EVAL counts: total == eligible + fallback (strict), fallback retained
        for d in ("PD", "SCZ"):
            pc = disk["provenance"][d]["per_candidate"]["C1"]
            assert pc["n_eval_batches_total"] == pc["n_eval_eligible_batches"] + pc["n_eval_fallback_batches"]
            assert pc["n_eval_fallback_batches"] > 0
        _expect(FileExistsError, lambda: D.freeze_dev_run(ctx, seven, outdir))      # non-overwrite
        import pickle as pk
        for d in ("PD", "SCZ"):
            sv = man["saved"][d]
            assert L._is_hex64(sv["predictor_file_sha256"]) and L._is_hex64(sv["c0_file_sha256"])
            with open(sv["predictor_path"], "rb") as f:
                art = pk.load(f)
            art.verify_integrity()
            assert art.artifact_sha256 == sv["predictor_sha256"] == res.refit_sha256[d]
            raw = open(sv["predictor_path"], "rb").read()
            with open(sv["predictor_path"], "wb") as f:
                f.write(raw[:len(raw) // 2])                                        # truncate -> reload fails closed
            _expect(Exception, lambda p=sv["predictor_path"]: pk.load(open(p, "rb")).verify_integrity())
        print("  [ok] forced-SELECT frozen runner: atomic; refit/execute once; saved==run; file+manifest SHA; EVAL total==elig+fallback; context evidence; tamper")


def _temp_git(tmp, *, dirty=False, untracked=False, make_tag=False):
    import subprocess
    repo = os.path.join(tmp, "repo" + str(int(dirty) * 1 + int(untracked) * 2 + int(make_tag) * 4)); os.makedirs(repo)
    def g(*a):
        subprocess.check_output(["git", "-C", repo, *a], stderr=subprocess.DEVNULL)
    g("init"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    with open(os.path.join(repo, "f.txt"), "w") as f:
        f.write("x")
    g("add", "."); g("commit", "-m", "init")
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"]).decode().strip()
    if make_tag:
        g("tag", "acar-v3-dev-design-v1")
    if dirty:
        with open(os.path.join(repo, "f.txt"), "w") as f:
            f.write("modified")
    if untracked:
        with open(os.path.join(repo, "new.py"), "w") as f:
            f.write("x")
    return repo, head


def test_binding_cli_fail_closed():
    import inspect
    import shlex
    import subprocess
    import acar.v3.run_dev_binding as RB
    import numpy as _np
    from acar.config import DISEASE
    with tempfile.TemporaryDirectory() as tmp:
        # NO bypass: run() takes only (input_manifest_path, output); verify_protocol has no require_tag; CLI no --protocol-commit
        assert list(inspect.signature(RB.run).parameters) == ["input_manifest_path", "output"]
        assert "require_tag" not in inspect.signature(RB.verify_protocol).parameters
        # verify_clean_worktree: clean ok; dirty/untracked fail
        clean_repo, head = _temp_git(tmp, make_tag=True); RB.verify_clean_worktree(clean_repo)
        dirty_repo, _ = _temp_git(tmp, dirty=True); _expect(ValueError, lambda: RB.verify_clean_worktree(dirty_repo))
        untr_repo, _ = _temp_git(tmp, untracked=True); _expect(ValueError, lambda: RB.verify_clean_worktree(untr_repo))
        # verify_protocol: HEAD==commit + tag->HEAD ok; absent tag / wrong commit fail (tag verified unconditionally)
        assert RB.verify_protocol(clean_repo, head) == head
        notag_repo, h2 = _temp_git(tmp); _expect(ValueError, lambda: RB.verify_protocol(notag_repo, h2))
        _expect(ValueError, lambda: RB.verify_protocol(clean_repo, "a" * 40))
        # input-manifest schema: full provenance required
        seven = _seven(tmp)
        cohorts = [{"dataset_id": ci.dataset_id, "disease": ci.disease, "path": ci.full_dump_path,
                    "full_dump_sha256": ci.manifest.full_dump_sha256, "source_fit_sha256": ci.manifest.source_fit_sha256,
                    "deployment_input_sha256": ci.manifest.deployment_input_sha256,
                    "label_sha256": ci.manifest.label_sha256, "subject_list_sha256": ci.manifest.subject_list_sha256,
                    "raw_pipeline_sha256": "c" * 64, "dataset_version": "v1.0"} for ci in seven]
        spec = {"protocol_commit": head, "cohorts": cohorts}
        RB.validate_input_manifest(spec)                                            # ok
        for drop in ("label_sha256", "raw_pipeline_sha256", "dataset_version"):     # each required field missing -> fail
            miss = {"protocol_commit": head, "cohorts": [{k: v for k, v in c.items() if k != drop} for c in cohorts]}
            _expect(ValueError, lambda mm=miss: RB.validate_input_manifest(mm))
        upper = {"protocol_commit": head, "cohorts": [dict(c) for c in cohorts]}; upper["cohorts"][0]["label_sha256"] = "A" * 64
        _expect(ValueError, lambda: RB.validate_input_manifest(upper))              # uppercase hash
        empty_v = {"protocol_commit": head, "cohorts": [dict(c) for c in cohorts]}; empty_v["cohorts"][0]["dataset_version"] = ""
        _expect(ValueError, lambda: RB.validate_input_manifest(empty_v))           # empty dataset_version
        # recorded command is shell-round-trippable even with spaces in paths
        cmd = shlex.join([RB.sys.executable, "-m", "acar.v3.run_dev_binding", "--input-manifest", "/a b/spec.json",
                          "--output", "/o ut/dir"])
        assert shlex.split(cmd)[-3:] == ["/a b/spec.json", "--output", "/o ut/dir"]
        # existing output -> refuse before anything
        spec_path = os.path.join(tmp, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        existing = os.path.join(tmp, "exists"); os.makedirs(existing)
        _expect(FileExistsError, lambda: RB.run(spec_path, existing))
        # full_dump RE-CHECK after load: substituting the file changes the computed full_dump_sha256
        ci0 = seven[0]; h_before = ci0.manifest.full_dump_sha256
        with open(ci0.full_dump_path, "ab") as f:
            f.write(b"TAMPER")
        h_after = L.build_cohort_input(ci0.full_dump_path, disease=ci0.disease, dataset_id=ci0.dataset_id).manifest.full_dump_sha256
        assert h_after != h_before                                                  # load-time recheck would catch it
        # no np.load before the preflight: real repo (no v3 tag) -> ValueError at verify_protocol, BEFORE np.load
        real_head = subprocess.check_output(["git", "-C", RB._repo_root(), "rev-parse", "HEAD"]).decode().strip()
        spec2 = {"protocol_commit": real_head, "cohorts": cohorts}; sp2 = os.path.join(tmp, "spec2.json")
        with open(sp2, "w") as f:
            json.dump(spec2, f)
        orig_load = _np.load
        _np.load = lambda *a, **k: (_ for _ in ()).throw(AssertionError("np.load before preflight!"))
        try:
            _expect(ValueError, lambda: RB.run(sp2, os.path.join(tmp, "o1")))       # fails at tag/clean, NOT np.load
        finally:
            _np.load = orig_load
        print(f"  [ok] binding CLI: no bypass; clean/dirty/untracked; tag/commit; required full-provenance schema; shlex-roundtrip cmd; full_dump recheck; no np.load before preflight {DISEASE['PD'][0]}")


def main():
    print("ACAR v3 split + DEV bake-off/gate guards (synthetic fixtures only):")
    for t in (test_split_partition_balance_determinism, test_split_permutation_independent,
              test_develop_multicohort_registry_and_leak_isolation, test_develop_fallback_eval_accounting,
              test_s2_c2_gate_boundaries_and_floor, test_s4_eligible_each_criterion_gates,
              test_s4_select_max_first_tie_and_dev_stop, test_c0_vector_is_v2_exact,
              test_fallback_changes_denominators, test_subject_weighting_unequal_batches,
              test_subject_macro_mean_and_q_finite, test_cohort_input_binding_and_swap, test_binding_seven_cohorts,
              test_json_safety, test_env_lock_drift, test_freeze_atomic_claim, test_frozen_runner_forced_select,
              test_binding_cli_fail_closed):
        t()
    print("ALL V3 DEVELOP/SPLIT GUARDS PASS")


if __name__ == "__main__":
    main()
