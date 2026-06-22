"""Structural real-loader guards for acar/v3/loader.py. SYNTHETIC .npz FIXTURES ONLY (mirror the erm_0 dump schema);
NO real DEV cohort is read. Proves: strict dtypes (no coercion), label firewall (deployment path never reads y_te),
WindowKey-aligned ΔR, ActionOutputsRecord/LabeledRiskRecord binding, v3 source-state ref, PD-on-SCZ disease gate.
Run: python -m acar.v3.tests.test_loader
"""
import os
import tempfile
import numpy as np

from acar.config import MIN_BATCH, N_CLS
from acar.v3.set_features import NON_IDENTITY, build_action_sets
from acar.v3.data import deployment_batch_digest, SubjectKey
from acar.v3.training import DeploymentFeatureRecord, fit_candidate_earlystop
from acar.v3 import loader as L


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _arrays(seed=0, d=6):
    """Clean fixture arrays. 6 full subjects (48 windows -> 32+16 eligible batches) + 1 tiny subject (fallback)."""
    rng = np.random.default_rng(seed)
    n_ev = 240
    yev = np.tile([0, 1], n_ev // 2).astype(np.int64)
    zev = rng.standard_normal((n_ev, d)) + yev[:, None] * 0.6
    sub, rec, win = [], [], []
    for s in range(6):
        for k in range(48):
            sub.append(f"sub-{s:03d}"); rec.append("rec-00"); win.append(k)
    for k in range(4):                                              # tiny subject -> fallback batch
        sub.append("sub-006"); rec.append("rec-00"); win.append(k)
    n = len(sub)
    zte = rng.standard_normal((n, d))
    yte = rng.integers(0, N_CLS, size=n).astype(np.int64)
    return dict(z_ev=zev, y_ev=yev, z_te=zte, y_te=yte,
                subject_id_te=np.array(sub), recording_id_te=np.array(rec),
                window_index_te=np.array(win, dtype=np.int64))


def _write(path, arrays):
    np.savez(path, **arrays)


def test_source_state_ref_and_deployment_build():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "audit_PD_dsX_erm_0.npz"); _write(p, _arrays())
        state, src_hash = L.load_source_state(p)
        assert L._is_hex64(src_hash)
        env = L.env_versions()
        ref = L.v3_source_state_ref(state, src_hash, env)
        assert L._is_hex64(ref) and ref == L.v3_source_state_ref(state, src_hash, env)        # deterministic, lowercase-64
        assert ref != L.v3_source_state_ref(state, "b" * 64, env)                              # source-dump-hash sensitive
        batches = L.load_deployment_batches(p, disease="PD", dataset_id="dsX", source_state_ref=ref)
        assert all(b.disease == "PD" and b.source_state_ref == ref for b in batches)
        assert any(b.fallback for b in batches) and any(not b.fallback for b in batches)        # tiny subject -> fallback
        # window order within a batch is the canonical acquisition order
        big = [b for b in batches if not b.fallback][0]
        assert [wk.window_index for wk in big.window_keys] == sorted(wk.window_index for wk in big.window_keys)
    print("  [ok] load_source_state + v3 source-state ref (64-hex, deterministic, dump-hash sensitive); deployment build")


def test_strict_dtypes_no_coercion():
    base = _arrays()
    with tempfile.TemporaryDirectory() as tmp:
        def run(mutate):
            a = dict(base); mutate(a); p = os.path.join(tmp, "m.npz"); _write(p, a)
            return L.load_deployment_batches(p, disease="PD", dataset_id="d", source_state_ref="a" * 64)
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("window_index_te", a["window_index_te"].astype(float))))  # float idx
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("window_index_te", a["window_index_te"].astype(bool))))   # bool idx
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("window_index_te",
                                                                np.array([str(x) for x in a["window_index_te"]]))))       # str idx
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("subject_id_te",
                                                                np.arange(len(a["subject_id_te"])))))                     # numeric id
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("z_te", a["z_te"][:5])))                                 # length mismatch
        _expect(ValueError, lambda: run(lambda a: a.pop("window_index_te")))                                             # missing field
    print("  [ok] window index must be true int (float/bool/str rejected); ids must be str (numeric rejected); length/field checks")


def test_action_outputs_and_labeled_risk_binding():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "audit_PD_dsX_erm_0.npz"); _write(p, _arrays())
        state, src = L.load_source_state(p); ref = L.v3_source_state_ref(state, src, L.env_versions())
        batches = L.load_deployment_batches(p, disease="PD", dataset_id="dsX", source_state_ref=ref)
        labels = L.load_labels_by_window(p, dataset_id="dsX")
        b = [x for x in batches if not x.fallback][0]
        ao = L.compute_action_outputs(state, b)
        assert L._is_hex64(ao.action_outputs_sha256); ao.verify_integrity()
        # immutability of stored probabilities
        def _w():
            ao.p0.flags.writeable = True
        _expect(ValueError, _w)
        lrr = L.labeled_risk_record(b, ao, labels)
        assert lrr.deployment_batch_digest == deployment_batch_digest(b) == ao.deployment_batch_digest
        assert lrr.action_outputs_sha256 == ao.action_outputs_sha256                            # ΔR bound to exact outputs
        assert tuple(a for a, _ in lrr.delta_r_by_action) == NON_IDENTITY
        # fallback batch has no action outputs
        fb = [x for x in batches if x.fallback][0]
        _expect(ValueError, lambda: L.compute_action_outputs(state, fb))
        # ΔR aligned by WindowKey: a permuted-but-consistent label dict yields the SAME record;
        # a label dict missing a key fails closed.
        permuted = {k: labels[k] for k in reversed(list(labels))}
        assert L.labeled_risk_record(b, ao, permuted).delta_r_by_action == lrr.delta_r_by_action
        _expect(ValueError, lambda: L.labeled_risk_record(b, ao, {k: v for k, v in labels.items()
                                                                  if k != b.window_keys[0]}))
    print("  [ok] ActionOutputsRecord 64-hex+immutable; LabeledRiskRecord binds digest+outputs; ΔR aligned by WindowKey")


def test_label_firewall_poison_proxy():
    """The deployment path (batches -> action outputs -> predictions) must be byte-identical whether y_te is clean or
    adversarially flipped. Only the labeled risk record (which legitimately reads y_te) may change."""
    clean = _arrays(); poison = dict(clean); poison["y_te"] = (N_CLS - 1 - clean["y_te"]).astype(np.int64)
    assert not np.array_equal(clean["y_te"], poison["y_te"])
    with tempfile.TemporaryDirectory() as tmp:
        pc = os.path.join(tmp, "clean.npz"); pp = os.path.join(tmp, "poison.npz")
        _write(pc, clean); _write(pp, poison)
        state, src = L.load_source_state(pc); ref = L.v3_source_state_ref(state, src, L.env_versions())
        bc = L.load_deployment_batches(pc, disease="PD", dataset_id="dsX", source_state_ref=ref)
        bp = L.load_deployment_batches(pp, disease="PD", dataset_id="dsX", source_state_ref=ref)
        # train ONE real artifact from the clean deployment path, then deploy on both
        exs = []
        labels = L.load_labels_by_window(pc, dataset_id="dsX")
        for b in [x for x in bc if not x.fallback]:
            ao = L.compute_action_outputs(state, b); lrr = L.labeled_risk_record(b, ao, labels)
            dr = dict(lrr.delta_r_by_action); sets = build_action_sets(state, np.asarray(b.z, float), b.window_keys)
            exs += DeploymentFeatureRecord("PD", b.subject, deployment_batch_digest(b),
                                           tuple((a, sets[a], dr[a]) for a in NON_IDENTITY)).to_train_examples()
        subs = sorted({e.subject_key for e in exs}, key=lambda s: s.subject_id)
        tr_subj = set(subs[:4]); tr = [e for e in exs if e.subject_key in tr_subj]
        va = [e for e in exs if e.subject_key not in tr_subj]
        artifact, _ = fit_candidate_earlystop("C1", "PD", tr, va, seed=0)
        for b1, b2 in zip(bc, bp):
            assert deployment_batch_digest(b1) == deployment_batch_digest(b2)                   # batch identity y_te-independent
            p1, p2 = L.predict_batch(artifact, state, b1), L.predict_batch(artifact, state, b2)
            assert p1 == p2                                                                     # predictions y_te-independent
        # but the labeled risk DOES change when labels are poisoned
        bclean = [x for x in bc if not x.fallback][0]
        ao = L.compute_action_outputs(state, bclean)
        lc = L.labeled_risk_record(bclean, ao, L.load_labels_by_window(pc, dataset_id="dsX"))
        lp = L.labeled_risk_record(bclean, ao, L.load_labels_by_window(pp, dataset_id="dsX"))
        assert lc.delta_r_by_action != lp.delta_r_by_action
    print("  [ok] label firewall: deployment digests + predictions byte-identical under y_te poisoning; only ΔR changes")


def test_disease_gate_before_prediction():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "audit_PD_dsX_erm_0.npz"); _write(p, _arrays())
        state, src = L.load_source_state(p); ref = L.v3_source_state_ref(state, src, L.env_versions())
        bpd = L.load_deployment_batches(p, disease="PD", dataset_id="dsX", source_state_ref=ref)
        bscz = L.load_deployment_batches(p, disease="SCZ", dataset_id="dsX", source_state_ref=ref)
        exs = []; labels = L.load_labels_by_window(p, dataset_id="dsX")
        for b in [x for x in bpd if not x.fallback]:
            ao = L.compute_action_outputs(state, b); dr = dict(L.labeled_risk_record(b, ao, labels).delta_r_by_action)
            sets = build_action_sets(state, np.asarray(b.z, float), b.window_keys)
            exs += DeploymentFeatureRecord("PD", b.subject, deployment_batch_digest(b),
                                           tuple((a, sets[a], dr[a]) for a in NON_IDENTITY)).to_train_examples()
        subs = sorted({e.subject_key for e in exs}, key=lambda s: s.subject_id)
        tr = [e for e in exs if e.subject_key in set(subs[:4])]; va = [e for e in exs if e.subject_key in set(subs[4:])]
        pd_artifact, _ = fit_candidate_earlystop("C1", "PD", tr, va, seed=0)
        scz_eligible = [x for x in bscz if not x.fallback][0]
        _expect(ValueError, lambda: pd_artifact.predict(build_action_sets(state, np.asarray(scz_eligible.z, float),
                                                                          scz_eligible.window_keys)) if False else
                pd_artifact.assert_disease(scz_eligible.disease))                               # gate raises...
        _expect(ValueError, lambda: L.predict_batch(pd_artifact, state, scz_eligible))          # ...before any forward pass
        assert L.predict_batch(pd_artifact, state, [x for x in bpd if not x.fallback][0]) is not None
        assert L.predict_batch(pd_artifact, state, [x for x in bpd if x.fallback][0]) is None   # fallback -> identity
    print("  [ok] PD artifact rejects SCZ batch via assert_disease BEFORE any prediction; fallback -> None")


def main():
    print("ACAR v3 structural loader guards (synthetic fixtures only):")
    for t in (test_source_state_ref_and_deployment_build, test_strict_dtypes_no_coercion,
              test_action_outputs_and_labeled_risk_binding, test_label_firewall_poison_proxy,
              test_disease_gate_before_prediction):
        t()
    print("ALL V3 LOADER GUARDS PASS")


if __name__ == "__main__":
    main()
