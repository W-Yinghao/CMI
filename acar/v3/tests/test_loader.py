"""Structural real-loader BINDING guards for acar/v3/loader.py. SYNTHETIC .npz FIXTURES ONLY (mirror the erm_0 dump
schema); NO real DEV cohort is read. Proves: field-separated provenance (y_te never touches deployment identity),
immutable SourceStateArtifact gating, canonical row identity, single-execution derivation, manifest, disease gate.
Run: python -m acar.v3.tests.test_loader
"""
import os
import tempfile
import numpy as np

from acar.config import N_CLS
from acar.v3.set_features import NON_IDENTITY
from acar.v3.data import deployment_batch_digest, canonical_row_digest
from acar.v3.training import fit_candidate_earlystop
from acar.v3 import loader as L


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _arrays(seed=0, d=6):
    """6 full subjects (48 windows -> 32+16 eligible) + 1 tiny subject (fallback)."""
    rng = np.random.default_rng(seed)
    n_ev = 240
    yev = np.tile([0, 1], n_ev // 2).astype(np.int64)
    zev = rng.standard_normal((n_ev, d)) + yev[:, None] * 0.6
    sub, rec, win = [], [], []
    for s in range(6):
        for k in range(48):
            sub.append(f"sub-{s:03d}"); rec.append("rec-00"); win.append(k)
    for k in range(4):
        sub.append("sub-006"); rec.append("rec-00"); win.append(k)
    n = len(sub)
    zte = rng.standard_normal((n, d))
    yte = rng.integers(0, N_CLS, size=n).astype(np.int64)
    return dict(z_ev=zev, y_ev=yev, z_te=zte, y_te=yte,
                subject_id_te=np.array(sub), recording_id_te=np.array(rec),
                window_index_te=np.array(win, dtype=np.int64))


def _write(path, arrays):
    np.savez(path, **arrays)


def _fit_artifact(state_art, batches, labels):
    exs = []
    for b in [x for x in batches if not x.fallback]:
        exe = state_art.execute(b)
        exs += exe.deployment_feature_record(state_art, b, labels).to_train_examples()
    subs = sorted({e.subject_key for e in exs}, key=lambda s: s.subject_id)
    tr = [e for e in exs if e.subject_key in set(subs[:4])]; va = [e for e in exs if e.subject_key in set(subs[4:])]
    art, _ = fit_candidate_earlystop("C1", "PD", tr, va, seed=0)
    return art


def test_field_separated_and_source_state_ref():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "audit_PD_dsX_erm_0.npz"); _write(p, _arrays())
        sa = L.load_source_artifact_from_dump(p, disease="PD")
        assert L._is_hex64(sa.source_state_ref) and L._is_hex64(sa.source_fit_sha256)
        sa.verify_integrity()
        batches = L.load_deployment_batches(p, disease="PD", dataset_id="dsX", source_state_ref=sa.source_state_ref)
        labels = L.load_labels_by_window(p, dataset_id="dsX")
        man = L.build_manifest(p, dataset_id="dsX", disease="PD", source_artifact=sa, batches=batches,
                               labels_by_window=labels)
        # five DISTINCT provenance fields
        hs = {man.full_dump_sha256, man.source_fit_sha256, man.deployment_input_sha256, man.label_sha256,
              man.subject_list_sha256}
        assert len(hs) == 5 and man.n_subjects == 7 and man.embedding_dim == 6
        assert any(b.fallback for b in batches) and any(not b.fallback for b in batches)
    print("  [ok] field-separated provenance manifest (5 distinct hashes); source-state ref from source_fit only")


def test_strict_dtypes_no_coercion():
    base = _arrays()
    with tempfile.TemporaryDirectory() as tmp:
        def run(mutate):
            a = dict(base); mutate(a); p = os.path.join(tmp, "m.npz"); _write(p, a)
            return L.load_deployment_batches(p, disease="PD", dataset_id="d", source_state_ref="a" * 64)
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("window_index_te", a["window_index_te"].astype(float))))
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("window_index_te", a["window_index_te"].astype(bool))))
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("window_index_te",
                                                                np.array([str(x) for x in a["window_index_te"]]))))
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("subject_id_te",
                                                                np.arange(len(a["subject_id_te"])))))
        _expect(ValueError, lambda: run(lambda a: a.__setitem__("z_te", a["z_te"][:5])))
        _expect(ValueError, lambda: run(lambda a: a.pop("window_index_te")))
    print("  [ok] window index true-int only; ids str only (no coercion); length/field checks")


def test_label_firewall_full_pipeline():
    """GUARD 1 + 8: run the WHOLE workflow on two .npz differing ONLY in y_te. Everything on the deployment path must be
    bit-identical; only full_dump/label hashes and ΔR may move."""
    clean = _arrays(); poison = dict(clean); poison["y_te"] = (N_CLS - 1 - clean["y_te"]).astype(np.int64)
    assert not np.array_equal(clean["y_te"], poison["y_te"])
    with tempfile.TemporaryDirectory() as tmp:
        pc = os.path.join(tmp, "clean.npz"); pp = os.path.join(tmp, "poison.npz"); _write(pc, clean); _write(pp, poison)
        # build EACH side from scratch (refs derived independently — not reusing the clean ref)
        sac = L.load_source_artifact_from_dump(pc, disease="PD"); sap = L.load_source_artifact_from_dump(pp, disease="PD")
        assert sac.source_fit_sha256 == sap.source_fit_sha256 and sac.source_state_ref == sap.source_state_ref
        bc = L.load_deployment_batches(pc, disease="PD", dataset_id="dsX", source_state_ref=sac.source_state_ref)
        bp = L.load_deployment_batches(pp, disease="PD", dataset_id="dsX", source_state_ref=sap.source_state_ref)
        manc = L.build_manifest(pc, dataset_id="dsX", disease="PD", source_artifact=sac, batches=bc,
                                labels_by_window=L.load_labels_by_window(pc, dataset_id="dsX"))
        manp = L.build_manifest(pp, dataset_id="dsX", disease="PD", source_artifact=sap, batches=bp,
                                labels_by_window=L.load_labels_by_window(pp, dataset_id="dsX"))
        assert manc.deployment_input_sha256 == manp.deployment_input_sha256
        assert manc.subject_list_sha256 == manp.subject_list_sha256
        assert manc.full_dump_sha256 != manp.full_dump_sha256 and manc.label_sha256 != manp.label_sha256
        art = _fit_artifact(sac, bc, L.load_labels_by_window(pc, dataset_id="dsX"))
        for b1, b2 in zip(bc, bp):
            assert deployment_batch_digest(b1) == deployment_batch_digest(b2)
            if b1.fallback:
                assert L.predict_batch(art, sac, b1) is None and L.predict_batch(art, sap, b2) is None
                continue
            e1, e2 = sac.execute(b1), sap.execute(b2)
            assert e1.execution_sha256 == e2.execution_sha256 and e1.action_outputs_sha256 == e2.action_outputs_sha256
            assert L.predict_batch(art, sac, b1) == L.predict_batch(art, sap, b2)
        # ΔR DOES change with poisoned labels
        bclean = [x for x in bc if not x.fallback][0]; exe = sac.execute(bclean)
        lc = exe.labeled_risk_record(L.load_labels_by_window(pc, dataset_id="dsX"))
        lp = exe.labeled_risk_record(L.load_labels_by_window(pp, dataset_id="dsX"))
        assert lc.delta_r_by_action != lp.delta_r_by_action
    print("  [ok] label firewall: deployment input/digests/executions/predictions bit-identical under y_te poison; full_dump changes don't propagate")


def test_state_and_disease_binding():
    """GUARD 2 + 3: a batch whose source_state_ref / disease / embedding_dim doesn't match the source artifact fails
    BEFORE any adapter."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.npz"); _write(p, _arrays())
        sa = L.load_source_artifact_from_dump(p, disease="PD")
        good = [b for b in L.load_deployment_batches(p, disease="PD", dataset_id="dsX",
                                                     source_state_ref=sa.source_state_ref) if not b.fallback][0]
        sa.execute(good)                                                              # ok
        wrong_ref = [b for b in L.load_deployment_batches(p, disease="PD", dataset_id="dsX",
                                                          source_state_ref="b" * 64) if not b.fallback][0]
        _expect(ValueError, lambda: sa.execute(wrong_ref))                            # declared state B, artifact A
        scz = [b for b in L.load_deployment_batches(p, disease="SCZ", dataset_id="dsX",
                                                    source_state_ref=sa.source_state_ref) if not b.fallback][0]
        _expect(ValueError, lambda: sa.execute(scz))                                  # disease mismatch
    # embedding-dim mismatch: artifact at d=6, batch at d=8
    with tempfile.TemporaryDirectory() as tmp:
        p6 = os.path.join(tmp, "d6.npz"); _write(p6, _arrays(d=6))
        p8 = os.path.join(tmp, "d8.npz"); _write(p8, _arrays(d=8))
        sa6 = L.load_source_artifact_from_dump(p6, disease="PD")
        b8 = [b for b in L.load_deployment_batches(p8, disease="PD", dataset_id="dsX",
                                                   source_state_ref=sa6.source_state_ref) if not b.fallback][0]
        _expect(ValueError, lambda: sa6.execute(b8))                                  # dim mismatch before forward pass
    print("  [ok] SourceStateArtifact gates ref/disease/embedding_dim before any forward pass")


def test_external_load_never_fits():
    """GUARD 7: the external/deployment path rebuilds f_0 from frozen params and does NOT call fit_source_state; the DEV
    path does. Reproduces the same source_state_ref."""
    import acar.v3.loader as M
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.npz"); _write(p, _arrays())
        sa = M.load_source_artifact_from_dump(p, disease="PD")
        blob = M.freeze_source_state_artifact(sa)
        orig = M.fit_source_state
        M.fit_source_state = lambda *a, **k: (_ for _ in ()).throw(AssertionError("external path fit!"))
        try:
            ext = M.load_frozen_source_state_artifact(blob)                           # must NOT fit
            assert ext.source_state_ref == sa.source_state_ref and ext.source_fit_sha256 == sa.source_fit_sha256
            sa.verify_integrity(); ext.verify_integrity()
            _expect(AssertionError, lambda: M.load_source_artifact_from_dump(p, disease="PD"))   # DEV path DOES fit
        finally:
            M.fit_source_state = orig
        # GUARD: tampering classes_ / env / any source-state byte fails the blob's own stored hash (repro #5 now closed)
        bad = dict(blob); bad["classes"] = blob["classes"][::-1].copy()
        _expect(ValueError, lambda: M.load_frozen_source_state_artifact(bad))
        bad2 = dict(blob); bad2["env_vals"] = list(blob["env_vals"]); bad2["env_vals"][0] = "tampered"
        _expect(ValueError, lambda: M.load_frozen_source_state_artifact(bad2))
        bad3 = dict(blob); bad3["coef"] = np.asarray(blob["coef"], float) + 1.0
        _expect(ValueError, lambda: M.load_frozen_source_state_artifact(bad3))
    print("  [ok] external load rebuilds frozen f_0 without fit (same ref); DEV path fits; tampering classes_/env/coef fails the stored hash")


def test_single_execution_binding():
    """GUARD 4 + 5 + 6: features and ΔR come from ONE execution (shared execution/action-outputs hashes); cross-pairing
    a record with the wrong batch or wrong source artifact fails; canonical row identity holds."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.npz"); _write(p, _arrays())
        sa = L.load_source_artifact_from_dump(p, disease="PD")
        batches = [b for b in L.load_deployment_batches(p, disease="PD", dataset_id="dsX",
                                                        source_state_ref=sa.source_state_ref) if not b.fallback]
        labels = L.load_labels_by_window(p, dataset_id="dsX")
        b0, b1 = batches[0], batches[1]
        exe = sa.execute(b0); exe.verify_integrity()
        dfr = exe.deployment_feature_record(sa, b0, labels)
        assert dfr.execution_sha256 == exe.execution_sha256 and dfr.action_outputs_sha256 == exe.action_outputs_sha256
        ao = exe.action_outputs_record()
        assert ao.action_outputs_sha256 == exe.action_outputs_sha256
        # cross-pairing this execution with a DIFFERENT batch fails (digest/row mismatch)
        _expect(ValueError, lambda: exe.deployment_feature_record(sa, b1, labels))
        # canonical row identity: digest unchanged, row digest defined and bound into the execution
        assert canonical_row_digest(b0) == exe.canonical_row_digest
        # immutability of captured probabilities
        def _w():
            np.asarray(exe.p0).flags.writeable = True
        _expect(ValueError, _w)
        # GUARD 6: end-to-end call count — identity + 3 actions EXACTLY once; features/ΔR do NOT re-execute
        import acar.v3.loader as M
        calls = []
        orig = M.apply_action
        M.apply_action = lambda name, st, zz: calls.append(name) or orig(name, st, zz)
        try:
            e2 = sa.execute(b0)
            e2.window_action_sets(sa); e2.labeled_risk_record(labels); e2.deployment_feature_record(sa, b0, labels)
        finally:
            M.apply_action = orig
        assert sorted(calls) == sorted(["identity", *NON_IDENTITY]), calls          # 4 calls total, no second pass
        # GUARD 9: subject-list hash permutation-insensitive (incl. duplicates) but add/remove sensitive
        subs = [b.subject for b in batches]
        uniq = sorted({b.subject for b in batches}, key=lambda s: s.subject_id)
        assert L.hash_subject_list(subs) == L.hash_subject_list(list(reversed(subs))) == L.hash_subject_list(uniq)
        assert L.hash_subject_list(uniq) != L.hash_subject_list(uniq[:-1])           # drop a whole subject -> changes
    print("  [ok] one execution -> features+ΔR share hashes; exactly 4 adapter calls (no re-exec); subject-list hash perm-insensitive/add-sensitive; wrong-batch rejected")


def test_disease_gate_before_prediction():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "a.npz"); _write(p, _arrays())
        sa = L.load_source_artifact_from_dump(p, disease="PD")
        bpd = L.load_deployment_batches(p, disease="PD", dataset_id="dsX", source_state_ref=sa.source_state_ref)
        art = _fit_artifact(sa, bpd, L.load_labels_by_window(p, dataset_id="dsX"))
        scz = [b for b in L.load_deployment_batches(p, disease="SCZ", dataset_id="dsX",
                                                    source_state_ref=sa.source_state_ref) if not b.fallback][0]
        _expect(ValueError, lambda: art.assert_disease(scz.disease))
        _expect(ValueError, lambda: L.predict_batch(art, sa, scz))                    # gate before any forward pass
        assert L.predict_batch(art, sa, [b for b in bpd if not b.fallback][0]) is not None
        assert L.predict_batch(art, sa, [b for b in bpd if b.fallback][0]) is None
    print("  [ok] PD artifact rejects SCZ batch before any prediction; fallback -> None")


def main():
    print("ACAR v3 structural loader-binding guards (synthetic fixtures only):")
    for t in (test_field_separated_and_source_state_ref, test_strict_dtypes_no_coercion,
              test_label_firewall_full_pipeline, test_state_and_disease_binding, test_external_load_never_fits,
              test_single_execution_binding, test_disease_gate_before_prediction):
        t()
    print("ALL V3 LOADER GUARDS PASS")


if __name__ == "__main__":
    main()
