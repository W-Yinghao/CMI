"""C86H F1->F2->H1 pipeline tests. F1 trains the 11-ch zoo (real frozen engine, tiny; or a fast
mock producing real untrained 11-ch weights), F2 generates the field + real-field manifest, and
the batch H1 consumes it. Validated on synthetic data only — no real EEG/label.
"""
import os
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mne")

from oaci.active_testing.c86h import f1f2, f1f2_train, batch_h1, field_spec
from oaci.active_testing.c86h import contract as K
from oaci.active_testing.c86h.f1f2_field import build_c86h_model


def test_train_cell_real_frozen_engine_tiny():
    """The frozen engine faithfully trains an 11-channel cell -> ERM + OACI/SRC trajectories."""
    X, y, d = f1f2_train.build_synthetic_source(n_domains=2, per_cell=6, seed=1)
    cands = f1f2_train.train_cell(X, y, d, seed=5, preset=f1f2_train.TINY)
    regimes = [c[0] for c in cands]
    assert regimes.count("ERM") == 1 and regimes.count("OACI") == 2 and regimes.count("SRC") == 2
    assert [c[1].epoch for c in cands if c[0] == "OACI"] == [4, 9]     # cadence


def _mock_cell(X, y, domain, seed, preset, sample_ids=None, groups=None):
    """Fast cell: real (untrained) 11-ch weights, no training — for pipeline structure tests."""
    def rec(order):
        m = build_c86h_model()
        return types.SimpleNamespace(model_state=m.state_dict(), model_hash="", epoch=order)
    return ([("ERM", rec(0), 0)]
            + [("OACI", rec(i), i) for i in (1, 2)]
            + [("SRC", rec(i), i) for i in (1, 2)])


def _synth_source(panel, seed, level):
    """(X, y, domain, sample_ids, groups, source_raw) — carries synthetic population identity +
    a source-raw ledger so the content-addressed manifest replay has all required objects."""
    X, y, d = f1f2_train.build_synthetic_source(n_domains=2, per_cell=4,
                                                seed=abs(hash((panel, seed, level))) % 1000)
    n = len(y)
    sids = [f"SRC|{panel}|se{seed}|lv{level}|t{i}" for i in range(n)]
    groups = [f"SRC|subject={int(d[i])}|session=0|run=0" for i in range(n)]
    raw = {f"SRC|{panel}|s{int(v)}": "0" * 64 for v in sorted(set(d.tolist()))}
    return X, y, d, sids, groups, raw


def _target_provider():
    rng = np.random.default_rng(0)
    for cohort, subj in (("SYN_A", 1), ("SYN_A", 2)):
        n = 88
        y = np.array([0, 1] * (n // 2))
        X = rng.standard_normal((n, 11, 480)) + y[:, None, None] * 0.5
        tids = [f"{subj}_t{i}" for i in range(n)]
        yield cohort, subj, X, y, tids, f"rawsha_{subj}"


def test_f1_f2_h1_pipeline_e2e(tmp_path):
    zoo_root = str(tmp_path / "zoo")
    # F1: 8-context zoo with the fast mock trainer (real 11-ch weights)
    zoo = f1f2_train.f1_train_zoo(_synth_source, zoo_root, preset=f1f2_train.TINY,
                                  cell_trainer=_mock_cell)
    assert zoo["n_models"] == 8 * 5                      # 8 contexts x 5 candidates
    assert len(set(zoo["candidates"])) == zoo["n_models"]        # unique candidate IDs
    assert all(c.startswith("c86_") for c in zoo["candidates"])  # C86 namespace
    # F2: field + real-field manifest from the zoo on synthetic targets
    field_root = str(tmp_path / "field")
    rfm = f1f2_train.f2_generate_predictions(zoo, zoo_root, _target_provider, field_root)
    assert rfm["construction_evaluation_overlap"] == 0
    assert rfm["n_targets"] == 2 and rfm["n_candidates_per_context"] == 5
    import os
    assert os.path.isfile(os.path.join(field_root, f1f2.REAL_FIELD_MANIFEST_NAME))
    # H1 consumes the F2-generated field byte-for-byte
    methods = list(K.METHOD_REGISTRY); chains = [0, 1]; exp = [("SYN_A", 1), ("SYN_A", 2)]
    odir = str(tmp_path / "orders"); h1 = str(tmp_path / "h1")
    batch_h1.run_h1a(os.path.join(field_root, "acquisition_unlabeled_pool"), odir, methods, chains)
    batch_h1.run_h1b_sealed(odir, os.path.join(field_root, "acquisition_label_oracle"),
                            os.path.join(field_root, "query_contribution_store"), h1, methods, chains)
    batch_h1.verify_h1(h1, odir, exp, methods, chains)
    sel = batch_h1.load_selections(h1, methods, exp, chains)
    assert len(sel) == 2 * 3 * 2                          # targets x methods x chains


def test_integrated_execute_wiring_with_injected_real_providers(tmp_path):
    """runner.execute truly completes F1 -> F2 -> content-addressed manifest replay -> H1..H4
    (not calling the lower helpers directly)."""
    from oaci.active_testing.c86h import runner
    contract = {"target_cohort": {("SYN_A", 1): "SYN_A", ("SYN_A", 2): "SYN_A"},
                "cohort_dataset": {"SYN_A": "SYN_A"}, "cohorts": {"SYN_A"},
                "n_targets": 2, "n_candidates_per_context": 5, "n_models": 8 * 5, "n_contexts": 16}
    m = runner.execute("授权 C86H", str(tmp_path / "camp"),
                       source_provider=_synth_source, target_provider=_target_provider,
                       preset=f1f2_train.TINY, chains=(0, 1), cell_trainer=_mock_cell,
                       contract=contract)
    assert m["stage"] == "C86H_H4_TERMINAL_RESULT"
    assert m["classification"]["formal_gate"] in K.FORMAL_GATE
    assert m["held_opened_after_freeze_verification"] is True
    # execute refuses without the token, and a re-run into the same attempt root is refused
    with pytest.raises(SystemExit):
        runner.execute("", str(tmp_path / "camp2"))
    with pytest.raises(RuntimeError):                     # fresh-attempt-root guard
        runner.execute("授权 C86H", str(tmp_path / "camp"),
                       source_provider=_synth_source, target_provider=_target_provider,
                       preset=f1f2_train.TINY, chains=(0, 1), cell_trainer=_mock_cell,
                       contract=contract)


def _synthetic_registered_source(panel, seed, level):
    """Multi-source (X,y,domain) built from synthetic_source_panel (passes the registered
    intervention) + row-aligned synthetic X, with contiguous cross-dataset domains."""
    from oaci.multidataset import c84l1_intervention as itv
    Xs, ys, subs = [], [], []
    for name in ("Lee2019_MI", "Cho2017", "PhysionetMI"):
        fx = itv.synthetic_source_panel(name, panel, rows_per_cell=8)
        labels = list(fx["labels"]); subjects = list(fx["subjects"]); tids = list(fx["trial_ids"])
        rng = np.random.default_rng(abs(hash((name, panel, seed, level))) % 10000)
        X = rng.standard_normal((len(labels), 11, 480)) + np.array(labels)[:, None, None] * 0.4
        app = itv.apply_level_intervention(dataset=name, panel=panel, level=int(level),
                                           source_subjects=subjects, source_labels=labels,
                                           source_trial_ids=tids)
        keep = list(app.keep_indices)
        Xs.append(X[keep]); ys.append(np.array(labels)[keep])
        subs.append([(name, subjects[k]) for k in keep])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    flat = [s for g in subs for s in g]
    dn = sorted(set(flat)); dmap = {s: i for i, s in enumerate(dn)}
    return X, y, np.array([dmap[s] for s in flat], dtype=int)


def test_registered_source_contract_and_level_intervention_replay():
    from oaci.multidataset import c84f_dual_level_training as f
    from oaci.multidataset import c84l1_intervention as itv
    from oaci.multidataset import c84l1_protocols as proto
    # exact hash-locked 12 train + 4 audit, disjoint
    train, audit = f._source_subject_contract("Lee2019_MI", "A")
    assert len(train) == 12 and len(audit) == 4 and not (set(train) & set(audit))
    # level 0 = full 24-cell graph ; level 1 = registered left_hand deletion -> exactly 23 cells
    fx = itv.synthetic_source_panel("Lee2019_MI", "A", rows_per_cell=8)
    a0 = itv.apply_level_intervention(dataset="Lee2019_MI", panel="A", level=0,
                                      source_subjects=fx["subjects"], source_labels=fx["labels"],
                                      source_trial_ids=fx["trial_ids"])
    a1 = itv.apply_level_intervention(dataset="Lee2019_MI", panel="A", level=1,
                                      source_subjects=fx["subjects"], source_labels=fx["labels"],
                                      source_trial_ids=fx["trial_ids"])
    assert len(a0.post_cell_counts) == 24 and len(a1.post_cell_counts) == 23
    assert a1.deleted_source_subject == proto.DELETED_SUBJECTS[("Lee2019_MI", "A")]
    assert a1.deleted_class == "left_hand" and len(a1.deleted_indices) >= 8
    # the registered multi-source builds CONTIGUOUS domains (0..n-1), and the frozen identity is deterministic
    X, y, d = _synthetic_registered_source("A", 5, 1)
    assert sorted(set(d.tolist())) == list(range(int(d.max()) + 1))       # contiguous
    assert f1f2_train.FAITHFUL["deterministic"] is True                   # frozen training identity


def _synth_moabb_loader(name, subs):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    out = {}
    for s in subs:
        y = np.array([0, 1] * 10)
        X = rng.standard_normal((20, 11, 480)) + y[:, None, None] * 0.3
        tids = [f"{name}|s{s}|sess0|run0|t{i}" for i in range(20)]
        groups = [f"{name}|subject={s}|session=0|run=0" for _ in range(20)]
        out[s] = (X, y, tids, groups)
    return out


def test_residual_population_identity_cuda_and_candidate_replay(tmp_path, monkeypatch):
    from oaci.active_testing.c86h import f1f2
    import glob
    import json
    import torch
    # (item 3 residual) _assemble_source yields REAL trial ids + dataset|subject|session|run groups
    # + real source-raw digests, and train_cell uses them (not syn|.../g...)
    X, y, d, sids, groups, raw = f1f2._assemble_source("A", 1, _synth_moabb_loader)
    assert sids and not any(str(s).startswith("syn|") for s in sids)
    assert all("subject=" in g and "session=" in g and "run=" in g for g in groups)
    assert all(g.startswith(("Lee2019_MI|", "Cho2017|", "PhysionetMI|")) for g in groups)
    assert raw and all(isinstance(v, str) and len(v) == 64 for v in raw.values())
    # train_cell ACCEPTS real-style trial ids + groups (small source so the frozen engine is fast)
    Xs, ys, ds = f1f2_train.build_synthetic_source(2, 4, seed=1)
    m = len(ys)
    r_sids = [f"Lee2019_MI|s{int(ds[i])}|sess0|run0|t{i}" for i in range(m)]
    r_groups = [f"Lee2019_MI|subject={int(ds[i])}|session=0|run=0" for i in range(m)]
    cands = f1f2_train.train_cell(Xs, ys, ds, seed=5, preset=f1f2_train.TINY,
                                  sample_ids=r_sids, groups=r_groups)
    assert len(cands) == 5                                # frozen engine accepts real identity
    # (item 4 residual) production FAITHFUL fail-closes without CUDA (no CPU fallback)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(f1f2.C86EError):
        f1f2.f1_train_zoo("授权 C86H", str(tmp_path / "zc"), source_provider=lambda p, s, l: None)
    # (item 5 residual) exact candidate-ID/order replay + source-raw registry
    zoo_root = str(tmp_path / "zoo")
    zoo = f1f2_train.f1_train_zoo(_synth_source, zoo_root, preset=f1f2_train.TINY,
                                  cell_trainer=_mock_cell)
    field_root = str(tmp_path / "field")
    f1f2_train.f2_generate_predictions(zoo, zoo_root, _target_provider, field_root)
    contract = {"target_cohort": {("SYN_A", 1): "SYN_A", ("SYN_A", 2): "SYN_A"},
                "n_targets": 2, "n_candidates_per_context": 5, "n_models": 8 * 5, "n_contexts": 16}
    assert f1f2.validate_real_field_manifest(field_root, contract=contract, zoo_root=zoo_root)
    # tamper a candidate's recorded panel -> recomputed c86_candidate_id != stored id -> C86-E
    zpath = os.path.join(zoo_root, "C86H_ZOO_MANIFEST.json")
    zman = json.load(open(zpath))
    cid = next(iter(zman["candidates"]))
    e = zman["candidates"][cid]; e["panel"] = "B" if e["panel"] == "A" else "A"
    json.dump(zman, open(zpath, "w"))
    with pytest.raises(f1f2.C86EError):
        f1f2.validate_real_field_manifest(field_root, contract=contract, zoo_root=zoo_root)


def test_f1f2_gated_not_stub(tmp_path):
    # gated entrypoints refuse without the token (SystemExit)
    with pytest.raises(SystemExit):
        f1f2.f1_train_zoo("", str(tmp_path))
    with pytest.raises(SystemExit):
        f1f2.f2_generate_predictions("", {}, str(tmp_path), str(tmp_path))
    # WITH the token it is real orchestration that reaches the provider (NOT an unconditional raise)
    def boom(panel, seed, level):
        raise RuntimeError("reached-source-provider")
    with pytest.raises(RuntimeError, match="reached-source-provider"):
        f1f2.f1_train_zoo("授权 C86H", str(tmp_path), preset=f1f2_train.TINY, source_provider=boom)
