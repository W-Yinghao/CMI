"""C86H F1->F2->H1 pipeline tests. F1 trains the 11-ch zoo (real frozen engine, tiny; or a fast
mock producing real untrained 11-ch weights), F2 generates the field + real-field manifest, and
the batch H1 consumes it. Validated on synthetic data only — no real EEG/label.
"""
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


def _mock_cell(X, y, domain, seed, preset):
    """Fast cell: real (untrained) 11-ch weights, no training — for pipeline structure tests."""
    def rec(order):
        m = build_c86h_model()
        return types.SimpleNamespace(model_state=m.state_dict(), model_hash="", epoch=order)
    return ([("ERM", rec(0), 0)]
            + [("OACI", rec(i), i) for i in (1, 2)]
            + [("SRC", rec(i), i) for i in (1, 2)])


def _synth_source(panel, seed, level):
    return f1f2_train.build_synthetic_source(n_domains=2, per_cell=4,
                                             seed=hash((panel, seed, level)) % 1000)


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
