"""C86H F1/F2 real-implementation tests — 11-channel model + interface epoching + the
ds007221 BIDS adapter + MOABB adapter, validated against SYNTHETIC fixtures / mocks. No real
EEG/label is touched.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mne = pytest.importorskip("mne")
pytest.importorskip("mne_bids")

from oaci.active_testing.c86h import f1f2_field as F
from oaci.active_testing.c86h import contract as K


def test_11_channel_model_forward():
    m = F.build_c86h_model()
    assert F.IN_CHANS == 11 and F.IN_TIMES == 480
    out = m(torch.randn(4, F.IN_CHANS, F.IN_TIMES))
    logits = out.logits if hasattr(out, "logits") else out
    assert tuple(logits.shape) == (4, 2)


def _synthetic_epochs(sfreq=200.0, n=20, extra=("EXTRA1",)):
    ch = list(K.INTERFACE_CHANNELS) + list(extra)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, len(ch), int(sfreq * 4))) * 1e-6
    info = mne.create_info(ch, sfreq, ch_types="eeg")
    ev = np.column_stack([np.arange(n), np.zeros(n, int),
                          np.array([1 if k % 2 else 2 for k in range(n)])])
    return mne.EpochsArray(data, info, events=ev,
                           event_id={"right_hand": 1, "left_hand": 2}, tmin=0.0, verbose="error")


def test_interface_epoching_maps_to_11ch_160hz():
    ep = _synthetic_epochs()
    X, y = F.interface_epochs(ep)
    assert X.shape[1] == 11 and X.shape[2] == 480
    assert set(y.tolist()) <= {0, 1}
    assert np.all(np.isfinite(X))


def test_interface_epoching_c86e_on_missing_channel():
    ep = _synthetic_epochs()
    ep = ep.drop_channels(["FC5"])                       # remove a required interface channel
    with pytest.raises(F.C86EError):
        F.interface_epochs(ep)


def test_ds007221_bids_adapter_roundtrip_and_deterministic(tmp_path):
    root = str(tmp_path / "bids")
    F.write_synthetic_ds007221_fixture(root, [37, 40], n_trials=24, seed=1)
    X, y, tids, sha = F.load_ds007221_bids(root, "sub-37")
    assert X.shape[1] == 11 and X.shape[2] == 480
    assert set(y.tolist()) <= {0, 1}
    assert len(tids) == len(y) == X.shape[0]
    assert len(set(tids)) == len(tids)                   # unique stable trial ids
    # deterministic replay: reload gives identical trial ids + provenance sha
    X2, y2, tids2, sha2 = F.load_ds007221_bids(root, "sub-37")
    assert tids == tids2 and sha == sha2
    assert np.array_equal(y, y2)


def test_ds007221_bids_adapter_c86e_subject_out_of_range(tmp_path):
    root = str(tmp_path / "bids")
    F.write_synthetic_ds007221_fixture(root, [37], n_trials=12, seed=2)
    with pytest.raises(F.C86EError):
        F.load_ds007221_bids(root, "sub-10")             # not in locked 37..73


def test_ds007221_bids_adapter_c86e_when_absent(tmp_path):
    root = str(tmp_path / "empty"); import os; os.makedirs(root, exist_ok=True)
    with pytest.raises(F.C86EError):
        F.load_ds007221_bids(root, "sub-37")             # no hybrid acquisition present


def test_moabb_adapter_with_mock_loader():
    def mock(name, subs):
        return {s: (np.random.randn(20, F.IN_CHANS, F.IN_TIMES),
                    ["left_hand", "right_hand"] * 10) for s in subs}
    r = F.load_moabb_dataset("Lee2019_MI", [1, 2], loader=mock)
    assert set(r) == {1, 2}
    X, y = r[1]
    assert X.shape == (20, 11, 480) and set(y.tolist()) <= {0, 1}


def _valid_real_field_manifest():
    ctx = 424; cands = 81
    return {
        "schema": "c86h_real_field_manifest_v1",
        "interface_id": K.COMMON_INTERFACE_ID,
        "field_training_manifest_sha256": K.FIELD_TRAINING_MANIFEST_SHA256,
        "n_targets": 53, "n_contexts": ctx, "n_candidates_per_context": cands,
        "n_candidate_context_slices": ctx * cands,
        "zoo": {"n_models": 648, "weight_sha256": {f"c86_{i:024x}": "0" * 64 for i in range(648)}},
        "prediction_context_sha256": {f"ctx{i}": "0" * 64 for i in range(ctx)},
        "construction_evaluation_overlap": 0,
        "split": {f"t{i}": {"pool": [f"t{i}_p{j}" for j in range(40)],
                            "held": [f"t{i}_h{j}" for j in range(40)]} for i in range(53)},
        "class_support": {f"t{i}": {"pool": {"0": 20, "1": 20}, "held": {"0": 20, "1": 20}}
                          for i in range(53)},
    }


def test_real_field_manifest_validator(tmp_path):
    from oaci.active_testing.c86h import f1f2
    import json
    root = str(tmp_path)
    man = _valid_real_field_manifest()
    json.dump(man, open(os.path.join(root, f1f2.REAL_FIELD_MANIFEST_NAME), "w"))
    assert f1f2.validate_real_field_manifest(root)["n_targets"] == 53   # valid passes
    # each corruption fails closed
    for mutate in (lambda m: m.update(n_targets=52),
                   lambda m: m["zoo"].update(n_models=647),
                   lambda m: m.update(construction_evaluation_overlap=1),
                   lambda m: m["split"]["t0"].update(held=m["split"]["t0"]["pool"]),
                   lambda m: m["class_support"]["t0"]["pool"].update({"0": 3})):
        bad = _valid_real_field_manifest(); mutate(bad)
        json.dump(bad, open(os.path.join(root, f1f2.REAL_FIELD_MANIFEST_NAME), "w"))
        with pytest.raises(f1f2.C86EError):
            f1f2.validate_real_field_manifest(root)
