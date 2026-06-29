"""Guards for acar/v4/heldout_reader.py — the held-out raw→windows selection/validation/key layer. SYNTHETIC mini-BIDS
fixtures + a SYNTHETIC signal_provider ONLY; NO real held-out raw, NO mne, NO encoder, NO inference. Run:
python -m acar.v4.tests.test_heldout_reader
"""
import json
import os
import shutil
import tempfile

import numpy as np

from acar.v4 import heldout_reader as HR
from acar.v4 import prepare_external_dump as PREP

CFG = PREP.FROZEN_PIPELINE                          # canon_channels 19, resample_fs 128, window_sec 4.0 → n_times 512


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                          # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


def _participants(base, rows):
    with open(os.path.join(base, "participants.tsv"), "w") as f:
        f.write("participant_id\tgroup\n" + "".join(f"{p}\t{g}\n" for p, g in rows))


def _recording(base, sub, task, *, fs=250, nch=65, run=None, ses=None, with_fs=True, with_nch=True, subdir="eeg"):
    bits = [f"sub-{sub}"] + ([f"ses-{ses}"] if ses is not None else []) + [f"task-{task}"] \
        + ([f"run-{run}"] if run is not None else [])
    stem = "_".join(bits)
    d = os.path.join(base, f"sub-{sub}", *( [f"ses-{ses}"] if ses is not None else []), subdir)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, stem + "_eeg.edf"), "w").close()
    meta = {}
    if with_fs:
        meta["SamplingFrequency"] = fs
    if with_nch:
        meta["EEGChannelCount"] = nch
    with open(os.path.join(d, stem + "_eeg.json"), "w") as f:
        json.dump(meta, f)


def _prov(nwin=2, ch=19, T=512):
    return lambda rec: np.zeros((ch, T * nwin), dtype=float)


def test_happy_path_ds007526():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "HC")])
        _recording(base, "01", "rest"); _recording(base, "02", "rest")
        out = HR.read_heldout("ds007526", base, pipeline_config=CFG, signal_provider=_prov(nwin=2))
        assert out["X"].shape == (4, 19, 512)
        assert set(out["y"].tolist()) == {0, 1} and out["y"].tolist() == [1, 1, 0, 0]
        assert "ds007526/sub-01" in out["subject_id"].tolist() and "ds007526/sub-02" in out["subject_id"].tolist()
        assert out["window_index"].tolist() == [0, 1, 0, 1]                     # resets per recording
        rows = list(zip(out["subject_id"].tolist(), out["recording_id"].tolist(), out["window_index"].tolist()))
        assert len(set(rows)) == 4                                              # unique keys
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_walking_dropped_not_raised():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "HC")])
        _recording(base, "01", "rest"); _recording(base, "01", "walking"); _recording(base, "02", "rest")
        idx = HR.build_heldout_index(base, "ds007526")
        tasks = sorted(r["task"] for r in idx)
        assert tasks == ["rest", "rest"]                                        # walking excluded (dropped, not raised)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_unknown_task_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "HC")])
        _recording(base, "01", "oddball"); _recording(base, "02", "rest")
        _expect(ValueError, lambda: HR.build_heldout_index(base, "ds007526"))   # 'oddball' neither resting nor excluded
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_missing_diagnosis_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "HC")])
        _recording(base, "01", "rest"); _recording(base, "03", "rest")          # sub-03 not in participants.tsv
        _expect(ValueError, lambda: HR.build_heldout_index(base, "ds007526"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_unknown_group_token_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "ZZZ")])             # ZZZ neither patient nor control
        _recording(base, "01", "rest"); _recording(base, "02", "rest")
        _expect(ValueError, lambda: HR.build_heldout_index(base, "ds007526"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_missing_channel_fs_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "SZ"), ("sub-02", "HC")])
        _recording(base, "01", "rest", fs=1000, nch=64, with_fs=False)         # no SamplingFrequency in sidecar
        _recording(base, "02", "rest", fs=1000, nch=64)
        _expect(ValueError, lambda: HR.build_heldout_index(base, "zenodo14808296"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_channel_mismatch_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "SZ"), ("sub-02", "HC")])
        _recording(base, "01", "rest", fs=1000, nch=32)                        # 32 != expected 64 for zenodo
        _recording(base, "02", "rest", fs=1000, nch=64)
        _expect(ValueError, lambda: HR.build_heldout_index(base, "zenodo14808296"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_recording_too_short_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "HC")])
        _recording(base, "01", "rest"); _recording(base, "02", "rest")
        idx = HR.build_heldout_index(base, "ds007526")
        short = lambda rec: np.zeros((19, 100), dtype=float)                    # < one 512-sample window
        _expect(ValueError, lambda: HR.assemble_windows(idx, pipeline_config=CFG, signal_provider=short))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_wrong_channel_signal_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "HC")])
        _recording(base, "01", "rest"); _recording(base, "02", "rest")
        idx = HR.build_heldout_index(base, "ds007526")
        bad = lambda rec: np.zeros((32, 512), dtype=float)                      # provider returns 32ch != canon 19
        _expect(ValueError, lambda: HR.assemble_windows(idx, pipeline_config=CFG, signal_provider=bad))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_duplicate_recording_id_fail_closed():
    base = tempfile.mkdtemp()
    try:
        _participants(base, [("sub-01", "PD"), ("sub-02", "HC")])
        _recording(base, "01", "rest"); _recording(base, "02", "rest")
        _recording(base, "01", "rest", subdir="eeg_copy")                      # same BIDS stem in a 2nd dir → dup rid
        _expect(ValueError, lambda: HR.build_heldout_index(base, "ds007526"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_real_signal_provider_is_gated():
    prov = HR.make_real_signal_provider(CFG)
    _expect(HR.RawSignalDSPNotWiredError, lambda: prov({"recording_id": "sub-01_task-rest"}))


def main():
    print("ACAR v4 heldout_reader guards (synthetic mini-BIDS fixtures only):")
    for t in (test_happy_path_ds007526, test_walking_dropped_not_raised, test_unknown_task_fail_closed,
              test_missing_diagnosis_fail_closed, test_unknown_group_token_fail_closed,
              test_missing_channel_fs_fail_closed, test_channel_mismatch_fail_closed,
              test_recording_too_short_fail_closed, test_wrong_channel_signal_fail_closed,
              test_duplicate_recording_id_fail_closed, test_real_signal_provider_is_gated):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 HELDOUT-READER GUARDS PASS")


if __name__ == "__main__":
    main()
