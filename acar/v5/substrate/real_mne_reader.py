"""ACAR V5 Stage-1B real mne DSP seam (NO heavy import at module load; numpy + mne are LAZY inside the functions). Turns a subject's
raw BIDS recording(s) into a validated, SIGNAL-ONLY SubjectWindows under the PINNED preprocessing config. The DSP is a fixed,
auditable pipeline (select+order 19 channels → average reference → 0.5–45 Hz bandpass → resample 128 Hz → 4 s / 512-sample
non-overlapping windows → per-trial per-channel z-score, microvolt units); no labels are ever read here.

BOUNDARY SAFETY: each recording is windowed INDEPENDENTLY (windows never span two recordings); the per-recording window arrays are
concatenated AFTER windowing. Recordings are discovered raw-BIDS-only (raw_recording_manifest: eeg/ and ses-*/eeg/ only).

Testability: `raw_to_windows(raw, ...)` is the mne-INDEPENDENT single-recording DSP core — it operates on any mne-Raw-like object
(duck-typed: .ch_names / .pick / .set_eeg_reference / .filter / .resample / .get_data). `preprocess_subject(...)` discovers the raw
recordings and reads each via a LAZY (or INJECTED, for fixtures) mne module, windows each independently, then concatenates the
window arrays. A synthetic FakeRaw/fake-mne adapter drives the whole path in tests.

Interpretation (reviewable): a recording must CONTAIN all 19 montage channels; they are picked and reordered to the canonical order
(a permuted input yields canonical output). Any of the 19 missing → fail-closed. Extra non-montage channels are dropped by the pick.
"""
from __future__ import annotations
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import subject_windows as SW
from acar.v5.substrate import raw_recording_manifest as RM

_MNE_LOADER = {".edf": "read_raw_edf", ".bdf": "read_raw_bdf", ".set": "read_raw_eeglab",
               ".vhdr": "read_raw_brainvision", ".fif": "read_raw_fif"}


class RealMneReaderError(RuntimeError):
    pass


def _per_trial_zscore(windows, np):
    """Per window, per channel z-score over the time axis (ddof=0). A zero-variance channel divides by 1 (stays finite)."""
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (windows - mean) / std


def _windows_from_raw(raw, np):
    """SINGLE-recording DSP core → float32 window array (n_win, 19, 512). Windows are taken WITHIN this recording only."""
    cfg = PC.PREPROCESSING_CONFIG
    ch = list(raw.ch_names)
    missing = [c for c in PC.CHANNELS_19 if c not in ch]      # required: all 19 canonical present (else fail)
    if missing:
        raise RealMneReaderError(f"recording missing montage channels {missing}")
    dups = [c for c in PC.CHANNELS_19 if ch.count(c) > 1]     # duplicate_channel_policy = fail_closed
    if dups:
        raise RealMneReaderError(f"recording has duplicate montage channels {dups}")
    raw = raw.pick(list(PC.CHANNELS_19))                      # select + reorder to canonical (extras dropped)
    if tuple(raw.ch_names) != PC.CHANNELS_19:
        raise RealMneReaderError("channels after pick are not the canonical 19-channel order")
    raw.set_eeg_reference("average", projection=False)
    raw.filter(l_freq=cfg["bandpass_hz"][0], h_freq=cfg["bandpass_hz"][1])
    raw.resample(cfg["resample_hz"])
    data = np.asarray(raw.get_data(units="uV"), dtype=np.float64)   # (19, n_times), microvolt
    if data.ndim != 2 or data.shape[0] != 19:
        raise RealMneReaderError(f"expected (19, n_times) data, got {data.shape}")
    w = cfg["window_samples"]
    n_win = data.shape[1] // w
    if n_win < 1:
        raise RealMneReaderError(f"recording too short for one {w}-sample window (n_times={data.shape[1]})")
    data = data[:, : n_win * w]                               # drop the trailing partial window (never spans recordings)
    windows = data.reshape(19, n_win, w).transpose(1, 0, 2)   # (n_win, 19, 512)
    return _per_trial_zscore(windows, np).astype(np.float32)


def _wrap(windows, disease, cohort, raw_subject_id, provenance="real_mne_reader"):
    import numpy as np
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    sw = SW.SubjectWindows(
        subject_key=f"{disease}/{cohort}/{raw_subject_id}", disease=disease, cohort=cohort, raw_subject_id=raw_subject_id,
        n_windows=int(windows.shape[0]), n_channels=19, n_samples=PC.PREPROCESSING_CONFIG["window_samples"],
        sfreq=PC.PREPROCESSING_CONFIG["resample_hz"], channels=PC.CHANNELS_19,
        preprocessing_config_sha256=PC.config_sha256(), windows=windows, provenance=provenance)
    SW.validate_subject_windows(sw)                           # fail-closed vs the pinned config + payload shape/finiteness
    return sw


def raw_to_windows(raw, disease, cohort, raw_subject_id):
    """mne-INDEPENDENT single-recording DSP → validated SubjectWindows. numpy imported lazily."""
    import numpy as np
    return _wrap(_windows_from_raw(raw, np), disease, cohort, raw_subject_id)


def _read_raw(path, mne):
    import os
    ext = os.path.splitext(path)[1].lower()
    loader = getattr(mne.io, _MNE_LOADER[ext])
    return loader(path, preload=True, verbose="ERROR")


def preprocess_subject(disease, cohort, raw_subject_id, subject_dir, *, mne=None):
    """Discover raw-BIDS recordings, window EACH independently (no cross-recording windows), concatenate the window arrays → one
    validated SubjectWindows. mne is lazy (or injected for fixtures)."""
    if mne is None:
        import mne as _mne  # lazy — never imported at module load
        mne = _mne
    import numpy as np
    manifest = RM.build_manifest(subject_dir)                 # raw-BIDS-only discovery + hashed manifest (incl. format sidecars)
    files = [e["path"] for e in manifest["files"] if e.get("role") == "primary"]
    per_recording = [_windows_from_raw(_read_raw(p, mne), np) for p in files]   # windowed WITHIN each recording
    windows = np.concatenate(per_recording, axis=0)           # concatenate WINDOWS, never raws
    provenance = f"real_mne_reader;raw_manifest_sha256={manifest['manifest_sha256']}"   # ties output to the exact raw files
    return _wrap(windows, disease, cohort, raw_subject_id, provenance=provenance)
