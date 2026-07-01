"""ACAR V5 Stage-1B real mne DSP seam (NO heavy import at module load; numpy + mne are LAZY inside the functions). Turns a subject's
raw BIDS recording(s) into a validated, SIGNAL-ONLY SubjectWindows under the PINNED preprocessing config. The DSP is a fixed,
auditable pipeline (select+order 19 channels → average reference → 0.5–45 Hz bandpass → resample 128 Hz → 4 s / 512-sample
non-overlapping windows → per-trial per-channel z-score, microvolt units); no labels are ever read here.

Testability: `raw_to_windows(raw, ...)` is the mne-INDEPENDENT DSP core — it operates on any mne-Raw-like object (duck-typed:
.ch_names / .pick / .set_eeg_reference / .filter / .resample / .get_data). `preprocess_subject(...)` does deterministic recording
discovery + read via a LAZY (or INJECTED, for fixtures) mne module, then calls the core. A synthetic FakeRaw/fake-mne adapter drives
the whole path in tests — no real DEV read, no real mne dependency.

Interpretation note (reviewable): a recording is required to CONTAIN all 19 montage channels; they are picked and reordered to the
canonical order (a permuted input yields canonical output). Any of the 19 missing → fail-closed. Extra non-montage channels are
dropped by the pick. The final SubjectWindows is validated to carry exactly the 19 canonical channels.
"""
from __future__ import annotations
import os
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import subject_windows as SW

EEG_EXTENSIONS = (".edf", ".bdf", ".set", ".vhdr", ".fif")   # sorted, deterministic discovery
_MNE_LOADER = {".edf": "read_raw_edf", ".bdf": "read_raw_bdf", ".set": "read_raw_eeglab",
               ".vhdr": "read_raw_brainvision", ".fif": "read_raw_fif"}


class RealMneReaderError(RuntimeError):
    pass


def discover_recordings(subject_dir):
    """Deterministic, sorted list of EEG recording files under subject_dir (recursive). Fail-closed if none."""
    if not subject_dir or not os.path.isdir(subject_dir):
        raise RealMneReaderError(f"subject dir not found: {subject_dir}")
    found = []
    for root, _dirs, files in os.walk(subject_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in EEG_EXTENSIONS:
                found.append(os.path.join(root, f))
    if not found:
        raise RealMneReaderError(f"no EEG recordings ({EEG_EXTENSIONS}) under {subject_dir}")
    return sorted(found)


def _read_concat_raw(files, mne):
    raws = []
    for path in files:                                        # already sorted → deterministic concatenation order
        ext = os.path.splitext(path)[1].lower()
        loader = getattr(mne.io, _MNE_LOADER[ext])
        raws.append(loader(path, preload=True, verbose="ERROR"))
    if len(raws) == 1:
        return raws[0]
    return mne.concatenate_raws(raws)


def _per_trial_zscore(windows, np):
    """Per window, per channel z-score over the time axis (ddof=0). A zero-variance channel divides by 1 (stays finite)."""
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (windows - mean) / std


def raw_to_windows(raw, disease, cohort, raw_subject_id):
    """mne-INDEPENDENT DSP core: raw (mne-Raw-like) → validated SubjectWindows under the pinned config. numpy imported lazily."""
    import numpy as np
    cfg = PC.PREPROCESSING_CONFIG
    missing = [c for c in PC.CHANNELS_19 if c not in list(raw.ch_names)]
    if missing:
        raise RealMneReaderError(f"recording missing montage channels {missing}")
    raw = raw.pick(list(PC.CHANNELS_19))                      # select + reorder to canonical order
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
    data = data[:, : n_win * w]
    windows = data.reshape(19, n_win, w).transpose(1, 0, 2)   # (n_win, 19, 512)
    windows = _per_trial_zscore(windows, np)
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    sw = SW.SubjectWindows(
        subject_key=f"{disease}/{cohort}/{raw_subject_id}", disease=disease, cohort=cohort, raw_subject_id=raw_subject_id,
        n_windows=int(n_win), n_channels=19, n_samples=w, sfreq=cfg["resample_hz"], channels=PC.CHANNELS_19,
        preprocessing_config_sha256=PC.config_sha256(), windows=windows, provenance="real_mne_reader")
    SW.validate_subject_windows(sw)                           # fail-closed vs the pinned config + payload shape/finiteness
    return sw


def preprocess_subject(disease, cohort, raw_subject_id, subject_dir, *, mne=None):
    """Deterministic discovery + read (lazy or injected mne) → raw_to_windows. No labels are read."""
    if mne is None:
        import mne as _mne  # lazy — never imported at module load
        mne = _mne
    files = discover_recordings(subject_dir)
    raw = _read_concat_raw(files, mne)
    return raw_to_windows(raw, disease, cohort, raw_subject_id)
