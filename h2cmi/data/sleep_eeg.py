"""W2 Sleep-EDF (Sleep-Cassette) offline loader for natural prevalence-shift sleep staging.

Frozen preprocessing (review W1_W2_FROZEN): Sleep-Cassette ONLY (no Sleep-Telemetry / temazepam),
channels EEG Fpz-Cz + Pz-Oz, 100 Hz, 30 s epochs, 5 classes W/N1/N2/N3/REM (S3+S4->N3; movement/
unscored discarded). Recording cropped to +-30 min around the scored sleep period (standard, removes
the daytime lights-on wake artifact). Per-epoch z-score per channel. Subject/night parsed from the
SC4ssN... filename. Labels are returned for EVALUATION only.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

SLEEP_ROOT = "/projects/EEG-foundation-model/datalake/raw/sleep-edf/sleep-cassette"
SLEEP_FS = 100.0
SLEEP_EPOCH_S = 30.0
SLEEP_N_TIMES = int(SLEEP_FS * SLEEP_EPOCH_S)            # 3000
SLEEP_CH = ["EEG Fpz-Cz", "EEG Pz-Oz"]
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
# annotation description -> stage code (S3 & S4 both -> N3)
_DESC2ID = {"Sleep stage W": 1, "Sleep stage 1": 2, "Sleep stage 2": 3,
            "Sleep stage 3": 4, "Sleep stage 4": 4, "Sleep stage R": 5}
_ID2CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}              # -> 0..4


@dataclass
class SleepEpochs:
    X: np.ndarray              # [n, 2, 3000] float32
    y: np.ndarray              # [n] int 0..4 (W/N1/N2/N3/REM)
    subject: np.ndarray        # [n] int subject id
    night: np.ndarray          # [n] int night (1 or 2)


def _pair_files():
    """Map subject-id -> {night: (psg_path, hyp_path)} for the Sleep-Cassette."""
    psg = sorted(glob.glob(os.path.join(SLEEP_ROOT, "SC*PSG.edf")))
    hyp = {os.path.basename(h)[:6]: h for h in glob.glob(os.path.join(SLEEP_ROOT, "SC*Hypnogram.edf"))}
    out = {}
    for p in psg:
        b = os.path.basename(p)
        subj = int(b[3:5]); night = int(b[5])
        h = hyp.get(b[:6])
        if h is None:
            continue
        out.setdefault(subj, {})[night] = (p, h)
    return out


def subject_list():
    return sorted(_pair_files())


def _load_record(psg_path, hyp_path):
    import warnings
    warnings.filterwarnings("ignore")
    import mne
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    have = [c for c in SLEEP_CH if c in raw.ch_names]
    if len(have) < len(SLEEP_CH):
        return None
    raw.pick(SLEEP_CH)
    if abs(raw.info["sfreq"] - SLEEP_FS) > 1e-6:
        raw.resample(SLEEP_FS, verbose=False)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)
    events, _ = mne.events_from_annotations(raw, event_id=_DESC2ID, chunk_duration=SLEEP_EPOCH_S,
                                            verbose=False)
    if len(events) == 0:
        return None
    # crop to +-30 min around the scored SLEEP period (drop daytime lights-on wake)
    cls = np.array([_ID2CLASS[c] for c in events[:, 2]])
    nonwake = np.where(cls != 0)[0]
    if len(nonwake) == 0:
        return None
    lo = max(0, nonwake[0] - 60); hi = min(len(events), nonwake[-1] + 60 + 1)
    events = events[lo:hi]
    tmax = SLEEP_EPOCH_S - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(raw, events, event_id={str(c): c for c in sorted(set(_DESC2ID.values()))},
                        tmin=0.0, tmax=tmax, baseline=None, preload=True, verbose=False,
                        on_missing="ignore")
    X = epochs.get_data(copy=False).astype(np.float32)
    if X.shape[2] >= SLEEP_N_TIMES:
        X = X[:, :, :SLEEP_N_TIMES]
    else:
        X = np.pad(X, ((0, 0), (0, 0), (0, SLEEP_N_TIMES - X.shape[2])))
    y = np.array([_ID2CLASS[c] for c in epochs.events[:, 2]], dtype=np.int64)
    mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-7
    X = (X - mu) / sd
    return X, y


def load_subjects(subjects) -> SleepEpochs:
    """Load all available nights for the given subject ids."""
    pairs = _pair_files()
    Xs, ys, ss, ns = [], [], [], []
    for s in subjects:
        for night, (p, h) in sorted(pairs.get(s, {}).items()):
            rec = _load_record(p, h)
            if rec is None:
                continue
            X, y = rec
            Xs.append(X); ys.append(y)
            ss.append(np.full(len(y), s, np.int64)); ns.append(np.full(len(y), night, np.int64))
    if not Xs:
        return SleepEpochs(np.zeros((0, 2, SLEEP_N_TIMES), np.float32), np.zeros(0, np.int64),
                           np.zeros(0, np.int64), np.zeros(0, np.int64))
    return SleepEpochs(np.concatenate(Xs), np.concatenate(ys),
                       np.concatenate(ss), np.concatenate(ns))
