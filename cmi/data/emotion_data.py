"""Emotion-recognition EEG loaders (SEED, DEAP) with the same (X, y, meta, classes)
interface as cmi.data.moabb_data.load — so the LOSO harness is task-agnostic.

SEED  : 15 subjects x 3 sessions, 62ch @200Hz, 3-class (negative/neutral/positive),
        15 film-clip trials per session; labels in label.mat (fixed order).
DEAP  : 32 subjects, 32 EEG ch @128Hz, 40 trials x 63s (first 3s = baseline, dropped);
        valence binarized at 5 -> {low,high}.

Long trials are cut into fixed windows; to bound dataset size we keep at most
`max_per_trial` evenly-spaced windows per trial.  Domain D = subject (LOSO).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

SEED_DIR = Path("/projects/EEG-foundation-model/SEED")
DEAP_DIR = Path("/projects/EEG-foundation-model/DEAP/data_preprocessed_python")
SEEDIV_DIR = Path("/projects/EEG-foundation-model/SEED_IV/eeg_raw_data")
SEED_FS, DEAP_FS, SEEDIV_FS = 200, 128, 200
SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1  # ->{0,1,2}
SEEDIV_LABELS = {  # per-session 24-trial labels (0 neutral,1 sad,2 fear,3 happy)
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]}


def _windows(trial, win, max_n):
    """trial [C,T] -> list of [C,win] non-overlapping windows, <= max_n evenly spaced."""
    T = trial.shape[1]
    n = T // win
    if n == 0:
        return []
    starts = np.arange(n) * win
    if max_n and n > max_n:
        starts = starts[np.linspace(0, n - 1, max_n).round().astype(int)]
    return [trial[:, s:s + win] for s in starts]


def _finalize(Xs, ys, subs, sess, normalize):
    X = np.stack(Xs).astype("float32")
    if normalize == "trial_zscore":
        m = X.mean(axis=2, keepdims=True); s = X.std(axis=2, keepdims=True) + 1e-7
        X = (X - m) / s
    meta = pd.DataFrame({"subject": subs, "session": sess, "run": 0})
    return X, np.array(ys, dtype="int64"), meta


def _load_seed(subjects, win_sec, max_per_trial, normalize):
    import scipy.io as sio
    win = int(win_sec * SEED_FS)
    files = sorted(SEED_DIR.glob("*.mat"))
    files = [f for f in files if f.name != "label.mat"]
    by_subj = {}
    for f in files:
        sid = int(f.name.split("_")[0]); by_subj.setdefault(sid, []).append(f)
    Xs, ys, subs, sess = [], [], [], []
    for sid in sorted(by_subj):
        if subjects and sid not in subjects:
            continue
        for si, f in enumerate(sorted(by_subj[sid], key=lambda p: p.name), start=1):
            m = sio.loadmat(str(f))
            trials = sorted([k for k in m if re.search(r"eeg\d+$", k)],
                            key=lambda k: int(re.search(r"(\d+)$", k).group(1)))
            for ti, k in enumerate(trials):
                for w in _windows(m[k], win, max_per_trial):
                    Xs.append(w); ys.append(int(SEED_LABELS[ti])); subs.append(sid); sess.append(si)
    return _finalize(Xs, ys, subs, sess, normalize), ["negative", "neutral", "positive"]


def _load_deap(subjects, win_sec, max_per_trial, normalize, target="valence"):
    """target: 'valence'/'arousal' (binary @midpoint-5) or 'quadrant' (4-class V×A)."""
    import pickle
    win = int(win_sec * DEAP_FS); base = 3 * DEAP_FS
    quad = target == "quadrant"
    col = None if quad else {"valence": 0, "arousal": 1}[target]
    Xs, ys, subs, sess = [], [], [], []
    for f in sorted(DEAP_DIR.glob("s*.dat")):
        sid = int(f.stem[1:])
        if subjects and sid not in subjects:
            continue
        d = pickle.load(open(f, "rb"), encoding="latin1")
        data, labels = d["data"][:, :32, base:], d["labels"]      # 32 EEG ch, drop baseline
        for t in range(data.shape[0]):
            if quad:                                              # 2*(valence>5) + (arousal>5)
                lab = 2 * int(labels[t, 0] > 5) + int(labels[t, 1] > 5)
            else:
                lab = int(labels[t, col] > 5)
            for w in _windows(data[t], win, max_per_trial):
                Xs.append(w); ys.append(lab); subs.append(sid); sess.append(1)
    classes = ["LVLA", "LVHA", "HVLA", "HVHA"] if quad else ["low", "high"]
    return _finalize(Xs, ys, subs, sess, normalize), classes


def _load_seed_iv(subjects, win_sec, max_per_trial, normalize):
    import scipy.io as sio
    win = int(win_sec * SEEDIV_FS)
    Xs, ys, subs, sess = [], [], [], []
    for s in (1, 2, 3):
        labs = SEEDIV_LABELS[s]
        for f in sorted((SEEDIV_DIR / str(s)).glob("*.mat")):
            sid = int(f.name.split("_")[0])
            if subjects and sid not in subjects:
                continue
            m = sio.loadmat(str(f))
            trials = sorted([k for k in m if re.search(r"eeg\d+$", k)],
                            key=lambda k: int(re.search(r"(\d+)$", k).group(1)))
            for ti, k in enumerate(trials[:len(labs)]):
                for w in _windows(m[k], win, max_per_trial):
                    Xs.append(w); ys.append(int(labs[ti])); subs.append(sid); sess.append(s)
    return _finalize(Xs, ys, subs, sess, normalize), ["neutral", "sad", "fear", "happy"]


def load(name, subjects=None, win_sec=4.0, max_per_trial=20, normalize="trial_zscore", **_):
    if name == "SEED":
        (X, y, meta), classes = _load_seed(subjects, win_sec, max_per_trial, normalize)
    elif name == "SEED_IV":
        (X, y, meta), classes = _load_seed_iv(subjects, win_sec, max_per_trial, normalize)
    elif name in ("DEAP", "DEAP_valence", "DEAP_arousal", "DEAP_quadrant"):
        tgt = "quadrant" if name.endswith("quadrant") else ("arousal" if name.endswith("arousal") else "valence")
        (X, y, meta), classes = _load_deap(subjects, win_sec, max_per_trial, normalize, tgt)
    else:
        raise ValueError(name)
    return X, y, meta, classes
