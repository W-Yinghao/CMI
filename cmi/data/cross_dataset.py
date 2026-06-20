"""Cross-dataset binary motor-imagery loader (Protocol C).

Loads several L/R-hand MI datasets restricted to a common channel set (aligned by name,
case-insensitive, reordered to a canonical order) and a common window/sampling rate, so a
model can train on some datasets and be tested on an entirely unseen dataset (different
device/montage/site). Domain D = dataset:subject. Labels fixed {left_hand:0, right_hand:1}.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from cmi.paths import configure_offline_moabb
configure_offline_moabb()
import mne  # noqa: E402
mne.set_log_level("ERROR")

# 21 channels shared by BNCI2014_001 (2a), Lee2019_MI, Cho2017 (verified).
CANON = ["C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CPz",
         "Cz", "FC1", "FC2", "FC3", "FC4", "Fz", "P1", "P2", "POz", "Pz"]
LABEL_MAP = {"left_hand": 0, "right_hand": 1}


def load_cross(datasets=("BNCI2014_001", "Lee2019_MI", "Cho2017"),
               tmin=0.5, tmax=3.5, resample=128, normalize="trial_zscore", max_subj=None):
    import moabb.datasets as D
    from moabb.paradigms import LeftRightImagery
    canon_up = [c.upper() for c in CANON]
    Xs, ys, dsl, subl = [], [], [], []
    for name in datasets:
        ds = getattr(D, name)()
        subs = ds.subject_list[:max_subj] if max_subj else ds.subject_list
        para = LeftRightImagery(fmin=8, fmax=30, tmin=tmin, tmax=tmax, resample=resample)
        epo, y, meta = para.get_data(ds, subjects=subs, return_epochs=True)
        chn = [c.upper() for c in epo.ch_names]
        idx = [chn.index(c) for c in canon_up]            # reorder to canonical
        X = epo.get_data(copy=False)[:, idx, :].astype("float32")
        if normalize == "trial_zscore":
            m = X.mean(axis=2, keepdims=True); s = X.std(axis=2, keepdims=True) + 1e-7
            X = (X - m) / s
        Xs.append(X)
        ys.extend(int(LABEL_MAP[v]) for v in y)
        dsl.extend([name] * len(y)); subl.extend(int(s) for s in meta["subject"])
    t = min(x.shape[2] for x in Xs)                   # datasets can differ by 1 sample (rounding)
    X = np.concatenate([x[:, :, :t] for x in Xs]).astype("float32")
    meta = pd.DataFrame({"dataset": dsl, "subject": subl})
    meta["domain"] = meta["dataset"] + ":" + meta["subject"].astype(str)
    return X, np.array(ys, dtype="int64"), meta, ["left_hand", "right_hand"]
