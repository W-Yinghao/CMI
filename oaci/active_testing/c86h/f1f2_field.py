"""C86H real F1/F2 building blocks — 11-channel model factory + interface epoching + target
adapters (Brandl MOABB / ds007221 BIDS).

These are REAL implementations, validated against synthetic tensors and a synthetic BIDS
fixture + mockable MOABB loader (the PM's "implement pre-authorization against synthetic BIDS
fixtures + mocked loaders"). Only real-data BEHAVIOUR is deferred to authorization: on a real
dataset that violates a locked assumption the caller returns a C86-E blocker, never an on-the-fly
code change.
"""
from __future__ import annotations

import numpy as np

from . import contract as K

IN_CHANS = len(K.INTERFACE_CHANNELS)                 # 11
IN_TIMES = int(K.INTERFACE_SFREQ_HZ * (K.INTERFACE_WINDOW_S[1] - K.INTERFACE_WINDOW_S[0]))  # 480
EVENT_ID = {"left_hand": 0, "right_hand": 1}


class C86EError(RuntimeError):
    """Raised when a real dataset violates a locked interface assumption -> C86-E blocker."""


# --------------------------------------------------------------- 11-channel model factory
def build_c86h_model(n_classes: int = 2):
    """The fresh 11-channel shallow_convnet (in_chans=11, in_times=480) — the C86H retarget of
    the C84 20-channel _model_factory."""
    from oaci.models import build_model
    return build_model("shallow_convnet", in_chans=IN_CHANS, in_times=IN_TIMES,
                       n_classes=n_classes, temporal_filters=40, temporal_kernel_samples=25,
                       pool_kernel_samples=75, pool_stride_samples=15, dropout=0.5)


# --------------------------------------------------------------- 11-channel interface epoching
def interface_epochs(epochs) -> tuple:
    """Map an mne.Epochs (left_hand/right_hand) onto the locked 11-ch/160Hz/[0,3)s/4-38Hz
    interface. Returns (X[n,11,480] z-scored per trial, y[n] in {0,1}). Assumption violations
    (missing channels, wrong events) raise C86EError -> C86-E."""
    import mne
    want = list(K.INTERFACE_CHANNELS)
    if not set(want).issubset(set(epochs.ch_names)):
        raise C86EError(f"interface channels {set(want) - set(epochs.ch_names)} absent")
    ep = epochs.copy().pick(want)
    if abs(ep.info["sfreq"] - K.INTERFACE_SFREQ_HZ) > 1e-6:
        ep = ep.resample(K.INTERFACE_SFREQ_HZ)
    ep = ep.filter(K.INTERFACE_BAND_HZ[0], K.INTERFACE_BAND_HZ[1], verbose="error")
    ep = ep.crop(tmin=K.INTERFACE_WINDOW_S[0],
                 tmax=K.INTERFACE_WINDOW_S[1] - 1.0 / K.INTERFACE_SFREQ_HZ)
    ep = ep.reorder_channels(want)
    X = ep.get_data(copy=True).astype(np.float64)
    if X.shape[1] != IN_CHANS:
        raise C86EError(f"interface channel count {X.shape[1]} != {IN_CHANS}")
    if X.shape[2] != IN_TIMES:                       # tolerate off-by-one from crop rounding
        X = X[:, :, :IN_TIMES] if X.shape[2] > IN_TIMES else np.pad(
            X, ((0, 0), (0, 0), (0, IN_TIMES - X.shape[2])))
    ev = epochs.events[:, -1]
    inv = {v: k for k, v in epochs.event_id.items()}
    y = np.array([EVENT_ID[inv[e].split("/")[0]] if inv[e].split("/")[0] in EVENT_ID
                  else EVENT_ID.get(inv[e], -1) for e in ev])
    if np.any(y < 0):
        raise C86EError("epochs contain events outside {left_hand,right_hand}")
    mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-7
    return ((X - mu) / sd), y.astype(int)


# --------------------------------------------------------------- ds007221 BIDS adapter (mne-bids)
DS007221_SUBJECTS = tuple(f"sub-{i}" for i in range(37, 74))     # 37..73 (locked)


def load_ds007221_bids(bids_root: str, subject: str) -> tuple:
    """Load one ds007221 hybrid subject via mne-bids and map to the 11-ch interface,
    exhaustively over every task-hybrid header for that subject. Returns
    (X[n,11,480], y[n], trial_ids[n], raw_file_sha256). C86-E on any assumption violation
    (subject not in 37..73, no hybrid acquisition, unparseable run, no left/right events)."""
    import hashlib
    from mne_bids import BIDSPath, read_raw_bids
    import mne
    if subject not in DS007221_SUBJECTS:
        raise C86EError(f"ds007221 subject {subject} not in locked 37..73")
    sub = subject.split("-")[-1]
    bp = BIDSPath(subject=sub, task="hybrid", datatype="eeg", root=bids_root, suffix="eeg")
    headers = {".vhdr", ".edf", ".bdf", ".set", ".fif"}
    matches = sorted([m for m in bp.match() if m.extension in headers],
                     key=lambda m: str(m.fpath))       # deterministic order (file-order agnostic)
    if not matches:
        raise C86EError(f"ds007221 sub-{sub} task-hybrid header not found under {bids_root}")
    Xs, ys, tids, shas = [], [], [], []
    for m in matches:
        run = m.run or "1"
        try:
            raw = read_raw_bids(m, verbose="error")
        except Exception as e:
            raise C86EError(f"ds007221 sub-{sub} run-{run} unparseable: {type(e).__name__}")
        with open(str(m.fpath), "rb") as fh:
            shas.append(hashlib.sha256(fh.read()).hexdigest())
        events, event_id = mne.events_from_annotations(raw, verbose="error")
        keep = {k: v for k, v in event_id.items()
                if k.split("/")[0] in EVENT_ID or k in EVENT_ID}
        if not keep:
            raise C86EError(f"ds007221 sub-{sub} run-{run}: no left_hand/right_hand annotations")
        epochs = mne.Epochs(raw, events, keep, tmin=K.INTERFACE_WINDOW_S[0],
                            tmax=K.INTERFACE_WINDOW_S[1], baseline=None, preload=True,
                            verbose="error")
        X, y = interface_epochs(epochs)
        Xs.append(X); ys.append(y)
        tids += [f"sub-{sub}_run-{run}_e{i}" for i in range(len(y))]   # stable deterministic
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)
    if len(set(tids)) != len(tids):
        raise C86EError(f"ds007221 sub-{sub}: duplicate trial ids")
    combined_sha = hashlib.sha256("|".join(shas).encode()).hexdigest()
    return X, y, tids, combined_sha


# --------------------------------------------------------------- MOABB source/Brandl adapter
def load_moabb_dataset(dataset_name: str, subjects, loader=None) -> dict:
    """Load MOABB source (Lee2019_MI/Cho2017/PhysionetMI) or Brandl2020 to the 11-ch interface.
    Returns {subject: (X[n,11,480], y[n], trial_ids[n], groups[n])}, where groups carry the real
    ``dataset|subject|session|run`` identity. ``loader`` may be injected for testing (returns
    {subject: mne.Epochs} or {subject: (X, y)} or {subject: (X, y, metadata_df)})."""
    if loader is None:
        def loader(name, subs):
            import moabb.datasets as md
            from moabb.paradigms import MotorImagery
            ds = getattr(md, name)()
            para = MotorImagery(n_classes=2, events=["left_hand", "right_hand"],
                                fmin=K.INTERFACE_BAND_HZ[0], fmax=K.INTERFACE_BAND_HZ[1],
                                tmin=K.INTERFACE_WINDOW_S[0], tmax=K.INTERFACE_WINDOW_S[1],
                                channels=list(K.INTERFACE_CHANNELS), resample=K.INTERFACE_SFREQ_HZ)
            out = {}
            for s in subs:
                Xs, ys, meta = para.get_data(dataset=ds, subjects=[s], return_epochs=False)
                out[s] = (Xs, ys, meta)              # meta: DataFrame with subject/session/run
            return out

    def _interface_XY(Xarr, ylabels):
        X = np.asarray(Xarr, dtype=np.float64)
        if X.shape[1] != IN_CHANS or X.shape[2] < IN_TIMES:
            raise C86EError(f"{dataset_name}: shape {X.shape} != (*,{IN_CHANS},{IN_TIMES})")
        X = X[:, :, :IN_TIMES]
        mu = X.mean(2, keepdims=True); sd = X.std(2, keepdims=True) + 1e-7
        y = np.array([EVENT_ID[str(v)] if str(v) in EVENT_ID else int(v) for v in ylabels], dtype=int)
        return (X - mu) / sd, y

    raw = loader(dataset_name, list(subjects))
    result = {}
    for s, val in raw.items():
        sess = run = None
        if hasattr(val, "get_data"):                 # mne.Epochs -> interface
            X, y = interface_epochs(val)
        elif isinstance(val, tuple) and len(val) >= 3 and hasattr(val[2], "columns"):
            X, y = _interface_XY(val[0], val[1])      # (X, y, metadata DataFrame)
            meta = val[2]
            sess = [str(v) for v in meta["session"]] if "session" in meta else None
            run = [str(v) for v in meta["run"]] if "run" in meta else None
        else:                                        # (X, y) already on the interface
            X, y = _interface_XY(val[0], val[1])
        n = len(y)
        sess = sess or ["0"] * n
        run = run or ["0"] * n
        trial_ids = [f"{dataset_name}|s{s}|sess{sess[i]}|run{run[i]}|t{i}" for i in range(n)]
        groups = [f"{dataset_name}|subject={s}|session={sess[i]}|run={run[i]}" for i in range(n)]
        result[s] = (X, y, trial_ids, groups)
    return result


# --------------------------------------------------------------- synthetic BIDS fixture (for tests)
def write_synthetic_ds007221_fixture(bids_root: str, subject_ints, n_trials: int = 40,
                                     seed: int = 0) -> None:
    """Write a tiny synthetic ds007221-like BIDS tree (mne-bids) with left_hand/right_hand
    annotations, so the BIDS adapter is validated without the real dataset."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids
    rng = np.random.default_rng(seed)
    sfreq = 200.0                                    # deliberately != 160 to exercise resample
    ch = list(K.INTERFACE_CHANNELS) + ["EXTRA1", "EXTRA2"]
    for si in subject_ints:
        n_s = int(sfreq * 5)
        data = rng.standard_normal((len(ch), n_s * n_trials)) * 1e-6
        info = mne.create_info(ch, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="error")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                        on_missing="ignore", verbose="error")
        onsets = np.arange(n_trials) * 5.0 + 0.5
        descs = ["left_hand" if k % 2 == 0 else "right_hand" for k in range(n_trials)]
        raw.set_annotations(mne.Annotations(onsets, [1.0] * n_trials, descs))
        bp = BIDSPath(subject=str(si), task="hybrid", datatype="eeg", root=bids_root)
        write_raw_bids(raw, bp, allow_preload=True, format="BrainVision",
                       overwrite=True, verbose="error")
