"""V2 real-EEG offline loader + frozen common-channel grid + adapt/eval split (review V2_FROZEN).

OFFLINE ONLY: points MNE/MOABB at the read-only datalake cache (no download). Loads binary
left/right motor imagery via MOABB's LeftRightImagery paradigm, harmonised by a FROZEN
deterministic preprocessing pipeline (band-pass + resample + fixed post-cue window). Channel-name
normalisation + a global ordered common-channel grid (intersection) are computed from channel
NAMES only -- never from outcomes. The adapt/eval split is contiguous (first-half adapt / second-
half eval) per the pre-registration; NEVER random (random split bleeds session drift across both).

Nothing here touches target labels for adaptation: y is returned for EVALUATION only; the runner
must keep the adapt block's labels out of source training / operator fitting / metadata.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

DATALAKE = "/projects/EEG-foundation-model/datalake/raw"


def configure_offline() -> None:
    """Point MNE/MOABB at the datalake; disable network. setdefault so an outer env wins."""
    for k in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH", "MNE_DATASETS_GIGADB_PATH",
              "MNE_DATASETS_EEGBCI_PATH", "MNE_DATASETS_SCHIRRMEISTER2017_PATH"):
        os.environ.setdefault(k, DATALAKE)
    os.environ.setdefault("MOABB_RESULTS", os.path.join(DATALAKE, "moabb_results"))


# Frozen preprocessing constants (harmonisation, NOT adaptation -- review §4).
FS = 250.0                 # common sampling rate (Hz)
FMIN, FMAX = 8.0, 30.0     # mu/beta MI band
WINDOW_S = 2.0             # post-cue analysis window (s); 500 samples @250Hz -- safe for all datasets
N_TIMES = int(WINDOW_S * FS)


@dataclass
class RealEpochs:
    X: np.ndarray              # [n, ch, t] float32, channels in the requested grid order
    y: np.ndarray              # [n] int (0=left_hand, 1=right_hand)
    subject: np.ndarray        # [n] int subject id
    session: np.ndarray        # [n] int session index (0-based, in sorted session order)
    run: np.ndarray            # [n] int run index (0-based within session)
    channels: list             # ordered channel names actually used (length ch)
    dataset: str
    session_names: list = field(default_factory=list)   # original session labels, sorted


def normalize_ch(name: str) -> str:
    """Normalisation KEY for cross-dataset channel matching (case/prefix-insensitive)."""
    s = name.upper().replace("EEG-", "").replace("EEG", "").replace(" ", "").strip()
    return s


def _label_to_int(y_str: np.ndarray) -> np.ndarray:
    return (np.asarray(y_str) == "right_hand").astype(np.int64)   # 0=left, 1=right


def load_dataset(name: str, subjects, *, channels: list | None = None,
                 return_session_names: bool = False) -> RealEpochs:
    """Offline-load one MOABB MI dataset, binary L/R, frozen preprocessing.

    channels: if given, a list of normalised KEYS to select (in this order); each must exist in the
    dataset (else ValueError -- the common grid is precomputed so this never silently drops). If None,
    use the dataset's full EEG channel set (native order).
    """
    configure_offline()
    import warnings
    warnings.filterwarnings("ignore")
    from moabb.paradigms import LeftRightImagery
    from h2cmi.data.real_metadata import MOABB_CLASS

    ds = MOABB_CLASS[name]()
    para = LeftRightImagery(fmin=FMIN, fmax=FMAX, resample=FS)
    ep, y_str, meta = para.get_data(dataset=ds, subjects=list(subjects), return_epochs=True)
    raw_names = list(ep.ch_names)
    keymap = {}                                   # normalised key -> original name (first wins)
    for nm in raw_names:
        keymap.setdefault(normalize_ch(nm), nm)
    if channels is None:
        sel_keys = [normalize_ch(nm) for nm in raw_names]
    else:
        sel_keys = list(channels)
        missing = [k for k in sel_keys if k not in keymap]
        if missing:
            raise ValueError(f"{name}: missing common-grid channels {missing}")
    pick = [keymap[k] for k in sel_keys]
    ep = ep.copy().pick(pick)                     # reorder to the requested grid
    X = ep.get_data(copy=False).astype(np.float32)     # [n, ch, t]
    X = X[:, :, :N_TIMES] if X.shape[2] >= N_TIMES else np.pad(X, ((0, 0), (0, 0), (0, N_TIMES - X.shape[2])))
    # per-trial z-score per channel (scale harmonisation across device gains; deterministic, label-free)
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-7
    X = (X - mu) / sd
    y = _label_to_int(y_str)
    subj = meta["subject"].to_numpy().astype(np.int64)
    sess_names = sorted(meta["session"].astype(str).unique().tolist())
    sess_map = {s: i for i, s in enumerate(sess_names)}
    sess = meta["session"].astype(str).map(sess_map).to_numpy().astype(np.int64)
    if "run" in meta:
        run_names = sorted(meta["run"].astype(str).unique().tolist())
        run_map = {r: i for i, r in enumerate(run_names)}
        run = meta["run"].astype(str).map(run_map).to_numpy().astype(np.int64)
    else:
        run = np.zeros(len(y), np.int64)
    out = RealEpochs(X=X, y=y, subject=subj, session=sess, run=run, channels=[keymap[k] for k in sel_keys],
                     dataset=name, session_names=sess_names)
    return out


def common_channel_grid(names_per_dataset: dict) -> tuple[list, str]:
    """Ordered global intersection of normalised channel keys across datasets. Deterministic
    (sorted), outcome-independent. Returns (ordered_keys, sha256_of_ordered_list)."""
    sets = [set(normalize_ch(n) for n in names) for names in names_per_dataset.values()]
    inter = set.intersection(*sets) if sets else set()
    ordered = sorted(inter)
    h = hashlib.sha256("\n".join(ordered).encode()).hexdigest()
    return ordered, h


def contiguous_split(ep: RealEpochs, subject: int, session: int) -> tuple[np.ndarray, np.ndarray]:
    """Mutually-exclusive adapt/eval indices for one subject+session: if >1 run, first-half runs ->
    adapt, second-half -> eval; else first 50% contiguous trials -> adapt, last 50% -> eval. Returns
    (adapt_idx, eval_idx) into ep arrays. Contiguous (NOT random) per the pre-registration."""
    m = np.where((ep.subject == subject) & (ep.session == session))[0]
    if len(m) < 2:
        return m[:0], m[:0]
    runs = ep.run[m]
    uruns = np.unique(runs)
    if len(uruns) >= 2:
        half = len(uruns) // 2
        adapt_runs, eval_runs = set(uruns[:half].tolist()), set(uruns[half:].tolist())
        adapt = m[np.isin(runs, list(adapt_runs))]
        evalm = m[np.isin(runs, list(eval_runs))]
    else:
        h = len(m) // 2                            # contiguous (m is in load order = time order)
        adapt, evalm = m[:h], m[h:]
    return adapt, evalm
