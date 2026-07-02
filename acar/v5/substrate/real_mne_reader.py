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

Channel handling (Stage-1B10 + Stage-1B11): raw channel names are case-normalized and aliased to the canonical 19 (modern 10-10
names T7/T8/P7/P8 → old T3/T4/T5/T6; Fp case-normalized); extra non-canonical channels are dropped by the pick; a duplicate logical
channel fails closed. A canonical channel that is missing after aliasing fails closed UNLESS it is in the reviewed per-cohort
montage-completion whitelist (montage_completion), in which case it is interpolated (spherical-spline over standard positions) and the
interpolation is recorded in the SubjectWindows provenance; the OUTPUT montage is always the old-10-20 canonical order.
"""
from __future__ import annotations
import os
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


def _windows_from_raw(raw, np, disease, cohort, mne=None):
    """SINGLE-recording DSP core → (float32 window array (n_win, 19, 512), interpolation provenance). Order: (2) alias → (3) detect
    missing → (4) interpolate WHITELISTED missing (else FAIL) → (5) pick canonical 19 → (6) avg ref → (7) bandpass → (8) resample →
    (9) 512-sample non-overlap windows → (10) per-trial z-score. Windows are taken WITHIN this recording only."""
    from acar.v5.substrate import channel_aliases as CA
    from acar.v5.substrate import montage_completion as MC
    cfg = PC.PREPROCESSING_CONFIG
    try:                                                      # (2-4) montage completion of whitelisted missing (or no-op)
        raw, interp = MC.complete_missing_channels(raw, disease, cohort, mne=mne)
    except MC.MontageCompletionError as e:
        raise RealMneReaderError(str(e))
    try:                                                      # alias raw names → canonical; fail-closed on dup-logical / missing
        ordered_src = CA.ordered_source_names(list(raw.ch_names))   # source names in CANONICAL order (length 19)
    except CA.ChannelAliasError as e:
        raise RealMneReaderError(str(e))
    raw = raw.pick(ordered_src)                               # (5) select the 19 source channels (extras/donors dropped)
    raw.set_eeg_reference("average", projection=False)        # (6)
    raw.filter(l_freq=cfg["bandpass_hz"][0], h_freq=cfg["bandpass_hz"][1])   # (7)
    raw.resample(cfg["resample_hz"])                          # (8)
    data = np.asarray(raw.get_data(units="uV"), dtype=np.float64)   # (n_selected, n_times), microvolt
    if data.ndim != 2 or data.shape[0] != 19:
        raise RealMneReaderError(f"expected (19, n_times) data, got {data.shape}")
    cur = list(raw.ch_names)                                  # reorder data ROWS to canonical order (robust to pick ordering)
    data = data[[cur.index(s) for s in ordered_src]]         # ordered_src is already in canonical order
    w = cfg["window_samples"]
    n_win = data.shape[1] // w
    if n_win < 1:
        raise RealMneReaderError(f"recording too short for one {w}-sample window (n_times={data.shape[1]})")
    data = data[:, : n_win * w]                               # (9) drop the trailing partial window (never spans recordings)
    windows = data.reshape(19, n_win, w).transpose(1, 0, 2)   # (n_win, 19, 512)
    return _per_trial_zscore(windows, np).astype(np.float32), interp   # (10) z-score


def _provenance(raw_manifest_sha256, montage):
    parts = ["real_mne_reader"]
    if raw_manifest_sha256:
        parts.append(f"raw_manifest_sha256={raw_manifest_sha256}")
    parts.append(f"channel_alias_policy_sha256={PC.channel_alias_policy_sha256()}")
    parts.append(f"montage_completion_policy_sha256={PC.montage_completion_policy_sha256()}")
    parts.append(f"interpolated={montage['interpolated']}")
    parts.append(f"n_interpolated={montage['n_interpolated']}")
    parts.append(f"donor_count={montage['donor_count']}")
    return ";".join(parts)


def _aggregate_montage(per_recording):
    """Combine per-recording interpolation into a subject-level montage_completion record."""
    interpolated = sorted({c for r in per_recording for c in r["interpolated"]})
    donor_counts = [r["donor_count"] for r in per_recording if r["n_interpolated"]]
    return {"interpolated": interpolated, "n_interpolated": len(interpolated),
            "donor_count": (min(donor_counts) if donor_counts else 0), "by_recording": per_recording}


def _wrap(windows, disease, cohort, raw_subject_id, raw_manifest_sha256=None, montage=None):
    import numpy as np
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    montage = montage or {"interpolated": [], "n_interpolated": 0, "donor_count": 0, "by_recording": []}
    sw = SW.SubjectWindows(
        subject_key=f"{disease}/{cohort}/{raw_subject_id}", disease=disease, cohort=cohort, raw_subject_id=raw_subject_id,
        n_windows=int(windows.shape[0]), n_channels=19, n_samples=PC.PREPROCESSING_CONFIG["window_samples"],
        sfreq=PC.PREPROCESSING_CONFIG["resample_hz"], channels=PC.CHANNELS_19,
        preprocessing_config_sha256=PC.config_sha256(), windows=windows,
        provenance=_provenance(raw_manifest_sha256, montage), montage_completion=montage)
    SW.validate_subject_windows(sw)                           # fail-closed vs the pinned config + payload shape/finiteness
    return sw


def raw_to_windows(raw, disease, cohort, raw_subject_id, mne=None):
    """mne-INDEPENDENT (unless interpolation needed) single-recording DSP → validated SubjectWindows. numpy imported lazily."""
    import numpy as np
    windows, interp = _windows_from_raw(raw, np, disease, cohort, mne=mne)
    montage = _aggregate_montage([{"recording": None, **interp}])
    return _wrap(windows, disease, cohort, raw_subject_id, montage=montage)


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
    per_windows, per_interp = [], []
    for p in files:                                           # window EACH recording independently
        wins, interp = _windows_from_raw(_read_raw(p, mne), np, disease, cohort, mne=mne)
        per_windows.append(wins)
        per_interp.append({"recording": os.path.basename(p), **interp})
    windows = np.concatenate(per_windows, axis=0)             # concatenate WINDOWS, never raws
    montage = _aggregate_montage(per_interp)
    return _wrap(windows, disease, cohort, raw_subject_id, raw_manifest_sha256=manifest["manifest_sha256"], montage=montage)
