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

Header read-repair (Stage-1B12, opt-in via `preprocess_subject(..., staging_dir=...)`): a few real cohorts have BrainVision *header*
defects (missing MarkerFile; internal DataFile/MarkerFile pointers left stale by BIDS renaming) that stop the pinned mne from even
opening the recording. `brainvision_read_repair` materializes an EPHEMERAL, audited repaired header (+ a minimal synthesized marker,
never inferred events) under the staging dir — the raw signal is NEVER modified, copied, or read for this — and the reader opens that
repaired header. The repair is recorded in SubjectWindows.read_repair + provenance. Only the two reviewed, whitelisted repair modes
apply; everything else opens the original header unchanged (fail-closed at mne if it is genuinely unreadable).
"""
from __future__ import annotations
import hashlib
import json
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


def _provenance(raw_manifest_sha256, montage, read_repair):
    parts = ["real_mne_reader"]
    if raw_manifest_sha256:
        parts.append(f"raw_manifest_sha256={raw_manifest_sha256}")
    parts.append(f"channel_alias_policy_sha256={PC.channel_alias_policy_sha256()}")
    parts.append(f"montage_completion_policy_sha256={PC.montage_completion_policy_sha256()}")
    parts.append(f"brainvision_read_repair_policy_sha256={PC.brainvision_read_repair_policy_sha256()}")
    parts.append(f"channel_name_repair_policy_sha256={PC.channel_name_repair_policy_sha256()}")
    parts.append(f"interpolated={montage['interpolated']}")
    parts.append(f"n_interpolated={montage['n_interpolated']}")
    parts.append(f"donor_count={montage['donor_count']}")
    parts.append(f"read_repaired={read_repair['repaired']}")
    parts.append(f"n_read_repaired={len(read_repair['repaired'])}")
    parts.append(f"channel_name_repaired={read_repair.get('channel_name_repaired', [])}")
    parts.append(f"n_channel_name_repaired={len(read_repair.get('channel_name_repaired', []))}")
    return ";".join(parts)


def _aggregate_montage(per_recording):
    """Combine per-recording interpolation into a subject-level montage_completion record."""
    interpolated = sorted({c for r in per_recording for c in r["interpolated"]})
    donor_counts = [r["donor_count"] for r in per_recording if r["n_interpolated"]]
    return {"interpolated": interpolated, "n_interpolated": len(interpolated),
            "donor_count": (min(donor_counts) if donor_counts else 0), "by_recording": per_recording}


def _aggregate_read_repair(manifests):
    """Combine per-recording BrainVision read-repair manifests into a subject-level read_repair record (empty if none repaired).
    `channel_name_repaired` = the recordings whose header channel names were rewritten from channels.tsv (Stage-1B13 mode C)."""
    from acar.v5.substrate import brainvision_read_repair as BR
    manifests = list(manifests or [])
    channel_named = sorted(m["recording"] for m in manifests if m.get("repair_mode") == BR.MODE_CHANNEL_NAMES_FROM_TSV)
    return {"repaired": sorted(m["recording"] for m in manifests), "channel_name_repaired": channel_named,
            "by_recording": manifests}


def _wrap(windows, disease, cohort, raw_subject_id, raw_manifest_sha256=None, montage=None, read_repair=None):
    import numpy as np
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    montage = montage or {"interpolated": [], "n_interpolated": 0, "donor_count": 0, "by_recording": []}
    read_repair = read_repair or {"repaired": [], "channel_name_repaired": [], "by_recording": []}
    sw = SW.SubjectWindows(
        subject_key=f"{disease}/{cohort}/{raw_subject_id}", disease=disease, cohort=cohort, raw_subject_id=raw_subject_id,
        n_windows=int(windows.shape[0]), n_channels=19, n_samples=PC.PREPROCESSING_CONFIG["window_samples"],
        sfreq=PC.PREPROCESSING_CONFIG["resample_hz"], channels=PC.CHANNELS_19,
        preprocessing_config_sha256=PC.config_sha256(), windows=windows,
        provenance=_provenance(raw_manifest_sha256, montage, read_repair), montage_completion=montage, read_repair=read_repair)
    SW.validate_subject_windows(sw)                           # fail-closed vs the pinned config + payload shape/finiteness
    return sw


def raw_to_windows(raw, disease, cohort, raw_subject_id, mne=None):
    """mne-INDEPENDENT (unless interpolation needed) single-recording DSP → validated SubjectWindows. numpy imported lazily."""
    import numpy as np
    windows, interp = _windows_from_raw(raw, np, disease, cohort, mne=mne)
    montage = _aggregate_montage([{"recording": None, **interp}])
    return _wrap(windows, disease, cohort, raw_subject_id, montage=montage)


def _read_raw(path, mne):
    ext = os.path.splitext(path)[1].lower()
    loader = getattr(mne.io, _MNE_LOADER[ext])
    return loader(path, preload=True, verbose="ERROR")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repair_aware_reads(disease, cohort, subject, files, mne, staging_dir):
    """Stage-1B12: for each discovered recording, apply the reviewed, header-only BrainVision read-repair (if any) into `staging_dir`
    and open the (repaired-or-original) header; also build an audit hash over the ORIGINAL header + effective data/marker targets. The
    raw signal is never modified/copied. Returns (list[(orig_path, raw)], manifests, raw_manifest_sha256)."""
    from acar.v5.substrate import brainvision_read_repair as BR
    reads, manifests, entries = [], [], []
    for p in files:
        read_path, manifest = BR.repaired_read_path(disease, cohort, subject, p, staging_dir)
        if manifest is None:                                   # native / no reviewed repair → normal sidecar-audited manifest entry
            entries.append(("primary", os.path.abspath(p), _sha256_file(p)))
            for sc in RM.resolve_sidecars(p):
                entries.append(("sidecar", os.path.abspath(sc), _sha256_file(sc)))
        else:
            manifests.append(manifest)
            entries.append(("primary", manifest["original_vhdr_path"], manifest["original_header_sha256"]))
            entries.append(("data_target", manifest["data_file_target"], _sha256_file(manifest["data_file_target"])))
            if manifest.get("generated_marker_sha256") is not None:   # SYNTHESIZED marker → its stable CONTENT hash (the marker
                entries.append(("marker_synth", manifest["generated_marker_sha256"]))   # file lives in the EPHEMERAL staging dir, so
            else:                                                                        # never fold its random path into the audit
                entries.append(("marker_target", manifest["marker_file_target"], _sha256_file(manifest["marker_file_target"])))
        reads.append((p, _read_raw(read_path, mne)))
    raw_manifest_sha256 = hashlib.sha256(json.dumps(sorted(entries), sort_keys=True).encode()).hexdigest()
    return reads, manifests, raw_manifest_sha256


def preprocess_subject(disease, cohort, raw_subject_id, subject_dir, *, mne=None, staging_dir=None):
    """Discover raw-BIDS recordings, window EACH independently (no cross-recording windows), concatenate the window arrays → one
    validated SubjectWindows. mne is lazy (or injected for fixtures). When `staging_dir` is given, the reviewed BrainVision
    read-repair layer (Stage-1B12) may materialize an EPHEMERAL repaired header there (raw signal never touched) so a header-defective
    recording can be opened; the repair is audited in SubjectWindows.read_repair + provenance."""
    if mne is None:
        import mne as _mne  # lazy — never imported at module load
        mne = _mne
    import numpy as np
    if staging_dir is None:
        manifest = RM.build_manifest(subject_dir)             # raw-BIDS-only discovery + hashed manifest (incl. format sidecars)
        files = [e["path"] for e in manifest["files"] if e.get("role") == "primary"]
        reads = [(p, _read_raw(p, mne)) for p in files]
        repair_manifests, raw_manifest_sha256 = [], manifest["manifest_sha256"]
    else:                                                     # repair-aware read path (discovery = listing only; repair per recording)
        subject = os.path.basename(os.path.normpath(subject_dir))
        files = RM.discover_raw_recordings(subject_dir)
        reads, repair_manifests, raw_manifest_sha256 = _repair_aware_reads(disease, cohort, subject, files, mne, staging_dir)
    per_windows, per_interp = [], []
    for p, raw in reads:                                      # window EACH recording independently
        wins, interp = _windows_from_raw(raw, np, disease, cohort, mne=mne)
        per_windows.append(wins)
        per_interp.append({"recording": os.path.basename(p), **interp})
    windows = np.concatenate(per_windows, axis=0)             # concatenate WINDOWS, never raws
    montage = _aggregate_montage(per_interp)
    read_repair = _aggregate_read_repair(repair_manifests)
    return _wrap(windows, disease, cohort, raw_subject_id, raw_manifest_sha256=raw_manifest_sha256, montage=montage,
                 read_repair=read_repair)
