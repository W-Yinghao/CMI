"""ACAR v4 — held-out raw→windows reader for the external Arm-B sites (Blocker 2; SYNTHETIC-TESTED, code-only).

NON-BINDING / NOT EXECUTED ON REAL HELD-OUT DATA HERE. This is the SELECTION + VALIDATION + WINDOWING + KEY-ASSEMBLY layer
that turns a BIDS-laid-out held-out site into (X, y, subject_ids, recording_ids, window_index) — the input the FROZEN
encoder then embeds into the erm_0 dump. It reuses the frozen per-site contract from prepare_external_dump
(DATASET_SPECS / resting_run_selector / parse_diagnosis_map / validate_channels_fs).

WHAT IS IMPLEMENTED + GUARDED (synthetic mini-BIDS fixtures, no mne, no real read):
  participants.tsv + BIDS-entity discovery; resting/exclude/UNKNOWN 3-way task classification (unknown → fail-closed);
  per-subject diagnosis binding; channel/Fs validation; cohort-namespaced subject ids; non-overlapping windowing into
  [n_win, canon_channels, n_times]; unique (subject, recording, window) keys; fail-closed on every listed defect.

WHAT IS DELIBERATELY GATED (the remaining real-data pieces; NOT run here):
  the actual raw-signal DSP (read EDF/BrainVision → 19-ch 10-20 montage harmonize → resample 128 Hz → 0.5–45 Hz bandpass)
  is supplied by a `signal_provider(record) -> np.ndarray [canon_channels, n_samples]` callback. Tests inject a SYNTHETIC
  provider; the REAL run must inject an mne-based provider — `make_real_signal_provider` raises RawSignalDSPNotWiredError
  until that is wired at substrate-provisioning time. The FROZEN encoder + reading real held-out raw remain forbidden until
  the protocol is tagged and both blockers (encoder + this reader's real provider) are resolved.
"""
from __future__ import annotations
import os

import numpy as np

from acar.v4.prepare_external_dump import (DATASET_SPECS, _spec, _task_tokens, parse_diagnosis_map,
                                           validate_channels_fs)

RAW_EEG_EXTS = (".edf", ".bdf", ".set", ".vhdr", ".fif", ".cnt", ".dat")


class RawSignalDSPNotWiredError(RuntimeError):
    """Raised when the real raw→canonical DSP provider is requested. The held-out reader's selection/validation/windowing
    layer is implemented + synthetic-tested, but the actual EDF/BrainVision read + 19ch/128Hz/0.5–45Hz harmonization needs
    mne + real held-out raw, which is gated until substrate provisioning. NEVER silently fabricates signal."""


# ----------------------------------------------------------------------------- BIDS discovery (pure; no signal read)

def _parse_entities(stem):
    """Parse a BIDS stem 'sub-01_ses-1_task-rest_run-1_eeg' → {sub, ses, task, run, suffix}. Suffix = last '_' token."""
    parts = stem.split("_")
    ent = {}
    for p in parts[:-1]:
        if "-" in p:
            k, v = p.split("-", 1)
            ent[k] = v
    ent["suffix"] = parts[-1] if parts else ""
    return ent


def _parse_participants_tsv(path):
    """Minimal TSV reader → list of row dicts. Fail-closed on missing file / empty / no header."""
    if not os.path.isfile(path):
        raise ValueError(f"participants.tsv not found: {path}")
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    if len(lines) < 2:
        raise ValueError(f"participants.tsv has no data rows: {path}")
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        fields = ln.split("\t")
        if len(fields) != len(header):
            raise ValueError(f"participants.tsv row has {len(fields)} fields, expected {len(header)}: {ln!r}")
        rows.append(dict(zip(header, fields)))
    return rows


def _read_sidecar_meta(raw_path):
    """Read (n_channels, fs) from the BIDS sidecars next to a raw file: SamplingFrequency from *_eeg.json; channel count
    from *_channels.tsv (data rows) else EEGChannelCount from the json. Returns (n_channels|None, fs|None)."""
    import json
    base = raw_path
    for ext in RAW_EEG_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    fs = None
    n_channels = None
    jpath = base + ".json"
    if os.path.isfile(jpath):
        with open(jpath) as f:
            meta = json.load(f)
        fs = meta.get("SamplingFrequency")
        n_channels = meta.get("EEGChannelCount")
    ch_tsv = base[: base.rfind("_")] + "_channels.tsv" if "_" in base else None
    if ch_tsv and os.path.isfile(ch_tsv):
        with open(ch_tsv) as f:
            data_rows = [ln for ln in f.read().splitlines() if ln.strip() != ""]
        n_channels = max(0, len(data_rows) - 1)              # minus header
    return n_channels, fs


def discover_recordings(bids_root, site):
    """Walk a BIDS-laid-out site → ordered list of recording descriptors {subject, recording_id, task, run, ses,
    n_channels, fs, path}. Pure metadata discovery (reads only sidecars, never the signal). Deterministic order."""
    _spec(site)                                              # fail-closed: unknown site
    recs = []
    for dirpath, _dirs, files in os.walk(bids_root):
        for fn in sorted(files):
            if "_eeg." not in fn or not fn.endswith(RAW_EEG_EXTS):
                continue
            stem = fn
            for ext in RAW_EEG_EXTS:
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            ent = _parse_entities(stem)
            if "sub" not in ent or "task" not in ent:
                raise ValueError(f"{site}: BIDS file missing sub-/task- entity: {fn!r}")
            path = os.path.join(dirpath, fn)
            n_channels, fs = _read_sidecar_meta(path)
            rid_bits = [f"sub-{ent['sub']}"]
            if "ses" in ent:
                rid_bits.append(f"ses-{ent['ses']}")
            rid_bits.append(f"task-{ent['task']}")
            if "run" in ent:
                rid_bits.append(f"run-{ent['run']}")
            recs.append({"sub": ent["sub"], "task": ent["task"], "run": ent.get("run"), "ses": ent.get("ses"),
                         "recording_id": "_".join(rid_bits), "n_channels": n_channels, "fs": fs, "path": path})
    recs.sort(key=lambda r: r["recording_id"])
    return recs


def _classify_task(task, site):
    """3-way: 'rest' (keep) / 'exclude' (drop) / 'unknown' (FAIL-CLOSED). A task matching neither the site's resting nor
    its exclude tokens is UNKNOWN → the caller raises (no silent inclusion/exclusion of an unrecognized task)."""
    spec = _spec(site)
    toks = _task_tokens(task)
    if set(spec["exclude_tokens"]) & toks:
        return "exclude"
    if set(spec["resting_tokens"]) & toks:
        return "rest"
    return "unknown"


# ----------------------------------------------------------------------------- index (selection + labels + validation)

def build_heldout_index(bids_root, site):
    """Return the ORDERED list of resting recordings to embed, each {subject_id (cohort-namespaced), recording_id, task,
    n_channels, fs, path, y}. FAIL-CLOSED: unknown task; missing per-subject diagnosis; missing channel/Fs metadata;
    channel/Fs mismatch (validate_channels_fs); duplicate recording_id; subject_id collision; no resting recording."""
    spec = _spec(site)
    diag = parse_diagnosis_map(_parse_participants_tsv(os.path.join(bids_root, "participants.tsv")), site)
    recs = discover_recordings(bids_root, site)
    index, seen_rid, subj_label = [], set(), {}
    for r in recs:
        cls = _classify_task(r["task"], site)
        if cls == "exclude":
            continue                                         # walking/gait etc. — expected, dropped
        if cls == "unknown":
            raise ValueError(f"{site}: unrecognized task {r['task']!r} (neither resting nor excluded) — fail-closed")
        pid = f"sub-{r['sub']}"
        if pid not in diag:
            raise ValueError(f"{site}: no diagnosis for {pid} in participants.tsv")
        if r["n_channels"] is None or r["fs"] is None:
            raise ValueError(f"{site}: missing channel/Fs metadata for {r['recording_id']}")
        validate_channels_fs(r["n_channels"], r["fs"], site)
        if r["recording_id"] in seen_rid:
            raise ValueError(f"{site}: duplicate recording_id {r['recording_id']!r}")
        seen_rid.add(r["recording_id"])
        subject_id = f"{site}/{pid}"                          # cohort-namespaced (no cross-cohort id collision)
        if subject_id in subj_label and subj_label[subject_id] != diag[pid]:
            raise ValueError(f"{site}: subject {subject_id} has mixed diagnosis labels")
        subj_label[subject_id] = diag[pid]
        index.append({"subject_id": subject_id, "recording_id": r["recording_id"], "task": r["task"],
                      "n_channels": r["n_channels"], "fs": r["fs"], "path": r["path"], "y": diag[pid]})
    if not index:
        raise ValueError(f"{site}: no resting recordings selected")
    return index


# ----------------------------------------------------------------------------- windowing (via injected signal_provider)

def assemble_windows(index, *, pipeline_config, signal_provider):
    """Window each recording's canonical signal into [n_win, canon_channels, n_times] and assemble label-free deployment
    arrays + labels + keys. `signal_provider(record) -> np.ndarray [canon_channels, n_samples]` at the canonical rate (DSP
    is the provider's job — synthetic in tests, mne at the real run). FAIL-CLOSED: wrong channel count, non-finite signal,
    recording too short (< one window), duplicate (subject,recording,window)."""
    canon_ch = int(pipeline_config["canon_channels"])
    n_times = int(round(pipeline_config["resample_fs"] * pipeline_config["window_sec"]))
    X, y, subj, rec, wi = [], [], [], [], []
    for r in index:
        sig = np.asarray(signal_provider(r), dtype="<f8")
        if sig.ndim != 2 or sig.shape[0] != canon_ch:
            raise ValueError(f"{r['recording_id']}: signal must be [canon_channels={canon_ch}, n_samples], got {sig.shape}")
        if not np.all(np.isfinite(sig)):
            raise ValueError(f"{r['recording_id']}: non-finite signal")
        n_win = sig.shape[1] // n_times
        if n_win < 1:
            raise ValueError(f"{r['recording_id']}: signal too short for one {n_times}-sample window")
        w = sig[:, : n_win * n_times].reshape(canon_ch, n_win, n_times).transpose(1, 0, 2)   # [n_win, C, T]
        X.append(w)
        y.extend([int(r["y"])] * n_win)
        subj.extend([r["subject_id"]] * n_win)
        rec.extend([r["recording_id"]] * n_win)
        wi.extend(range(n_win))
    X = np.concatenate(X, axis=0)
    subject_id = np.array(subj); recording_id = np.array(rec)
    window_index = np.asarray(wi, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    rows = list(zip(subject_id.tolist(), recording_id.tolist(), window_index.tolist()))
    if len(set(rows)) != len(rows):
        raise ValueError("(subject_id, recording_id, window_index) rows must be unique")
    return {"X": X, "y": y, "subject_id": subject_id, "recording_id": recording_id, "window_index": window_index}


def read_heldout(site, bids_root, *, pipeline_config, signal_provider):
    """Full held-out read: build_heldout_index + assemble_windows. Returns the windowed deployment arrays + labels + keys.
    NOTE: this reads the real held-out BIDS tree + (via the provider) the real signal — it must NOT be invoked before the
    protocol is tagged and a real signal_provider is supplied. Synthetic guards inject a synthetic provider."""
    index = build_heldout_index(bids_root, site)
    return assemble_windows(index, pipeline_config=pipeline_config, signal_provider=signal_provider)


def make_real_signal_provider(pipeline_config):
    """Return the REAL (mne-based) raw→canonical signal provider. GATED: not wired here — the EDF/BrainVision read + 19-ch
    10-20 montage harmonize + resample 128 Hz + 0.5–45 Hz bandpass needs mne + real held-out raw. Calling the returned
    provider raises RawSignalDSPNotWiredError until that is implemented at substrate-provisioning time."""
    def _provider(record):
        raise RawSignalDSPNotWiredError(
            f"real raw-signal DSP not wired for {record.get('recording_id')!r}: implement the mne EDF/BrainVision read + "
            f"19ch 10-20 / 128Hz / 0.5-45Hz harmonization (FROZEN_PIPELINE) at substrate-provisioning time; see "
            f"notes/ACAR_V4_SUBSTRATE_REGEN_PLAN.md.")
    return _provider
