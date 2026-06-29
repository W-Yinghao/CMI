"""ACAR v4 — FROZEN external-input preparation layer for the held-out Arm-B sites.

NON-BINDING UNTIL TAGGED. This module FREEZES the raw-EEG→erm_0-feature-dump contract for the two admissible held-out
strata (notes/ACAR_V4_EXTERNAL_INPUT_SCHEMA.md / ACAR_FROZEN_v4.md §4) so the held-out signal is processed by exactly the
same pipeline as DEV. The PURE parts (resting-run selector, diagnosis-map parser, channel/Fs check, provenance hashers,
output-dump schema validator) are synthetic-tested in base env. The actual raw download + signal processing + frozen
encoder run (`prepare_dump`) is a GATED real step that runs ONLY after `acar-v4-protocol` is tagged — it is NOT executed
here and reads no real data in this module.

Firewall: the held-out diagnosis labels are written to the dump's `y_te` (used downstream ONLY for ΔR at CAL λ* / EVAL
scoring). They are NEVER used to fit the encoder or the source state — the source state is the DEV-frozen artifact.
"""
from __future__ import annotations
import hashlib
import json

import numpy as np

# Per-site frozen metadata. Values marked None must be confirmed at prep time (metadata-only) and recorded; the verified
# fields are from notes/ACAR_V4_LOCKBOX_AUDIT.md (primary-source 2026-06-29).
DATASET_SPECS = {
    "zenodo14808296": {
        "disease": "SCZ", "expected_channels": 64, "expected_fs": 1000,
        "resting_tokens": ("rest", "ec", "eyesclosed", "eyes-closed"), "exclude_tokens": (),
        "group_field": "group", "patient_tokens": ("sz", "scz", "schizophrenia", "patient", "1"),
        "control_tokens": ("hc", "control", "healthy", "0"),
    },
    "ds007526": {
        "disease": "PD", "expected_channels": None, "expected_fs": None,      # confirm ch/Fs at prep (metadata-only)
        "resting_tokens": ("rest",), "exclude_tokens": ("walk", "walking", "gait"),
        "group_field": "group", "patient_tokens": ("pd", "parkinson", "1"),
        "control_tokens": ("hc", "control", "healthy", "0"),
    },
}

# the erm_0 dump field contract the v3 loader reads (label-free deployment fields + a separate label field).
OUTPUT_DUMP_SCHEMA = {
    "z_te": "float (N, d)", "subject_id_te": "str (N,)", "recording_id_te": "str (N,)",
    "window_index_te": "int (N,)", "y_te": "int (N,) in {0,1}", "feat_hash_te": "str",
}


def _spec(site):
    if site not in DATASET_SPECS:
        raise ValueError(f"unknown held-out site {site!r}; admissible: {sorted(DATASET_SPECS)}")
    return DATASET_SPECS[site]


# ----------------------------------------------------------------------------- pure selectors / parsers / validators

def resting_run_selector(runs, site):
    """From BIDS run descriptors (each a dict with a 'task' key, optionally 'run'/'acq'), return the resting runs for
    `site`: a run is kept iff its task contains a resting token AND no exclude token (e.g. walking). Fail-closed if none."""
    spec = _spec(site)
    kept = []
    for r in runs:
        task = str(r.get("task", "")).lower()
        if any(tok in task for tok in spec["resting_tokens"]) and not any(x in task for x in spec["exclude_tokens"]):
            kept.append(r)
    if not kept:
        raise ValueError(f"{site}: no resting runs found (tokens {spec['resting_tokens']}, exclude {spec['exclude_tokens']})")
    return kept


def parse_diagnosis_map(rows, site):
    """rows: list of dicts with 'participant_id' + the site's group field. Return {participant_id: 0/1} (0=HC, 1=patient).
    Fail-closed on unknown group token, duplicate id, or missing field."""
    spec = _spec(site)
    gf = spec["group_field"]
    out = {}
    for r in rows:
        pid = r.get("participant_id")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{site}: missing/invalid participant_id in {r!r}")
        if pid in out:
            raise ValueError(f"{site}: duplicate participant_id {pid!r}")
        if gf not in r:
            raise ValueError(f"{site}: row {pid} missing group field {gf!r}")
        g = str(r[gf]).strip().lower()
        if g in spec["patient_tokens"]:
            out[pid] = 1
        elif g in spec["control_tokens"]:
            out[pid] = 0
        else:
            raise ValueError(f"{site}: unknown group {r[gf]!r} for {pid} (not patient/control)")
    if not out:
        raise ValueError(f"{site}: empty diagnosis map")
    if not (0 in out.values() and 1 in out.values()):
        raise ValueError(f"{site}: diagnosis map missing a class (need both patient and control)")
    return out


def validate_channels_fs(n_channels, fs, site):
    """Confirm channel count / sampling rate against the frozen spec when known; raise on mismatch. None expected =
    'confirm at prep' (returns the observed for recording, no raise)."""
    spec = _spec(site)
    if spec["expected_channels"] is not None and int(n_channels) != spec["expected_channels"]:
        raise ValueError(f"{site}: channels {n_channels} != expected {spec['expected_channels']}")
    if spec["expected_fs"] is not None and int(fs) != spec["expected_fs"]:
        raise ValueError(f"{site}: Fs {fs} != expected {spec['expected_fs']}")
    return {"site": site, "n_channels": int(n_channels), "fs": int(fs)}


def validate_dump_schema(arrays):
    """Validate an erm_0 dump (dict of arrays) against OUTPUT_DUMP_SCHEMA. Checks presence, dtype family, consistent N,
    and y_te ∈ {0,1}. Pure (no file I/O)."""
    for k in ("z_te", "subject_id_te", "recording_id_te", "window_index_te", "y_te"):
        if k not in arrays:
            raise ValueError(f"dump missing field {k!r}")
    z = np.asarray(arrays["z_te"])
    if z.ndim != 2 or not np.issubdtype(z.dtype, np.floating):
        raise ValueError("z_te must be 2-D float [N, d]")
    n = z.shape[0]
    for k in ("subject_id_te", "recording_id_te"):
        a = np.asarray(arrays[k])
        if a.ndim != 1 or a.shape[0] != n or a.dtype.kind not in ("U", "S", "O"):
            raise ValueError(f"{k} must be 1-D string of length N")
    wi = np.asarray(arrays["window_index_te"])
    if wi.ndim != 1 or wi.shape[0] != n or not np.issubdtype(wi.dtype, np.integer):
        raise ValueError("window_index_te must be 1-D int of length N")
    y = np.asarray(arrays["y_te"])
    if y.ndim != 1 or y.shape[0] != n or not np.issubdtype(y.dtype, np.integer) or set(np.unique(y)) - {0, 1}:
        raise ValueError("y_te must be 1-D int of length N with values in {0,1}")
    return n


# ----------------------------------------------------------------------------- provenance hashers (injective)

def _u(b):
    return len(b).to_bytes(8, "big") + b


def raw_pipeline_sha256(params):
    """Deterministic injective digest of the frozen raw→feature pipeline params (resample/filter/window/montage/encoder
    ref). Pinned in the dump + the external manifest so DEV and held-out share one pipeline."""
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def subject_list_sha256(subject_ids):
    h = hashlib.sha256()
    for s in sorted(set(subject_ids)):
        h.update(_u(str(s).encode()))
    return h.hexdigest()


def diagnosis_mapping_sha256(mapping):
    h = hashlib.sha256()
    for k in sorted(mapping):
        h.update(_u(str(k).encode())); h.update(_u(str(int(mapping[k])).encode()))
    return h.hexdigest()


def resting_selection_sha256(selected_runs):
    h = hashlib.sha256()
    for r in sorted(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in selected_runs):
        h.update(_u(r.encode()))
    return h.hexdigest()


# ----------------------------------------------------------------------------- gated real path (post-tag, eeg2025)

def prepare_dump(site, raw_bids_dir, out_path, *, frozen_pipeline_params, frozen_encoder_ref):
    """GATED real step (runs ONLY after acar-v4-protocol is tagged): read the held-out raw BIDS data, select resting
    runs, apply the FROZEN DEV pipeline (resample/filter/window per frozen_pipeline_params) + the FROZEN encoder
    (frozen_encoder_ref) to produce the erm_0 dump (OUTPUT_DUMP_SCHEMA) + provenance hashes. NOT IMPLEMENTED here — it
    requires the held-out raw signal and the frozen encoder checkpoint, and must not run before the protocol is tagged.
    The diagnosis labels go ONLY into y_te; the encoder/source-state are the DEV-frozen artifacts (no refit)."""
    raise NotImplementedError(
        "prepare_dump is a GATED post-tag step: run the frozen DEV pipeline + frozen encoder on the held-out raw data "
        "to emit the erm_0 dump (validate_dump_schema) + provenance (raw_pipeline_sha256 == DEV pipeline). See "
        "notes/ACAR_V4_EXTERNAL_INPUT_SCHEMA.md. It is not executed pre-tag and reads no real data in this module.")
