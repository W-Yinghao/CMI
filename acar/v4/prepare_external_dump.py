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
import os
import re

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

def _task_tokens(task):
    """Tokenize a BIDS task label: lowercase, split camelCase, then split on non-alphanumerics → set of tokens. This
    makes matching exact-per-token (so 'arrest'/'prestim' do NOT match the 'rest' token)."""
    s = str(task)
    out, cur = [], []
    for ch in s:
        if ch.isupper() and cur:
            out.append("".join(cur)); cur = [ch]
        else:
            cur.append(ch)
    out.append("".join(cur))
    toks = set()
    for piece in out:
        for t in re.split(r"[^0-9a-zA-Z]+", piece.lower()):
            if t:
                toks.add(t)
    return toks


def resting_run_selector(runs, site):
    """From BIDS run descriptors (each a dict with a 'task' key), return the resting runs for `site`: kept iff a resting
    token EQUALS one of the task's tokens AND no exclude token does (e.g. walking). EXACT-token match avoids substring
    false-positives ('arrest'/'prestimulus' are not resting). Fail-closed if none match."""
    spec = _spec(site)
    rest, excl = set(spec["resting_tokens"]), set(spec["exclude_tokens"])
    kept = []
    for r in runs:
        toks = _task_tokens(r.get("task", ""))
        if (rest & toks) and not (excl & toks):
            kept.append(r)
    if not kept:
        raise ValueError(f"{site}: no resting runs (resting tokens {sorted(rest)}, exclude {sorted(excl)})")
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


_ALLOWED_DUMP_KEYS = {"z_te", "subject_id_te", "recording_id_te", "window_index_te", "y_te", "feat_hash_te"}


def validate_dump_schema(arrays, *, embedding_dim=16):
    """Validate an erm_0 dump (dict of arrays) to the v3 loader's invariants (so a dump that passes here passes the v3
    read): ALLOW-LIST (no source-fit fields like z_ev/y_ev), all fields present, z finite 2-D [N>=1, d>=1] (d==
    embedding_dim), non-empty string ids, non-negative window_index, UNIQUE (subject,recording,window) rows, y∈{0,1},
    and a 64-hex feat_hash_te. Pure (no file I/O)."""
    extra = set(arrays) - _ALLOWED_DUMP_KEYS
    if extra:
        raise ValueError(f"dump has forbidden/extra fields {sorted(extra)} (label-free deployment dump must be minimal; "
                         "source-fit fields z_ev/y_ev are not allowed)")
    for k in _ALLOWED_DUMP_KEYS:
        if k not in arrays:
            raise ValueError(f"dump missing field {k!r}")
    z = np.asarray(arrays["z_te"])
    if z.ndim != 2 or not np.issubdtype(z.dtype, np.floating):
        raise ValueError("z_te must be 2-D float [N, d]")
    if z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError("z_te must be non-empty with d>=1")
    if not np.all(np.isfinite(z)):
        raise ValueError("z_te contains non-finite values")
    if embedding_dim is not None and z.shape[1] != embedding_dim:
        raise ValueError(f"z_te d={z.shape[1]} != expected embedding_dim {embedding_dim}")
    n = z.shape[0]
    sids = np.asarray(arrays["subject_id_te"]); rids = np.asarray(arrays["recording_id_te"])
    for nm, a in (("subject_id_te", sids), ("recording_id_te", rids)):
        if a.ndim != 1 or a.shape[0] != n or a.dtype.kind not in ("U", "S", "O"):
            raise ValueError(f"{nm} must be 1-D string of length N")
        if a.dtype.kind == "O" and not all(isinstance(x, str) for x in a.tolist()):
            raise ValueError(f"{nm} object array must contain only str (v3 _str_list forbids coercion)")   # SCHEMA-1
        if any(str(x) == "" for x in a.tolist()):
            raise ValueError(f"{nm} contains empty id(s)")
    wi = np.asarray(arrays["window_index_te"])
    if wi.ndim != 1 or wi.shape[0] != n or not np.issubdtype(wi.dtype, np.integer):
        raise ValueError("window_index_te must be 1-D int of length N")
    if np.any(wi < 0):
        raise ValueError("window_index_te must be non-negative")
    rows = list(zip((str(x) for x in sids.tolist()), (str(x) for x in rids.tolist()), wi.tolist()))
    if len(set(rows)) != n:
        raise ValueError("(subject_id, recording_id, window_index) rows must be unique")
    y = np.asarray(arrays["y_te"])
    if y.ndim != 1 or y.shape[0] != n or not np.issubdtype(y.dtype, np.integer) or (set(np.unique(y).tolist()) - {0, 1}):
        raise ValueError("y_te must be 1-D int of length N with values in {0,1}")
    fh = arrays["feat_hash_te"]
    fhs = fh.item() if hasattr(fh, "item") else fh
    if not (isinstance(fhs, str) and len(fhs) == 64 and all(c in "0123456789abcdef" for c in fhs)):
        raise ValueError("feat_hash_te must be a 64-char lowercase sha-256")
    return n


# ----------------------------------------------------------------------------- provenance hashers (injective)
# These compute the THREE prep-only manifest hashes (raw_pipeline / diagnosis_mapping / resting_selection) — the ones the
# Arm-B CLI cannot recompute from the .npz, so they are carried in the frozen-prep provenance sidecar (see
# provenance_sidecar_dict). NOTE (PROV-2): the manifest's subject_list_sha256 / deployment_input_sha256 / label_sha256 are
# NOT computed here — they MUST come from acar.v3.loader.{hash_subject_list, hash_deployment_input, hash_labels}, which the
# CLI RE-COMPUTES and aborts on mismatch; a different same-named digest here would only cause a fail-closed abort.

def _u(b):
    return len(b).to_bytes(8, "big") + b


def raw_pipeline_sha256(params):
    """Deterministic injective digest of the frozen raw→feature pipeline params (resample/filter/window/montage/encoder
    ref). Pinned in the dump + the external manifest so DEV and held-out share one pipeline."""
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


# ----------------------------------------------------------------------------- executable, FAIL-CLOSED real prep (gated)

class FrozenEncoderMissingError(RuntimeError):
    """Raised when prepare_dump is invoked without a complete, on-disk, hash-verified DEV-frozen encoder + source-state.
    prepare_dump NEVER trains/regenerates an encoder; the absence of an archived encoder is a hard, explicit blocker."""


class ExternalReaderNotWiredError(RuntimeError):
    """Raised when prepare_dump reaches the held-out raw→embedding step but no held-out BIDS reader is wired. The cmi
    `load_crossdataset` only indexes registered COHORTS and KeyErrors for the held-out sites (ds007526/zenodo14808296),
    so embedding them needs a dedicated reader (DATASET_SPECS + resting_run_selector + parse_diagnosis_map +
    validate_channels_fs → X/y/subjects). This is the SECOND hard executability blocker (alongside the missing encoder);
    see notes/ACAR_V4_ENCODER_ARTIFACT_DECISION.md. prepare_dump NEVER silently mis-routes through load_crossdataset."""


# the frozen DEV pipeline (== feat_dump_v4): cmi.run_scps_crossdataset --configs erm:0 (EEGNet, 19ch 10-20, 128Hz,
# 0.5-45 Hz bandpass, 4s/512 windows, embedding_dim 16). The held-out pipeline MUST equal this.
FROZEN_PIPELINE = {"resample_fs": 128, "bandpass": [0.5, 45.0], "window_sec": 4.0, "canon_channels": 19,
                   "encoder": "EEGNet", "embedding_dim": 16}
ENCODER_ARTIFACT_FIELDS = ("encoder_checkpoint_path", "encoder_checkpoint_sha256", "encoder_architecture",
                           "encoder_training_command", "encoder_training_data_scope", "encoder_seed", "determinism",
                           "torch_version", "braindecode_version", "embedding_dim",
                           "source_state_path", "source_state_sha256", "source_state_ref")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon_hash(z):
    """feat_hash_te definition (matches cmi): sha256 of canonical little-endian float64 C-contiguous bytes."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(z, dtype="<f8")).tobytes()).hexdigest()


def validate_pipeline_config(cfg):
    """The held-out raw→feature pipeline MUST equal the frozen DEV pipeline; any deviation raises."""
    if not isinstance(cfg, dict):
        raise ValueError("raw_pipeline_config must be a dict")
    for k, v in FROZEN_PIPELINE.items():
        if cfg.get(k) != v:
            raise ValueError(f"raw_pipeline_config[{k!r}]={cfg.get(k)!r} != frozen DEV {v!r}")
    return True


def require_encoder_artifact(meta):
    """Fail-closed: the frozen encoder + source-state artifact must be a COMPLETE dict, present on disk, and
    hash-verified. Missing/incomplete → FrozenEncoderMissingError; on-disk hash mismatch / wrong embedding_dim →
    ValueError. This is the executable gate that keeps prepare_dump from ever implicitly retraining an encoder."""
    if not isinstance(meta, dict):
        raise FrozenEncoderMissingError("encoder_artifact must be a dict")
    missing = [f for f in ENCODER_ARTIFACT_FIELDS if f not in meta or meta[f] in (None, "")]
    if missing:
        raise FrozenEncoderMissingError(f"frozen encoder/source-state artifact incomplete: missing {missing}")
    if int(meta["embedding_dim"]) != FROZEN_PIPELINE["embedding_dim"]:
        raise ValueError(f"encoder embedding_dim {meta['embedding_dim']} != frozen {FROZEN_PIPELINE['embedding_dim']}")
    for pf in ("encoder_checkpoint_path", "source_state_path"):
        if not os.path.exists(meta[pf]):
            raise FrozenEncoderMissingError(f"{pf} not on disk: {meta[pf]!r} (archive the frozen DEV encoder first)")
    if _sha256_file(meta["encoder_checkpoint_path"]) != meta["encoder_checkpoint_sha256"]:
        raise ValueError("encoder_checkpoint_sha256 mismatch")
    if _sha256_file(meta["source_state_path"]) != meta["source_state_sha256"]:
        raise ValueError("source_state_sha256 mismatch")
    return True


SIDECAR_SCHEMA = "acar_v4_external_provenance/1"
# the manifest's 5 v3-recomputable hashes (re-verified by the CLI) + the 3 prep-only hashes the CLI cannot recompute.
_SIDECAR_HASH_FIELDS = ("full_dump_sha256", "deployment_input_sha256", "label_sha256", "subject_list_sha256",
                        "diagnosis_mapping_sha256", "resting_selection_sha256", "raw_pipeline_sha256",
                        "source_state_sha256")


def provenance_sidecar_dict(*, source_state_ref, **hashes):
    """Build the frozen-prep provenance record carried alongside the dump as `<dump>.provenance.json`. Must contain all 8
    manifest hash fields (PROV-1 / closure point B) + source_state_ref; the CLI sha-pins this file and asserts every
    manifest field equals it, so the 3 prep-only hashes can't be hand-filled. Pure (no I/O)."""
    missing = [f for f in _SIDECAR_HASH_FIELDS if f not in hashes]
    if missing:
        raise ValueError(f"provenance sidecar incomplete: missing {missing}")
    extra = [f for f in hashes if f not in _SIDECAR_HASH_FIELDS]
    if extra:
        raise ValueError(f"provenance sidecar has unexpected fields {extra}")
    d = {"schema": SIDECAR_SCHEMA, "source_state_ref": source_state_ref}
    d.update({f: hashes[f] for f in _SIDECAR_HASH_FIELDS})
    return d


def write_provenance_sidecar(dump_path, sidecar):
    """Atomically write the provenance sidecar next to the dump as `<dump>.provenance.json`."""
    path = dump_path + ".provenance.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sidecar, f, sort_keys=True, indent=2)
    os.replace(tmp, path)
    return path


def _embed_heldout_raw(site, raw_bids_root, encoder_artifact, raw_pipeline_config):
    """Held-out raw EEG → (z_te, y_te, subject_ids) via the FROZEN pipeline + FROZEN encoder. The selection/validation/
    windowing/key layer is now IMPLEMENTED + synthetic-tested in `acar.v4.heldout_reader` (read_heldout → X/y/keys); the
    real path still has TWO gated remainders, so this stays FAIL-CLOSED: (1) the real raw-signal DSP provider
    (`heldout_reader.make_real_signal_provider` → RawSignalDSPNotWiredError; needs mne + real held-out raw), and (2) the
    FROZEN encoder (already gated upstream by require_encoder_artifact). cmi.load_crossdataset is NOT usable (it KeyErrors
    on non-COHORTS held-out sites). To run post-provisioning: heldout_reader.read_heldout(site, raw_bids_root,
    pipeline_config=raw_pipeline_config, signal_provider=<mne provider>) → embed X with the frozen encoder."""
    raise ExternalReaderNotWiredError(
        f"{site}: held-out reader's selection/key layer is implemented (acar.v4.heldout_reader, synthetic-tested), but the "
        f"real raw-signal DSP provider + frozen encoder are gated. Wire heldout_reader.make_real_signal_provider (mne "
        f"EDF/BrainVision → 19ch/128Hz/0.5-45Hz) + the archived encoder at substrate provisioning. See "
        f"notes/ACAR_V4_SUBSTRATE_REGEN_PLAN.md.")


def prepare_dump(site, raw_bids_root, output_npz, *, encoder_artifact, raw_pipeline_config):
    """FAIL-CLOSED scaffold (post-tag gated). Build the held-out erm_0 dump + provenance sidecar by applying the FROZEN DEV
    pipeline + FROZEN encoder to held-out raw EEG. TWO hard executability blockers, both fail-closed BEFORE any heavy
    import / raw read: (1) require_encoder_artifact → FrozenEncoderMissingError if the frozen encoder+source-state are not
    archived/hash-verified (NEVER retrains); (2) _embed_heldout_raw → ExternalReaderNotWiredError because cmi's
    load_crossdataset cannot read the held-out sites and no held-out BIDS reader is wired. Held-out diagnosis labels go
    ONLY into y_te. The dump+sidecar assembly below is the documented continuation, reachable once BOTH blockers clear."""
    require_encoder_artifact(encoder_artifact)                # blocker 1: frozen encoder/source-state (fail-closed)
    validate_pipeline_config(raw_pipeline_config)
    spec = _spec(site)
    if output_npz and os.path.exists(output_npz):
        raise FileExistsError(f"output dump already exists: {output_npz}")
    z_te, y_te, subject_ids = _embed_heldout_raw(site, raw_bids_root, encoder_artifact, raw_pipeline_config)  # blocker 2
    # --- assembly (reachable once a held-out reader + encoder exist) ---
    z_te = np.asarray(z_te, dtype="<f8")
    subject_id_te = np.asarray(subject_ids)                   # already cohort-namespaced by the held-out reader (no re-prefix)
    arrays = {"z_te": z_te, "subject_id_te": subject_id_te, "recording_id_te": subject_id_te.copy(),
              "window_index_te": np.arange(len(z_te), dtype=np.int64), "y_te": np.asarray(y_te, dtype=np.int64),
              "feat_hash_te": _canon_hash(z_te)}
    validate_dump_schema(arrays, embedding_dim=FROZEN_PIPELINE["embedding_dim"])   # self-check vs the v3-read contract
    tmp = output_npz + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, output_npz)                               # atomic publish
    return output_npz
