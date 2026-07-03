"""ACAR V5 Stage-1B12 BrainVision READ-REPAIR (pure/stdlib; nothing heavy at import — no mne, no numpy). The Stage-1B11P preflight
found that the pinned mne cannot OPEN some real DEV recordings because of HEADER defects (not signal defects): ds003944/ds003947
BrainVision headers declare no MarkerFile (and no .vmrk exists), and ds004000/sub-042's two recordings declare DataFile/MarkerFile
pointers to files that were renamed at BIDS-ification. This module produces an EPHEMERAL, audited *header* repair so the pinned reader
can open the recording — WITHOUT ever modifying, copying, or reading the raw signal.

Three exact, whitelisted repair modes (everything else → no plan / fail-closed):
  * missing_markerfile_minimal_vmrk  (cohorts ds003944, ds003947): the .vhdr has a DataFile but NO MarkerFile and there is no .vmrk
    on disk → synthesize a MINIMAL marker file (a single `New Segment` at position 1 — NO task/stimulus/event inference) and a
    repaired header that points DataFile at the ORIGINAL .eeg and MarkerFile at the synthesized marker.
  * broken_internal_pointer_rewrite  (ds004000, sub-042, the two exact recordings from Stage-1B11P): the .vhdr's declared
    DataFile/MarkerFile do not exist → a repaired header that points DataFile/MarkerFile at the EXISTING BIDS sibling
    (<recording-stem>.eeg|.dat and <recording-stem>.vmrk). No marker is synthesized.
  * channel_names_from_channels_tsv_for_generic_brainvision  (Stage-1B13; cohorts ds003944, ds003947 ONLY): the .vhdr [Channel Infos]
    carries only GENERIC sequential placeholder names (EEG001..EEG0NN) while the BIDS channels.tsv holds the real electrode labels →
    rewrite the staged header's [Channel Infos] names from the placeholders to the channels.tsv names by ROW ORDER (BIDS requires
    channels.tsv rows in EEG-data-file order), composing with the marker fix above when the recording is also marker-less. Allowed
    ONLY when the header is exactly generic-sequential AND channels.tsv exists / has a `name` column / row count == header channel
    count / names unique after strip+casefold / all 19 canonical resolve via the Stage-1B10 aliases. No fuzzy/heuristic/partial
    matching; never used for other cohorts; never overrides a header that already carries real (non-generic) names.

Hard invariants (fail-closed): repaired files are written ONLY under the caller's ephemeral staging dir (never the raw tree; a
staging dir inside the recording's directory is rejected); the original raw files are never opened for writing; targets must be
existing NON-symlink regular files; no event synthesis beyond the single New Segment. Every repair emits a manifest
(original_header_sha256, repaired_header_sha256, generated_marker_sha256|None, repair_mode, cohort/subject/recording, targets, reason);
`assert_manifest_consistent` re-verifies the on-disk hashes so the reader consumes the repaired header ONLY after the manifest
validates. The pinned policy lives in preprocessing_config (part of preprocessing_config_sha256).
"""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass
from acar.v5.substrate import channel_aliases as CA          # pure/stdlib (only pulls preprocessing_config)
from acar.v5.substrate import preprocessing_config as PC

MODE_MISSING_MARKER = "missing_markerfile_minimal_vmrk"
MODE_POINTER_REWRITE = "broken_internal_pointer_rewrite"
MODE_CHANNEL_NAMES_FROM_TSV = "channel_names_from_channels_tsv_for_generic_brainvision"


class BrainvisionReadRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairPlan:
    mode: str
    disease: str
    cohort: str
    subject: str
    recording: str                 # original .vhdr basename
    original_vhdr_path: str        # absolute
    data_target: str               # absolute path the repaired DataFile must point at (an existing raw file)
    marker_target: str             # absolute path the repaired MarkerFile must point at, or "" to synthesize a minimal marker
    reason: str
    channel_name_map: tuple = ()   # Stage-1B13: channels.tsv names in ROW ORDER to rename [Channel Infos] (empty = no rename)
    channels_tsv_path: str = ""    # absolute path of the channels.tsv used as the rename source (empty = none)


def _norm_sub(s):
    s = str(s).strip()
    if s.lower().startswith("sub-"):
        s = s[len("sub-"):]
    return s.casefold()


def _is_regular_file(p):
    return bool(p) and os.path.isfile(p) and not os.path.islink(p)


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p):
    with open(p, "rb") as f:
        return _sha256_bytes(f.read())


def _read_common_infos(vhdr_path):
    """Parse the [Common Infos] DataFile/MarkerFile of a BrainVision .vhdr the way mne (configparser) does: case-insensitive keys,
    whitespace tolerant. Returns {'datafile': str|None, 'markerfile': str|None} (raw values as written; None if the key is absent)."""
    out = {"datafile": None, "markerfile": None}
    with open(vhdr_path, encoding="latin-1", errors="replace") as f:
        text = f.read()
    in_common = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_common = (s.casefold() == "[common infos]")
            continue
        if not in_common or "=" not in line:
            continue
        key, val = line.split("=", 1)
        k = key.strip().casefold()
        if k in out and out[k] is None:
            out[k] = val.strip()
    return out


def _resolve_bare(d, name):
    """Resolve a header DataFile/MarkerFile value relative to the header dir (mne semantics: os.path.join). None if name is falsy."""
    if not name:
        return None
    return os.path.abspath(os.path.join(d, name))


def _bids_sibling(vhdr_path, ext):
    """<recording-stem><ext> in the same dir (the BIDS-renamed sibling)."""
    d = os.path.dirname(vhdr_path)
    stem = os.path.basename(vhdr_path)[: -len(".vhdr")]
    return os.path.join(d, stem + ext)


def _sibling_data_file(vhdr_path):
    """The first existing BIDS sibling data file (<stem>.eeg or <stem>.dat) — the actual binary for this recording."""
    for ext in PC.PREPROCESSING_CONFIG["brainvision_read_repair_data_extensions"]:
        cand = _bids_sibling(vhdr_path, ext)
        if _is_regular_file(cand):
            return os.path.abspath(cand)
    return None


def _channels_tsv_path(vhdr_path):
    """The BIDS channels.tsv path for a recording: <stem-without-_eeg>_channels.tsv in the same dir (or None if absent)."""
    d = os.path.dirname(vhdr_path)
    base = os.path.basename(vhdr_path)
    if base.lower().endswith("_eeg.vhdr"):
        ct = os.path.join(d, base[: -len("_eeg.vhdr")] + "_channels.tsv")
        return ct if _is_regular_file(ct) else None
    return None


def _read_channels_tsv_names(ct_path):
    """Ordered `name` column of a channels.tsv, or None if the file has no rows / no name column."""
    import csv
    with open(ct_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return None
    col = next((c for c in rows[0] if str(c).strip().casefold() == "name"), None)
    return [r[col] for r in rows] if col else None


def _parse_channel_infos(vhdr_path):
    """Ordered list of the [Channel Infos] channel names (the first comma field of each `Ch<i>=<name>,...`), or None if the section
    is absent/empty or the Ch indices are not the contiguous 1..N. Header text only (mne semantics)."""
    with open(vhdr_path, encoding="latin-1", errors="replace") as f:
        text = f.read()
    in_ch, pairs = False, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_ch = (s.casefold() == "[channel infos]")
            continue
        if not in_ch or s.startswith(";") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        k = key.strip()
        if not (k[:2].lower() == "ch" and k[2:].isdigit()):
            continue
        name = val.split(",", 1)[0].strip()
        pairs.append((int(k[2:]), name))
    if not pairs:
        return None
    pairs.sort(key=lambda p: p[0])
    if [i for i, _ in pairs] != list(range(1, len(pairs) + 1)):
        return None                                            # Ch indices must be the contiguous 1..N
    return [n for _, n in pairs]


def _is_generic_sequential(names):
    """True iff the i-th (1-based) name is exactly the generic placeholder EEG<i> (any zero-padding), for every channel."""
    if not names:
        return False
    for i, n in enumerate(names, 1):
        s = str(n).strip()
        if s[:3].upper() != "EEG" or not s[3:].isdigit() or int(s[3:]) != i:
            return False
    return True


def _validate_channels_tsv_for_rename(vhdr_path, n_header):
    """(tsv_names_or_None, ok): channels.tsv exists + has a name column + row count == n_header + names unique after strip/casefold +
    all 19 canonical resolve via the Stage-1B10 aliases with no logical duplicate. No fuzzy matching anywhere."""
    ct = _channels_tsv_path(vhdr_path)
    if ct is None:
        return None, False
    names = _read_channels_tsv_names(ct)
    if names is None:
        return None, False
    if len(names) != int(n_header):
        return names, False
    norm = [str(x).strip().casefold() for x in names]
    if len(set(norm)) != len(norm):
        return names, False
    for nm in names:                                           # BrainVision headers are latin-1; a non-encodable name → no rename
        try:
            str(nm).encode("latin-1")
        except UnicodeEncodeError:
            return names, False
    try:
        CA.resolve_canonical_sources(names)                    # raises on a logical duplicate OR any missing canonical channel
    except CA.ChannelAliasError:
        return names, False
    return names, True


def _plan_channel_name_map(vhdr_path, cohort):
    """(tuple channels.tsv names in row order, channels.tsv abs path) if a channels.tsv rename is authorized for this generic
    BrainVision header, else None. Requires the cohort to be whitelisted AND the header exactly generic-sequential AND channels.tsv
    valid for a row-order rename. Never overrides a header that already carries real (non-generic) names."""
    if cohort not in PC.PREPROCESSING_CONFIG["channel_name_repair_cohorts"]:
        return None
    header_names = _parse_channel_infos(vhdr_path)
    if header_names is None or not _is_generic_sequential(header_names):
        return None
    tsv_names, ok = _validate_channels_tsv_for_rename(vhdr_path, len(header_names))
    if not ok:
        return None
    return tuple(tsv_names), os.path.abspath(_channels_tsv_path(vhdr_path))


def plan_repair(disease, cohort, subject, recording_path):
    """Header-ONLY decision (reads only the small .vhdr / channels.tsv text + checks sibling existence): return a RepairPlan for one
    of the three whitelisted modes, or None if no reviewed repair applies. Fail-closed on a non-.vhdr / missing header."""
    if os.path.splitext(recording_path)[1].lower() != ".vhdr":
        return None
    if not _is_regular_file(recording_path):
        raise BrainvisionReadRepairError(f"recording header is not a regular file: {recording_path}")
    vhdr = os.path.abspath(recording_path)
    d = os.path.dirname(vhdr)
    ci = _read_common_infos(vhdr)
    base = os.path.basename(vhdr)

    # (a) missing-markerfile + (c) channels.tsv channel-name rename — the two whitelisted cohorts (ds003944/ds003947) share this
    # branch. Prefer the mode-C rename when the header is generic-sequential AND channels.tsv is valid; otherwise fall back to the
    # marker-only fix when marker-less (so the header opens and the generic-name defect is surfaced downstream), else no plan.
    if cohort in PC.PREPROCESSING_CONFIG["brainvision_read_repair_missing_markerfile_cohorts"]:
        declared_data = _resolve_bare(d, ci["datafile"])
        if not (ci["datafile"] and _is_regular_file(declared_data)):
            return None
        sib_vmrk = _bids_sibling(vhdr, ".vmrk")
        declared_mrk = _resolve_bare(d, ci["markerfile"])
        marker_less = (not ci["markerfile"]) and (not os.path.exists(sib_vmrk))
        if marker_less:
            marker_target = ""                                 # synthesize a minimal marker
        elif ci["markerfile"] and declared_mrk and os.path.exists(declared_mrk):
            marker_target = os.path.abspath(declared_mrk)      # keep the existing (resolvable) marker
        else:
            marker_target = None                               # ambiguous marker state → do not guess
        cmap = _plan_channel_name_map(vhdr, cohort)            # generic-sequential header + valid channels.tsv?
        if cmap is not None and marker_target is not None:
            names, tsv_path = cmap
            return RepairPlan(mode=MODE_CHANNEL_NAMES_FROM_TSV, disease=disease, cohort=cohort, subject=str(subject),
                              recording=base, original_vhdr_path=vhdr, data_target=declared_data, marker_target=marker_target,
                              channel_name_map=names, channels_tsv_path=tsv_path,
                              reason="generic EEG00N header names; rename [Channel Infos] from channels.tsv by row order"
                                     + ("" if marker_target else " and synthesize a minimal marker"))
        if marker_less:
            return RepairPlan(mode=MODE_MISSING_MARKER, disease=disease, cohort=cohort, subject=str(subject),
                              recording=base, original_vhdr_path=vhdr, data_target=declared_data, marker_target="",
                              reason="header declares DataFile but no MarkerFile and no .vmrk exists; synthesize minimal marker")
        return None

    # (b) broken internal pointer rewrite — only ds004000/sub-042's two exact recordings, only when a declared pointer is missing
    pr = PC.PREPROCESSING_CONFIG["brainvision_read_repair_pointer_rewrite"]
    if cohort == pr["cohort"] and _norm_sub(subject) == _norm_sub(pr["subject"]) and base in tuple(pr["recordings"]):
        declared_data = _resolve_bare(d, ci["datafile"])
        declared_mrk = _resolve_bare(d, ci["markerfile"])
        broken = (declared_data is not None and not os.path.exists(declared_data)) or \
                 (declared_mrk is not None and not os.path.exists(declared_mrk))
        if not broken:
            return None                                          # pointers already resolve → no repair
        sib_data = _sibling_data_file(vhdr)
        sib_mrk = _bids_sibling(vhdr, ".vmrk")
        if _is_regular_file(sib_data) and _is_regular_file(sib_mrk):
            return RepairPlan(mode=MODE_POINTER_REWRITE, disease=disease, cohort=cohort, subject=str(subject), recording=base,
                              original_vhdr_path=vhdr, data_target=os.path.abspath(sib_data),
                              marker_target=os.path.abspath(sib_mrk),
                              reason="declared DataFile/MarkerFile do not exist; repoint at existing BIDS sibling data/marker files")
        return None
    return None


def _rewrite_common_infos(text, data_abs, marker_abs):
    """Return the .vhdr text with the [Common Infos] DataFile set to data_abs and MarkerFile set to marker_abs (inserted right after
    DataFile if absent). Only the [Common Infos] section is touched; every other line is preserved verbatim."""
    lines = text.splitlines()
    out = []
    in_common = False
    wrote_marker = False
    seen_datafile = False
    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            if in_common and not wrote_marker and seen_datafile:
                out.append(f"MarkerFile={marker_abs}")          # marker line was absent → add before leaving the section
                wrote_marker = True
            in_common = (s.casefold() == "[common infos]")
            out.append(line)
            continue
        if in_common and "=" in line:
            k = line.split("=", 1)[0].strip().casefold()
            if k == "datafile":
                out.append(f"DataFile={data_abs}")
                seen_datafile = True
                continue
            if k == "markerfile":
                out.append(f"MarkerFile={marker_abs}")
                wrote_marker = True
                continue
        out.append(line)
    if in_common and not wrote_marker and seen_datafile:         # [Common Infos] was the final section and had no MarkerFile
        out.append(f"MarkerFile={marker_abs}")
        wrote_marker = True
    if not (seen_datafile and wrote_marker):
        raise BrainvisionReadRepairError("repaired header must contain exactly one DataFile and one MarkerFile in [Common Infos]")
    return "\n".join(out) + "\n"


def _minimal_vmrk(data_abs):
    """A minimal BrainVision marker file: a SINGLE `New Segment` at position 1 — no task/stimulus/event markers are ever inferred."""
    return ("Brain Vision Data Exchange Marker File, Version 1.0\n\n"
            "[Common Infos]\nCodepage=UTF-8\n"
            f"DataFile={data_abs}\n\n"
            "[Marker Infos]\nMk1=New Segment,,1,1,0\n")


def _rewrite_channel_infos(text, names):
    """Return the .vhdr text with each [Channel Infos] `Ch<i>=<name>,<rest>` line's NAME field replaced by names[i-1] (row order),
    preserving the reference/resolution/unit fields and every other line. Raises unless exactly len(names) Ch lines are renamed."""
    out, in_ch, n = [], False, 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_ch = (s.casefold() == "[channel infos]")
            out.append(line)
            continue
        if in_ch and "=" in line and not s.startswith(";"):
            key, val = line.split("=", 1)
            k = key.strip()
            if k[:2].lower() == "ch" and k[2:].isdigit() and 1 <= int(k[2:]) <= len(names):
                parts = val.split(",", 1)
                rest = ("," + parts[1]) if len(parts) > 1 else ""
                out.append(f"{k}={names[int(k[2:]) - 1]}{rest}")
                n += 1
                continue
        out.append(line)
    if n != len(names):
        raise BrainvisionReadRepairError(f"channel rename touched {n} of {len(names)} channels")
    return "\n".join(out) + "\n"


def _names_sha256(names):
    return _sha256_bytes(json.dumps([str(x) for x in names], separators=(",", ":")).encode())


def _mapping_sha256(names):
    return _sha256_bytes(json.dumps([[i + 1, str(x)] for i, x in enumerate(names)], separators=(",", ":")).encode())


def _staging_is_safe(staging_dir, original_vhdr_path):
    """The staging dir must be an existing real dir OUTSIDE the recording's directory (never write into the raw tree)."""
    if not (staging_dir and os.path.isdir(staging_dir) and not os.path.islink(staging_dir)):
        raise BrainvisionReadRepairError(f"staging dir must be an existing non-symlink directory: {staging_dir}")
    stg = os.path.realpath(staging_dir)
    raw = os.path.realpath(os.path.dirname(original_vhdr_path))
    if stg == raw or stg.startswith(raw + os.sep) or raw.startswith(stg + os.sep):
        raise BrainvisionReadRepairError("staging dir must not overlap the raw recording directory (no writing into the raw tree)")


def _out_stem(plan):
    return f"{plan.disease}_{plan.cohort}_{_norm_sub(plan.subject)}__{plan.recording[:-len('.vhdr')]}"


def apply_repair(plan, staging_dir):
    """Materialize the repaired header (+ a synthesized marker when marker_target is empty; + a channels.tsv [Channel Infos] rename
    for mode C) under `staging_dir`, NEVER touching the raw tree, and return (repaired_vhdr_path, manifest). The manifest is
    re-verified (assert_manifest_consistent) before return → fail-closed."""
    if not isinstance(plan, RepairPlan):
        raise BrainvisionReadRepairError("apply_repair requires a RepairPlan")
    if plan.mode not in (MODE_MISSING_MARKER, MODE_POINTER_REWRITE, MODE_CHANNEL_NAMES_FROM_TSV):
        raise BrainvisionReadRepairError(f"unknown repair mode {plan.mode!r}")
    _staging_is_safe(staging_dir, plan.original_vhdr_path)
    if not _is_regular_file(plan.data_target):
        raise BrainvisionReadRepairError(f"data target is not a regular file: {plan.data_target}")
    with open(plan.original_vhdr_path, "rb") as f:
        original_bytes = f.read()
    stem = _out_stem(plan)
    repaired_vhdr = os.path.abspath(os.path.join(staging_dir, stem + ".repaired.vhdr"))
    generated_marker_sha256 = None

    # marker action: synthesize iff marker_target is empty (mode A always; mode C when the recording is marker-less); pointer-rewrite
    # must point at an existing marker; otherwise keep the given existing marker.
    if plan.marker_target == "":
        if plan.mode == MODE_POINTER_REWRITE:
            raise BrainvisionReadRepairError("pointer-rewrite must point at an existing marker (marker_target must not be empty)")
        synth_vmrk = os.path.abspath(os.path.join(staging_dir, stem + ".repaired.vmrk"))
        marker_bytes = _minimal_vmrk(plan.data_target).encode("utf-8")
        with open(synth_vmrk, "wb") as f:
            f.write(marker_bytes)
        generated_marker_sha256 = _sha256_bytes(marker_bytes)
        marker_abs = synth_vmrk
    else:
        if not _is_regular_file(plan.marker_target):
            raise BrainvisionReadRepairError(f"marker target is not a regular file: {plan.marker_target}")
        marker_abs = plan.marker_target

    text = _rewrite_common_infos(original_bytes.decode("latin-1"), plan.data_target, marker_abs)
    channel_fields = {}
    if plan.mode == MODE_CHANNEL_NAMES_FROM_TSV:
        if not plan.channel_name_map:
            raise BrainvisionReadRepairError("channels.tsv rename mode requires a non-empty channel_name_map")
        orig_names = _parse_channel_infos(plan.original_vhdr_path)
        if orig_names is None or len(orig_names) != len(plan.channel_name_map):
            raise BrainvisionReadRepairError("original header channel count does not match the channels.tsv rename map")
        text = _rewrite_channel_infos(text, list(plan.channel_name_map))
        channel_fields = {
            "channel_name_source": PC.PREPROCESSING_CONFIG["channel_name_repair_source"],
            "channels_tsv_path": plan.channels_tsv_path, "channels_tsv_sha256": _sha256_file(plan.channels_tsv_path),
            "channel_name_mapping_sha256": _mapping_sha256(plan.channel_name_map),
            "original_header_channel_names_sha256": _names_sha256(orig_names),
            "repaired_header_channel_names_sha256": _names_sha256(plan.channel_name_map),
            "channel_name_repair_policy_sha256": PC.channel_name_repair_policy_sha256(),
        }
    try:
        repaired_bytes = text.encode("latin-1")               # BrainVision headers are latin-1; fail-closed if not encodable
    except UnicodeEncodeError as e:
        raise BrainvisionReadRepairError(f"repaired header is not latin-1 encodable (non-BrainVision channel names?): {e}")
    with open(repaired_vhdr, "wb") as f:
        f.write(repaired_bytes)

    manifest = {
        "repair_mode": plan.mode, "disease": plan.disease, "cohort": plan.cohort, "subject": str(plan.subject),
        "recording": plan.recording, "original_vhdr_path": plan.original_vhdr_path, "repaired_vhdr_path": repaired_vhdr,
        "data_file_target": plan.data_target, "marker_file_target": marker_abs,
        "original_header_sha256": _sha256_bytes(original_bytes), "repaired_header_sha256": _sha256_bytes(repaired_bytes),
        "generated_marker_sha256": generated_marker_sha256,
        "brainvision_read_repair_policy_sha256": PC.brainvision_read_repair_policy_sha256(), "reason": plan.reason,
        **channel_fields,
    }
    assert_manifest_consistent(manifest)                         # re-verify on-disk hashes/targets before the reader consumes it
    return repaired_vhdr, manifest


def assert_manifest_consistent(manifest):
    """Fail-closed re-verification: known mode; the repaired header + (synthesized marker) + (mode-C renamed channel names +
    channels.tsv) on disk hash to the manifest values; data/marker targets are existing non-symlink regular files; the repaired header
    (and any synthesized marker) live outside the raw recording dir; a mode-C repaired header's names resolve the canonical 19."""
    if not isinstance(manifest, dict):
        raise BrainvisionReadRepairError("manifest must be a dict")
    mode = manifest.get("repair_mode")
    if mode not in (MODE_MISSING_MARKER, MODE_POINTER_REWRITE, MODE_CHANNEL_NAMES_FROM_TSV):
        raise BrainvisionReadRepairError(f"unknown repair_mode {mode!r}")
    rep = manifest.get("repaired_vhdr_path")
    if not _is_regular_file(rep):
        raise BrainvisionReadRepairError(f"repaired header missing: {rep}")
    if _sha256_file(rep) != manifest.get("repaired_header_sha256"):
        raise BrainvisionReadRepairError("repaired_header_sha256 mismatch (tampered / stale manifest)")
    if not _is_regular_file(manifest.get("data_file_target")):
        raise BrainvisionReadRepairError("data_file_target is not a regular file")
    if not _is_regular_file(manifest.get("marker_file_target")):
        raise BrainvisionReadRepairError("marker_file_target is not a regular file")
    orig = manifest.get("original_vhdr_path")
    if not _is_regular_file(orig) or _sha256_file(orig) != manifest.get("original_header_sha256"):
        raise BrainvisionReadRepairError("original header missing or original_header_sha256 mismatch")
    raw = os.path.realpath(os.path.dirname(orig))
    if os.path.realpath(rep) == raw or os.path.realpath(rep).startswith(raw + os.sep):
        raise BrainvisionReadRepairError("repaired header must not live inside the raw recording directory")

    # marker: a marker is SYNTHESIZED iff generated_marker_sha256 is set → verify its bytes + staging containment
    gm = manifest.get("generated_marker_sha256")
    if gm is not None:
        if not (isinstance(gm, str) and len(gm) == 64):
            raise BrainvisionReadRepairError("generated_marker_sha256 must be 64-hex")
        if _sha256_file(manifest["marker_file_target"]) != gm:
            raise BrainvisionReadRepairError("generated_marker_sha256 mismatch")
        if os.path.realpath(manifest["marker_file_target"]).startswith(raw + os.sep):
            raise BrainvisionReadRepairError("synthesized marker must not live inside the raw recording directory")
    if mode == MODE_MISSING_MARKER and gm is None:
        raise BrainvisionReadRepairError("missing-marker repair must synthesize a marker")
    if mode == MODE_POINTER_REWRITE and gm is not None:
        raise BrainvisionReadRepairError("pointer-rewrite must NOT synthesize a marker")

    if mode == MODE_CHANNEL_NAMES_FROM_TSV:
        need = ("channel_name_source", "channels_tsv_path", "channels_tsv_sha256", "channel_name_mapping_sha256",
                "original_header_channel_names_sha256", "repaired_header_channel_names_sha256", "channel_name_repair_policy_sha256")
        missing = [k for k in need if k not in manifest]
        if missing:
            raise BrainvisionReadRepairError(f"channel-name repair manifest missing fields {missing}")
        ctp = manifest["channels_tsv_path"]
        if not _is_regular_file(ctp) or _sha256_file(ctp) != manifest["channels_tsv_sha256"]:
            raise BrainvisionReadRepairError("channels.tsv missing or channels_tsv_sha256 mismatch")
        orig_names, rep_names = _parse_channel_infos(orig), _parse_channel_infos(rep)
        if orig_names is None or rep_names is None:
            raise BrainvisionReadRepairError("could not re-parse [Channel Infos] of the original/repaired header")
        if _names_sha256(orig_names) != manifest["original_header_channel_names_sha256"]:
            raise BrainvisionReadRepairError("original_header_channel_names_sha256 mismatch")
        if _names_sha256(rep_names) != manifest["repaired_header_channel_names_sha256"]:
            raise BrainvisionReadRepairError("repaired_header_channel_names_sha256 mismatch")
        if _mapping_sha256(rep_names) != manifest["channel_name_mapping_sha256"]:
            raise BrainvisionReadRepairError("channel_name_mapping_sha256 mismatch")
        try:
            CA.resolve_canonical_sources(rep_names)             # the renamed header MUST resolve all 19 canonical, no logical dup
        except CA.ChannelAliasError as e:
            raise BrainvisionReadRepairError(f"repaired header names do not resolve the canonical 19: {e}")
    return True


def repaired_read_path(disease, cohort, subject, recording_path, staging_dir):
    """Convenience for the reader: return (path_to_open, manifest_or_None). If a reviewed repair applies, materialize + validate it
    and return the repaired header path; otherwise return the original path and None."""
    plan = plan_repair(disease, cohort, subject, recording_path)
    if plan is None:
        return os.path.abspath(recording_path), None
    return apply_repair(plan, staging_dir)


def _manifest_digest(m):
    """The audited, order-independent slice of a manifest used for the set hash (paths excluded — only identity + content hashes)."""
    return {k: m.get(k) for k in ("repair_mode", "disease", "cohort", "subject", "recording",
                                  "original_header_sha256", "repaired_header_sha256", "generated_marker_sha256",
                                  "brainvision_read_repair_policy_sha256", "channel_name_mapping_sha256",
                                  "channels_tsv_sha256", "channel_name_repair_policy_sha256")}


def manifest_set_sha256(manifests):
    """Deterministic hash of a set of repair manifests (empty set → a fixed sentinel hash). Order-independent."""
    digests = sorted((_manifest_digest(m) for m in (manifests or [])),
                     key=lambda d: (d["disease"], d["cohort"], d["subject"], d["recording"], d["repair_mode"]))
    return hashlib.sha256(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


EMPTY_MANIFEST_SET_SHA256 = manifest_set_sha256([])
