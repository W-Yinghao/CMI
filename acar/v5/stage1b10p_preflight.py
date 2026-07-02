"""ACAR V5 Stage-1B10P metadata/channel PREFLIGHT (read-only; NOT a Stage-1B build).

Confirms, for every subject of the 7 frozen DEV cohorts, that under the Stage-1B10 semantics:
  * the cohort label spec (cohort_label_spec) resolves the subject's control/case label, and
  * the channel-alias layer (channel_aliases) resolves ALL 19 canonical channels for EVERY raw recording that Stage-1B would
    process (per-recording, NOT a subject-union rule).

STRICTLY metadata/header only:
  * participants.tsv (label), channels.tsv (channel names) — plain TSV reads;
  * if a recording has no channels.tsv, its channel names are read from the raw HEADER with preload=False (no signal into memory).
  NO signal load / NO get_data / NO DSP / NO filtering / NO resampling / NO full-file hashing / NO build_manifest / NO training /
  NO embedding / NO artifact / NO registry / NO SLURM. Aggregate control/case counts only — no per-subject label values are written.
"""
from __future__ import annotations
import csv
import os

from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import raw_recording_manifest as RM   # discover_raw_recordings ONLY (path listing; no hashing)

PROTOCOL_TAG_TARGET_SHA = "4278435975a72b1127803dd2cffab420c083e430"
IMPLEMENTATION_BASE_SHA = "79447dcad7fce65c6468a89dd2bc518a6b9c8672"

_BASE = "/projects/EEG-foundation-model/datalake/raw/scps"
COHORT_ROOTS = {
    ("PD", "ds002778"): f"{_BASE}/PD/ds002778",
    ("PD", "ds003490"): f"{_BASE}/PD/ds003490",
    ("PD", "ds004584"): f"{_BASE}/PD/ds004584",
    ("SCZ", "ds003944"): f"{_BASE}/SCZ/ds003944",
    ("SCZ", "ds003947"): f"{_BASE}/SCZ/ds003947",
    ("SCZ", "ds004000"): f"{_BASE}/SCZ/ds004000",
    ("SCZ", "ds004367"): f"{_BASE}/SCZ/ds004367",
}
_EEG_EXT = (".edf", ".bdf", ".set", ".vhdr", ".fif")
_MNE_LOADER = {".edf": "read_raw_edf", ".bdf": "read_raw_bdf", ".set": "read_raw_eeglab",
               ".vhdr": "read_raw_brainvision", ".fif": "read_raw_fif"}


def _list_subjects(cohort_root):
    if not os.path.isdir(cohort_root):
        raise FileNotFoundError(f"cohort root not found: {cohort_root}")
    return sorted(d for d in os.listdir(cohort_root)
                  if d.startswith("sub-") and os.path.isdir(os.path.join(cohort_root, d)))


def _channels_tsv_for(rec):
    base, d = os.path.basename(rec), os.path.dirname(rec)
    for suf in _EEG_EXT:
        marker = "_eeg" + suf
        if base.lower().endswith(marker):
            return os.path.join(d, base[: -len(marker)] + "_channels.tsv")
    return os.path.join(d, os.path.splitext(base)[0] + "_channels.tsv")


def _read_channel_names_from_tsv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return None
    col = next((c for c in rows[0] if str(c).strip().casefold() == "name"), None)
    if col is None:
        return None
    return [r[col] for r in rows]


def _channel_names(rec, mne_state):
    """(names, source). Prefer channels.tsv (pure metadata); else the raw HEADER via mne preload=False (no signal loaded)."""
    ct = _channels_tsv_for(rec)
    if ct and os.path.isfile(ct):
        names = _read_channel_names_from_tsv(ct)
        if names:
            return names, "channels.tsv"
    if mne_state.get("mod") is None:
        import mne  # lazy — only if a channels.tsv is missing
        mne_state["mod"] = mne
    ext = os.path.splitext(rec)[1].lower()
    loader = getattr(mne_state["mod"].io, _MNE_LOADER[ext])
    raw = loader(rec, preload=False, verbose="ERROR")         # HEADER only — no signal into memory
    return list(raw.ch_names), "mne_header(preload=False)"


def _resolve_channels(names):
    """Mirror channel_aliases.resolve_canonical_sources for reporting (missing / duplicates / dropped extras)."""
    canon_to_src, dups, extras = {}, [], []
    for n in names:
        c = CA.normalize_channel(n)
        if c is None:
            extras.append(n)
            continue
        if c in canon_to_src:
            dups.append(c)
        else:
            canon_to_src[c] = n
    missing = [c for c in PC.CHANNELS_19 if c not in canon_to_src]
    return {"resolved": (not missing and not dups), "missing": missing, "duplicates": sorted(set(dups)),
            "n_dropped_extra": len(extras)}


def run():
    mne_state = {"mod": None}
    per_cohort, failures, naming_families = [], [], {}
    for (disease, cohort), root in COHORT_ROOTS.items():
        row = {"disease": disease, "cohort": cohort, "n_subject_dirs": 0, "n_label_resolved": 0, "n_label_failed": 0,
               "n_control": 0, "n_case": 0, "n_recordings_checked": 0, "n_channel_resolved": 0, "n_channel_failed": 0,
               "channel_sources": set(), "n_dropped_extra_total": 0}
        participants = os.path.join(root, "participants.tsv")
        try:
            subjects = _list_subjects(root)
        except OSError as e:
            failures.append((disease, cohort, "*", "cohort_root", "cohort_list_failed", str(e)))
            per_cohort.append(row)
            continue
        row["n_subject_dirs"] = len(subjects)
        for sub in subjects:
            # --- label (metadata only) ---
            try:
                lbl = CLS.resolve_label(disease, cohort, sub, participants)
                row["n_label_resolved"] += 1
                row["n_control" if lbl == 0 else "n_case"] += 1
            except CLS.CohortLabelSpecError as e:
                row["n_label_failed"] += 1
                failures.append((disease, cohort, sub, "participants.tsv", "label_unresolvable", str(e)))
            # --- channels per recording (per-recording, not subject-union) ---
            subject_dir = os.path.join(root, sub)
            try:
                recs = RM.discover_raw_recordings(subject_dir)   # path listing only (no hashing / no signal)
            except RM.RawManifestError as e:
                failures.append((disease, cohort, sub, "eeg-discovery", "no_raw_recording", str(e)))
                continue
            for rec in recs:
                row["n_recordings_checked"] += 1
                try:
                    names, src = _channel_names(rec, mne_state)
                except Exception as e:  # noqa: BLE001 — header read failure is a preflight failure, reported
                    row["n_channel_failed"] += 1
                    failures.append((disease, cohort, sub, os.path.basename(rec), "channel_names_read_failed", str(e)))
                    continue
                row["channel_sources"].add(src)
                naming_families.setdefault(_family(names), 0)
                naming_families[_family(names)] += 1
                res = _resolve_channels(names)
                row["n_dropped_extra_total"] += res["n_dropped_extra"]
                if res["resolved"]:
                    row["n_channel_resolved"] += 1
                else:
                    row["n_channel_failed"] += 1
                    ft = "missing_canonical" if res["missing"] else "duplicate_logical"
                    failures.append((disease, cohort, sub, os.path.basename(rec), ft,
                                     f"missing={res['missing']} duplicates={res['duplicates']}"))
        per_cohort.append(row)
    return per_cohort, failures, naming_families


def _family(names):
    has_modern = any(str(n).strip().upper() in ("T7", "T8", "P7", "P8") for n in names)
    has_old = any(str(n).strip().upper() in ("T3", "T4", "T5", "T6") for n in names)
    if has_modern and not has_old:
        return "modern_10_10 (T7/T8/P7/P8)"
    if has_old and not has_modern:
        return "old_10_20 (T3/T4/T5/T6)"
    if has_modern and has_old:
        return "mixed (T3+T7 present)"
    return "other/unknown"


def _fmt(per_cohort, failures, naming_families):
    lines = []
    passed = not failures
    lines.append(f"STATUS: {'PASS' if passed else 'FAIL'}")
    lines.append(f"implementation_base_sha: {IMPLEMENTATION_BASE_SHA}")
    lines.append(f"protocol_tag_target_sha: {PROTOCOL_TAG_TARGET_SHA}")
    lines.append("scope: participants.tsv + channels.tsv/raw headers only; no signal/DSP/train/embed/registry")
    lines.append("")
    hdr = ("disease", "cohort", "subj", "lbl_ok", "lbl_fail", "ctrl", "case", "recs", "ch_ok", "ch_fail", "src", "extras")
    lines.append("  ".join(f"{h:>8}" for h in hdr))
    for r in per_cohort:
        lines.append("  ".join(f"{v:>8}" for v in (
            r["disease"], r["cohort"], r["n_subject_dirs"], r["n_label_resolved"], r["n_label_failed"],
            r["n_control"], r["n_case"], r["n_recordings_checked"], r["n_channel_resolved"], r["n_channel_failed"],
            ",".join(sorted(r["channel_sources"])) or "-", r["n_dropped_extra_total"])))
    lines.append("")
    lines.append("observed channel naming families: " + "; ".join(f"{k}×{v}" for k, v in sorted(naming_families.items())))
    lines.append("")
    if failures:
        lines.append(f"FAILURES ({len(failures)}):")
        lines.append("  ".join(("disease", "cohort", "subject", "source", "failure_type", "message")))
        for f in failures[:200]:
            lines.append("  ".join(str(x) for x in f))
        if len(failures) > 200:
            lines.append(f"... (+{len(failures) - 200} more)")
    else:
        lines.append("FAILURES: none")
    return "\n".join(lines)


def main():
    per_cohort, failures, naming_families = run()
    print(_fmt(per_cohort, failures, naming_families))


if __name__ == "__main__":
    main()
