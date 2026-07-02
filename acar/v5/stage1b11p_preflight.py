"""ACAR V5 Stage-1B11P metadata/header/geometry PREFLIGHT (read-only; NOT a Stage-1B build).

For every recording of the 7 frozen DEV cohorts, using the Stage-1B10/1B11 semantics, classify:
  * native_19_pass                 — all 19 canonical channels present after aliasing;
  * montage_completion_required    — the missing set is non-empty AND a subset of the per-cohort whitelist, with enough donor
                                     geometry (reports the exact missing set + donor estimate);
  * fail                           — a non-whitelisted missing canonical channel, a RAW-HEADER duplicate logical channel, or
                                     insufficient donor geometry.
Also runs the RAW-HEADER-decisive duplicate adjudication (a channels.tsv-only duplicate with a clean raw header is a non-fatal WARN).

STRICTLY metadata/header/geometry only:
  * participants.tsv (label via cohort_label_spec) + channels.tsv (names) — plain TSV reads;
  * raw HEADER channel names via mne read_raw_*(preload=False) — NO signal loaded;
  * donor geometry from channel names vs the standard_1020 position set — NO signal, NO interpolation.
NO get_data / NO raw.load_data / NO DSP / NO filtering/resampling / NO interpolation on real signals / NO full-file hashing / NO
build_manifest / NO complete_missing_channels / NO preprocess_subject / NO _windows_from_raw / NO training / NO embedding / NO
artifact / NO registry / NO SLURM.
"""
from __future__ import annotations
import csv
import os

from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.substrate import montage_completion as MC
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import raw_recording_manifest as RM   # discover_raw_recordings ONLY (path listing; NO hashing)

PROTOCOL_TAG_TARGET_SHA = "4278435975a72b1127803dd2cffab420c083e430"
IMPLEMENTATION_BASE_SHA = "fd0e1a77139fb8261588a8ba4ced3b6102255451"

_BASE = "/projects/EEG-foundation-model/datalake/raw/scps"
COHORT_ROOTS = {
    ("PD", "ds002778"): f"{_BASE}/PD/ds002778", ("PD", "ds003490"): f"{_BASE}/PD/ds003490",
    ("PD", "ds004584"): f"{_BASE}/PD/ds004584", ("SCZ", "ds003944"): f"{_BASE}/SCZ/ds003944",
    ("SCZ", "ds003947"): f"{_BASE}/SCZ/ds003947", ("SCZ", "ds004000"): f"{_BASE}/SCZ/ds004000",
    ("SCZ", "ds004367"): f"{_BASE}/SCZ/ds004367",
}
_EEG_EXT = (".edf", ".bdf", ".set", ".vhdr", ".fif")
_MNE_LOADER = {".edf": "read_raw_edf", ".bdf": "read_raw_bdf", ".set": "read_raw_eeglab",
               ".vhdr": "read_raw_brainvision", ".fif": "read_raw_fif"}


def _list_subjects(root):
    return sorted(d for d in os.listdir(root) if d.startswith("sub-") and os.path.isdir(os.path.join(root, d)))


def _channels_tsv_names(rec):
    base, d = os.path.basename(rec), os.path.dirname(rec)
    ct = None
    for suf in _EEG_EXT:
        if base.lower().endswith("_eeg" + suf):
            ct = os.path.join(d, base[: -len("_eeg" + suf)] + "_channels.tsv")
            break
    if ct is None or not os.path.isfile(ct):
        return None
    with open(ct, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return None
    col = next((c for c in rows[0] if str(c).strip().casefold() == "name"), None)
    return [r[col] for r in rows] if col else None


def _raw_header_names(rec, mne_state):
    if mne_state.get("mod") is None:
        import mne
        mne_state["mod"] = mne
    ext = os.path.splitext(rec)[1].lower()
    loader = getattr(mne_state["mod"].io, _MNE_LOADER[ext])
    raw = loader(rec, preload=False, verbose="ERROR")          # HEADER ONLY — no signal loaded, no get_data
    return list(raw.ch_names)


def _std_positions_casefold(mne_state):
    if mne_state.get("std") is None:
        if mne_state.get("mod") is None:
            import mne
            mne_state["mod"] = mne
        m = mne_state["mod"].channels.make_standard_montage("standard_1020")
        pos = m.get_positions()["ch_pos"]
        import numpy as np
        mne_state["std"] = {name.casefold() for name, p in pos.items() if p is not None and bool(np.isfinite(list(p)).all())}
    return mne_state["std"]


def _classify(disease, cohort, tsv_names, raw_names, std_cf):
    adj = CA.adjudicate_channel_source(tsv_names, raw_names)
    warn = adj["verdict"] == "WARN_TSV_DUPLICATE"
    if adj["verdict"] == "FAIL":
        return {"status": "fail", "failure_type": "raw_header_duplicate", "message": adj["reason"],
                "missing": [], "donor_estimate": 0, "warn": False}
    canon = {}
    for n in raw_names:
        c = CA.normalize_channel(n)
        if c is not None:
            canon[c] = n
    missing = [c for c in PC.CHANNELS_19 if c not in canon]
    donor_estimate = sum(1 for n in raw_names if str(n).strip().casefold() in std_cf)
    if not missing:
        return {"status": "native_19_pass", "failure_type": "", "message": "", "missing": [],
                "donor_estimate": donor_estimate, "warn": warn}
    allowed = MC.allowed_missing_for(cohort)
    if not set(missing) <= allowed:
        return {"status": "fail", "failure_type": "missing_not_whitelisted",
                "message": f"missing {missing} not in cohort whitelist {sorted(allowed)}",
                "missing": missing, "donor_estimate": donor_estimate, "warn": warn}
    if len(missing) > int(PC.PREPROCESSING_CONFIG["max_interpolated_canonical_channels_per_recording"]):
        return {"status": "fail", "failure_type": "too_many_missing", "message": f"{len(missing)} missing > max",
                "missing": missing, "donor_estimate": donor_estimate, "warn": warn}
    if donor_estimate < int(PC.PREPROCESSING_CONFIG["min_donor_channels"]):
        return {"status": "fail", "failure_type": "insufficient_donor_geometry",
                "message": f"donor_estimate {donor_estimate} < min {PC.PREPROCESSING_CONFIG['min_donor_channels']}",
                "missing": missing, "donor_estimate": donor_estimate, "warn": warn}
    return {"status": "montage_completion_required", "failure_type": "", "message": "",
            "missing": sorted(missing), "donor_estimate": donor_estimate, "warn": warn}


def run():
    st = {"mod": None, "std": None}
    std_cf = _std_positions_casefold(st)
    per_cohort, failures, completion, naming = [], [], [], {}
    for (disease, cohort), root in COHORT_ROOTS.items():
        row = {"disease": disease, "cohort": cohort, "n_subjects": 0, "n_labels_resolved": 0, "n_label_failures": 0,
               "n_recordings_checked": 0, "n_native_19_pass": 0, "n_montage_completion_required": 0,
               "n_warn_tsv_duplicate_raw_clean": 0, "n_fail": 0}
        participants = os.path.join(root, "participants.tsv")
        try:
            subjects = _list_subjects(root)
        except OSError as e:
            failures.append((disease, cohort, "*", "cohort_root", "cohort_list_failed", str(e)))
            per_cohort.append(row)
            continue
        row["n_subjects"] = len(subjects)
        for sub in subjects:
            try:
                CLS.resolve_label(disease, cohort, sub, participants)
                row["n_labels_resolved"] += 1
            except CLS.CohortLabelSpecError as e:
                row["n_label_failures"] += 1
                failures.append((disease, cohort, sub, "participants.tsv", "label_unresolvable", str(e)))
            try:
                recs = RM.discover_raw_recordings(os.path.join(root, sub))
            except RM.RawManifestError as e:
                failures.append((disease, cohort, sub, "eeg-discovery", "no_raw_recording", str(e)))
                continue
            for rec in recs:
                row["n_recordings_checked"] += 1
                try:
                    raw_names = _raw_header_names(rec, st)
                except Exception as e:  # noqa: BLE001
                    row["n_fail"] += 1
                    failures.append((disease, cohort, sub, os.path.basename(rec), "raw_header_read_failed", str(e)))
                    continue
                tsv_names = _channels_tsv_names(rec)
                r = _classify(disease, cohort, tsv_names, raw_names, std_cf)
                naming[_family(raw_names)] = naming.get(_family(raw_names), 0) + 1
                if r["warn"]:
                    row["n_warn_tsv_duplicate_raw_clean"] += 1
                if r["status"] == "native_19_pass":
                    row["n_native_19_pass"] += 1
                elif r["status"] == "montage_completion_required":
                    row["n_montage_completion_required"] += 1
                    completion.append((disease, cohort, sub, os.path.basename(rec), r["missing"], r["donor_estimate"], r["status"]))
                else:
                    row["n_fail"] += 1
                    failures.append((disease, cohort, sub, os.path.basename(rec), r["failure_type"], r["message"]))
        per_cohort.append(row)
    return per_cohort, failures, completion, naming


def _family(names):
    up = {str(n).strip().upper() for n in names}
    modern, old = up & {"T7", "T8", "P7", "P8"}, up & {"T3", "T4", "T5", "T6"}
    if modern and not old:
        return "modern_10_10"
    if old and not modern:
        return "old_10_20"
    if modern and old:
        return "mixed"
    return "other"


def _fmt(per_cohort, failures, completion, naming):
    L = []
    ok = not failures
    L.append(f"STATUS: {'PASS' if ok else 'FAIL'}")
    L.append(f"implementation_base_sha: {IMPLEMENTATION_BASE_SHA}")
    L.append(f"protocol_tag_target_sha: {PROTOCOL_TAG_TARGET_SHA}")
    L.append("scope: participants.tsv + channels.tsv + raw headers(preload=False) + standard_1020 geometry;"
             " no signal load / no DSP / no interpolation / no training / no embedding / no registry")
    L.append("")
    hdr = ("disease", "cohort", "subj", "lbl_ok", "lbl_fail", "recs", "native19", "completion", "tsv_warn", "fail")
    L.append("  ".join(f"{h:>10}" for h in hdr))
    for r in per_cohort:
        L.append("  ".join(f"{v:>10}" for v in (
            r["disease"], r["cohort"], r["n_subjects"], r["n_labels_resolved"], r["n_label_failures"],
            r["n_recordings_checked"], r["n_native_19_pass"], r["n_montage_completion_required"],
            r["n_warn_tsv_duplicate_raw_clean"], r["n_fail"])))
    L.append("")
    L.append("observed channel naming families: " + "; ".join(f"{k}×{v}" for k, v in sorted(naming.items())))
    L.append("")
    L.append(f"montage-completion-required recordings ({len(completion)}):")
    seen = {}
    for (d, c, s, rec, miss, don, st) in completion:
        seen.setdefault((c, tuple(miss)), [0, don])
        seen[(c, tuple(miss))][0] += 1
        seen[(c, tuple(miss))][1] = min(seen[(c, tuple(miss))][1], don)
    for (c, miss), (n, mindon) in sorted(seen.items()):
        L.append(f"  {c}: missing={list(miss)} × {n} recordings; min donor_estimate={mindon}")
    L.append("")
    if failures:
        L.append(f"FAILURES ({len(failures)}):")
        L.append("  ".join(("disease", "cohort", "subject", "recording", "failure_type", "message")))
        for f in failures[:300]:
            L.append("  ".join(str(x) for x in f))
        if len(failures) > 300:
            L.append(f"... (+{len(failures) - 300} more)")
    else:
        L.append("FAILURES: none")
    return "\n".join(L)


def main():
    per_cohort, failures, completion, naming = run()
    print(_fmt(per_cohort, failures, completion, naming))


if __name__ == "__main__":
    main()
