"""ACAR V5 Stage-1B14P metadata/header/repair/name/geometry PREFLIGHT (read-only; NOT a Stage-1B build).

Identical 8-way classifier to Stage-1B13P, but the shared channels.tsv channel-NAME repair now accepts the WIDENED ordinal-placeholder
header pattern (Stage-1B14): the i-th header name may be <PREFIX>i with PREFIX in {EEG,EOG,ECG} and the integer equal to the 1-based
data-column position (ds003944/ds003947 only) — so type-prefixed headers (EOG/ECG on the eye/cardiac channels) are renamed from
channels.tsv by row order like the pure-EEG ones. For every recording of the 7 frozen DEV cohorts it: plans the reviewed BrainVision
repair (marker synth / pointer rewrite / channels.tsv rename), materializes the ephemeral repaired header into a staging dir, opens the
(repaired-or-original) header at preload=False, adjudicates duplicates, computes the
missing canonical channels + standard_1020 donor geometry, and classifies EXACTLY one of:
  native_19_pass · montage_completion_required · read_repair_required · channel_name_repair_required ·
  read_repair_plus_channel_name_repair_required · read_repair_plus_montage_completion_required ·
  read_repair_plus_channel_name_repair_plus_montage_completion_required · fail.

STRICTLY metadata/header/repair-readability only (same forbidden set as Stage-1B12P): NO signal preload/get_data/load_data, NO DSP,
NO real interpolation, NO full raw-file hashing, NO preprocess_subject/_windows_from_raw/complete_missing_channels on real recordings,
NO training/embedding/registry/SLURM. The only files written are the ephemeral repaired headers/markers under the staging dir.
"""
from __future__ import annotations
import csv
import os
import statistics
import tempfile

from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.substrate import channel_aliases as CA
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.substrate import montage_completion as MC
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import raw_recording_manifest as RM

PROTOCOL_TAG_TARGET_SHA = "4278435975a72b1127803dd2cffab420c083e430"
IMPLEMENTATION_BASE_SHA = "3fe885245133e2bc141651955da33bb7fd7adeac"   # reviewed Stage-1B14 commit

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

_STATUS_KEYS = ("native_19_pass", "montage_completion_required", "read_repair_required", "channel_name_repair_required",
                "read_repair_plus_channel_name_repair_required", "read_repair_plus_montage_completion_required",
                "read_repair_plus_channel_name_repair_plus_montage_completion_required", "fail")
# (read_repair, name_repair, needs_completion) -> status
_COMBO = {
    (False, False, False): "native_19_pass",
    (False, False, True): "montage_completion_required",
    (True, False, False): "read_repair_required",
    (True, False, True): "read_repair_plus_montage_completion_required",
    (False, True, False): "channel_name_repair_required",
    (True, True, False): "read_repair_plus_channel_name_repair_required",
    (True, True, True): "read_repair_plus_channel_name_repair_plus_montage_completion_required",
}


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


def _raw_header_names(read_path, mne_state):
    if mne_state.get("mod") is None:
        import mne
        mne_state["mod"] = mne
    ext = os.path.splitext(read_path)[1].lower()
    loader = getattr(mne_state["mod"].io, _MNE_LOADER[ext])
    raw = loader(read_path, preload=False, verbose="ERROR")
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


def _fail(ftype, msg, repair_mode=None, warn=False, missing=(), donor=0):
    return {"status": "fail", "failure_type": ftype, "message": msg, "repair_mode": repair_mode, "warn": warn,
            "missing": list(missing), "donor_estimate": donor}


def classify(disease, cohort, subject, recording_path, staging_dir, mne_state, std_cf, counters):
    read_path, repair_mode = recording_path, None
    marker_repair = name_repair = False
    plan = BR.plan_repair(disease, cohort, subject, recording_path)
    if plan is not None:
        try:
            read_path, manifest = BR.apply_repair(plan, staging_dir)
            BR.assert_manifest_consistent(manifest)
        except BR.BrainvisionReadRepairError as e:
            counters[f"mode_{plan.mode}"] = counters.get(f"mode_{plan.mode}", 0) + 1
            return _fail("repair_manifest_invalid", str(e), repair_mode=plan.mode)
        repair_mode = manifest["repair_mode"]
        counters[f"mode_{repair_mode}"] = counters.get(f"mode_{repair_mode}", 0) + 1
        counters["manifest_validated"] = counters.get("manifest_validated", 0) + 1
        name_repair = repair_mode == BR.MODE_CHANNEL_NAMES_FROM_TSV
        if name_repair:                                        # Stage-1B14 report telemetry: pure_eeg vs type_prefixed ordinal
            counters[f"subtype_{manifest.get('channel_name_repair_subtype')}"] = \
                counters.get(f"subtype_{manifest.get('channel_name_repair_subtype')}", 0) + 1
        marker_repair = repair_mode in (BR.MODE_MISSING_MARKER, BR.MODE_POINTER_REWRITE) or \
            (name_repair and manifest.get("generated_marker_sha256") is not None)

    try:
        raw_names = _raw_header_names(read_path, mne_state)
    except Exception as e:  # noqa: BLE001
        if plan is not None:
            counters["repaired_preload_false_fail"] = counters.get("repaired_preload_false_fail", 0) + 1
            return _fail("repaired_header_preload_false_fail", str(e), repair_mode=repair_mode)
        return _fail("header_unreadable_no_repair", str(e))
    if plan is not None:
        counters["repaired_preload_false_pass"] = counters.get("repaired_preload_false_pass", 0) + 1

    adj = CA.adjudicate_channel_source(_channels_tsv_names(recording_path), raw_names)
    if adj["verdict"] == "FAIL":
        return _fail("raw_header_duplicate", adj["reason"], repair_mode=repair_mode)
    warn = adj["verdict"] == "WARN_TSV_DUPLICATE"

    canon = {}
    for n in raw_names:
        c = CA.normalize_channel(n)
        if c is not None:
            canon[c] = n
    missing = [c for c in PC.CHANNELS_19 if c not in canon]
    donor_estimate = sum(1 for n in raw_names if str(n).strip().casefold() in std_cf)

    needs_completion = False
    if missing:
        if len(missing) == 19:
            tsv_canon = len({CA.normalize_channel(n) for n in (_channels_tsv_names(recording_path) or [])
                             if CA.normalize_channel(n) is not None})
            note = " (real names in channels.tsv, not consulted by the pinned reader)" if tsv_canon >= 19 else ""
            return _fail("header_channel_names_non_canonical",
                         f"header names {list(raw_names)[:4]}… resolve 0/19; channels.tsv resolves {tsv_canon}/19{note}",
                         repair_mode=repair_mode, warn=warn, missing=missing, donor=donor_estimate)
        allowed = MC.allowed_missing_for(cohort)
        if not set(missing) <= allowed:
            return _fail("missing_not_whitelisted", f"missing {missing} not in cohort whitelist {sorted(allowed)}",
                         repair_mode=repair_mode, warn=warn, missing=missing, donor=donor_estimate)
        try:
            MC._require_conditional(cohort, missing, raw_names)
        except MC.MontageCompletionError as e:
            return _fail("conditional_variant_pattern_absent", str(e), repair_mode=repair_mode, warn=warn,
                         missing=missing, donor=donor_estimate)
        if len(missing) > int(PC.PREPROCESSING_CONFIG["max_interpolated_canonical_channels_per_recording"]):
            return _fail("too_many_missing", f"{len(missing)} missing > max", repair_mode=repair_mode, warn=warn,
                         missing=missing, donor=donor_estimate)
        if donor_estimate < int(PC.PREPROCESSING_CONFIG["min_donor_channels"]):
            return _fail("insufficient_donor_geometry", f"donor_estimate {donor_estimate} < min", repair_mode=repair_mode,
                         warn=warn, missing=missing, donor=donor_estimate)
        needs_completion = True

    status = _COMBO.get((marker_repair, name_repair, needs_completion))
    if status is None:                                         # e.g. channel-name repair + completion with NO marker repair
        return _fail("unexpected_repair_combination",
                     f"read_repair={marker_repair} name_repair={name_repair} completion={needs_completion}",
                     repair_mode=repair_mode, warn=warn, missing=missing, donor=donor_estimate)
    return {"status": status, "failure_type": "", "message": "", "repair_mode": repair_mode, "warn": warn,
            "missing": sorted(missing), "donor_estimate": donor_estimate}


def run():
    st = {"mod": None, "std": None}
    std_cf = _std_positions_casefold(st)
    staging = tempfile.mkdtemp(prefix="acar_v5_1b14p_staging_")
    per_cohort, failures, completion, counters = [], [], [], {}
    for (disease, cohort), root in COHORT_ROOTS.items():
        row = {"disease": disease, "cohort": cohort, "n_subjects": 0, "n_labels_resolved": 0, "n_label_failures": 0,
               "n_recordings_checked": 0}
        for k in _STATUS_KEYS:
            row["n_" + k] = 0
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
                r = classify(disease, cohort, sub, rec, staging, st, std_cf, counters)
                row["n_" + r["status"]] += 1
                if r["status"].endswith("montage_completion_required"):
                    completion.append((cohort, tuple(r["missing"]), r["donor_estimate"]))
                elif r["status"] == "fail":
                    failures.append((disease, cohort, sub, os.path.basename(rec), r["failure_type"], r["message"]))
        per_cohort.append(row)
    return per_cohort, failures, completion, counters, staging


def _fmt(per_cohort, failures, completion, counters, staging):
    L = [f"STATUS: {'PASS' if not failures else 'FAIL'}", f"implementation_base_sha: {IMPLEMENTATION_BASE_SHA}",
         f"protocol_tag_target_sha: {PROTOCOL_TAG_TARGET_SHA}",
         "scope: participants.tsv + channels.tsv + raw headers(preload=False) + ephemeral repaired-header(preload=False) +"
         " standard_1020 geometry; no signal load / no DSP / no real interpolation / no training / no embedding / no registry",
         f"staging(ephemeral repaired headers/markers only): {staging}", ""]
    short = ("native19", "montComp", "readRep", "nameRep", "rep+name", "rep+comp", "rep+name+comp", "fail")
    hdr = ("disease", "cohort", "subj", "lblOK", "recs") + short
    L.append("  ".join(f"{h:>13}" for h in hdr))
    for r in per_cohort:
        vals = (r["disease"], r["cohort"], r["n_subjects"], r["n_labels_resolved"], r["n_recordings_checked"]) + \
               tuple(r["n_" + k] for k in _STATUS_KEYS)
        L.append("  ".join(f"{v:>13}" for v in vals))
    L += ["", "repair summary:",
          f"  n_missing_markerfile_minimal_vmrk     = {counters.get('mode_' + BR.MODE_MISSING_MARKER, 0)}",
          f"  n_broken_internal_pointer_rewrite     = {counters.get('mode_' + BR.MODE_POINTER_REWRITE, 0)}",
          f"  n_channel_names_from_channels_tsv     = {counters.get('mode_' + BR.MODE_CHANNEL_NAMES_FROM_TSV, 0)}",
          f"  n_pure_eeg_ordinal                    = {counters.get('subtype_pure_eeg_ordinal', 0)}",
          f"  n_type_prefixed_ordinal               = {counters.get('subtype_type_prefixed_ordinal', 0)}",
          f"  n_repair_manifest_validated           = {counters.get('manifest_validated', 0)}",
          f"  n_repaired_header_preload_false_pass  = {counters.get('repaired_preload_false_pass', 0)}",
          f"  n_repaired_header_preload_false_fail  = {counters.get('repaired_preload_false_fail', 0)}", "", "montage summary:"]
    by_set = {}
    for (c, miss, don) in completion:
        by_set.setdefault((c, miss), []).append(don)
    for (c, miss), dons in sorted(by_set.items()):
        L.append(f"  {c}: missing={list(miss)} × {len(dons)}; donor_estimate min={min(dons)} median={int(statistics.median(dons))}")
    L.append("")
    if failures:
        ft = {}
        for f in failures:
            ft[f[4]] = ft.get(f[4], 0) + 1
        L.append(f"FAILURES ({len(failures)}): " + "; ".join(f"{k}×{v}" for k, v in sorted(ft.items())))
        L.append("  ".join(("disease", "cohort", "subject", "recording", "failure_type", "message")))
        for f in failures[:400]:
            L.append("  ".join(str(x) for x in f))
    else:
        L.append("FAILURES: none")
    return "\n".join(L)


def main():
    per_cohort, failures, completion, counters, staging = run()
    print(_fmt(per_cohort, failures, completion, counters, staging))


if __name__ == "__main__":
    main()
