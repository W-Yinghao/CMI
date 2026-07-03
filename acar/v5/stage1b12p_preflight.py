"""ACAR V5 Stage-1B12P metadata/header/repair/geometry PREFLIGHT (read-only; NOT a Stage-1B build).

For every recording of the 7 frozen DEV cohorts, using the Stage-1B10/1B11/1B12 semantics, classify EXACTLY one of:
  * native_19_pass                                  — all 19 canonical channels present after aliasing; no repair, no completion;
  * montage_completion_required                     — missing set exactly authorized (ds004584:{Pz}, ds004000:{F3,F4,P3,P4},
                                                      ds004367:{F7}+F7-0/F7-1 variant pattern); no header repair;
  * read_repair_required                            — a reviewed BrainVision header repair is required, but no montage completion;
  * read_repair_plus_montage_completion_required    — both a reviewed read-repair AND a reviewed montage completion are required;
  * fail                                            — any unreviewed missing channel, a raw-header duplicate logical channel, a
                                                      BrainVision repair not whitelisted, an invalid repair manifest, a repaired
                                                      header still unreadable at preload=False, or insufficient donor geometry.

STRICTLY metadata/header/repair-readability only:
  * participants.tsv (label via cohort_label_spec) + channels.tsv (names) — plain TSV reads;
  * raw HEADER channel names via mne read_raw_*(preload=False) — NO signal loaded;
  * BrainVision repair PLANNING (plan_repair) + EPHEMERAL repaired-header/marker creation in a staging dir + repaired-header
    open test with preload=False + repair-manifest validation — the raw SIGNAL is never modified/copied/loaded;
  * donor geometry from channel names vs the standard_1020 position set — NO signal, NO interpolation.
NEVER: get_data / raw.load_data / preload=True / DSP / filtering / resampling / interpolate_bads on real signal / full raw-file
hashing / preprocess_subject / _windows_from_raw / complete_missing_channels on real recordings / training / embedding / registry /
SLURM. The only files written are the ephemeral repaired headers/markers under the staging dir.
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
from acar.v5.substrate import raw_recording_manifest as RM   # discover_raw_recordings ONLY (path listing; NO hashing)

PROTOCOL_TAG_TARGET_SHA = "4278435975a72b1127803dd2cffab420c083e430"
IMPLEMENTATION_BASE_SHA = "b67ca3b95f3767043f92a574fb6b13732ae799ab"

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


def _raw_header_names(read_path, mne_state):
    if mne_state.get("mod") is None:
        import mne
        mne_state["mod"] = mne
    ext = os.path.splitext(read_path)[1].lower()
    loader = getattr(mne_state["mod"].io, _MNE_LOADER[ext])
    raw = loader(read_path, preload=False, verbose="ERROR")     # HEADER ONLY — no signal loaded, no get_data
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
    """Header/repair/geometry classification of ONE recording (no signal load). Mutates `counters` for the repair summary."""
    # (1) BrainVision read-repair planning + ephemeral materialization (header-only; raw signal untouched)
    read_path, repair_mode = recording_path, None
    plan = BR.plan_repair(disease, cohort, subject, recording_path)
    if plan is not None:
        try:
            read_path, manifest = BR.apply_repair(plan, staging_dir)   # writes ONLY into staging; re-verifies manifest hashes
            BR.assert_manifest_consistent(manifest)
        except BR.BrainvisionReadRepairError as e:
            counters[f"mode_{plan.mode}"] = counters.get(f"mode_{plan.mode}", 0) + 1
            return _fail("repair_manifest_invalid", str(e), repair_mode=plan.mode)
        repair_mode = manifest["repair_mode"]
        counters[f"mode_{repair_mode}"] = counters.get(f"mode_{repair_mode}", 0) + 1
        counters["manifest_validated"] = counters.get("manifest_validated", 0) + 1

    # (2) open the (repaired-or-original) header at preload=False
    try:
        raw_names = _raw_header_names(read_path, mne_state)
    except Exception as e:  # noqa: BLE001
        if plan is not None:
            counters["repaired_preload_false_fail"] = counters.get("repaired_preload_false_fail", 0) + 1
            return _fail("repaired_header_preload_false_fail", str(e), repair_mode=repair_mode)
        return _fail("header_unreadable_no_repair", str(e))
    if plan is not None:
        counters["repaired_preload_false_pass"] = counters.get("repaired_preload_false_pass", 0) + 1

    # (3) raw-header-decisive duplicate adjudication
    tsv_names = _channels_tsv_names(recording_path)            # channels.tsv of the ORIGINAL recording
    adj = CA.adjudicate_channel_source(tsv_names, raw_names)
    if adj["verdict"] == "FAIL":
        return _fail("raw_header_duplicate", adj["reason"], repair_mode=repair_mode)
    warn = adj["verdict"] == "WARN_TSV_DUPLICATE"

    # (4) missing canonical + donor geometry (name-only)
    canon = {}
    for n in raw_names:
        c = CA.normalize_channel(n)
        if c is not None:
            canon[c] = n
    missing = [c for c in PC.CHANNELS_19 if c not in canon]
    donor_estimate = sum(1 for n in raw_names if str(n).strip().casefold() in std_cf)
    needs_repair = plan is not None

    if not missing:
        status = "read_repair_required" if needs_repair else "native_19_pass"
        return {"status": status, "failure_type": "", "message": "", "repair_mode": repair_mode, "warn": warn,
                "missing": [], "donor_estimate": donor_estimate}

    if len(missing) == 19:                                     # header carries NON-canonical (e.g. generic EEG001..) channel names
        tsv_canon = len({CA.normalize_channel(n) for n in (tsv_names or []) if CA.normalize_channel(n) is not None})
        note = " (real electrode names are in channels.tsv, which the pinned reader does NOT consult)" if tsv_canon >= 19 else ""
        return _fail("header_channel_names_non_canonical",
                     f"header names {list(raw_names)[:4]}… resolve 0/19 canonical; channels.tsv resolves {tsv_canon}/19{note}",
                     repair_mode=repair_mode, warn=warn, missing=missing, donor=donor_estimate)

    allowed = MC.allowed_missing_for(cohort)
    if not set(missing) <= allowed:
        return _fail("missing_not_whitelisted", f"missing {missing} not in cohort whitelist {sorted(allowed)}",
                     repair_mode=repair_mode, warn=warn, missing=missing, donor=donor_estimate)
    try:                                                       # ds004367 F7 requires the F7-0/F7-1 variant pattern (pure name check)
        MC._require_conditional(cohort, missing, raw_names)
    except MC.MontageCompletionError as e:
        return _fail("conditional_variant_pattern_absent", str(e), repair_mode=repair_mode, warn=warn,
                     missing=missing, donor=donor_estimate)
    if len(missing) > int(PC.PREPROCESSING_CONFIG["max_interpolated_canonical_channels_per_recording"]):
        return _fail("too_many_missing", f"{len(missing)} missing > max", repair_mode=repair_mode, warn=warn,
                     missing=missing, donor=donor_estimate)
    if donor_estimate < int(PC.PREPROCESSING_CONFIG["min_donor_channels"]):
        return _fail("insufficient_donor_geometry",
                     f"donor_estimate {donor_estimate} < min {PC.PREPROCESSING_CONFIG['min_donor_channels']}",
                     repair_mode=repair_mode, warn=warn, missing=missing, donor=donor_estimate)
    status = "read_repair_plus_montage_completion_required" if needs_repair else "montage_completion_required"
    return {"status": status, "failure_type": "", "message": "", "repair_mode": repair_mode, "warn": warn,
            "missing": sorted(missing), "donor_estimate": donor_estimate}


_STATUS_KEYS = ("native_19_pass", "montage_completion_required", "read_repair_required",
                "read_repair_plus_montage_completion_required", "fail")


def run():
    st = {"mod": None, "std": None}
    std_cf = _std_positions_casefold(st)
    staging = tempfile.mkdtemp(prefix="acar_v5_1b12p_staging_")
    per_cohort, failures, completion = [], [], []
    counters = {}
    f7_variant_count = 0
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
                if cohort == "ds004367" and "F7" in r.get("missing", []):
                    f7_variant_count += 1
                if r["status"] in ("montage_completion_required", "read_repair_plus_montage_completion_required"):
                    completion.append((disease, cohort, sub, os.path.basename(rec), r["missing"], r["donor_estimate"], r["status"]))
                elif r["status"] == "fail":
                    failures.append((disease, cohort, sub, os.path.basename(rec), r["failure_type"], r["message"]))
        per_cohort.append(row)
    return per_cohort, failures, completion, counters, f7_variant_count, staging


def _fmt(per_cohort, failures, completion, counters, f7_variant_count, staging):
    L = []
    ok = not failures
    L.append(f"STATUS: {'PASS' if ok else 'FAIL'}")
    L.append(f"implementation_base_sha: {IMPLEMENTATION_BASE_SHA}")
    L.append(f"protocol_tag_target_sha: {PROTOCOL_TAG_TARGET_SHA}")
    L.append("scope: participants.tsv + channels.tsv + raw headers(preload=False) + ephemeral repaired-header(preload=False) +"
             " standard_1020 geometry; no signal load / no DSP / no real interpolation / no training / no embedding / no registry")
    L.append(f"staging(ephemeral repaired headers/markers only): {staging}")
    L.append("")
    hdr = ("disease", "cohort", "subj", "lbl_ok", "lbl_fail", "recs", "native19", "mont_compl", "read_rep",
           "rep+compl", "fail")
    L.append("  ".join(f"{h:>11}" for h in hdr))
    for r in per_cohort:
        L.append("  ".join(f"{v:>11}" for v in (
            r["disease"], r["cohort"], r["n_subjects"], r["n_labels_resolved"], r["n_label_failures"],
            r["n_recordings_checked"], r["n_native_19_pass"], r["n_montage_completion_required"],
            r["n_read_repair_required"], r["n_read_repair_plus_montage_completion_required"], r["n_fail"])))
    L.append("")
    L.append("repair summary:")
    L.append(f"  n_missing_markerfile_minimal_vmrk        = {counters.get('mode_' + BR.MODE_MISSING_MARKER, 0)}")
    L.append(f"  n_broken_internal_pointer_rewrite        = {counters.get('mode_' + BR.MODE_POINTER_REWRITE, 0)}")
    L.append(f"  n_repair_manifest_validated              = {counters.get('manifest_validated', 0)}")
    L.append(f"  n_repaired_header_preload_false_pass     = {counters.get('repaired_preload_false_pass', 0)}")
    L.append(f"  n_repaired_header_preload_false_fail     = {counters.get('repaired_preload_false_fail', 0)}")
    L.append("")
    L.append("montage summary:")
    by_set = {}
    for (d, c, s, rec, miss, don, status) in completion:
        by_set.setdefault((c, tuple(miss)), []).append(don)
    for (c, miss), dons in sorted(by_set.items()):
        L.append(f"  {c}: missing={list(miss)} × {len(dons)} recordings; donor_estimate min={min(dons)} "
                 f"median={int(statistics.median(dons))}")
    L.append(f"  ds004367 F7-variant-pattern completions   = {f7_variant_count}")
    unreviewed = [f for f in failures if f[4] in ("missing_not_whitelisted", "conditional_variant_pattern_absent",
                                                  "too_many_missing", "header_channel_names_non_canonical")]
    L.append(f"  unreviewed missing-channel cases          = {len(unreviewed)}")
    ftype_counts = {}
    for f in failures:
        ftype_counts[f[4]] = ftype_counts.get(f[4], 0) + 1
    L.append("  failure_type histogram: " + "; ".join(f"{k}×{v}" for k, v in sorted(ftype_counts.items())))
    L.append("")
    if failures:
        L.append(f"FAILURES ({len(failures)}):")
        L.append("  ".join(("disease", "cohort", "subject", "recording", "failure_type", "message")))
        for f in failures[:400]:
            L.append("  ".join(str(x) for x in f))
        if len(failures) > 400:
            L.append(f"... (+{len(failures) - 400} more)")
    else:
        L.append("FAILURES: none")
    return "\n".join(L)


def main():
    per_cohort, failures, completion, counters, f7v, staging = run()
    print(_fmt(per_cohort, failures, completion, counters, f7v, staging))


if __name__ == "__main__":
    main()
