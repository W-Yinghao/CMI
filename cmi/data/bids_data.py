"""Generic BIDS-EEG loader for the cross-dataset SCPS benchmark (Protocol C, hierarchical D).

Each OpenNeuro cohort is BIDS. We read the resting (or specified) task per subject, label it via a
per-cohort rule (COHORTS registry), canonicalize every recording to a common 19-ch 10-20 montage, and
tag the trial with BOTH its cohort and subject so the domain can be taken at either granularity —
`D=(cohort, subject)` — which is what the hierarchical pi_y needs.

  X,y,meta,classes = load_crossdataset("PD")          # pool all PD cohorts, leave-one-cohort-out
  meta columns: subject (globally-unique 'dsid/sub'), cohort (dsid), session, run

Cohort label rules live in COHORTS; add a cohort by appending one entry. label_fn returns the int
class or None (skip the subject, e.g. unlabeled / wrong group)."""
import os, glob, csv, re, tempfile
import numpy as np
import mne

mne.set_log_level("ERROR")

# canonical 19-ch 10-20 (modern names); old names mapped in ALIAS
CH19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ"]
ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _norm(ch):
    c = ch.upper().replace("EEG ", "").strip()
    for suf in ("-REF", "-LE", "-A1", "-A2"):
        if c.endswith(suf):
            c = c[: -len(suf)]
    return ALIAS.get(c, c)


_GENERIC_CH = re.compile(r"^(EEG|CH)\s*0*\d+$", re.IGNORECASE)  # placeholder names like EEG001


def _apply_channels_tsv(raw, fp):
    """If channel names are generic placeholders (EEG001..) and a sibling BIDS
    *_channels.tsv exists with a matching count, rename channels positionally to the
    real names and set non-EEG channel types (EOG/ECG/MISC) from the tsv 'type' column.
    No-op when names already look real or the tsv is absent/mismatched (backward-safe)."""
    if not any(_GENERIC_CH.match(c) for c in raw.ch_names):
        return raw
    tsv = re.sub(r"_eeg\.(vhdr|set|edf|bdf)$", "_channels.tsv", fp)
    if not os.path.exists(tsv):
        return raw
    try:
        rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    except Exception:
        return raw
    names = [str(r.get("name", "")).strip() for r in rows]
    if len(names) != len(raw.ch_names) or not all(names):
        return raw                                  # count mismatch -> leave untouched
    raw.rename_channels({o: n for o, n in zip(raw.ch_names, names)})
    type_map = {"EOG": "eog", "ECG": "ecg", "EKG": "ecg", "MISC": "misc",
                "EMG": "emg", "TRIG": "stim", "VEOG": "eog", "HEOG": "eog"}
    setc = {n: type_map[str(r.get("type", "")).strip().upper()]
            for n, r in zip(names, rows)
            if str(r.get("type", "")).strip().upper() in type_map}
    if setc:
        try:
            raw.set_channel_types(setc, verbose="ERROR")
        except Exception:
            pass
    return raw


def _read_brainvision_robust(fp):
    """read_raw_brainvision, tolerant of .vhdr headers that omit the MarkerFile= line
    (configparser NoOptionError 'markerfile'). On that failure, write a patched temp .vhdr
    (next to the .eeg so the relative DataFile resolves) that points MarkerFile at either a
    sibling .vmrk, or — if none exists — a minimal stub .vmrk, so MNE reads the data without
    requiring real events. Cleans up temp files afterwards."""
    try:
        return mne.io.read_raw_brainvision(fp, preload=True)
    except Exception as e:
        if "markerfile" not in str(e).lower():
            raise
    d = os.path.dirname(fp)
    base = os.path.basename(fp)[:-5]                  # strip '.vhdr'
    sib_vmrk = os.path.join(d, base + ".vmrk")
    tmp_vmrk = None
    if os.path.exists(sib_vmrk):
        marker_name = os.path.basename(sib_vmrk)
    else:                                             # no events on disk -> minimal stub
        eeg_name = base + ".eeg"
        if not os.path.exists(os.path.join(d, eeg_name)):
            # fall back to whatever DataFile the header declares
            for line in open(fp, errors="replace"):
                if line.strip().lower().startswith("datafile="):
                    eeg_name = line.split("=", 1)[1].strip(); break
        fd, tmp_vmrk = tempfile.mkstemp(prefix="_cmivmrk_", suffix=".vmrk", dir=d)
        with os.fdopen(fd, "w") as fh:
            fh.write("Brain Vision Data Exchange Marker File, Version 1.0\n\n"
                     "[Common Infos]\nDataFile=%s\n\n"
                     "[Marker Infos]\nMk1=New Segment,,1,1,0,0\n" % eeg_name)
        marker_name = os.path.basename(tmp_vmrk)
    lines = open(fp, errors="replace").read().splitlines()
    out, injected = [], False
    for line in lines:
        out.append(line)
        if not injected and line.strip().lower().startswith("[common infos]"):
            out.append("MarkerFile=" + marker_name); injected = True
    if not injected:
        out.append("[Common Infos]"); out.append("MarkerFile=" + marker_name)
    fd, tmp_vhdr = tempfile.mkstemp(prefix="_cmivhdr_", suffix=".vhdr", dir=d)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(out) + "\n")
    try:
        return mne.io.read_raw_brainvision(tmp_vhdr, preload=True)
    finally:
        for f in (tmp_vhdr, tmp_vmrk):
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


def _read_raw(fp):
    if fp.endswith(".set"):
        return mne.io.read_raw_eeglab(fp, preload=True)
    if fp.endswith(".vhdr"):
        raw = _read_brainvision_robust(fp)
        return _apply_channels_tsv(raw, fp)          # fix generic EEG001.. names from channels.tsv
    if fp.endswith(".bdf"):
        return mne.io.read_raw_bdf(fp, preload=True)
    if fp.endswith(".edf"):
        return mne.io.read_raw_edf(fp, preload=True)
    raise ValueError(f"unknown EEG format: {fp}")


def _windows(data, win, max_w):
    n = data.shape[1] // win
    return [data[:, i * win:(i + 1) * win] for i in range(min(n, max_w))]


def _canon_19(raw):
    """Rename channels to canonical 10-20, set a standard montage, and INTERPOLATE any of the 19
    that are missing natively (e.g. an online-reference channel like Pz). Returns the raw restricted
    to the 19, in CH19 order, or None if too many (>4) channels would need interpolating."""
    new, used = {}, set()
    for c in raw.ch_names:
        n = _norm(c)
        if n in used:                       # avoid duplicate target names
            n = c
        used.add(n); new[c] = n
    raw.rename_channels(new)
    present = [c for c in CH19 if c in raw.ch_names]
    missing = [c for c in CH19 if c not in raw.ch_names]
    if len(present) < 15:                   # not really a 10-20 recording
        return None
    raw.set_montage("standard_1020", on_missing="ignore", match_case=False, verbose="ERROR")
    # drop channels with no valid position (non-10-20 like I1/I2) — they break interpolate_bads
    pos = raw.get_montage().get_positions()["ch_pos"]
    nopos = [c for c in raw.ch_names if c not in pos or np.isnan(np.asarray(pos[c], float)).any()]
    if nopos:
        raw.drop_channels(nopos)
    if missing:
        info = mne.create_info(missing, raw.info["sfreq"], ["eeg"] * len(missing))
        pad = mne.io.RawArray(np.zeros((len(missing), raw.n_times), "float64"), info, verbose="ERROR")
        raw.add_channels([pad], force_update_info=True)
        raw.set_montage("standard_1020", on_missing="ignore", match_case=False, verbose="ERROR")
        raw.info["bads"] = missing
        try:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        except Exception:
            return None
    return raw


def _read_participants(ds):
    p = os.path.join(ds, "participants.tsv")
    if not os.path.exists(p):
        return {}
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    return {r["participant_id"]: r for r in rows}


def load_cohort(ds, task, label_fn, fmin=0.5, fmax=45.0, resample=128, win_sec=4.0,
                max_per_subject=40, crop_sec=180, normalize="trial_zscore"):
    """Load one BIDS cohort -> (X[B,19,T], y, subjects(list of sub-ids), classes-as-ints present)."""
    part = _read_participants(ds)
    win = int(win_sec * resample)
    Xs, ys, subs = [], [], []
    for sd in sorted(glob.glob(os.path.join(ds, "sub-*"))):
        sid = os.path.basename(sd)
        lab = label_fn(part.get(sid, {}), sid)
        if lab is None:
            continue
        # find the task file (prefer .set/.vhdr/.bdf/.edf), any session
        cands = []
        for ext in (".set", ".vhdr", ".bdf", ".edf"):
            cands += glob.glob(os.path.join(sd, "**", "eeg", f"*task-{task}*_eeg{ext}"), recursive=True)
        if not cands:
            continue
        try:
            raw = _read_raw(sorted(cands)[0])
            raw.pick("eeg")                                   # drop EOG/ECG/misc before canon
            raw = _canon_19(raw)                              # rename + montage + interpolate missing
        except Exception:
            continue
        if raw is None:
            continue
        if fmin or fmax:
            raw.filter(fmin, fmax, verbose="ERROR")
        if int(raw.info["sfreq"]) != resample:
            raw.resample(resample, verbose="ERROR")
        data = raw.get_data(picks=CH19).astype("float32")     # [19, T] in CH19 order
        data = data[:, : int(crop_sec * resample)]            # cap length
        ws = _windows(data, win, max_per_subject)
        for w in ws:
            if w.shape[1] == win:
                Xs.append(w); ys.append(int(lab)); subs.append(sid)
    if not Xs:
        return None
    X = np.stack(Xs).astype("float32")
    if normalize == "trial_zscore":
        mu = X.mean(2, keepdims=True); sd = X.std(2, keepdims=True) + 1e-7
        X = (X - mu) / sd
    return X, np.array(ys, "int64"), subs


# ---- per-cohort registry: (condition, BIDS task token, label_fn -> int|None) ----
def _grp(col, mapping):
    return lambda row, sid: mapping.get(str(row.get(col, "")).strip(), None)

def _prefix(pos="pd", neg="hc"):
    return lambda row, sid: (1 if sid.lower().startswith(f"sub-{pos}") else
                             0 if sid.lower().startswith(f"sub-{neg}") else None)

COHORTS = {
    # ---- Parkinson (PD=1 / HC=0); resting cohorts first, then task-based ----
    "ds002778": dict(cond="PD", task="rest",  label=_prefix("pd", "hc")),
    "ds003490": dict(cond="PD", task="Rest",  label=_grp("Group", {"PD": 1, "CTL": 0})),
    "ds004584": dict(cond="PD", task="Rest",  label=_grp("GROUP", {"PD": 1, "Control": 0})),
    "ds003509": dict(cond="PD", task="SimonConflict", label=_grp("Group", {"PD": 1, "CTL": 0})),
    "ds004574": dict(cond="PD", task="Oddball", label=_grp("GROUP", {"PD": 1, "Control": 0})),
    "ds004580": dict(cond="PD", task="Simon",  label=_grp("GROUP", {"PD": 1, "Control": 0})),
    # ---- Schizophrenia/Psychosis (P=1 / HC=0) ----
    "ds003944": dict(cond="SCZ", task="Rest",  label=_grp("type", {"Psychosis": 1, "Control": 0})),
    "ds003947": dict(cond="SCZ", task="rest",  label=_grp("type", {"Psychosis": 1, "Control": 0})),
    "ds004000": dict(cond="SCZ", task="proposer", label=_grp("group", {"P": 1, "HC": 0})),
    "ds004367": dict(cond="SCZ", task="rdk",   label=_grp("Group", {"Patient": 1, "Control": 0})),
    # ---- Depression (MDD=1 / HC=0); ds003478 uses the SCID diagnosis column ----
    "ds003478": dict(cond="DEP", task="Rest", label=_grp(
        "SCID", {"Current MDD": 1, "Past MDD": 1, "Do not meet criterion for current or past MDD": 0})),
    # ---- Alzheimer (AD=1 / CN=0; drop FTD). ds004504 == ADFTD ----
    "ds004504": dict(cond="AD", task="eyesclosed", label=_grp("Group", {"A": 1, "C": 0})),
}

ROOT = "/projects/EEG-foundation-model/datalake/raw/scps"


def load_crossdataset(condition, cohorts=None, resample=128, win_sec=4.0, fmin=0.5, fmax=45.0,
                      max_per_subject=40, **kw):
    """Pool all cohorts of `condition` into one Protocol-C dataset with hierarchical domain tags.
    Returns X[B,19,T], y, meta(DataFrame: subject, cohort, session, run), classes=['HC','Patient']."""
    import pandas as pd
    ids = cohorts or [k for k, v in COHORTS.items() if v["cond"] == condition]
    Xs, ys, subj, coh = [], [], [], []
    for dsid in ids:
        ds = os.path.join(ROOT, COHORTS[dsid]["cond"], dsid)
        if not os.path.isdir(ds):
            print(f"  [skip] {dsid} not on disk"); continue
        r = load_cohort(ds, COHORTS[dsid]["task"], COHORTS[dsid]["label"],
                        fmin=fmin, fmax=fmax, resample=resample, win_sec=win_sec,
                        max_per_subject=max_per_subject, **kw)
        if r is None:
            print(f"  [skip] {dsid} no usable subjects"); continue
        X, y, subs = r
        Xs.append(X); ys.append(y)
        subj += [f"{dsid}/{s}" for s in subs]; coh += [dsid] * len(subs)
        print(f"  [{dsid}] {X.shape[0]} trials, {len(set(subs))} subj, classes={np.bincount(y)}")
    X = np.concatenate(Xs).astype("float32"); y = np.concatenate(ys)
    meta = pd.DataFrame({"subject": subj, "cohort": coh, "session": 1, "run": 0})
    return X, y, meta, ["HC", "Patient"]
