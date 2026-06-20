"""Build a PD MEDICATION-STATE cache (Y = OFF/ON levodopa), so the concept-shift harness can treat med-state
as a domain/label. Unlike the disease task, each PD subject contributes BOTH ON and OFF recordings, so for the
ON-vs-OFF task `D=subject` is NON-degenerate (each subject spans both classes) — and levodopa demonstrably
changes resting EEG (β-band), so this doubles as a real signal we EXPECT to find, contrasting the disease null.

Cohorts with paired ON/OFF resting sessions:
  ds002778 : explicit ses-off / ses-on (.bdf), 15 PD subjects.
  ds003490 : ses-01 / ses-02 (.set) mapped to ON/OFF per-subject via participants sess1_Med/sess2_Med
             (order varies per subject!), 25 PD subjects (CTL excluded).
Output: {ROOT}/cache/PDMED.npz with the same schema as the disease caches
        (X[N,19,512], y[OFF=0/ON=1], subject='dsid/sub-xxx', cohort, classes=['OFF','ON']).
"""
import csv, glob, os
import numpy as np

from cmi.data.bids_data import _read_raw, _canon_19, _windows, CH19, ROOT


def load_med_recording(fp, fmin=0.5, fmax=45.0, resample=128, win=512, max_w=40):
    try:
        raw = _read_raw(fp)
        raw.pick("eeg")
        raw = _canon_19(raw)
    except Exception as e:
        print(f"    [skip {os.path.basename(fp)}: {e}]"); return []
    if raw is None:
        return []
    raw.filter(fmin, fmax, verbose="ERROR")
    if int(raw.info["sfreq"]) != resample:
        raw.resample(resample, verbose="ERROR")
    data = raw.get_data(picks=CH19).astype("float32")[:, : 180 * resample]
    return [w for w in _windows(data, win, max_w) if w.shape[1] == win]


def build(smoke=False):
    resample = 128; win = int(4 * resample)                       # 512 samples @128Hz, matches disease cache
    Xs, ys, subj, coh = [], [], [], []

    # ---- ds002778: ses-off=0, ses-on=1 ----
    subs = sorted(glob.glob(f"{ROOT}/PD/ds002778/sub-pd*"))
    if smoke: subs = subs[:2]
    for sd in subs:
        sid = os.path.basename(sd)
        for med, lab in [("off", 0), ("on", 1)]:
            cands = glob.glob(f"{sd}/ses-{med}/eeg/*task-rest*_eeg.bdf")
            if not cands:
                continue
            ws = load_med_recording(sorted(cands)[0], resample=resample, win=win)
            for w in ws:
                Xs.append(w); ys.append(lab); subj.append(f"ds002778/{sid}"); coh.append("ds002778")
        print(f"  [ds002778/{sid}] cum windows={len(Xs)}", flush=True)

    # ---- ds003490: map ses-01/02 -> med via participants (order varies per subject) ----
    part = {r["participant_id"]: r for r in
            csv.DictReader(open(f"{ROOT}/PD/ds003490/participants.tsv"), delimiter="\t")}
    subs = sorted(glob.glob(f"{ROOT}/PD/ds003490/sub-*"))
    if smoke: subs = [s for s in subs if part.get(os.path.basename(s), {}).get("Group") == "PD"][:2]
    for sd in subs:
        sid = os.path.basename(sd); row = part.get(sid, {})
        if row.get("Group") != "PD":
            continue
        for ses, col in [("01", "sess1_Med"), ("02", "sess2_Med")]:
            lab = {"ON": 1, "OFF": 0}.get(str(row.get(col, "")).strip().upper())
            if lab is None:
                continue
            cands = glob.glob(f"{sd}/ses-{ses}/eeg/*task-Rest*_eeg.set")
            if not cands:
                continue
            ws = load_med_recording(sorted(cands)[0], resample=resample, win=win)
            for w in ws:
                Xs.append(w); ys.append(lab); subj.append(f"ds003490/{sid}"); coh.append("ds003490")
        print(f"  [ds003490/{sid}] cum windows={len(Xs)}", flush=True)

    X = np.stack(Xs).astype("float32")
    mu = X.mean(2, keepdims=True); sd_ = X.std(2, keepdims=True) + 1e-7
    X = (X - mu) / sd_                                              # trial z-score (matches disease cache)
    y = np.array(ys, "int64"); subj = np.array(subj); coh = np.array(coh)
    out = f"{ROOT}/cache/PDMED{'_smoke' if smoke else ''}.npz"
    np.savez(out, X=X, y=y, subject=subj, cohort=coh, classes=np.array(["OFF", "ON"]))
    import collections
    print(f"\nsaved {out}  X={X.shape}  y(OFF,ON)={np.bincount(y).tolist()}  "
          f"cohorts={collections.Counter(coh.tolist())}  n_subj={len(set(subj.tolist()))}", flush=True)
    # per-subject class span (confirms D=subject is non-degenerate for the med task)
    bysub = collections.defaultdict(set)
    for s, yy in zip(subj, y): bysub[s].add(int(yy))
    both = sum(1 for v in bysub.values() if len(v) == 2)
    print(f"subjects spanning BOTH ON&OFF: {both}/{len(bysub)} (D=subject non-degenerate iff most span both)")


if __name__ == "__main__":
    import sys
    build(smoke="--smoke" in sys.argv)
