#!/usr/bin/env python
"""S2P Phase 9A — TUEG corpus inventory (see docs/S2P_01). Reads the processed TUEG metadata (4704743c config:
13446 subjects, 57700 recordings, 23187 h, 200 Hz, TUH 10-20 -LE montage), builds the per-subject manifest +
hours distribution, and the fixed-hours subset feasibility for the subject-scaling grid. Subject IDs are the
processed `subject` index (stable; mapped to original TUH patient IDs via infos.json original_subjects). NO
pretraining here; pure inventory. Run with the icml python (has pyarrow). Writes results/s2p_inventory/*.csv."""
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd

TUEG = "/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"
OUT = Path("results/s2p_inventory")
SFREQ = 200
# 19 common 10-20 channels (CBraMod/CodeBrain pretraining montage) in TUH -LE naming
COMMON19 = ["EEG FP1-LE","EEG FP2-LE","EEG F3-LE","EEG F4-LE","EEG C3-LE","EEG C4-LE","EEG P3-LE","EEG P4-LE",
            "EEG O1-LE","EEG O2-LE","EEG F7-LE","EEG F8-LE","EEG T3-LE","EEG T4-LE","EEG T5-LE","EEG T6-LE",
            "EEG FZ-LE","EEG CZ-LE","EEG PZ-LE"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = json.load(open(f"{TUEG}/infos.json"))
    m = pd.read_parquet(f"{TUEG}/metadata.parquet")
    m["hours"] = m["n_timepoints"] / SFREQ / 3600.0
    # channel coverage: fraction of recordings containing all 19 common channels
    def has19(chstr):
        try:
            chs = set(json.loads(chstr))
        except Exception:
            chs = set()
        return all(c in chs for c in COMMON19)
    m["has_common19"] = m["channels"].map(has19)

    subj = m.groupby("subject").agg(n_recordings=("recording_id", "count"),
                                    n_sessions=("session", "nunique"),
                                    total_hours=("hours", "sum"),
                                    frac_common19=("has_common19", "mean")).reset_index()
    subj = subj.sort_values("subject")
    # subject manifest + hours-by-subject
    subj.to_csv(OUT / "tueg_subject_manifest.csv", index=False)
    subj[["subject", "total_hours"]].to_csv(OUT / "tueg_hours_by_subject.csv", index=False)

    hh = subj["total_hours"].values
    cov = float(m["has_common19"].mean())
    # per-subject-hours thresholds -> how many subjects have >= X hours (fixed-hours feasibility)
    thr_counts = {f"ge_{t}h": int((hh >= t).sum()) for t in [0.25, 0.5, 1, 2, 5, 10]}
    # fixed-hours subset feasibility: for hours_budget H0 and N subjects, per-subject cap = H0/N;
    # feasible if at least N subjects have >= H0/N hours.
    def feasible(H0, N):
        cap = H0 / N
        return int((hh >= cap).sum()) >= N, round(cap, 3), int((hh >= cap).sum())
    feas_rows = []
    for H0 in [100, 250, 500, 1000]:
        for N in [32, 128, 512, 2048]:
            ok, cap, avail = feasible(H0, N)
            feas_rows.append(dict(hours_budget=H0, n_subjects=N, per_subject_cap_h=cap,
                                  subjects_with_cap=avail, fixed_hours_feasible=ok))
    with open(OUT / "fixed_hours_subset_feasibility.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(feas_rows[0].keys())); w.writeheader(); w.writerows(feas_rows)

    inv = dict(config="4704743c", dataset="TUEG", sfreq=SFREQ, n_channels_stored=33,
               n_subjects=int(subj.shape[0]), n_recordings=int(m.shape[0]),
               total_hours=round(float(m["hours"].sum()), 1),
               per_subject_hours=dict(median=round(float(np.median(hh)), 3), mean=round(float(hh.mean()), 3),
                                      max=round(float(hh.max()), 1), min=round(float(hh.min()), 4)),
               subjects_by_hours_threshold=thr_counts,
               common19_channel_coverage_frac=round(cov, 4),
               subject_ids_reliable=True, subject_id_source="processed `subject` index + infos.json original_subjects (TUH patient ids)",
               sample_by_subject_feasible=True, preprocessing=info.get("preprocessing"))
    (OUT / "tueg_inventory_summary.json").write_text(json.dumps(inv, indent=2) + "\n")
    print(json.dumps(inv, indent=2))
    print("\nfixed-hours feasibility (H0 x N):")
    for r in feas_rows:
        print(f"  H0={r['hours_budget']}h N={r['n_subjects']}: cap={r['per_subject_cap_h']}h avail={r['subjects_with_cap']} feasible={r['fixed_hours_feasible']}")


if __name__ == "__main__":
    main()
