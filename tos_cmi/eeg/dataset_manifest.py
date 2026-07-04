"""Dump manifest / completeness gate for the real-EEG validation (PM P0). For each
(dataset, backbone, seed) reports expected vs completed vs missing LOSO folds and the dump metadata, so a
hardcoded-fold-cap or a silent skip can never again pass unnoticed. Reads the npz headers only (fast).
  python -m tos_cmi.eeg.dataset_manifest [DATASET ...]      # default: all validation datasets
Writes tos_cmi/results/tos_cmi_eeg_frozen/validation_manifest.json and prints a table.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys
import numpy as np

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
CHANNELS = {"BNCI2014_001": 22, "BNCI2014_004": 3, "Lee2019_MI": 62, "Cho2017": 64, "Schirrmeister2017": 128}
BACKBONES = ["TSMNet", "EEGNet"]
SEEDS = [0, 1, 2]


def _subject_lists():
    p = "artifact_build/subject_lists.json"
    if os.path.exists(p):
        return json.load(open(p))
    import moabb.datasets as D
    return {n: [int(s) for s in getattr(D, n)().subject_list]
            for n in ["Lee2019_MI", "Cho2017", "Schirrmeister2017", "BNCI2014_004"]}


def _cell(dataset, backbone, seed, subj_list):
    d = "%s/%s_%s_LOSO" % (RESULTS, dataset, backbone)
    paths = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (d, seed)))
    have = sorted({int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)) for p in paths})
    missing = [s for s in subj_list if s not in have]
    rec = {"dataset": dataset, "backbone": backbone, "seed": seed,
           "n_subjects": len(subj_list), "n_folds_expected": len(subj_list),
           "n_folds_completed": len(have), "n_folds_missing": len(missing),
           "missing_folds": missing, "channels": CHANNELS.get(dataset)}
    if paths:
        z = np.load(paths[0], allow_pickle=True)
        rec.update({"z_dim": int(z["z_dim"]), "class_count": int(z["n_cls"]),
                    "chance_task_bAcc": round(1.0 / int(z["n_cls"]), 4),
                    "source_subject_count_per_fold": int(z["n_dom_source"]),
                    "chance_subject_decode": round(1.0 / int(z["n_dom_source"]), 4)})
    else:
        rec.update({"z_dim": None, "class_count": None, "chance_task_bAcc": None,
                    "source_subject_count_per_fold": None, "chance_subject_decode": None})
    return rec


def main():
    datasets = sys.argv[1:] or ["BNCI2014_001", "BNCI2014_004", "Lee2019_MI", "Cho2017", "Schirrmeister2017"]
    subs = _subject_lists()
    subs.setdefault("BNCI2014_001", list(range(1, 10)))
    out = []
    print("%-18s %-7s %-4s  exp comp miss  z_dim ch cls chance srcSubj  status" % ("dataset", "backbone", "seed"))
    for ds in datasets:
        sl = subs.get(ds, list(range(1, 10)))
        for bb in BACKBONES:
            for s in SEEDS:
                r = _cell(ds, bb, s, sl)
                out.append(r)
                status = "COMPLETE" if r["n_folds_missing"] == 0 and r["n_folds_completed"] > 0 \
                    else ("EMPTY" if r["n_folds_completed"] == 0 else "MISSING:%d" % r["n_folds_missing"])
                print("%-18s %-7s %-4d  %3s %4s %4s  %5s %2s %3s %5s %6s   %s"
                      % (ds, bb, s, r["n_folds_expected"], r["n_folds_completed"], r["n_folds_missing"],
                         r["z_dim"], r["channels"], r["class_count"], r["chance_task_bAcc"],
                         r["source_subject_count_per_fold"], status))
    os.makedirs(RESULTS, exist_ok=True)
    mp = "%s/validation_manifest.json" % RESULTS
    prev = json.load(open(mp)) if os.path.exists(mp) else []          # MERGE: keep other datasets' cells
    prev = [c for c in prev if c["dataset"] not in set(datasets)]
    merged = prev + out
    json.dump(merged, open(mp, "w"), indent=1)
    print("\nwrote %s (%d cells this run, %d total)" % (mp, len(out), len(merged)))
    print("MANIFEST_DONE")


if __name__ == "__main__":
    main()
