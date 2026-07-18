"""Freeze the per-dataset session_split_manifest BEFORE running the readout ladder (pre-registration). Reads the
frozen EEGNet dumps' session metadata and records source / target-calibration / target-future-query sessions + trial
and class counts + fallback/exclusion reason. Sessions are chosen deterministically (session_split: earliest = cal,
rest = query) and NEVER by result. Writes configs/session_manifests/{DS}_session_split_manifest.csv.

  python scripts/freeze_session_manifests.py
"""
from __future__ import annotations
import csv, glob, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump
from tos_cmi.eval.targetx_observability import session_split
from scripts.run_readout_label_efficiency import FEAT_ROOTS, DATASETS


def main():
    outd = REPO / "configs" / "session_manifests"; outd.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        rows = []
        for p in sorted(glob.glob(f"{FEAT_ROOTS[ds]}/sub*_erm_lam0_seed*.npz")):
            f = feat_from_tos_dump(p)
            yt = np.asarray(f["y_target"]).astype(int); st = np.asarray(f["session_target"]).astype(str)
            ss = np.asarray(f["session_source"]).astype(str); ys = np.asarray(f["y_source"]).astype(int)
            cal, qry, info = session_split(f["session_target"], yt, int(f.get("seed", 0)))
            ycal = yt[cal]
            rows.append(dict(dataset=ds, subject=str(f.get("heldout_subject")), seed=int(f.get("seed", -1)),
                             n_cls=int(f.get("n_cls", len(np.unique(ys)))),
                             source_sessions="|".join(sorted(set(ss))),
                             cal_session="|".join(info["cal_sessions"]), query_sessions="|".join(map(str, info["query_sessions"])),
                             n_cal=int(cal.sum()), n_query=int(qry.sum()),
                             cal_per_class=";".join(f"{int(c)}:{int((ycal == c).sum())}" for c in np.unique(ycal)),
                             fallback_used=bool(info["fallback_used"]),
                             reason=("SINGLE_SESSION_TEMPORAL_HALF" if info["fallback_used"] else "OK")))
        fn = outd / f"{ds}_session_split_manifest.csv"
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        cal_sess = sorted(set(r["cal_session"] for r in rows)); q_sess = sorted(set(r["query_sessions"] for r in rows))
        min_calpc = min(min(int(x.split(":")[1]) for x in r["cal_per_class"].split(";")) for r in rows)
        print(f"{ds}: {len(rows)} cells -> {fn.name} | cal={cal_sess} query={q_sess} | min cal/class={min_calpc} | fallback={sum(r['fallback_used'] for r in rows)}")


if __name__ == "__main__":
    main()
