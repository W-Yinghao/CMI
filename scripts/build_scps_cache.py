"""Load a cross-dataset SCPS condition once and cache to npz (BIDS reading is slow).
  python scripts/build_scps_cache.py PD ds002778,ds003490,ds004584"""
import sys, os, numpy as np
from cmi.data.bids_data import load_crossdataset

cond = sys.argv[1]
cohorts = sys.argv[2].split(",") if len(sys.argv) > 2 else None
X, y, meta, classes = load_crossdataset(cond, cohorts=cohorts, resample=128, win_sec=4.0,
                                        fmin=0.5, fmax=45.0, max_per_subject=40)
out = f"/projects/EEG-foundation-model/datalake/raw/scps/cache/{cond}.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, X=X.astype("float32"), y=y.astype("int64"),
                    subject=meta["subject"].values.astype(str),
                    cohort=meta["cohort"].values.astype(str), classes=np.array(classes))
print(f"saved {out}  X={X.shape}  cohorts={sorted(set(meta['cohort']))}  y={np.bincount(y)}")
