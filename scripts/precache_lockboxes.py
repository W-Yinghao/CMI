import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.data import moabb_data as MD
out={}
for name, tmin, tmax in [("Stieger2021",0.0,3.0),("Shin2017A",0.5,3.5)]:
    X,y,meta,classes=MD.load(name, tmin=tmin, tmax=tmax)   # all subjects -> caches
    sess=meta["session"].astype(str); subj=meta["subject"].astype(str)
    per={str(s):sorted(sess[subj==s].unique().tolist()) for s in list(subj.unique())[:3]}
    out[name]=dict(X_shape=list(X.shape), n_cls=len(classes), classes=list(map(str,classes)),
                   n_subjects=int(subj.nunique()), sessions=sorted(sess.unique().tolist()),
                   sample_subject_sessions=per, n_trials=int(len(y)))
    print(f"{name}: CACHED X={X.shape} n_cls={len(classes)} classes={classes} sessions={sorted(sess.unique().tolist())} subjects={subj.nunique()}", flush=True)
Path("results").mkdir(exist_ok=True); Path("results/lockbox_precache.json").write_text(json.dumps(out,indent=2,default=str))
print("PRECACHE_DONE", flush=True)
