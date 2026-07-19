import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.data import moabb_data as MD
name = sys.argv[1]
tmin, tmax = (0.0, 3.0) if name == "Stieger2021" else (0.5, 3.5)
t0 = time.time()
X, y, meta, classes = MD.load(name, tmin=tmin, tmax=tmax)
print(f"CONSOLIDATED {name}: X{X.shape} y{y.shape} classes={classes} "
      f"nsubj={meta['subject'].nunique()} nsess={meta['session'].nunique()} ({time.time()-t0:.0f}s)", flush=True)
