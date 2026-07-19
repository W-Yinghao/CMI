import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.data import moabb_data as MD
name, s = sys.argv[1], int(sys.argv[2])
tmin, tmax = (0.0, 3.0) if name == "Stieger2021" else (0.5, 3.5)
marker = REPO / "results" / "precache_done" / f"{name}_{s}.done"
if marker.exists():
    print(f"SKIP {name} sub{s}"); sys.exit(0)
out = MD.precache_subject(name, s, tmin=tmin, tmax=tmax)
marker.write_text(str(out) + "\n")
print(f"CACHED {name} sub{s} -> {out}", flush=True)
