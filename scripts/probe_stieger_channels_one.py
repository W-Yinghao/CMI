import gc, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.data import moabb_data as MD
s = int(sys.argv[1])
ds = MD.construct_dataset("Stieger2021")
data = ds.get_data(subjects=[s]); chs = None
for sess, runs in data[s].items():
    for run, raw in runs.items():
        chs = [c for c in raw.ch_names if c.upper() not in ("STIM", "STI 014", "STATUS")]; break
    if chs: break
del data; gc.collect()
Path("results/stieger_ch").mkdir(parents=True, exist_ok=True)
Path(f"results/stieger_ch/sub{s}.json").write_text(json.dumps({"subject": s, "n": len(chs), "channels": chs}))
print(f"sub{s}: {len(chs)} channels")
