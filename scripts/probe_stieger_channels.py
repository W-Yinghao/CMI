"""Probe Stieger2021 per-subject channel sets to find the intersection (fixes the 59-vs-60 concat error). Reads only
raw headers (no epoching). Writes the common channel set for the loader."""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.data import moabb_data as MD


def _subject_channels(ds, s):
    """ch_names for subject s from ONE run, freeing the full raw immediately (memory-safe)."""
    data = ds.get_data(subjects=[s])
    chs = None
    for sess, runs in data[s].items():
        for run, raw in runs.items():
            chs = [c for c in raw.ch_names if c.upper() not in ("STIM", "STI 014", "STATUS")]
            break
        if chs:
            break
    del data; gc.collect()
    return chs


def main():
    ds = MD.construct_dataset("Stieger2021")
    subs = [int(x) for x in ds.subject_list]
    ch_sets = {}; common = None
    for s in subs:
        try:
            chs = _subject_channels(ds, s)
            ch_sets[s] = (len(chs), chs)
            common = set(chs) if common is None else (common & set(chs))
            print(f"sub{s}: {len(chs)} channels (running common={len(common)})", flush=True)
        except Exception as e:
            ch_sets[s] = ("ERROR", str(e)[:120]); print(f"sub{s}: ERROR {str(e)[:100]}", flush=True)
        gc.collect()
    sizes = sorted(set(v[0] for v in ch_sets.values() if isinstance(v[0], int)))
    common = sorted(common) if common else []
    out = dict(n_subjects=len(subs), channel_set_sizes=sizes, n_common=len(common), common_channels=common,
               per_subject_nch={str(k): v[0] for k, v in ch_sets.items()})
    Path("results").mkdir(exist_ok=True)
    Path("results/stieger_channels.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"CHANNEL_PROBE_DONE sizes={sizes} n_common={len(common)}", flush=True)


if __name__ == "__main__":
    main()
