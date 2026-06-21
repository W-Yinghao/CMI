"""Strict, exact-key merge of per-task grid shards (review §"parallel writes" + P0-3).

Each SLURM array task writes its OWN shard JSONL (+ manifest); concurrent appends to one
file are not a safe multi-writer protocol. This tool merges a shard directory, requiring:

  * every shard has a manifest (a manifest-less shard is rejected, not silently accepted);
  * all shards share one experiment_signature (and global grid);
  * the merged key set EXACTLY equals the global expected key set from the manifest
    (a missing legitimate key + an extra wrong key with the same count is rejected);
  * no duplicate keys across shards.

It then writes the merged JSONL AND a merged manifest atomically (temp + os.replace).

  python -m h2cmi.merge_grid_shards --in-dir results/h2cmi/qxu/shards \
      --out results/h2cmi/qxu_dev.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from h2cmi.grid_io import manifest_path, global_expected_keys


def merge_shards(in_dir: str, out: str, *, item_field: str | None = None) -> dict:
    shards = [s for s in sorted(glob.glob(os.path.join(in_dir, "*.jsonl")))
              if not s.endswith(".tmp") and os.path.abspath(s) != os.path.abspath(out)]
    if not shards:
        raise FileNotFoundError(f"no *.jsonl shards in {in_dir}")

    manifests = []
    for s in shards:
        mp = manifest_path(s)
        if not os.path.exists(mp):
            raise RuntimeError(f"shard {s} has no manifest {mp}; refusing to merge")
        with open(mp) as f:
            manifests.append(json.load(f))
    exp_sigs = {m.get("experiment_signature") for m in manifests}
    if len(exp_sigs) != 1:
        raise RuntimeError(f"shards span multiple experiment_signatures: {exp_sigs}")
    g = manifests[0]
    item_field = item_field or g["item_field"]

    keys: set = set()
    lines: list[str] = []
    for s in shards:
        with open(s) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["data_seed"], row["target_site"], row["scenario"],
                       row[item_field], row["cmi"])
                if key in keys:
                    raise ValueError(f"duplicate key across shards at {s}:{line_no}: {key}")
                keys.add(key)
                lines.append(line)

    expected = global_expected_keys(g)
    if keys != expected:
        missing = sorted(expected - keys)[:5]
        extra = sorted(keys - expected)[:5]
        raise ValueError(f"merge is not the exact global key set: "
                         f"missing={len(expected - keys)} (e.g. {missing}), "
                         f"extra={len(keys - expected)} (e.g. {extra})")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    # merged manifest: same experiment, shard_spec spanning the full global grid
    merged_manifest = dict(g, shard_spec={"seeds": g["global_seeds"], "sites": g["global_sites"]},
                           merged_from=[os.path.basename(s) for s in shards])
    mtmp = manifest_path(out) + ".tmp"
    with open(mtmp, "w") as f:
        json.dump(merged_manifest, f, indent=2, default=str)
    os.replace(mtmp, manifest_path(out))
    return dict(shards=len(shards), rows=len(lines), unique_keys=len(keys),
                experiment_signature=g["experiment_signature"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--item-field", default=None, help="default: read from shard manifests")
    args = ap.parse_args()
    info = merge_shards(args.in_dir, args.out, item_field=args.item_field)
    print(f"[merge] {info['shards']} shards -> {args.out}: {info['rows']} rows, "
          f"{info['unique_keys']} keys, exp_sig={info['experiment_signature']} "
          f"(+ {manifest_path(args.out)})")


if __name__ == "__main__":
    main()
