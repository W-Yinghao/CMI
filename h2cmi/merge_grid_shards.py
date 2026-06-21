"""Strict merge of per-task grid shards into one JSONL (review §"parallel writes").

For SLURM array parallelism, each task writes its OWN shard (e.g. one --out per
(seed,site)); concurrent appends to a single file are NOT a safe multi-writer protocol.
This tool merges a directory of shards, verifying:
  * all shard manifests share one run_signature;
  * no duplicate result keys across shards;
  * (optional) an expected total row count;
then writes atomically via a temp file + os.replace.

  python -m h2cmi.merge_grid_shards --in-dir results/h2cmi/qxu/shards \
      --out results/h2cmi/qxu_dev.jsonl --item-field action [--expect-count N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from h2cmi.grid_io import manifest_path


def merge_shards(in_dir: str, out: str, *, item_field: str, expect_count: int | None = None) -> dict:
    shards = sorted(glob.glob(os.path.join(in_dir, "*.jsonl")))
    if not shards:
        raise FileNotFoundError(f"no *.jsonl shards in {in_dir}")
    sigs = set()
    for s in shards:
        mp = manifest_path(s)
        if os.path.exists(mp):
            with open(mp) as f:
                sigs.add(json.load(f).get("run_signature"))
    if len(sigs) > 1:
        raise RuntimeError(f"shards have differing run_signatures: {sigs}")
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
    if expect_count is not None and len(lines) != expect_count:
        raise ValueError(f"expected {expect_count} rows, merged {len(lines)}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    return dict(shards=len(shards), rows=len(lines), unique_keys=len(keys),
                run_signature=next(iter(sigs)) if sigs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--item-field", default="action")
    ap.add_argument("--expect-count", type=int, default=None)
    args = ap.parse_args()
    info = merge_shards(args.in_dir, args.out, item_field=args.item_field,
                        expect_count=args.expect_count)
    print(f"[merge] {info['shards']} shards -> {args.out}: {info['rows']} rows, "
          f"{info['unique_keys']} unique keys, run_sig={info['run_signature']}")


if __name__ == "__main__":
    main()
