"""Strict, exact-key, provenance-bound merge of per-task grid shards (review §"parallel
writes" + audit §3).

Each SLURM array task writes its OWN shard JSONL (+ manifest); concurrent appends to one file
are not a safe multi-writer protocol. This tool merges a shard directory, requiring:

  * every shard has a manifest (a manifest-less shard is rejected, not silently accepted);
  * all shards share one experiment_signature (and global grid);
  * EVERY result row is provenance-checked against its shard manifest (schema_version,
    experiment_signature, config_signature, runner commit) -- a row from a different
    experiment that happens to share a (seed,site,scenario,item,cmi) key is rejected;
  * each shard's row keys EXACTLY equal that shard's shard_spec (no missing / no foreign key);
  * the merged key set EXACTLY equals the global expected key set from the manifest;
  * no duplicate keys across shards.

It writes the merged JSONL AND a merged manifest atomically (temp + os.replace). The merged
manifest records the SHA-256 of every input shard JSONL + manifest and of the merged output,
spans the full global grid in its shard_spec, and clears the per-shard CLI fields so it no
longer advertises one shard's `--shard-target-sites`.

  python -m h2cmi.merge_grid_shards --in-dir results/h2cmi/qxu/shards \
      --out results/h2cmi/qxu_dev.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from h2cmi.grid_io import (manifest_path, global_expected_keys, shard_expected_keys,
                           validate_result_row, sha256_file)


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
    input_hashes: list[dict] = []
    for s, m in zip(shards, manifests):
        shard_keys: set = set()
        with open(s) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = validate_result_row(row, m, item_field=item_field, line_ref=f"{s}:{line_no}")
                if key not in shard_expected_keys(m):
                    raise ValueError(f"{s}:{line_no}: row {key} is outside its shard_spec {m['shard_spec']}")
                if key in keys:
                    raise ValueError(f"duplicate key across shards at {s}:{line_no}: {key}")
                keys.add(key); shard_keys.add(key); lines.append(line)
        expected_shard = shard_expected_keys(m)
        if shard_keys != expected_shard:
            missing = sorted(expected_shard - shard_keys)[:5]
            extra = sorted(shard_keys - expected_shard)[:5]
            raise ValueError(f"shard {s} keys != its shard_spec: missing={len(expected_shard - shard_keys)} "
                             f"(e.g. {missing}), extra={len(shard_keys - expected_shard)} (e.g. {extra})")
        input_hashes.append(dict(shard=os.path.basename(s), jsonl_sha256=sha256_file(s),
                                 manifest_sha256=sha256_file(manifest_path(s)),
                                 shard_spec=m["shard_spec"], rows=len(shard_keys)))

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
    merged_sha = sha256_file(out)
    # merged manifest: full global grid; clear the per-shard CLI fields so it no longer
    # advertises a single shard's --shard-* selection; record input + output hashes.
    merged_cli = dict(g.get("cli", {}))
    for k in ("shard_seeds", "shard_target_sites"):
        if k in merged_cli:
            merged_cli[k] = ""
    merged_manifest = dict(g, cli=merged_cli,
                           shard_spec={"seeds": g["global_seeds"], "sites": g["global_sites"]},
                           merged_from=[os.path.basename(s) for s in shards],
                           input_hashes=input_hashes, merged_jsonl_sha256=merged_sha)
    mtmp = manifest_path(out) + ".tmp"
    with open(mtmp, "w") as f:
        json.dump(merged_manifest, f, indent=2, default=str)
    os.replace(mtmp, manifest_path(out))
    return dict(shards=len(shards), rows=len(lines), unique_keys=len(keys),
                experiment_signature=g["experiment_signature"], merged_jsonl_sha256=merged_sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--item-field", default=None, help="default: read from shard manifests")
    args = ap.parse_args()
    info = merge_shards(args.in_dir, args.out, item_field=args.item_field)
    print(f"[merge] {info['shards']} shards -> {args.out}: {info['rows']} rows, "
          f"{info['unique_keys']} keys, exp_sig={info['experiment_signature']}, "
          f"sha256={info['merged_jsonl_sha256'][:12]} (+ {manifest_path(args.out)})")


if __name__ == "__main__":
    main()
