#!/usr/bin/env python
"""Merge per-seed shard CSVs (process-parallel 7C run) into the canonical results dir. Concatenates every CSV
basename found across shard dirs, preserving a single header; merges the firewall JSON row lists."""
import glob, json, sys
from pathlib import Path

OUT = Path("results/fsr_head_only_label_conflict")


def main():
    prefix = sys.argv[1]  # e.g. "shard_gate_" or "shard_full_"
    shards = sorted(glob.glob(str(OUT / f"{prefix}*")))
    if not shards:
        print(f"no shards match {prefix}*"); sys.exit(1)
    basenames = set()
    for sh in shards:
        for f in glob.glob(str(Path(sh) / "*.csv")):
            basenames.add(Path(f).name)
    for bn in sorted(basenames):
        header, body = None, []
        for sh in shards:
            p = Path(sh) / bn
            if not p.exists():
                continue
            lines = p.read_text().splitlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
            body.extend(lines[1:])
        if header is not None:
            (OUT / bn).write_text("\n".join([header] + body) + "\n")
            print(f"merged {bn}: {len(body)} rows from {len(shards)} shards")
    # merge firewall JSONs if present (full stage)
    fws = [Path(sh) / "target_label_firewall.json" for sh in shards]
    fws = [f for f in fws if f.exists()]
    if fws:
        rows, n = [], 0
        for f in fws:
            j = json.load(open(f)); rows.extend(j.get("rows", [])); n += j.get("n", 0)
        (OUT / "target_label_firewall.json").write_text(json.dumps(
            dict(n=n, rows=rows, target_labels_used_for_fit=False, target_labels_used_for_selection=False,
                 target_labels_used_for_final_eval_only=True,
                 note="all target reads via p4e.TargetScorer; l4 label-free; reliance dose on SOURCE held-out"), indent=2) + "\n")
        print(f"merged firewall: {n} fold-seed rows")


if __name__ == "__main__":
    main()
