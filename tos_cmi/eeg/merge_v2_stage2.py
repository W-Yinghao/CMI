"""Merge the 5 Stage-2 world x backbone shards into global outputs, then run the global report + stop-condition
check (NOT a raw CSV concat). Summary cells are disjoint across shards (keyed incl world|backbone); manifest
cells (keyed ds|bb|seed, world-independent) are unioned. Launch plumbing only -- no reporter/config changes.

  python -m tos_cmi.eeg.merge_v2_stage2 --outdir tos_cmi/results/method_deepen/v2_stage2 --tag stage2
Writes v2_<tag>_{summary.json, manifest.json, rows.csv} then invokes report_v2 (v2_<tag>_{report.md,
ceiling_scatter.png, naive_controller_table.csv}) and prints the global stop-condition verdict.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys

from tos_cmi.eeg.run_v2_stage2_scoped import stop_conditions


def _merge_manifest(shard_manifests):
    out = {}
    for man in shard_manifests:
        for k, v in man.items():
            if k not in out:
                out[k] = dict(v)
            else:                                  # same (ds,bb,seed) across world-shards -> union fold coverage
                o = out[k]
                o["valid_folds"] = sorted(set(o["valid_folds"]) | set(v["valid_folds"]))
                o["degenerate_folds"] = sorted(set(o["degenerate_folds"]) | set(v["degenerate_folds"]))
                o["n_valid"] = len(o["valid_folds"]); o["n_degenerate"] = len(o["degenerate_folds"])
                o["frac_skipped"] = round((o["expected_folds"] - o["n_valid"]) / o["expected_folds"], 3) if o["expected_folds"] else 0.0
                o["status"] = ("VALID" if o["n_valid"] == o["expected_folds"] and not o["degenerate_folds"]
                               else "DEGENERATE" if not o["valid_folds"] else "PARTIAL")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="tos_cmi/results/method_deepen/v2_stage2")
    ap.add_argument("--tag", default="stage2")
    ap.add_argument("--shard-glob", default="v2_shard_*_summary.json")
    a = ap.parse_args()
    sums = sorted(glob.glob("%s/%s" % (a.outdir, a.shard_glob)))
    if not sums:
        print("NO SHARD SUMMARIES FOUND at %s/%s" % (a.outdir, a.shard_glob)); sys.exit(2)
    print("merging %d shard summaries: %s" % (len(sums), [os.path.basename(s) for s in sums]))
    merged, manifests, cfg_hashes, params, thr, nt, nf, nd = {}, [], set(), None, None, 0, 0, 0
    for sp in sums:
        S = json.load(open(sp)); merged.update(S["summary"])
        cfg_hashes.add(S["config_hash"]); params = params or S["params"]; thr = thr or S["thresholds"]
        nt += S.get("n_tasks", 0); nf += S.get("n_fail", 0); nd += S.get("n_degenerate", 0)
        mp = sp.replace("_summary.json", "_manifest.json")
        if os.path.exists(mp):
            manifests.append(json.load(open(mp))["manifest"])
    assert len(cfg_hashes) == 1, "shards used DIFFERENT config hashes: %s" % cfg_hashes
    cfg_hash = cfg_hashes.pop()
    manifest = _merge_manifest(manifests)
    json.dump({"config_hash": cfg_hash, "thresholds": thr, "params": params,
               "n_tasks": nt, "n_fail": nf, "n_degenerate": nd, "summary": merged},
              open("%s/v2_%s_summary.json" % (a.outdir, a.tag), "w"), indent=1)
    json.dump({"config_hash": cfg_hash, "manifest": manifest},
              open("%s/v2_%s_manifest.json" % (a.outdir, a.tag), "w"), indent=1)
    # concat shard rows CSVs (header once)
    rows_files = sorted(glob.glob("%s/v2_shard_*_rows.csv" % a.outdir))
    with open("%s/v2_%s_rows.csv" % (a.outdir, a.tag), "w") as out:
        for i, rf in enumerate(rows_files):
            with open(rf) as fh:
                lines = fh.readlines()
            out.writelines(lines if i == 0 else lines[1:])
    # global report (reuses the tested report_v2; reads the merged summary)
    subprocess.run([sys.executable, "-m", "tos_cmi.eeg.report_v2", "--tag", a.tag, "--outdir", a.outdir], check=True)
    # global stop-condition verdict
    bt = thr["benefit_lcb"]
    halt, findings = stop_conditions(merged, manifest, bt)
    print("\n=== GLOBAL stop conditions (merged) ===")
    print("  config hash: %s ; tasks %d ; fail %d ; degenerate %d" % (cfg_hash, nt, nf, nd))
    for name, val in findings:
        print("  %-45s : %s" % (name, val))
    print("  STAGE2_MERGE_HALT" if halt else "  STAGE2_MERGE_CLEAN")
    print("V2_STAGE2_MERGE_DONE")


if __name__ == "__main__":
    main()
