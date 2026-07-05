#!/usr/bin/env python
"""Merge the R1-hardened n_perm=1000 per-method chunk CSVs into the final evidence table, adding target_subject
(from each fold's .audit.npz) and ordering columns to the frozen schema. Append-only; no recompute."""
from __future__ import annotations
import csv, glob, sys
from pathlib import Path
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from cmi.eval.audit_npz import load_audit_npz

GATE = Path("results/cigl/r2_seed0_gate")
FINAL = Path("results/cigl_r123/final")
COLS = ["dataset", "method", "seed", "fold", "target_subject", "representation", "observed_kl", "perm_mean",
        "perm_std", "num_exceed", "n_perm", "exact_p", "bh_fdr_q"]
METHODS = ["erm", "cigl_graph_node", "cdan"]


def main():
    subj = {}                                                  # (dataset, fold) -> heldout target_subject
    for p in glob.glob(str(GATE / "*" / "audit" / "*_erm_seed0.audit.npz")):
        d = load_audit_npz(p)
        subj[(str(d.get("dataset", "")), int(np.asarray(d.get("fold", -1))))] = str(d.get("target_subject", ""))
    rows = []
    for m in METHODS:
        c = GATE / f"R1_chunk_{m}_nperm1000.csv"
        if not c.exists():
            raise SystemExit(f"[append] missing chunk {c} — wait for it to finish")
        for r in csv.DictReader(open(c)):
            r["target_subject"] = subj.get((r["dataset"], int(r["fold"])), "")
            rows.append(r)
    out = FINAL / "r1_hardened_nperm1000.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLS})
    # summary
    nsig = sum(1 for r in rows if r["bh_fdr_q"] and float(r["bh_fdr_q"]) <= 0.05)
    print(f"[append] wrote {out} ({len(rows)} rows); FDR-sig {nsig}/{len(rows)}; "
          f"max num_exceed={max(int(r['num_exceed']) for r in rows)}; "
          f"exact_p in {sorted(set(r['exact_p'] for r in rows))[:3]}")
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["method"], r["representation"]), []).append(r)
    for k in sorted(by):
        g = by[k]; import statistics as st
        print(f"  {k[0]:14s} {k[1]:16s} {k[2]:5s}: n={len(g)} kl={st.mean(float(r['observed_kl']) for r in g):.3f} "
              f"FDR-sig={sum(1 for r in g if float(r['bh_fdr_q'])<=0.05)}/{len(g)}")


if __name__ == "__main__":
    sys.exit(main())
