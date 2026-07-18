"""Combine the RW0 per-cell weight-audit outputs (wcells/*.json) into the 3 CSVs + a source-only audit summary
(all-zero fraction, effective support, single-subject domination, seed stability across the 3 seed-warm-ups).
Refuses until all cells are .done. Manuscript FROZEN.

  python scripts/aggregate_rw_weight_audit.py --from-dir results/cmi_trace_risk_weighted_mcc --expect 63
"""
from __future__ import annotations
import argparse, csv, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_risk_weighted_mcc"); ap.add_argument("--expect", type=int, default=63)
    a = ap.parse_args()
    wd = Path(a.from_dir) / "wcells"; done = sorted(wd.glob("cell_*.done"))
    if len(done) < a.expect:
        print(f"[rw-audit-agg] INCOMPLETE {len(done)}/{a.expect} cells -> REFUSING."); raise SystemExit(2)
    cells = [json.loads(open(c).read()) for c in sorted(wd.glob("cell_*.json"))]
    summaries = [c["summary"] for c in cells]; rows = [r for c in cells for r in c["rows"]]
    comp = [dict(dataset=s["dataset"], subject=s["subject"], seed=s["seed"], status="ok", weight_status=s["status"]) for s in summaries]
    outd = Path(a.from_dir)

    def _w(fp, data, keys):
        with open(fp, "w", newline="") as fh:
            wt = csv.DictWriter(fh, fieldnames=keys); wt.writeheader(); [wt.writerow({k: d.get(k) for k in keys}) for d in data]
    _w(outd / "risk_weight_fold_summary.csv", summaries, list(summaries[0].keys()))
    _w(outd / "risk_weight_rows.csv", rows, list(rows[0].keys()))
    _w(outd / "risk_weight_completeness.csv", comp, ["dataset", "subject", "seed", "status", "weight_status"])

    # ---- source-only audit ----
    print(f"[rw-audit-agg] {len(summaries)} bundles")
    for ds in ["BNCI2014_001", "BNCI2015_001"]:
        S = [s for s in summaries if s["dataset"] == ds]
        if not S:
            continue
        nz = sum(1 for s in S if s["status"] == "NO_POSITIVE_SOURCE_TRANSFER_GAP")
        eff = np.array([s["effective_weight_support"] for s in S]); top = np.array([s["top_subject_share"] for s in S])
        # seed stability: per (dataset, subject) top-subject-share std across the 3 seeds
        bysubj = defaultdict(list)
        for s in S:
            bysubj[s["subject"]].append(s["top_subject_share"])
        seed_std = float(np.mean([np.std(v) for v in bysubj.values() if len(v) > 1]))
        print(f"  {ds}: NO_POSITIVE={nz}/{len(S)} | eff_support med={np.median(eff):.1f} | top_subj_share med={np.median(top):.2f} "
              f"max={top.max():.2f} (>0.5 = single-subject domination) | seed_std(top_share)={seed_std:.3f} | true!=perm all={all(s['true_vs_perm_loss_diff']>1e-4 or s['status']!='ok' for s in S)}")
    json.dump(dict(n_bundles=len(summaries), no_positive=sum(1 for s in summaries if s["status"] == "NO_POSITIVE_SOURCE_TRANSFER_GAP"),
                   discipline="source-only weight characterization; no target; manuscript FROZEN"),
              open(outd / "risk_weight_audit_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
