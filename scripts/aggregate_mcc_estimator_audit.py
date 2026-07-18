"""Aggregate the MCC estimator audit (63 cells) into the E1/E2/E3/Mixed routing (CPU, env c84c). Reads the
per-cell diagnostics written by run_mcc_estimator_audit.py and decides whether the K=4 episodic MCC gradient is a
poor estimate of the EXACT full-source MCC gradient. Frozen diagnostic: tests estimator quality ONLY, NOT
geometry->DG. REFUSES until all cells are .done. Manuscript FROZEN; only the project owner stops a scientific line.

  python scripts/aggregate_mcc_estimator_audit.py --from-dir results/cmi_trace_mcc_estimator_audit --expect 63
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _route(per_ds):
    """E1 requires ALL: both-dataset median A_4<0.5; A16-A4>0.25; dw_full >= 2*dw_k4; not 1-2-subject-driven."""
    med_A4 = {ds: per_ds[ds]["median_A_4"] for ds in per_ds}
    a4_low = all(v < 0.5 for v in med_A4.values())
    a16_gain = all(per_ds[ds]["median_A_16"] - per_ds[ds]["median_A_4"] > 0.25 for ds in per_ds)
    dw_ratio_ok = all(per_ds[ds]["median_dw_full"] >= 2.0 * max(per_ds[ds]["median_dw_k4"], 1e-9) for ds in per_ds)
    robust = all(per_ds[ds]["A_4_frac_below_0.5"] >= 0.5 for ds in per_ds)   # not driven by 1-2 subjects
    if a4_low and a16_gain and dw_ratio_ok and robust:
        return dict(verdict="EPISODIC_MCC_ESTIMATOR_VARIANCE_LIMITED",
                    next="approve ONE source-only EMA / memory-bank prototype MCC round, then a full A/B/C GPU matrix to test DG")
    a4_high = all(per_ds[ds]["median_A_4"] >= 0.8 for ds in per_ds)
    a16_no_gain = all(per_ds[ds]["median_A_16"] - per_ds[ds]["median_A_4"] <= 0.1 for ds in per_ds)
    if a4_high and a16_no_gain:
        return dict(verdict="E2_K4_ESTIMATES_FULL_WELL", next="risk_weighted_MCC (global equal-weight target is DG-irrelevant, NOT estimator-limited)")
    weak = all(per_ds[ds]["median_true_vs_shuffle_cos"] > 0.9 or per_ds[ds]["median_full_grad_norm"] < 1e-6 for ds in per_ds)
    if weak:
        return dict(verdict="E3_FULL_GRADIENT_WEAK_OR_NEAR_SHUFFLE", next="risk_weighted_MCC (even a zero-variance estimator lacks independent signal)")
    # datasets disagree or partial E1 -> not confirmed variance-limited
    return dict(verdict="MIXED_OR_NOT_VARIANCE_LIMITED",
                next="add episode DRAWS to cut MC uncertainty (NOT new seeds/training); default to risk_weighted_MCC (geometry-DG decoupling is the stronger standing evidence), do NOT commit an EMA round")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_mcc_estimator_audit"); ap.add_argument("--expect", type=int, default=63)
    a = ap.parse_args()
    d = Path(a.from_dir); done = sorted(d.glob("cell_*.done"))
    if len(done) < a.expect:
        print(f"[mcc-audit-agg] INCOMPLETE {len(done)}/{a.expect} cells -> REFUSING to route."); raise SystemExit(2)
    rows = [json.loads(open(c).read()) for c in sorted(d.glob("cell_*.json"))]
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    per_ds = {}
    for ds in DATASETS:
        R = by.get(ds, [])
        if not R:
            continue
        A4 = np.array([r["A_4"] for r in R]); A16 = np.array([r["A_16"] for r in R])
        per_ds[ds] = dict(n_cells=len(R), median_A_4=float(np.median(A4)), median_A_16=float(np.median(A16)),
                          median_SNR_4=float(np.median([r["SNR_4"] for r in R])), median_SNR_16=float(np.median([r["SNR_16"] for r in R])),
                          median_B_4=float(np.median([r["B_4"] for r in R])),
                          median_dw_full=float(np.median([r["dw_full"] for r in R])), median_dw_k4=float(np.median([r["dw_k4"] for r in R])),
                          median_true_vs_shuffle_cos=float(np.median([r["true_vs_shuffle_cos"] for r in R])),
                          median_full_grad_norm=float(np.median([r["full_grad_norm"] for r in R])),
                          A_4_frac_below_0.5=float(np.mean(A4 < 0.5)))
    route = _route(per_ds)
    json.dump(dict(per_dataset=per_ds, routing=route, n_cells=len(rows),
                   discipline="frozen diagnostic: estimator quality ONLY, not geometry->DG; no training committed; manuscript FROZEN"),
              open(d / "mcc_estimator_audit_verdict.json", "w"), indent=2, default=float)
    print(f"[mcc-audit-agg] {len(rows)} cells; routing={route['verdict']}")
    for ds, v in per_ds.items():
        print(f"  {ds}: median A_4={v['median_A_4']:+.3f} A_16={v['median_A_16']:+.3f} (gain {v['median_A_16']-v['median_A_4']:+.3f}) "
              f"SNR_4={v['median_SNR_4']:.2f} SNR_16={v['median_SNR_16']:.2f} dw_full/k4={v['median_dw_full']:+.4f}/{v['median_dw_k4']:+.4f} "
              f"frac(A4<0.5)={v['A_4_frac_below_0.5']:.2f} trueVSshuf_cos={v['median_true_vs_shuffle_cos']:+.3f}")
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
