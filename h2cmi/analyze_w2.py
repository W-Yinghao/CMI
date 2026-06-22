"""W2 analysis: Sleep-EDF cross-subject natural-prevalence staging (review W1_W2_FROZEN).

Per method: mean paired delta vs identity, macro-F1 delta, harm rate, worst-quartile, transform
displacement, subject-bootstrap CI. KEY contrast: per-subject (delta_CC - delta_pooled) -- is canonical
fixed-prior CC RELATIVELY more robust to natural prevalence variation than pooled? Frozen MECHANISM
analysis (not routing): regress (delta_CC - delta_pooled) on JS(rho_T, rho_S). Conclusion bound: even
if CC wins it is NOT prevalence-invariant (V2P). Unit = target subject.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

METHODS = ["identity", "always_pooled", "always_canonical_CC", "current_joint",
           "euclidean_alignment", "spdim", "metadata_only"]


def _wq(d):
    d = np.sort(np.asarray(d)); k = max(1, len(d) // 4)
    return float(d[:k].mean()) if len(d) else float("nan")


def _boot_mean(vals, n=10000, seed=0):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.var(x) < 1e-12:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def _boot_slope(x, y, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        bs.append(_slope(x[idx], y[idx]))
    bs = [b for b in bs if b == b]
    return _slope(x, y), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, nargs="+")
    ap.add_argument("--out", default="results/h2cmi/w2.report.json")
    args = ap.parse_args()
    rows = []
    for p in args.inp:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    units = defaultdict(dict)
    for r in rows:
        units[r["target_subject"]][r["method"]] = r
    rep = {"panel": "W2", "n_rows": len(rows), "n_subjects": len(units), "methods": {}}
    for m in METHODS:
        d = [u[m]["delta"] for u in units.values() if m in u]
        df1 = [u[m]["delta_f1"] for u in units.values() if m in u]
        harm = [u[m]["harm"] for u in units.values() if m in u]
        tn = [u[m]["transform_norm"] for u in units.values() if m in u and u[m]["transform_norm"] == u[m]["transform_norm"]]
        mn, lo, hi = _boot_mean(d)
        rep["methods"][m] = dict(n=len(d), mean_delta=mn, delta_ci95=[lo, hi],
                                 mean_delta_f1=float(np.mean(df1)) if df1 else float("nan"),
                                 harm_rate=float(np.mean(harm)) if harm else float("nan"),
                                 worst_quartile=_wq(d), mean_transform_norm=float(np.mean(tn)) if tn else float("nan"))
    # KEY contrast: per-subject (delta_CC - delta_pooled)
    contrast, js = [], []
    dCC, dPool = [], []
    for u in units.values():
        if "always_canonical_CC" in u and "always_pooled" in u:
            c = u["always_canonical_CC"]["delta"] - u["always_pooled"]["delta"]
            contrast.append(c); js.append(u["always_canonical_CC"]["js_target_source"])
            dCC.append(u["always_canonical_CC"]["delta"]); dPool.append(u["always_pooled"]["delta"])
    cm, clo, chi = _boot_mean(contrast)
    sm, slo, shi = _boot_slope(js, contrast)
    rep["cc_vs_pooled"] = dict(mean=cm, ci95=[clo, chi], excludes_0=bool(clo > 0 or chi < 0), n=len(contrast))
    rep["mechanism_js_regression"] = dict(
        slope_contrast_vs_js=sm, slope_ci95=[slo, shi], excludes_0=bool(slo > 0 or shi < 0),
        slope_pooled_vs_js=_slope(js, dPool), slope_cc_vs_js=_slope(js, dCC), n=len(js))
    json.dump(rep, open(args.out, "w"), indent=2)

    print(f"=== W2 Sleep-EDF cross-subject natural-prevalence staging (n_subjects={rep['n_subjects']}) ===")
    print("  method                 mean_Δ vs id [ci95]        ΔmacroF1  harm   wq      |T-I|")
    for m in METHODS:
        s = rep["methods"][m]
        print(f"  {m:22s} {s['mean_delta']:+.3f} [{s['delta_ci95'][0]:+.3f},{s['delta_ci95'][1]:+.3f}]  "
              f"{s['mean_delta_f1']:+.3f}   {s['harm_rate']:.2f}  {s['worst_quartile']:+.3f}  {s['mean_transform_norm']:.3f}")
    cv = rep["cc_vs_pooled"]; mj = rep["mechanism_js_regression"]
    print(f"\n  KEY: (Δ_CC - Δ_pooled) per subject = {cv['mean']:+.4f} [{cv['ci95'][0]:+.4f},{cv['ci95'][1]:+.4f}] "
          f"(excludes 0: {cv['excludes_0']}, n={cv['n']})")
    print(f"  MECHANISM: slope (Δ_CC-Δ_pooled) vs JS(ρ_T,ρ_S) = {mj['slope_contrast_vs_js']:+.4f} "
          f"[{mj['slope_ci95'][0]:+.4f},{mj['slope_ci95'][1]:+.4f}] (excludes 0: {mj['excludes_0']})")
    print(f"            slope Δ_pooled vs JS = {mj['slope_pooled_vs_js']:+.4f}; slope Δ_CC vs JS = {mj['slope_cc_vs_js']:+.4f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
