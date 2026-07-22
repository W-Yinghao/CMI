#!/usr/bin/env python
"""E1 firming-up — partial correlation corr(tau_erm, delta_lambda | lambda_erm).

The raw corr(tau, dLambda) is significantly negative, but dLambda = lambda_cigl - lambda_erm is mechanically
anti-correlated with lambda_erm (high-lambda directions have more room to fall). This controls that confound:
report (a) the raw corr, (b) the two confound legs corr(tau, lambda_erm) and corr(lambda_erm, dLambda), and
(c) the PARTIAL corr(tau, dLambda | lambda_erm) = corr of residuals after regressing out lambda_erm from both.

If the partial corr CI includes 0, the negative raw corr is the regression artifact -> E1 = "no support for the
predicted corr>0 ordering" (clean), rather than a positive claim about the negative direction. Subject-cluster
(fold) bootstrap, seeds grouped, matching the main E1 aggregate. Pure re-analysis of committed per-fold JSONs.
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FOLDS = {"BNCI2014_001": 9, "BNCI2015_001": 12}


def _resid(x, z):
    """Residual of x after linear regression on z (with intercept)."""
    A = np.column_stack([z, np.ones_like(z)])
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ beta


def _corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _partial(tau, dlam, lam):
    return _corr(_resid(tau, lam), _resid(dlam, lam))


def _fold_boot(fold_arrays, stat, n_boot=10000, seed=0):
    folds = list(fold_arrays)
    if not folds:
        return {"point": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    point = stat(folds)
    boots = []
    for _ in range(n_boot):
        samp = [folds[i] for i in rng.integers(0, len(folds), len(folds))]
        v = stat(samp)
        if np.isfinite(v):
            boots.append(v)
    return {"point": float(point), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}


def _pool(fold_entries, key):
    return np.concatenate([np.asarray(e[key]) for e in fold_entries])


def run(spec_dir, dataset, pairing_key, n_boot):
    by_fold = defaultdict(list)
    for fp in glob.glob(str(spec_dir / f"{dataset}_seed*_fold*.json")):
        c = json.loads(Path(fp).read_text())
        pairs = c[pairing_key]
        by_fold[c["fold"]].append({
            "tau": [p["tau_erm"] for p in pairs],
            "lam": [p["lambda_erm"] for p in pairs],
            "dlam": [p["delta_lambda"] for p in pairs],
        })
    # merge seeds within a fold (pool their pairs)
    folds = []
    for fold, cs in by_fold.items():
        folds.append({k: np.concatenate([np.asarray(c[k]) for c in cs]) for k in ("tau", "lam", "dlam")})
    raw = _fold_boot(folds, lambda F: _corr(_pool(F, "tau"), _pool(F, "dlam")), n_boot)
    leg1 = _fold_boot(folds, lambda F: _corr(_pool(F, "tau"), _pool(F, "lam")), n_boot)
    leg2 = _fold_boot(folds, lambda F: _corr(_pool(F, "lam"), _pool(F, "dlam")), n_boot)
    part = _fold_boot(folds, lambda F: _partial(_pool(F, "tau"), _pool(F, "dlam"), _pool(F, "lam")), n_boot)
    return {"raw_corr_tau_dlambda": raw, "corr_tau_lambda_erm": leg1,
            "corr_lambda_erm_dlambda": leg2, "partial_corr_tau_dlambda_given_lambda_erm": part}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", default=str(REPO / "results" / "spectrum"))
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    spec_dir = Path(args.spec_dir)
    out = {"experiment": "E1_partial_correlation", "n_boot": args.n_boot, "datasets": {}}
    for ds in FOLDS:
        out["datasets"][ds] = {pk: run(spec_dir, ds, pk, args.n_boot) for pk in ("pairs", "pairs_rank")}
    (spec_dir / "partial_corr.json").write_text(json.dumps(out, indent=2))
    for ds in FOLDS:
        print(f"\n=== {ds} ===")
        for pk in ("pairs", "pairs_rank"):
            r = out["datasets"][ds][pk]
            def f(x): return f'{x["point"]:+.3f} [{x["ci95"][0]:+.3f},{x["ci95"][1]:+.3f}]'
            print(f"  [{pk}] raw={f(r['raw_corr_tau_dlambda'])}  "
                  f"partial(|lambda_erm)={f(r['partial_corr_tau_dlambda_given_lambda_erm'])}")
            print(f"         confound legs: corr(tau,lam_erm)={f(r['corr_tau_lambda_erm'])}  "
                  f"corr(lam_erm,dlam)={f(r['corr_lambda_erm_dlambda'])}")
    print(f"\n-> {spec_dir/'partial_corr.json'}")


if __name__ == "__main__":
    main()
