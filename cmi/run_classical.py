"""Classical (non-deep) EEG baselines under the SAME LOSO protocol: Tangent-Space + LR,
MDM (Riemannian minimum-distance), and CSP + LDA. These are the baselines EEG reviewers
expect; they have no learned Z / leakage term (accuracy only). CPU.

  python -m cmi.run_classical --dataset BNCI2014_001 --out results/classical_2a.json
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from mne.decoding import CSP

from cmi.run_loso import load, _global_metrics
from cmi.data.moabb_data import loso_splits


def pipelines(n_csp=6):
    return {
        "TS+LR": make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                               LogisticRegression(max_iter=2000)),
        "MDM": make_pipeline(Covariances("oas"), MDM()),
        "CSP+LDA": make_pipeline(CSP(n_components=n_csp, reg="ledoit_wolf"),
                                 LinearDiscriminantAnalysis()),
    }


def run(args):
    X, y, meta, classes = load(args.dataset, subjects=None, tmin=args.tmin, tmax=args.tmax,
                               resample=args.resample)
    X = X.astype("float64")
    print(f"[{args.dataset}] X={X.shape} classes={classes}", flush=True)
    results = {k: [] for k in pipelines()}
    pooled = {k: [] for k in pipelines()}
    t0 = time.time()
    for tgt, tr, te in loso_splits(meta):
        for name, pipe in pipelines().items():
            try:
                pipe.fit(X[tr], y[tr]); pred = pipe.predict(X[te])
            except Exception as e:
                print(f"  tgt={tgt} {name} FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
            ba = balanced_accuracy_score(y[te], pred)
            results[name].append(dict(target=str(tgt), balanced_acc=float(ba),
                                      accuracy=float((pred == y[te]).mean())))
            pooled[name].append((y[te], pred, str(tgt)))
            print(f"  tgt={tgt} {name:8s} bAcc={ba*100:5.1f} ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    print(f"\n=== {args.dataset} classical (LOSO) ===")
    print(f"{'pipeline':10s} {'PerTgtBAcc':>12s} {'PoolBAcc':>9s} {'Worst':>6s}")
    for name in results:
        if not results[name]:
            continue
        ba = np.array([r["balanced_acc"] for r in results[name]])
        summary[name] = dict(per_target_balanced_acc_mean=float(ba.mean()),
                             per_target_balanced_acc_std=float(ba.std()),
                             worst_target_balanced_acc=float(ba.min()),
                             **_global_metrics(pooled[name]), per_target=results[name])
        s = summary[name]
        print(f"{name:10s} {s['per_target_balanced_acc_mean']*100:6.1f}±{s['per_target_balanced_acc_std']*100:4.1f} "
              f"{s['pooled_balanced_acc']*100:8.1f} {s['worst_target_balanced_acc']*100:6.1f}")
    if args.out:
        json.dump(dict(config=vars(args), classes=classes, summary=summary), open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--resample", type=int, default=128)
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
