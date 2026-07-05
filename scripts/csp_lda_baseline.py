#!/usr/bin/env python3
"""P9 CSP+LDA classical baseline for the CSP-init FBCSP-LGG sidecar. Source-only LOSO, target labels
eval-only. Two protocols: csp_all_source (all source subjects) and csp_sourceval_matched (exclude the same
seeded source-val subject the neural early-stop holds out). Reports mean/worst bAcc + BNCI2014 CSP-decodable
{1,3,8,9} vs hard subsets. CPU-only. CSP uses the same one-vs-rest/binary filters as P8-A.

P9-G: repo-root path resolved dynamically (no hardcoded machine path) so the baseline is reproducible from any
checkout; adds --dataset/--protocol/--max_subjects for a quick smoke."""
import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # repo root (scripts/ -> ..)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from cmi.data.moabb_data import load, domain_labels, loso_splits
from cmi.models.csp_init import source_csp_filters

DEC = {1, 3, 8, 9}
PROTOCOLS = ("csp_all_source", "csp_sourceval_matched")


def csp_feat(X, W):
    """log-variance of CSP-projected trials. X [N,C,T], W [F,C] -> [N,F]."""
    Xc = X - X.mean(axis=2, keepdims=True)
    proj = np.einsum("fc,nct->nft", W, Xc)                 # [N,F,T]
    return np.log(np.clip(proj.var(axis=2), 1e-8, None))   # [N,F]


def run(dataset, protocols, m=4, shrinkage=0.1, seeds=(0, 1, 2), max_subjects=0):
    X, y, meta, classes = load(dataset, tmin=0.5, tmax=3.5, resample=128)
    if max_subjects:                                       # smoke: truncate the SUBJECT pool
        keep = sorted(meta["subject"].unique())[:max_subjects]
        mk = meta["subject"].isin(keep).to_numpy()
        X, y, meta = X[mk], y[mk], meta[mk].reset_index(drop=True)
    dom, _ = domain_labels(meta, "subject")
    n_cls = len(classes)
    splits = list(loso_splits(meta))                       # (tgt, train_mask, test_mask)
    rows = {"csp_all_source": {}, "csp_sourceval_matched": {s: {} for s in seeds}}
    for tgt, trm, tem in splits:
        Xtr, ytr, dtr = X[trm], y[trm], dom[trm]
        Xte, yte = X[tem], y[tem]                          # target labels used for EVAL only
        if "csp_all_source" in protocols:
            W, _, _ = source_csp_filters(Xtr, ytr, n_cls, m, shrinkage=shrinkage)
            lda = LDA().fit(csp_feat(Xtr, W), ytr)
            rows["csp_all_source"][int(tgt)] = float(balanced_accuracy_score(yte, lda.predict(csp_feat(Xte, W))))
        if "csp_sourceval_matched" in protocols:
            for s in seeds:
                sval = int(np.random.default_rng(s).permutation(np.unique(dtr))[0])
                mk = dtr != sval                           # exclude the same seeded source-val subject
                Ws, _, _ = source_csp_filters(Xtr[mk], ytr[mk], n_cls, m, shrinkage=shrinkage)
                lda2 = LDA().fit(csp_feat(Xtr[mk], Ws), ytr[mk])
                rows["csp_sourceval_matched"][s][int(tgt)] = float(
                    balanced_accuracy_score(yte, lda2.predict(csp_feat(Xte, Ws))))
    return rows, [int(t) for t, _, _ in splits]


def summarize(dataset, rows, protocols):
    is2a = dataset == "BNCI2014_001"
    def stats(perfold):
        v = list(perfold.values())
        d = {"mean": float(np.mean(v)), "worst": float(np.min(v))}
        if is2a:
            d["decodable"] = float(np.mean([perfold[t] for t in perfold if t in DEC]))
            d["hard"] = float(np.mean([perfold[t] for t in perfold if t not in DEC]))
        return d
    out = []
    if "csp_all_source" in protocols and rows["csp_all_source"]:
        out.append(("csp_all_source", "det", stats(rows["csp_all_source"])))
    if "csp_sourceval_matched" in protocols:
        for s in rows["csp_sourceval_matched"]:
            if rows["csp_sourceval_matched"][s]:
                out.append(("csp_sourceval_matched", f"seed{s}", stats(rows["csp_sourceval_matched"][s])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["BNCI2014_001", "BNCI2015_001"])
    ap.add_argument("--dataset", default=None, help="single dataset (overrides --datasets; for smoke)")
    ap.add_argument("--protocol", choices=["csp_all_source", "csp_sourceval_matched", "both"], default="both")
    ap.add_argument("--max_subjects", type=int, default=0, help="truncate SOURCE+target subject pool (smoke)")
    ap.add_argument("--out", default="results/baseline_sidecar/CSP_LDA_BASELINE.json")
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    protos = PROTOCOLS if args.protocol == "both" else (args.protocol,)
    result = {}
    print(f"{'dataset':14s} {'protocol':22s} {'ver':7s} {'mean':>6s} {'worst':>6s} {'2a_dec':>7s} {'2a_hard':>7s}")
    for ds in datasets:
        rows, tgts = run(ds, protos, max_subjects=args.max_subjects)
        result[ds] = {"per_fold": rows, "targets": tgts, "summary": summarize(ds, rows, protos)}
        for proto, ver, st in result[ds]["summary"]:
            print(f"{ds:14s} {proto:22s} {ver:7s} {st['mean']:.4f} {st['worst']:.4f} "
                  f"{st.get('decodable', float('nan')):.4f} {st.get('hard', float('nan')):.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
