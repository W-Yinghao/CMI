"""Track G -- concept-erasure baselines vs TOS deletion on frozen EEG latents (BCI-IV-2a).
Answers the reviewer question "is TOS-CMI just linear concept erasure (LEACE/RLACE/INLP)?". Self-contained
implementations (the concept-erasure / rlace packages are not installed; LEACE correctness is validated
empirically -- after LEACE the LINEAR subject decode collapses to chance, its defining property).

Per ERM npz (trial split A=fit / B=probe), erase subject identity D with each method, then probe subject
and task decode (linear + MLP) on the held-out B split:
  full Z | LEACE (closed-form, linear) | INLP (iterative nullspace) | TOS V_D deletion | random-k removal.
Compares conditional-D|Y TOS deletion against unconditional linear concept erasure of D.
  python -m tos_cmi.eeg.erasure_baselines <BNCI2014_001_TSMNet_LOSO dir>   (env TOS_SEED_FILTER optional)
"""
from __future__ import annotations
import glob
import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _cross_fit_fisher, _SplitPlan,
                                  candidate_order, _m_proj)


def _ids(a):
    u = {v: i for i, v in enumerate(sorted(set(map(str, a))))}
    return np.array([u[str(v)] for v in a]), len(u)


def _lin(Xtr, ltr, Xte, lte):
    if len(np.unique(ltr)) < 2:
        return float("nan")
    return float((LogisticRegression(max_iter=200, C=1.0).fit(Xtr, ltr).predict(Xte) == lte).mean())


def _mlp(Xtr, ltr, Xte, lte):
    try:
        if len(np.unique(ltr)) < 2:
            return float("nan")
        from sklearn.neural_network import MLPClassifier
        return float((MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0)
                      .fit(Xtr, ltr).predict(Xte) == lte).mean())
    except Exception:
        return float("nan")


def leace_eraser(Xa, d_onehot):
    """Closed-form LEACE (Belrose+2023): oblique projection orthogonal in the Sigma^{-1} metric that
    removes ALL linear information about the concept while minimally perturbing X. Returns apply(X)."""
    mu = Xa.mean(0); Xc = Xa - mu
    Sigma = (Xc.T @ Xc) / len(Xc)
    ev, V = np.linalg.eigh(Sigma); ev = np.clip(ev, 1e-8, None)
    Wh = V @ np.diag(ev ** -0.5) @ V.T          # Sigma^{-1/2}
    Wh_inv = V @ np.diag(ev ** 0.5) @ V.T        # Sigma^{1/2}
    Zc = d_onehot - d_onehot.mean(0)
    Cxz = (Xc.T @ Zc) / len(Xc)                  # cross-covariance [d, c]
    M = Wh @ Cxz                                 # whitened cross-cov
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    U = U[:, s > 1e-6]                           # concept-correlated directions (whitened)
    P = Wh_inv @ U @ U.T @ Wh                    # eraser projector (original space)
    I = np.eye(P.shape[0])
    return lambda X: (X - mu) @ (I - P).T + mu


def inlp_eraser(Xa, da, k=12, tol=0.02):
    """Iterative Nullspace Projection (Ravfogel+2020): repeatedly null the rowspace of a linear domain
    classifier until linear domain decode ~ chance. Returns apply(X)."""
    d = Xa.shape[1]; P = np.eye(d); chance = 1.0 / len(np.unique(da))
    for _ in range(k):
        Xc = Xa @ P.T
        if len(np.unique(da)) < 2:
            break
        clf = LogisticRegression(max_iter=200).fit(Xc, da)
        if float((clf.predict(Xc) == da).mean()) <= chance + tol:
            break
        W = np.atleast_2d(clf.coef_)             # [c, d]
        Pn = np.eye(d) - W.T @ np.linalg.pinv(W @ W.T) @ W
        P = Pn @ P
    return lambda X: X @ P.T


def analyze(npz_path, cfg=None, seed=0):
    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z_source"].astype(np.float64); y = d["y_source"]
    subj, ns = _ids(d["subject_source"]); n_cls = int(d["n_cls"]); zdim = Z.shape[1]
    cfg = cfg or ScoreFisherConfig()
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); cut = len(y) // 2
    A = np.zeros(len(y), bool); A[perm[:cut]] = True; B = ~A
    oh = np.eye(ns)[subj]
    reps = {"full": lambda X: X,
            "LEACE": leace_eraser(Z[A], oh[A]),
            "INLP": inlp_eraser(Z[A], subj[A])}
    # TOS V_D deletion (conditional D|Y) + random-k, fit on A
    plan = _SplitPlan(int(A.sum()), cfg.n_folds, 1); M = _metric(Z[A], y[A], n_cls, cfg)
    G_Y = _cross_fit_fisher(Z[A], y[A], None, n_cls, zdim, 0, cfg, plan, 0)
    G_DgY = _cross_fit_fisher(Z[A], subj[A], np.eye(n_cls)[y[A]], ns, zdim, n_cls, cfg, plan, 100)
    V_D = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]; k = int(V_D.shape[1])
    reps["TOS_VD"] = (lambda X: X) if k == 0 else (lambda X: X - X @ _m_proj(V_D, M).T)
    Vr = rng.standard_normal((zdim, max(k, 1)))
    reps["random_k"] = (lambda X: X) if k == 0 else (lambda X: X - X @ _m_proj(Vr, M).T)
    out = {"tag": os.path.basename(npz_path), "backbone": str(d["backbone"]), "z_dim": zdim,
           "n_subj": ns, "chance_subj": 1.0 / ns, "chance_task": 1.0 / n_cls, "nDcand": k}
    for nm, fn in reps.items():
        Ztr, Zte = fn(Z[A]), fn(Z[B])
        out["subj_%s_lin" % nm] = _lin(Ztr, subj[A], Zte, subj[B])
        out["subj_%s_mlp" % nm] = _mlp(Ztr, subj[A], Zte, subj[B])
        out["task_%s_lin" % nm] = _lin(Ztr, y[A], Zte, y[B])
        out["task_%s_mlp" % nm] = _mlp(Ztr, y[A], Zte, y[B])
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO"
    sd = os.environ.get("TOS_SEED_FILTER")
    paths = [p for p in sorted(glob.glob("%s/sub*_erm_lam0_seed*.npz" % base))
             if (not sd or p.endswith("_seed%s.npz" % sd))]
    rows = []
    for p in paths:
        try:
            r = analyze(p); rows.append(r)
            print("[%s] LEACE subj lin %.2f mlp %.2f | INLP %.2f/%.2f | TOS %.2f/%.2f | rand %.2f/%.2f "
                  "(ch %.2f) | task LEACE %.2f INLP %.2f TOS %.2f" % (r["tag"],
                  r["subj_LEACE_lin"], r["subj_LEACE_mlp"], r["subj_INLP_lin"], r["subj_INLP_mlp"],
                  r["subj_TOS_VD_lin"], r["subj_TOS_VD_mlp"], r["subj_random_k_lin"], r["subj_random_k_mlp"],
                  r["chance_subj"], r["task_LEACE_lin"], r["task_INLP_lin"], r["task_TOS_VD_lin"]), flush=True)
        except Exception as e:
            print("[FAIL] %s : %r" % (p.split("/")[-1], e), flush=True)
    if rows:
        m = lambda k: float(np.nanmean([r[k] for r in rows if k in r]))
        agg = {k: m(k) for k in rows[0] if k.startswith(("subj_", "task_")) or k in ("nDcand",)}
        agg["chance_subj"] = rows[0]["chance_subj"]; agg["chance_task"] = rows[0]["chance_task"]
        agg["backbone"] = rows[0]["backbone"]; agg["z_dim"] = rows[0]["z_dim"]; agg["n"] = len(rows)
        json.dump({"rows": rows, "aggregate": agg}, open("%s/erasure_report.json" % base, "w"), indent=1)
        print("\n=== %s (z=%d, n=%d, pooled) subject decode ===" % (agg["backbone"], agg["z_dim"], len(rows)))
        for nm in ["full", "LEACE", "INLP", "TOS_VD", "random_k"]:
            print("  %-9s subj lin=%.3f mlp=%.3f | task lin=%.3f mlp=%.3f"
                  % (nm, agg["subj_%s_lin" % nm], agg["subj_%s_mlp" % nm],
                     agg["task_%s_lin" % nm], agg["task_%s_mlp" % nm]))
        print("  chance subj=%.3f task=%.3f ; nDcand=%.1f" % (agg["chance_subj"], agg["chance_task"], agg["nDcand"]))
    print("ERASURE_BASELINES_DONE")


if __name__ == "__main__":
    main()
