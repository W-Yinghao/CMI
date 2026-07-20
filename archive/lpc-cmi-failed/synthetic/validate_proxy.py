"""Validate the LPC-CMI *neural* leakage proxy against an INDEPENDENT kNN estimate of
I(Z;D|Y) on the synthetic. If the neural frozen-encoder probe KL(q||pi_y) tracks the
kNN conditional MI across methods/lambdas, the proxy is a faithful leakage measure.

kNN estimator (license-clean, sklearn): I(Z;D|Y) = sum_y p(y) * I(Z;D|Y=y), where each
stratum's I(Z;D|Y=y) is sklearn's kNN mutual information (Ross 2014) between continuous Z
and discrete D, summed over Z-dims. Independent of the neural posterior used in training.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sanity_check import DGP, make_data, train_one, embed, leakage_probe


def knn_cmi(Z, y, d, n_cls):
    """kNN estimate of I(Z;D|Y) (nats), stratified by Y."""
    tot = 0.0
    for c in range(n_cls):
        m = y == c
        if m.sum() < 20 or len(np.unique(d[m])) < 2:
            continue
        mi = mutual_info_classif(Z[m], d[m], discrete_features=False, random_state=0)
        tot += m.mean() * float(mi.sum())
    return tot


def main():
    dgp = DGP()
    pts = []
    for seed in range(3):
        data = make_data(dgp, seed)
        Xv, yv, dv = data["val"]
        for method in ["erm", "marginal", "lpc_uniform", "lpc_prior"]:
            for lam in [0.0, 0.5, 2.0, 8.0]:
                if method == "erm" and lam != 0.0:
                    continue
                enc, _, _ = train_one(method, data, dgp, lam=lam, epochs=60, seed=seed)
                neural_kl = leakage_probe(enc, data, dgp, seed=seed)["leakage_kl"]
                Zv = embed(enc, Xv)
                knn = knn_cmi(Zv, yv, dv, 2)
                pts.append(dict(method=method, lam=lam, seed=seed, neural_kl=neural_kl, knn_cmi=knn))
                print(f"  {method:12s} lam={lam:4.1f} seed={seed} | neural_KL={neural_kl:.3f}  kNN_CMI={knn:.3f}", flush=True)

    nk = np.array([p["neural_kl"] for p in pts])
    kn = np.array([p["knn_cmi"] for p in pts])
    r = np.corrcoef(nk, kn)[0, 1]
    rho = _spearman(nk, kn)
    print(f"\n=== proxy validation: Pearson r={r:.3f}  Spearman rho={rho:.3f}  (n={len(pts)}) ===")
    plt.figure(figsize=(5, 4.4))
    col = {"erm": "k", "marginal": "C1", "lpc_uniform": "C0", "lpc_prior": "C2"}
    for mth in col:
        s = [p for p in pts if p["method"] == mth]
        if s:
            plt.scatter([p["neural_kl"] for p in s], [p["knn_cmi"] for p in s],
                        c=col[mth], label=mth, s=36, alpha=.8, edgecolors="k", linewidths=.3)
    plt.xlabel("neural proxy  KL(q_probe ‖ π_y)")
    plt.ylabel("independent kNN  Î(Z;D|Y)  (nats)")
    plt.title(f"Leakage-proxy validation\nPearson r={r:.2f}, Spearman ρ={rho:.2f}")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("synthetic/proxy_validation.png", dpi=150)
    print("saved -> synthetic/proxy_validation.png")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
