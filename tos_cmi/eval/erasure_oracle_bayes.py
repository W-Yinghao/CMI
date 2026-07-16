"""Synthetic Bayes leakage-erasure ORACLE (Priority 3) — the theoretical existence oracle on a KNOWN DGP.

Completes `bayes_oracle.py` (which gives only the task-safety side Delta_Y*) with the subject-leakage side and
the constrained search:

    Delta_Y*(P) = H(Y|K_P) - H(Y|Z) = I(Y ; deleted | kept)     (task information lost — from bayes_oracle)
    Delta_D*(P) = H(D|Y,K_P) - H(D|Y,Z) = I(deleted ; D | Y, kept)   (conditional subject leakage removed)

    P*_{k,delta} = argmax_P Delta_D*(P)   s.t.   Delta_Y*(P) <= delta,  rank(P) <= k.

If no non-zero P clears (Delta_D* >= gamma_D, Delta_Y* <= delta) the oracle returns P=0 (identity fallback is
the CORRECT answer, not a method failure). The oracle also emits the (Delta_Y*, Delta_D*) Pareto frontier with
identity / LEACE / random / TOS-candidate / oracle marked, so TOS can be evaluated as an oracle-approximation
(oracle REGRET Delta_D*(P*) - Delta_D*(P_TOS); safety VIOLATION max(0, Delta_Y*(P_TOS) - delta)).

Everything is EXACT on the true Gaussian mixture z|y,d ~ N(mu_{y,d}, diag(sigma^2)); pure numpy.
"""
from __future__ import annotations
from itertools import combinations, chain

import numpy as np

from tos_cmi.eval.bayes_oracle import _gaussian_logpost_entropy


def _cond_subject_entropy(X, mu_cells, cov_inv, pdy, y):
    """Per-sample H(D | Y=y_i, X=x_i) under the true mixture: posterior over d WITHIN the true class y_i,
    p(d|y,x) ∝ p(d|y) N(x; mu_{y,d}, cov). Returns the mean over samples."""
    n_cls, n_dom = mu_cells.shape[:2]
    H = np.zeros(len(X))
    for c in range(n_cls):
        m = y == c
        if not m.any():
            continue
        Xc = X[m]
        comps = np.stack([np.log(pdy[c, e] + 1e-12)
                          - 0.5 * np.einsum("ij,jk,ik->i", Xc - mu_cells[c, e], cov_inv, Xc - mu_cells[c, e])
                          for e in range(n_dom)], 1)
        mx = comps.max(1, keepdims=True); p = np.exp(comps - mx); p /= p.sum(1, keepdims=True)
        H[m] = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)
    return float(H.mean())


def _draw(mu_yd, std, py, pdy, n_mc, seed):
    n_cls, n_dom, Dd = mu_yd.shape
    rng = np.random.default_rng(seed)
    ys = rng.choice(n_cls, n_mc, p=py / py.sum())
    ds = np.array([rng.choice(n_dom, p=pdy[c] / pdy[c].sum()) for c in ys])
    Z = mu_yd[ys, ds] + std * rng.standard_normal((n_mc, Dd))
    return Z, ys, ds


def bayes_deltas(mu_yd, sigma, py, pdy, P, n_mc=20000, seed=0):
    """EXACT (Delta_Y*, Delta_D*) for a projector P on an INDEPENDENT MC draw from the true mixture.
    Returns dict(delta_Y, delta_D, H_YZ, H_YU, H_DZ, H_DU, rank_kept)."""
    mu_yd = np.asarray(mu_yd, float); py = np.asarray(py, float); pdy = np.asarray(pdy, float)
    n_cls, n_dom, Dd = mu_yd.shape
    std = np.broadcast_to(np.asarray(sigma, float), (Dd,)).copy()
    covZ = np.diag(std ** 2); covZ_inv = np.diag(1.0 / std ** 2)
    Z, ys, ds = _draw(mu_yd, std, py, pdy, n_mc, seed)
    # kept component U = A Z, A = orthobasis(range(I-P))^T (I-P)
    Ur, sr, _ = np.linalg.svd(np.eye(Dd) - np.asarray(P, float))
    B_R = Ur[:, sr > 1e-8]
    A = B_R.T @ (np.eye(Dd) - np.asarray(P, float))
    U = Z @ A.T
    muU = np.einsum("md,ced->cem", A, mu_yd)
    covU_inv = np.linalg.pinv(A @ covZ @ A.T)
    H_YZ = _gaussian_logpost_entropy(Z, mu_yd, covZ_inv, py).mean()
    H_YU = _gaussian_logpost_entropy(U, muU, covU_inv, py).mean()
    H_DZ = _cond_subject_entropy(Z, mu_yd, covZ_inv, pdy, ys)
    H_DU = _cond_subject_entropy(U, muU, covU_inv, pdy, ys)
    return {"delta_Y": float(H_YU - H_YZ), "delta_D": float(H_DU - H_DZ),
            "H_YZ": float(H_YZ), "H_YU": float(H_YU), "H_DZ": float(H_DZ), "H_DU": float(H_DU),
            "rank_kept": int(B_R.shape[1])}


def axis_projector(axes, Dd):
    """Orthogonal projector onto the coordinate axes in `axes` (subset of range(Dd))."""
    P = np.zeros((Dd, Dd))
    for a in axes:
        P[a, a] = 1.0
    return P


def oracle_search(mu_yd, sigma, py, pdy, k=None, delta=0.02, gamma_D=1e-3, n_mc=20000, seed=0):
    """Exhaustive axis-subset search for P*_{k,delta} = argmax Delta_D* s.t. Delta_Y* <= delta, rank<=k.
    (The synthetic DGP is defined in the intrinsic axis frame, so axis subsets are the natural candidates.)
    Returns the full frontier (all subsets with their deltas), the constrained oracle, and identity."""
    Dd = np.asarray(mu_yd, float).shape[2]
    k = Dd if k is None else int(k)
    subsets = list(chain.from_iterable(combinations(range(Dd), m) for m in range(1, k + 1)))
    rows = [{"axes": (), "delta_Y": 0.0, "delta_D": 0.0}]     # identity
    for S in subsets:
        d = bayes_deltas(mu_yd, sigma, py, pdy, axis_projector(S, Dd), n_mc=n_mc, seed=seed)
        rows.append({"axes": list(S), "delta_Y": d["delta_Y"], "delta_D": d["delta_D"]})
    feasible = [r for r in rows if r["delta_Y"] <= delta and r["delta_D"] >= gamma_D]
    oracle = max(feasible, key=lambda r: r["delta_D"]) if feasible else {"axes": [], "delta_Y": 0.0, "delta_D": 0.0}
    return {"frontier": rows, "oracle": oracle, "delta": delta, "gamma_D": gamma_D,
            "oracle_is_identity": bool(len(oracle["axes"]) == 0),
            "n_candidates": len(rows)}


def pareto_front(frontier):
    """Non-dominated (min Delta_Y, max Delta_D) points of the frontier."""
    pts = sorted(frontier, key=lambda r: (r["delta_Y"], -r["delta_D"]))
    out, best = [], -np.inf
    for r in pts:
        if r["delta_D"] > best:
            out.append(r); best = r["delta_D"]
    return out


def oracle_regret(oracle_delta_D, method_delta_D):
    """Regret_D = Delta_D*(P*) - Delta_D*(P_method) (how far a method is below the oracle's removed leakage)."""
    return float(oracle_delta_D - method_delta_D)


def safety_violation(method_delta_Y, delta):
    """Violation_Y = max(0, Delta_Y*(P_method) - delta) (how much a method exceeds the task-loss budget)."""
    return float(max(0.0, method_delta_Y - delta))
