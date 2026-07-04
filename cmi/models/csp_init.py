"""CIGL_52 P8-A — source-only CSP spatial filters for initializing the FBCSP-LGG spatial branch.

Motivation: P7a showed a covariance-tangent spatial feature underfits the near-rank-deficient 4-class 2a
covariances (−0.123 on the CSP-decodable subset). The logvar spatial branch IS load-bearing, and classical
CSP still beats the neural spatial branch on several subjects. Rather than change the feature form, we make
the neural spatial filters START from source-estimated CSP filters, then train them normally (not frozen).

FIREWALL: CSP is fit ONLY on source-train trials passed in (the LOSO caller has already removed the held-out
target subject; the source-val subject is removed before this is called). Target labels are never touched.
"""
import numpy as np


def _class_cov(X, shrinkage):
    """Mean trace-normalized, shrinkage-regularized covariance of trials X [N, C, T] -> [C, C]."""
    covs = []
    for xi in X:
        xi = xi - xi.mean(axis=1, keepdims=True)
        c = xi @ xi.T / max(xi.shape[1] - 1, 1)
        tr = np.trace(c)
        covs.append(c / tr if tr > 0 else c)
    C = np.mean(covs, axis=0)
    d = C.shape[0]
    return (1.0 - shrinkage) * C + shrinkage * np.eye(d) * (np.trace(C) / d)


def _csp_pair(Ca, Cb, m):
    """Top-m CSP spatial filters most discriminative for class a vs b. Returns (filters [m, C], disc [m])."""
    comp = Ca + Cb
    ev, U = np.linalg.eigh(comp)                       # whiten the composite covariance
    ev = np.clip(ev, 1e-10, None)
    P = np.diag(1.0 / np.sqrt(ev)) @ U.T               # whitening: P comp P^T = I
    Sa = P @ Ca @ P.T
    lam, V = np.linalg.eigh(Sa)                         # ascending
    W = V.T @ P                                        # rows = spatial filters in channel space
    order = np.argsort(lam)[::-1]                       # descending: most class-a-discriminative first
    W, lam = W[order], lam[order]
    return W[:m], np.abs(lam[:m] - 0.5)                 # disc = distance from the non-discriminative 0.5


def source_csp_filters(X, y, n_cls, m, shrinkage=0.1):
    """Source-only CSP filter bank. n_cls==2 -> binary CSP (m per class end -> 2m). n_cls>2 -> one-vs-rest
    (m per class -> n_cls*m). Returns (W [n_filters, C] unit-norm, disc [n_filters], present_classes)."""
    X, y = np.asarray(X, dtype="float64"), np.asarray(y)
    present = [c for c in range(n_cls) if int((y == c).sum()) > 0]
    covs = {c: _class_cov(X[y == c], shrinkage) for c in present}
    W_list, disc_list = [], []
    if n_cls == 2:
        a, b = present[0], present[-1]
        for hi, lo in ((a, b), (b, a)):
            f, d = _csp_pair(covs[hi], covs[lo], m)
            W_list.append(f); disc_list.append(d)
    else:
        for c in present:                              # one-vs-rest
            c_rest = np.mean([covs[k] for k in present if k != c], axis=0)
            f, d = _csp_pair(covs[c], c_rest, m)
            W_list.append(f); disc_list.append(d)
    W = np.concatenate(W_list, axis=0)
    disc = np.concatenate(disc_list, axis=0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)   # unit-norm filters
    return W.astype("float32"), disc.astype("float32"), present
