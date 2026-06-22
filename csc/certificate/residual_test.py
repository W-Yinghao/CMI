"""
csc.certificate.residual_test — source-side concept-evidence test (CSC-P0 rewrite).

Statistic (cross-fitted, out-of-fold):

    T = CE( h0(Y | Z, D) ) - CE( h(Y | Z, D) )

  h0 : domain-INTERCEPT only      -> features [Z, D_dummies]        (shared boundary + label shift)
  h  : domain-DEPENDENT boundary  -> features [Z, D_dummies, Z(x)D_dummies]

THREE fixes over the v0 scaffold (per review):

1. REFERENCE CODING. The v0 design put a full domain one-hot next to the LR intercept, and
   a full Z(x)D block next to shared Z -> deterministic rank deficiency (h0: 1, h: 1+d).
   Under L2 that silently re-weights the penalty, so CE(h0)-CE(h) was not "is the
   interaction useful". We now DROP the reference domain column in both the dummy and the
   interaction block: the reference domain's boundary IS the shared-Z coefficient, and
   non-reference domains add identified adjustments. No deterministic collinearity remains.

2. VALID NULL = PARAMETRIC BOOTSTRAP UNDER FITTED h0 (not within-Y permutation of D).
   Permuting D within Y assumed exchangeability of D | Y, which fails under covariate shift
   (Z and D are dependent given Y). Instead we CONDITION on (Z, D) and resample
   Y* ~ p_hat0(y | z, d) from the fitted domain-intercept model. This is exactly the null
   "the boundary is domain-independent" with the observed P(Z, D) held fixed. T* recomputed
   per draw gives a calibrated one-sided p-value for T > 0.

3. SUPPORT GATE checks IDENTIFIABILITY, not just degree (see check_support_graph): bipartite
   (domain,class) connectivity, a minimum per-cell sample count, and the numerical
   conditioning of the interaction design -- the single-class / disconnected / ill-posed
   cases the v0 gate let through.

The raw I(Y;D|Z) is a predictive-insufficiency diagnostic (H2-CMI P0-4); only the
intercept-vs-boundary split + this gate make T interpretable as boundary movement. We do
NOT call it "precise CMI".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import log_loss


@dataclass
class SupportGraph:
    valid: bool
    reasons: list
    n_domains: int
    n_classes: int
    min_classes_per_domain: int
    min_domains_per_class: int
    connected: bool
    n_components: int
    min_cell_count: int
    design_condition: float
    design_rank: int = -1
    design_ncols: int = -1
    full_rank: bool = True


@dataclass
class ResidualTestResult:
    status: str               # "VALID" | "INVALID"
    T: float                  # cross-fitted CE(h0) - CE(h)
    p_value: float            # parametric-bootstrap p (under fitted h0); 1.0 if INVALID
    significant: bool
    null_mean: float
    null_q: float             # (1-alpha) quantile of the bootstrap null
    support: SupportGraph
    ce_h0: float = float("nan")
    ce_h: float = float("nan")


# ---------------------------------------------------------------------------------------
# support-graph validity gate
# ---------------------------------------------------------------------------------------
def _connected_components(Y, D, classes, domains) -> int:
    """Union-find over the bipartite (domain, class) graph; an edge per observed cell.
    Returns the number of connected components among nodes that actually appear."""
    nd = len(domains)
    parent = list(range(nd + len(classes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    di = {d: i for i, d in enumerate(domains)}
    ci = {c: nd + j for j, c in enumerate(classes)}
    for d, c in zip(D, Y):
        union(di[d], ci[c])
    roots = set()
    for d in domains:
        roots.add(find(di[d]))
    for c in classes:
        roots.add(find(ci[c]))
    return len(roots)


def _design_rank_cond(Zs, D, domains):
    """Rank, #cols and condition number of the LR design [1, reference-coded interaction X].
    The intercept is included (the LR fits it), so a column collinear with the intercept is a
    genuine rank deficiency. RANK is on the actual (uncentered) design so exact-zero
    directions DROP it (the v0 condition number dropped sub-1e-12 singular values and so
    never reached inf); the condition number is on the centred feature block."""
    X = _features(Zs, D, domains, interaction=True)
    Xi = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    ncols = Xi.shape[1]
    rank = int(np.linalg.matrix_rank(Xi))
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    nz = s[s > max(Xc.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)]
    cond = float(nz[0] / nz[-1]) if nz.size else np.inf
    return rank, ncols, cond


def check_support_graph(Y, D, Z=None,
                        min_classes_per_domain: int = 2,
                        min_domains_per_class: int = 2,
                        min_cell: int = 10,
                        max_condition: float = 1e8) -> SupportGraph:
    Y = np.asarray(Y); D = np.asarray(D)
    classes = list(np.unique(Y)); domains = list(np.unique(D))
    reasons = []

    cpd = min(int(np.unique(Y[D == d]).size) for d in domains)
    if cpd < min_classes_per_domain:
        reasons.append(f"a domain spans only {cpd} class(es) (need >= "
                       f"{min_classes_per_domain}); residual decoder degenerates")
    dpc = min(int(np.unique(D[Y == c]).size) for c in classes)
    if dpc < min_domains_per_class:
        reasons.append(f"a class appears in only {dpc} domain(s) (need >= "
                       f"{min_domains_per_class}); boundary not comparable")

    n_comp = _connected_components(Y, D, classes, domains)
    connected = (n_comp == 1)
    if not connected:
        reasons.append(f"domain-class support graph is DISCONNECTED ({n_comp} components); "
                       "the boundary is not jointly identifiable across the partition")

    cells = [int(((D == d) & (Y == c)).sum()) for d in domains for c in classes]
    nonempty = [n for n in cells if n > 0]
    min_cell_count = min(nonempty) if nonempty else 0
    if nonempty and min(nonempty) < min_cell:
        reasons.append(f"smallest occupied (domain,class) cell has {min(nonempty)} samples "
                       f"(need >= {min_cell}); per-cell boundary estimate is unreliable")

    cond = float("nan")
    design_rank, design_ncols, full_rank = -1, -1, True
    if Z is not None:
        Zs = _standardise(Z)
        design_rank, design_ncols, cond = _design_rank_cond(Zs, D, domains)
        full_rank = (design_rank == design_ncols)
        if not full_rank:
            reasons.append(f"interaction design is RANK-DEFICIENT (rank {design_rank} < "
                           f"{design_ncols} cols); boundary contrasts not all identifiable")
        elif cond > max_condition:
            reasons.append(f"interaction design is ill-conditioned (cond={cond:.1e} > "
                           f"{max_condition:.0e}); boundary contrasts numerically unstable")

    return SupportGraph(valid=len(reasons) == 0, reasons=reasons,
                        n_domains=len(domains), n_classes=len(classes),
                        min_classes_per_domain=cpd, min_domains_per_class=dpc,
                        connected=connected, n_components=n_comp,
                        min_cell_count=min_cell_count, design_condition=cond,
                        design_rank=design_rank, design_ncols=design_ncols,
                        full_rank=full_rank)


# ---------------------------------------------------------------------------------------
# reference-coded designs + cross-fitted CE
# ---------------------------------------------------------------------------------------
def _onehot_ref(D, domains) -> np.ndarray:
    """One-hot of D with the FIRST (reference) domain column dropped."""
    idx = {d: i for i, d in enumerate(domains)}
    O = np.zeros((len(D), len(domains)))
    for i, d in enumerate(D):
        O[i, idx[d]] = 1.0
    return O[:, 1:]


def _features(Z, D, domains, interaction: bool) -> np.ndarray:
    O = _onehot_ref(D, domains)                         # (n, n_domains-1)
    if not interaction:
        return np.concatenate([Z, O], axis=1)
    inter = (Z[:, :, None] * O[:, None, :]).reshape(len(Z), -1)   # Z (x) non-ref dummies
    return np.concatenate([Z, O, inter], axis=1)


def _splits(X, Y, n_folds, seed, groups=None):
    """Cross-fit splits. With `groups` (subject/session ids) use StratifiedGroupKFold so no
    cluster is split across train/test -- the cluster-aware path needed for real EEG."""
    if groups is not None:
        ng = len(np.unique(groups))
        k = max(2, min(n_folds, ng))
        return StratifiedGroupKFold(n_splits=k).split(X, Y, groups=groups)
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, Y)


def _xfit_ce(Zs, Y, D, domains, interaction, classes, n_folds, C, seed, groups=None) -> float:
    X = _features(Zs, D, domains, interaction)
    ce_sum, n = 0.0, 0
    cl = list(classes)
    for tr, te in _splits(X, Y, n_folds, seed, groups):
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], Y[tr])
        proba = clf.predict_proba(X[te])
        full = np.full((len(te), len(cl)), 1e-12)
        for j, c in enumerate(clf.classes_):
            full[:, cl.index(c)] = proba[:, j]
        full /= full.sum(1, keepdims=True)
        ce_sum += log_loss(Y[te], full, labels=cl) * len(te)
        n += len(te)
    return ce_sum / n


def fit_h0_proba(Zs, Y, D, domains, classes, C=1.0) -> np.ndarray:
    """Fit the domain-intercept model on ALL data; return p_hat0(y|z,d) aligned to
    `classes` (used to resample Y* under the null)."""
    X = _features(Zs, D, domains, interaction=False)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    clf.fit(X, Y)
    proba = clf.predict_proba(X)
    cl = list(classes)
    full = np.full((len(Zs), len(cl)), 1e-12)
    for j, c in enumerate(clf.classes_):
        full[:, cl.index(c)] = proba[:, j]
    full /= full.sum(1, keepdims=True)
    return full


def sample_labels(proba, classes, rng) -> np.ndarray:
    """Vectorised categorical sampling Y* ~ proba (rows), returning values in `classes`."""
    cl = np.asarray(classes)
    cdf = np.cumsum(proba, axis=1)
    u = rng.random((proba.shape[0], 1))
    idx = (u > cdf).sum(axis=1)
    idx = np.clip(idx, 0, len(cl) - 1)
    return cl[idx]


def _standardise(Z):
    Z = np.asarray(Z, float)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)


# ---------------------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------------------
def residual_decoder_test(Z, Y, D,
                          n_folds: int = 4,
                          n_boot: int = 100,
                          alpha: float = 0.05,
                          C: float = 1.0,
                          group_ids=None,
                          seed: int = 0) -> ResidualTestResult:
    Y = np.asarray(Y); D = np.asarray(D)
    Zs = _standardise(Z)
    groups = None if group_ids is None else np.asarray(group_ids)
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    support = check_support_graph(Y, D, Z=Zs)
    if not support.valid:
        return ResidualTestResult("INVALID", float("nan"), 1.0, False,
                                  float("nan"), float("nan"), support)

    ce_h0 = _xfit_ce(Zs, Y, D, domains, False, classes, n_folds, C, seed, groups)
    ce_h = _xfit_ce(Zs, Y, D, domains, True, classes, n_folds, C, seed, groups)
    T = ce_h0 - ce_h

    p0 = fit_h0_proba(Zs, Y, D, domains, classes, C)
    rng = np.random.default_rng(seed + 1)
    null = np.empty(n_boot)
    for b in range(n_boot):
        Yb = sample_labels(p0, classes, rng)
        counts = np.array([(Yb == c).sum() for c in classes])
        if counts.min() < n_folds:                # keep stratified folds well-defined
            null[b] = -np.inf
            continue
        c0 = _xfit_ce(Zs, Yb, D, domains, False, classes, n_folds, C, seed, groups)
        c1 = _xfit_ce(Zs, Yb, D, domains, True, classes, n_folds, C, seed, groups)
        null[b] = c0 - c1
    valid_null = null[np.isfinite(null)]
    p_value = (1.0 + np.sum(valid_null >= T)) / (1.0 + valid_null.size)
    null_q = float(np.quantile(valid_null, 1.0 - alpha)) if valid_null.size else float("nan")
    return ResidualTestResult("VALID", float(T), float(p_value),
                              bool(p_value <= alpha),
                              float(valid_null.mean()) if valid_null.size else float("nan"),
                              null_q, support, float(ce_h0), float(ce_h))


if __name__ == "__main__":
    from csc.sim.shift_simulator import SimConfig, make_source
    cfg = SimConfig(seed=1)
    src = make_source(cfg, n_domains=8, concept_domains=3)
    r = residual_decoder_test(src.Z, src.Y, src.D, n_boot=60)
    print(f"status={r.status}  T={r.T:+.4f}  p={r.p_value:.3f}  sig={r.significant}")
    print(f"  CE(h0)={r.ce_h0:.4f}  CE(h)={r.ce_h:.4f}  null_mean={r.null_mean:+.4f}")
    print(f"  support: connected={r.support.connected} min_cell={r.support.min_cell_count} "
          f"cond={r.support.design_condition:.1e}")
