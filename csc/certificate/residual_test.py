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
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import log_loss


def stage_seed(root, stage: str) -> int:
    """Named-stage RNG derivation shared across modules: the runtime ROOT seed drives every
    stage via a stable hash, so each stage (residual-null, geometry-null, cov-bootstrap,
    calibration, target-bootstrap, CV) gets an INDEPENDENT seed -- no hand-written seed+1/+7/+11.
    Uses a CRYPTOGRAPHIC hash (sha256), NOT Python's built-in hash() -- the latter is subject to
    per-process PYTHONHASHSEED randomization and would not be reproducible across processes.
    The root is execution context, not part of the method manifest."""
    return int(hashlib.sha256(f"{int(root)}::{stage}".encode()).hexdigest()[:8], 16)


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
    null_invalid: int = 0     # degenerate null replicates -- COUNTED, never silently dropped


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


def check_support_graph(Y, D, Z=None, group_ids=None,
                        min_classes_per_domain: int = 2,
                        min_domains_per_class: int = 2,
                        min_cell: int = 10,
                        min_cell_subjects: int = 1,   # per-cell = CONNECTIVITY (>=1 subject);
                        min_subjects_per_class: int = 4,  # the substantive POWER gate is per-class
                        n_folds: int = 4,
                        check_design: bool = True,    # False -> skip the (expensive) interaction
                                                      # design rank/condition SVD (cov-bootstrap
                                                      # per-replicate only needs cell support)
                        max_condition: float = 1e8) -> SupportGraph:
    Y = np.asarray(Y); D = np.asarray(D)
    g = None if group_ids is None else np.asarray(group_ids)
    classes = list(np.unique(Y)); domains = list(np.unique(D))
    reasons = []
    # grouped-fold feasibility: each class needs >= n_folds INDEPENDENT subjects so a
    # StratifiedGroupKFold over subjects is well-defined (epoch count never substitutes).
    if g is not None:
        spc = min(int(np.unique(g[Y == c]).size) for c in classes)
        if spc < max(min_subjects_per_class, n_folds):
            reasons.append(f"a class has only {spc} independent subjects (need >= "
                           f"{max(min_subjects_per_class, n_folds)} for grouped {n_folds}-fold CV)")

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

    # cell count uses the ANALYSIS UNIT: independent SUBJECTS per (domain,class) when
    # group_ids are given (rows otherwise). On real EEG epochs-per-cell overstates evidence.
    def _cell(d, c):
        m = (D == d) & (Y == c)
        return int(np.unique(g[m]).size) if g is not None else int(m.sum())
    cells = [_cell(d, c) for d in domains for c in classes]
    unit = "subjects" if g is not None else "samples"
    thr = min_cell_subjects if g is not None else min_cell
    nonempty = [n for n in cells if n > 0]
    min_cell_count = min(nonempty) if nonempty else 0
    if nonempty and min(nonempty) < thr:
        reasons.append(f"smallest occupied (domain,class) cell has {min(nonempty)} {unit} "
                       f"(need >= {thr}); per-cell boundary estimate is unreliable")

    cond = float("nan")
    design_rank, design_ncols, full_rank = -1, -1, True
    if Z is not None and check_design:
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


def _condition_cells(groups, D):
    """Map each row to its (subject, condition) cell. The CONDITION is the domain D (e.g. PD
    ON/OFF). Returns (cell_id[n], subject_of_cell{cell:subject})."""
    g = np.asarray(groups); d = np.asarray(D)
    keys = list(zip(g.tolist(), d.tolist()))
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    cell = np.array([uniq[k] for k in keys])
    subj_of_cell = {uniq[k]: k[0] for k in uniq}
    return cell, subj_of_cell


def _subject_condition_weights(groups, D):
    """Fit weights w_sue ∝ 1/(|U_s| * n_su) (subject-CONDITION unit), normalised to MEAN 1. A
    condition with more epochs no longer dominates a subject's loss, and a subject with more
    conditions does not get extra total weight -- one vote per biological subject."""
    g = np.asarray(groups)
    cell, subj_of_cell = _condition_cells(g, D)
    n_su = np.bincount(cell)                                # epochs per (subject,condition) cell
    # |U_s| = number of distinct conditions for each subject
    US = {}
    for c, s in subj_of_cell.items():
        US[s] = US.get(s, 0) + 1
    w = np.array([1.0 / (US[subj_of_cell[c]] * n_su[c]) for c in cell])
    return w * (len(w) / w.sum())


def _aligned(clf, X, cl):
    proba = clf.predict_proba(X)
    full = np.full((len(X), len(cl)), 1e-12)
    for j, c in enumerate(clf.classes_):
        full[:, cl.index(c)] = proba[:, j]
    return full / full.sum(1, keepdims=True)


def aggregate_subject_loss(per_epoch_loss, groups, D):
    """The CSC-P1.4.2 #2 subject-CONDITION estimand aggregation:

        l_s = (1/|U_s|) sum_{u in U_s} ( (1/n_su) sum_e loss_e ),   u = (subject, condition/domain).

    Inner (1/n_su) mean => duplicating epochs within a (subject,condition) cell leaves l_s
    unchanged; outer (1/|U_s|) mean => unequal ON/OFF epoch counts do NOT reweight conditions.
    Returns loss_per_BIOLOGICAL_subject. (Separated out so it is directly unit-testable.)"""
    loss = np.asarray(per_epoch_loss, float)
    cell, subj_of_cell = _condition_cells(groups, D)
    subj_cells = {}
    for c in np.unique(cell):
        subj_cells.setdefault(subj_of_cell[c], []).append(loss[cell == c].mean())
    return {s: float(np.mean(v)) for s, v in subj_cells.items()}


def _xfit_subject_loss(Zs, Y, D, groups, domains, interaction, classes, n_folds, C, seed):
    """Cross-fitted OOF per-epoch loss (group-CV by BIOLOGICAL subject; fit weights
    1/(|U_s| n_su) so a paired subject is one vote), then the subject-condition aggregation
    above. Returns loss_per_biological_subject."""
    X = _features(Zs, D, domains, interaction)
    cl = list(classes)
    w = _subject_condition_weights(groups, D)
    oof = np.full(len(Y), np.nan)
    for tr, te in _splits(X, Y, n_folds, seed, groups):     # split by biological subject
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], Y[tr], sample_weight=w[tr])
        p = _aligned(clf, X[te], cl)
        yi = np.searchsorted(cl, Y[te])
        oof[te] = -np.log(p[np.arange(len(te)), yi])
    seen = ~np.isnan(oof)
    return aggregate_subject_loss(oof[seen], np.asarray(groups)[seen], np.asarray(D)[seen])


def _xfit_T(Zs, Y, D, groups, domains, classes, n_folds, C, seed):
    """T = (1/S) sum_s (l_{s,h0} - l_{s,h}) over BIOLOGICAL subjects -- subject-condition estimand."""
    l0 = _xfit_subject_loss(Zs, Y, D, groups, domains, False, classes, n_folds, C, seed)
    l1 = _xfit_subject_loss(Zs, Y, D, groups, domains, True, classes, n_folds, C, seed)
    keys = [g for g in l0 if g in l1]
    return float(np.mean([l0[g] - l1[g] for g in keys])) if keys else float("nan")


def fit_h0_proba(Zs, Y, D, domains, classes, C=1.0, groups=None) -> np.ndarray:
    """Domain-intercept model on ALL data (subject-condition weighted if groups given); p0(y|z,d)."""
    X = _features(Zs, D, domains, interaction=False)
    w = None if groups is None else _subject_condition_weights(groups, D)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X, Y, sample_weight=w)
    return _aligned(clf, X, list(classes))


def subject_null_labels(p0, groups, classes, rng, D=None) -> np.ndarray:
    """Cluster-consistent h0 null: ONE label per (subject, condition) CELL, q(y) ∝ exp[mean_e
    log p0(y)] (geometric mean over the cell's epochs), broadcast to the cell. Reduces to one
    label per subject when each subject is a single condition (the source). Passing D makes it
    paired-correct (a subject in ON and OFF gets an independent draw per condition)."""
    cl = np.asarray(classes)
    g = np.asarray(groups)
    cell, _ = _condition_cells(g, g if D is None else D)
    out = np.empty(len(g), dtype=cl.dtype)
    logp = np.log(np.clip(p0, 1e-12, 1.0))
    for u in np.unique(cell):
        m = cell == u
        q = np.exp(logp[m].mean(0)); q = q / q.sum()
        y = cl[int((rng.random() > np.cumsum(q)).sum())]
        out[m] = y
    return out


def sample_labels(proba, classes, rng) -> np.ndarray:
    """Per-row categorical sampling Y* ~ proba (epoch-level fallback when no clusters)."""
    cl = np.asarray(classes)
    cdf = np.cumsum(proba, axis=1)
    idx = np.clip((rng.random((proba.shape[0], 1)) > cdf).sum(axis=1), 0, len(cl) - 1)
    return cl[idx]


def _standardise(Z):
    Z = np.asarray(Z, float)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)


# ---------------------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------------------
def _conservative_p(valid_null, n_invalid, T, n_boot):
    """p = (1 + N_extreme_valid + N_invalid) / (1 + B). Invalid replicates are charged as
    extreme (raise p) -> selective dropping can never LOWER the p-value (CSC-P1.4.2 #1)."""
    n_extreme = int(np.sum(np.asarray(valid_null) >= T)) if len(valid_null) else 0
    return (1.0 + n_extreme + n_invalid) / (1.0 + n_boot)


def residual_decoder_test(Z, Y, D,
                          n_folds: int = 4,
                          n_boot: int = 100,
                          alpha: float = 0.05,
                          C: float = 1.0,
                          group_ids=None,
                          invalid_frac_max: float = 0.20,
                          seed: int = 0) -> ResidualTestResult:
    Y = np.asarray(Y); D = np.asarray(D)
    Zs = _standardise(Z)
    # the inference cluster is the biological subject; with no group_ids each EPOCH is its own
    # cluster (component/epoch-level fallback).
    groups = np.asarray(group_ids) if group_ids is not None else np.arange(len(Y))
    classes = list(np.unique(Y)); domains = list(np.unique(D))

    support = check_support_graph(Y, D, Z=Zs, group_ids=group_ids, n_folds=n_folds)
    if not support.valid:
        return ResidualTestResult("INVALID", float("nan"), 1.0, False,
                                  float("nan"), float("nan"), support)

    s_cv = stage_seed(seed, "residual_cv")             # fold split (shared observed + null)
    s_null = stage_seed(seed, "residual_null")         # Y* sampling
    T = _xfit_T(Zs, Y, D, groups, domains, classes, n_folds, C, s_cv)

    # cluster-consistent NULL: one label per (subject,condition) cell ~ geom-mean q; refit T*.
    p0 = fit_h0_proba(Zs, Y, D, domains, classes, C, groups)
    rng = np.random.default_rng(s_null)
    null, n_invalid = [], 0
    for b in range(n_boot):
        Yb = subject_null_labels(p0, groups, classes, rng, D=D)
        if len(np.unique(Yb)) < 2:                  # degenerate replicate -> charged as extreme
            n_invalid += 1
            continue
        try:
            t = _xfit_T(Zs, Yb, D, groups, domains, classes, n_folds, C, s_cv)
            null.append(t) if np.isfinite(t) else None
            if not np.isfinite(t):
                n_invalid += 1
        except Exception:
            n_invalid += 1
    null = np.asarray(null)
    # too many invalid replicates -> the null is not estimable -> source test INVALID (fail closed)
    if n_invalid > invalid_frac_max * n_boot:
        support.reasons.append(f"residual null not estimable: {n_invalid}/{n_boot} invalid replicates")
        return ResidualTestResult("INVALID", float(T), 1.0, False,
                                  float("nan"), float("nan"), support, null_invalid=n_invalid)
    p_value = _conservative_p(null, n_invalid, T, n_boot)
    null_q = float(np.quantile(null, 1.0 - alpha)) if null.size else float("nan")
    return ResidualTestResult("VALID", float(T), float(p_value),
                              bool(p_value <= alpha),
                              float(null.mean()) if null.size else float("nan"),
                              null_q, support, float("nan"), float("nan"),
                              null_invalid=n_invalid)


if __name__ == "__main__":
    from csc.sim.shift_simulator import SimConfig, make_source
    cfg = SimConfig(seed=1)
    src = make_source(cfg, n_domains=8, concept_domains=3)
    r = residual_decoder_test(src.Z, src.Y, src.D, n_boot=60)
    print(f"status={r.status}  T={r.T:+.4f}  p={r.p_value:.3f}  sig={r.significant}")
    print(f"  CE(h0)={r.ce_h0:.4f}  CE(h)={r.ce_h:.4f}  null_mean={r.null_mean:+.4f}")
    print(f"  support: connected={r.support.connected} min_cell={r.support.min_cell_count} "
          f"cond={r.support.design_condition:.1e}")
