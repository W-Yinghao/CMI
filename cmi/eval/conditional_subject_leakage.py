"""CMI-Trace P1.1 — reusable FLAT-feature conditional-subject-leakage ruler (the same posterior-KL CMI
ruler used for graph/node objects, applied to arbitrary flat latents: full / TOS_VD / LEACE / RLACE / INLP /
random-k erased representations).

Refactor, not duplicate: the estimator, within-label prior, and fully-retrained within-label permutation null
all come from cmi.eval.graph_leakage (fit_conditional_domain_probe, _permutation_null, _perm_summary,
compute_label_domain_prior); the support-aware trial split comes from cmi.eval.probe_splits.

CROSS-FITTING (P1.1 requirement): never fit an eraser and a CMI posterior on the same evaluation samples. We
use a THREE-WAY trial-disjoint split — eraser-fit / posterior-train / posterior-eval — each preserving (Y,D)
support and keeping a trial intact, from SOURCE data only. The SAME (ptrain, peval) split is reused across all
transformations so their leakage numbers are paired.

Reported per representation:
  * posterior_kl_nats            raw held-out posterior-KL (NOT exact/calibrated bits)
  * excess_over_null             posterior_kl - retrained-within-label-permutation-null mean
  * perm_p                       one-sided permutation p-value
  * normalized_leakage           posterior_kl / H_hat(D|Y)   (secondary cross-dataset scale)
  * critic_acc / prior_acc / critic_advantage   critic vs within-label prior baseline
  * subject_residual_linear / _mlp   label-conditional subject-decoding bAcc (linear + MLP)
"""
from __future__ import annotations
import numpy as np

from cmi.eval.graph_leakage import (fit_conditional_domain_probe, _permutation_null, _perm_summary)
from cmi.eval.probe_splits import stratified_trial_split_by_y_d


# --------------------------------------------------------------------- three-way cross-fitting split
def three_way_support_split(y, d, seed=0, eraser_frac=0.34, min_per_cell=2):
    """Trial-disjoint 3-way split preserving (Y,D) support: (eraser_idx, ptrain_idx, peval_idx). Two nested
    support-aware splits (eraser-fit vs rest; then rest -> posterior-train/eval 50/50). Reuse the returned
    (ptrain_idx, peval_idx) across ALL transformations for paired leakage comparison."""
    y = np.asarray(y).astype(np.int64); d = np.asarray(d).astype(np.int64)
    eraser_idx, rest_idx, diag1 = stratified_trial_split_by_y_d(
        y, d, train_frac=eraser_frac, seed=seed, min_per_cell=min_per_cell)
    if rest_idx.size < 2:
        return eraser_idx, rest_idx, rest_idx, {"warning": "rest split too small", "diag_eraser": diag1}
    rel_tr, rel_ev, diag2 = stratified_trial_split_by_y_d(
        y[rest_idx], d[rest_idx], train_frac=0.5, seed=seed + 1, min_per_cell=min_per_cell)
    ptrain_idx = np.sort(rest_idx[rel_tr]); peval_idx = np.sort(rest_idx[rel_ev])
    diag = {"n_eraser": int(eraser_idx.size), "n_ptrain": int(ptrain_idx.size), "n_peval": int(peval_idx.size),
            "disjoint": bool(len(np.intersect1d(eraser_idx, ptrain_idx)) == 0
                             and len(np.intersect1d(eraser_idx, peval_idx)) == 0
                             and len(np.intersect1d(ptrain_idx, peval_idx)) == 0),
            "eraser_split": diag1, "posterior_split": diag2}
    return eraser_idx, ptrain_idx, peval_idx, diag


# --------------------------------------------------------------------- scales / residuals
def conditional_entropy_d_given_y(y, d, n_dom, smoothing=1e-3):
    """H_hat(D|Y) in nats from the empirical smoothed label-conditional domain prior p(D|Y)."""
    y = np.asarray(y).astype(np.int64); d = np.asarray(d).astype(np.int64)
    H, n = 0.0, len(y)
    for c in np.unique(y):
        m = y == c
        p = np.bincount(d[m], minlength=n_dom).astype(np.float64) + smoothing
        p = p / p.sum()
        H += (m.sum() / n) * float(-(p * np.log(p)).sum())
    return float(H)


def subject_residual(features, y, d, seed=0, kind="linear"):
    """Label-conditional subject-decoding balanced accuracy on `features` (a residual-leakage witness):
    within each label, decode subject on a source train/val split, average over labels. kind='linear'
    (logistic) or 'mlp' (one hidden layer). Matches the I(Z;D|Y) estimand's label-conditioning."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import balanced_accuracy_score
    X = np.asarray(features, float); y = np.asarray(y).astype(np.int64); d = np.asarray(d).astype(np.int64)
    rng = np.random.default_rng(seed)
    accs = []
    for c in np.unique(y):
        m = y == c
        zz, dd = X[m], d[m]
        if len(np.unique(dd)) < 2:
            continue
        idx = rng.permutation(len(zz)); cut = int(0.7 * len(idx))
        tr, ev = idx[:cut], idx[cut:]
        if len(np.unique(dd[tr])) < 2 or len(ev) == 0:
            continue
        clf = (LogisticRegression(max_iter=500) if kind == "linear"
               else MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=int(seed)))
        clf.fit(zz[tr], dd[tr])
        accs.append(balanced_accuracy_score(dd[ev], clf.predict(zz[ev])))
    return float(np.mean(accs)) if accs else float("nan")


# --------------------------------------------------------------------- the flat CMI ruler
def flat_conditional_cmi(features, y, d, n_cls, n_dom, ptrain_idx, peval_idx, *,
                         n_perm=50, seed=0, device="cpu", hidden_dim=64, epochs=100,
                         with_residual=True):
    """Posterior-KL conditional-subject-leakage ruler on FLAT features [N, F], using a FIXED (ptrain, peval)
    split (both disjoint from the eraser-fit split). Reuses the graph/node estimator + retrained within-label
    permutation null. Returns a report dict (raw nats + excess over null + perm p + normalized leakage +
    critic diagnostics + optional linear/MLP subject residual). NOT exact CMI / not calibrated in bits."""
    y = np.asarray(y).astype(np.int64); d = np.asarray(d).astype(np.int64)
    ptrain_idx = np.asarray(ptrain_idx, np.int64); peval_idx = np.asarray(peval_idx, np.int64)

    def fit(d_arr):
        return fit_conditional_domain_probe(features, y, d_arr, n_cls, n_dom,
                                            train_idx=ptrain_idx, val_idx=peval_idx,
                                            hidden_dim=hidden_dim, epochs=epochs, seed=seed, device=device)
    g = fit(d)
    ps = _perm_summary(g["kl_mean"], _permutation_null(fit, y, d, n_perm, seed, permute_idx=ptrain_idx))
    hcond = conditional_entropy_d_given_y(y[np.r_[ptrain_idx, peval_idx]], d[np.r_[ptrain_idx, peval_idx]], n_dom)
    out = {"posterior_kl_nats": float(g["kl_mean"]),
           "excess_over_null": float(ps["excess_over_null"]),
           "null_mean": float(ps["permutation_mean"]), "perm_p": float(ps["permutation_p"]),
           "conditional_entropy_d_given_y": hcond,
           "normalized_leakage": float(g["kl_mean"] / hcond) if hcond > 1e-9 else float("nan"),
           "critic_acc": float(g["domain_acc"]), "prior_acc": float(g["prior_acc"]),
           "critic_advantage": float(g["leakage_advantage"]),
           "n_ptrain": int(g["n_train"]), "n_peval": int(g["n_val"])}
    if with_residual:
        pool = np.r_[ptrain_idx, peval_idx]
        Xp = np.asarray(features, float)[pool]
        out["subject_residual_linear"] = subject_residual(Xp, y[pool], d[pool], seed=seed, kind="linear")
        out["subject_residual_mlp"] = subject_residual(Xp, y[pool], d[pool], seed=seed, kind="mlp")
    return out


def cmi_ruler_across_transforms(transformed, y, d, n_cls, n_dom, ptrain_idx, peval_idx, *,
                                n_perm=50, seed=0, device="cpu", hidden_dim=64, epochs=100,
                                with_residual=True):
    """Apply the SAME flat CMI ruler to a dict {name: features_[N,F]} of representations (full + erased),
    reusing the SAME (ptrain, peval) split for paired comparison. Erasers must already be fit on the DISJOINT
    eraser-fit split by the caller and applied to ALL rows. Returns {name: report}. This is the table that
    connects CMI amount -> linear/nonlinear removability -> (target effect, when the caller adds it)."""
    return {name: flat_conditional_cmi(Z, y, d, n_cls, n_dom, ptrain_idx, peval_idx,
                                       n_perm=n_perm, seed=seed, device=device, hidden_dim=hidden_dim,
                                       epochs=epochs, with_residual=with_residual)
            for name, Z in transformed.items()}
