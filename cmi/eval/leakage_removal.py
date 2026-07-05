"""CIGL R3 (flagship) — does CMI-measured label-conditional subject leakage become FUNCTIONALLY load-bearing
for the task classifier? We fit a k-dim subject-predictive subspace on SOURCE only, remove it from the frozen
representation, and measure the task-accuracy drop — for ERM vs CIGL. If ERM's task drop >> CIGL's, CIGL has
reduced the classifier/representation's *reliance* on subject leakage, not merely its decodability.

Artifact-driven: consumes a .audit.npz sidecar; NEVER retrains the backbone. Two evaluation modes:
  head-replay  (task_head in the sidecar): logits_removed = head(z_removed) -> CLASSIFIER reliance.
  probe fallback (no head): source-fit task probe on z -> REPRESENTATION reliance (weaker claim; labeled).

FIREWALL: the subspace, label means, and task/subject probes are fit on SOURCE only (d != target_domain).
Target trials are eval-only; target labels never influence any fit. k is fixed / curve-reported, never chosen
by target performance.
"""
from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from cmi.eval.audit_npz import has_task_head, replay_head

CONDITIONINGS = ("label_conditional", "marginal_domain", "random_subspace")
DEFAULT_K_CURVE = (1, 2, 4, 8)
PRIMARY_K = 2
PRIMARY_CONDITIONING = "label_conditional"


def fit_leakage_subspace(z, y, d, k, conditioning="label_conditional", seed=0):
    """Fit the k-dim subject-predictive subspace on the given (SOURCE) z,y,d. Returns (P, dirs) where dirs is
    [k, Zdim] orthonormal and P = I - dirs^T dirs removes the subspace. Deterministic under `seed`."""
    z = np.asarray(z, dtype=float); y = np.asarray(y); d = np.asarray(d)
    Zdim = z.shape[1]
    if conditioning == "label_conditional":
        # Delta_{y,d} = mean(z | y,d) - mu_y  (subject-within-label offset), weighted by sqrt(count)
        rows = []
        for yy in np.unique(y):
            my = y == yy
            mu_y = z[my].mean(0)
            for dd in np.unique(d[my]):
                m = my & (d == dd)
                if m.sum() > 0:
                    rows.append(np.sqrt(m.sum()) * (z[m].mean(0) - mu_y))
        M = np.stack(rows) if rows else np.zeros((1, Zdim))
    elif conditioning == "marginal_domain":
        gm = z.mean(0)                                          # ignores label -> control
        M = np.stack([np.sqrt((d == dd).sum()) * (z[d == dd].mean(0) - gm) for dd in np.unique(d)])
    elif conditioning == "random_subspace":
        M = np.random.default_rng(seed).standard_normal((max(2 * k, Zdim), Zdim))   # deterministic control
    else:
        raise ValueError(conditioning)
    # top-k directions = leading right singular vectors of the offset matrix
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    kk = min(k, Vt.shape[0], Zdim)
    dirs = Vt[:kk]                                              # [k, Zdim] orthonormal
    P = np.eye(Zdim) - dirs.T @ dirs
    return P, dirs


def remove_subspace(z, P):
    return np.asarray(z, dtype=float) @ P.T


def _bacc(pred, true):
    return float(balanced_accuracy_score(np.asarray(true), np.asarray(pred)))


def _task_bacc_headreplay(data, z_eval, y_eval):
    return _bacc(np.argmax(replay_head(data, z_eval), 1), y_eval)


def _task_bacc_probe(z_src, y_src, z_eval, y_eval):
    clf = LDA().fit(z_src, y_src)                               # source-fit task probe (representation reliance)
    return _bacc(clf.predict(z_eval), y_eval)


def _subject_bacc(z, d, y, seed):
    """LABEL-CONDITIONAL subject-decoding balanced accuracy (matches the CIGL estimand I(Z;D|Y)): within each
    label decode subject on a source train/val split, average over labels. A marginal decoder would miss the
    label-conditional leakage that CIGL targets (subject info that cancels when you pool over labels)."""
    rng = np.random.default_rng(seed)
    accs = []
    for yy in np.unique(y):
        m = y == yy
        zz, dd = z[m], d[m]
        idx = rng.permutation(len(zz)); cut = int(0.7 * len(idx))
        tr, ev = idx[:cut], idx[cut:]
        if len(np.unique(dd[tr])) < 2 or len(ev) == 0:
            continue
        clf = LogisticRegression(max_iter=500).fit(zz[tr], dd[tr])
        accs.append(_bacc(clf.predict(zz[ev]), dd[ev]))
    return float(np.mean(accs)) if accs else float("nan")


def evaluate_reliance(data, target_domain, k=PRIMARY_K, conditioning=PRIMARY_CONDITIONING, seed=0,
                      representation="graph_z"):
    """One reliance row: fit the subspace on SOURCE, remove it, measure task/subject bAcc before/after.
    `data` = a loaded .audit.npz dict; `target_domain` = the int d value of the held-out target subject."""
    z = np.asarray(data[representation], dtype=float)
    y = np.asarray(data["y"]); d = np.asarray(data["d"])
    src = d != target_domain
    tgt = d == target_domain
    firewall_ok = bool(src.sum() > 0 and tgt.sum() >= 0 and target_domain not in np.unique(d[src]))
    P, _ = fit_leakage_subspace(z[src], y[src], d[src], k, conditioning, seed)   # SOURCE-only fit
    z_rm = remove_subspace(z, P)

    head = has_task_head(data) and data.get("task_head_input", representation) == representation
    mode = "head_replay" if head else "probe_replay"
    if head:
        s_before = _task_bacc_headreplay(data, z[src], y[src]); s_after = _task_bacc_headreplay(data, z_rm[src], y[src])
        t_before = _task_bacc_headreplay(data, z[tgt], y[tgt]) if tgt.sum() else float("nan")
        t_after = _task_bacc_headreplay(data, z_rm[tgt], y[tgt]) if tgt.sum() else float("nan")
    else:
        s_before = _task_bacc_probe(z[src], y[src], z[src], y[src])
        s_after = _task_bacc_probe(z_rm[src], y[src], z_rm[src], y[src])
        t_before = _task_bacc_probe(z[src], y[src], z[tgt], y[tgt]) if tgt.sum() else float("nan")
        t_after = _task_bacc_probe(z_rm[src], y[src], z_rm[tgt], y[tgt]) if tgt.sum() else float("nan")
    subj_before = _subject_bacc(z[src], d[src], y[src], seed)
    subj_after = _subject_bacc(z_rm[src], d[src], y[src], seed)
    task_drop = (t_before - t_after) if tgt.sum() else (s_before - s_after)
    return {
        "dataset": data.get("dataset", ""), "fold": int(np.asarray(data.get("fold", -1))),
        "seed": int(np.asarray(data.get("seed", seed))), "target_subject": data.get("target_subject", ""),
        "method": data.get("method", ""), "representation": representation, "removal_mode": mode,
        "conditioning": conditioning, "k": int(k),
        "source_task_bacc_before": s_before, "source_task_bacc_after": s_after,
        "target_task_bacc_before": t_before, "target_task_bacc_after": t_after, "task_drop": float(task_drop),
        "source_subject_bacc_before": subj_before, "source_subject_bacc_after": subj_after,
        "subject_leakage_drop": float(subj_before - subj_after) if subj_before == subj_before else float("nan"),
        "head_replay_available": bool(has_task_head(data)), "probe_replay_used": (mode == "probe_replay"),
        "firewall_passed": firewall_ok,
    }


def reliance_curve(data, target_domain, ks=DEFAULT_K_CURVE, conditionings=CONDITIONINGS, seed=0,
                   representation="graph_z"):
    """Full fixed k-curve x conditioning (label_conditional primary; marginal_domain + random_subspace
    controls). Returns a list of rows. k is NEVER selected by target performance."""
    return [evaluate_reliance(data, target_domain, k=k, conditioning=c, seed=seed, representation=representation)
            for c in conditionings for k in ks]
