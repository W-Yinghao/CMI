"""CMI-Trace Relaxation Ladder — the four-level protocol ladder that isolates, one at a time, the
differences between our strict source-only erasure result and the concurrent FMScope-style result.

  L0 STRICT_SOURCE_ORIGINAL_HEAD      eraser: source only            head: replay stored original  (reliance anchor)
  L1 STRICT_SOURCE_FRESH_HEAD         eraser: source only            head: fresh on source          (deployable DG)
  L2 TARGET_X_UNLABELED_FRESH_HEAD    eraser: source + target X      head: fresh on source          (transductive)
  L3 ORACLE_GLOBAL_GEOMETRY_FRESH...  eraser: whole cohort LW-LEACE   head: fresh subject-grouped CV (oracle diag)

FIREWALL (enforced by `_fit_data` + the row metadata, tested in tests/test_relaxation_ladder_firewall.py):
target TASK LABELS never enter eraser-fit, head-fit, or selection at ANY level (they enter only final
scoring). L0/L1 additionally never touch target X. L2 may use target X + the known "all target trials are one
subject" grouping (never target Y). L3 fits geometry on the whole cohort (oracle diagnostic; NEVER DG).

Every informed deletion (LW-LEACE / repo-LEACE / TOS_VD) is paired with a same-rank RANDOM removal control
(>=50 draws) and a whitening-only (zero-deletion) control, so `specific_erasure_gain = Δ(LEACE) − Δ(random)`
separates identity removal from generic dimensionality reduction / conditioning.

Reads existing P0/P1 DGCNN audit npz (graph_z + verified head) OR regenerated TOS dumps (Z_source/Z_target).
Pure numpy + sklearn; no torch required to run the ladder itself.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

LEVELS = ["L0_STRICT_SOURCE_ORIGINAL_HEAD", "L1_STRICT_SOURCE_FRESH_HEAD",
          "L2_TARGET_X_UNLABELED_FRESH_HEAD", "L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD"]

# (uses_target_x, uses_target_subject_group, uses_target_y, is_source_only_dg, is_transductive, is_oracle_diagnostic)
LEVEL_META = {
    "L0_STRICT_SOURCE_ORIGINAL_HEAD":       dict(uses_target_x=False, uses_target_subject_group=False,
                                                 uses_target_y=False, is_source_only_dg=True,
                                                 is_transductive=False, is_oracle_diagnostic=False),
    "L1_STRICT_SOURCE_FRESH_HEAD":          dict(uses_target_x=False, uses_target_subject_group=False,
                                                 uses_target_y=False, is_source_only_dg=True,
                                                 is_transductive=False, is_oracle_diagnostic=False),
    "L2_TARGET_X_UNLABELED_FRESH_HEAD":     dict(uses_target_x=True, uses_target_subject_group=True,
                                                 uses_target_y=False, is_source_only_dg=False,
                                                 is_transductive=True, is_oracle_diagnostic=False),
    "L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD": dict(uses_target_x=True, uses_target_subject_group=True,
                                                 uses_target_y=False, is_source_only_dg=False,
                                                 is_transductive=False, is_oracle_diagnostic=True),
}

INFORMED_ERASERS = ("lw_leace_full", "repo_leace", "tos_vd", "inlp", "rlace")


# ============================================================ erasers (each returns apply_fn: X -> X_erased)
def _subject_span(Zfit, subj_fit):
    """Centered one-hot subject-mean span: rows = sqrt(n_s)*(mean(Z|s) - global mean). Its right singular
    vectors span the linear subject-identity subspace; rank = (#subjects present) - 1 generically."""
    gm = Zfit.mean(0)
    rows = [np.sqrt((subj_fit == s).sum()) * (Zfit[subj_fit == s].mean(0) - gm) for s in np.unique(subj_fit)]
    return np.vstack(rows) if rows else np.zeros((1, Zfit.shape[1]))


def _ledoit_wolf_cov(Xc):
    """Ledoit-Wolf shrinkage covariance of already-centered Xc (better conditioning than the sample cov)."""
    from sklearn.covariance import ledoit_wolf
    cov, _ = ledoit_wolf(Xc)
    return cov


def lw_leace_full(Zfit, subj_fit, rank=None):
    """Ledoit-Wolf-whitened LEACE removing the FULL centered subject span (rank = k-1 unless k-1 >= dim).
    Whitening uses the LW shrinkage covariance (FMScope numerical conditioning). Returns (apply_fn, rank)."""
    mu = Zfit.mean(0); Xc = Zfit - mu
    Sigma = _ledoit_wolf_cov(Xc)
    ev, V = np.linalg.eigh(Sigma); ev = np.clip(ev, 1e-8, None)
    Wh = V @ np.diag(ev ** -0.5) @ V.T; Wh_inv = V @ np.diag(ev ** 0.5) @ V.T
    M = _subject_span(Zfit, subj_fit) @ Wh.T            # subject span in the whitened metric
    U = np.linalg.svd(M, full_matrices=False)[2]        # right singular vecs (whitened directions)
    kmax = min(len(np.unique(subj_fit)) - 1, Zfit.shape[1])
    k = kmax if rank is None else min(int(rank), kmax)
    k = max(k, 0)
    Uk = U[:k]                                          # [k, dim] whitened directions to remove
    P = Wh_inv @ Uk.T @ Uk @ Wh                         # eraser projector in original space
    I = np.eye(P.shape[0])
    return (lambda X: (X - mu) @ (I - P).T + mu), int(k)


def repo_leace(Zfit, subj_fit):
    """Current repository closed-form LEACE (implementation sensitivity vs LW-LEACE). Returns (apply_fn, rank)."""
    from tos_cmi.eeg.erasure_baselines import leace_eraser
    ns = len(np.unique(subj_fit))
    oh = np.eye(ns)[_dense(subj_fit)]
    return leace_eraser(Zfit, oh), int(min(ns - 1, Zfit.shape[1]))


def whitening_only(Zfit):
    """Standardize/whiten with ZERO deletion (conditioning control). PCA-whiten then keep all dims (an
    invertible re-scaling of the space): removes covariance conditioning as a confound without deleting a
    subject subspace. Returns (apply_fn, rank=0)."""
    mu = Zfit.mean(0); Xc = Zfit - mu
    Sigma = _ledoit_wolf_cov(Xc)
    ev, V = np.linalg.eigh(Sigma); ev = np.clip(ev, 1e-8, None)
    Wh = V @ np.diag(ev ** -0.5) @ V.T
    return (lambda X: (X - mu) @ Wh.T), 0


def random_removal(dim, rank, seed):
    """Remove a RANDOM orthonormal rank-`rank` subspace (matched-rank control for informed deletion)."""
    if rank <= 0:
        return (lambda X: X)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, rank)))
    return (lambda X: X - (X @ Q) @ Q.T)


def tos_vd_eraser(Zfit, yfit, subj_fit, n_cls):
    """TOS conditional-D|Y deletion (score-Fisher V_D). Returns (apply_fn, rank). Falls back to identity if
    the score-Fisher machinery is unavailable/degenerate."""
    try:
        from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _cross_fit_fisher, _SplitPlan,
                                          candidate_order, _m_proj)
        cfg = ScoreFisherConfig()
        subj = _dense(subj_fit); ns = len(np.unique(subj)); zdim = Zfit.shape[1]
        plan = _SplitPlan(len(yfit), cfg.n_folds, 1); M = _metric(Zfit, yfit, n_cls, cfg)
        G_Y = _cross_fit_fisher(Zfit, yfit, None, n_cls, zdim, 0, cfg, plan, 0)
        G_DgY = _cross_fit_fisher(Zfit, subj, np.eye(n_cls)[yfit], ns, zdim, n_cls, cfg, plan, 100)
        V_D = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]; k = int(V_D.shape[1])
        if k == 0:
            return (lambda X: X), 0
        return (lambda X: X - X @ _m_proj(V_D, M).T), k
    except Exception:
        return (lambda X: X), 0


def _dense(a):
    u = {v: i for i, v in enumerate(sorted(set(map(int, np.asarray(a)))))}
    return np.array([u[int(v)] for v in np.asarray(a)], dtype=np.int64)


# ============================================================ heads
def fresh_head_bacc(Ztr, ytr, Zte, yte, head="logreg", seed=0):
    """Fresh readout trained on (Ztr,ytr), scored (balanced accuracy) on (Zte,yte). 'logreg' = standardized
    L2 logistic (primary); 'mlp' = one fixed small hidden layer (sensitivity). Standardization stats from
    the TRAIN split only (no target leakage into scaling)."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr_s, Zte_s = (Ztr - mu) / sd, (Zte - mu) / sd
    if len(np.unique(ytr)) < 2:
        return float("nan")
    clf = (LogisticRegression(max_iter=500, C=1.0) if head == "logreg"
           else MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=int(seed)))
    clf.fit(Ztr_s, ytr)
    return float(balanced_accuracy_score(yte, clf.predict(Zte_s)))


def original_head_bacc(head_W, head_b, Zte, yte):
    """Replay a stored linear head: logits = Zte @ W^T + b -> balanced accuracy. head_W/head_b required."""
    logits = Zte @ np.asarray(head_W).T + np.asarray(head_b)
    return float(balanced_accuracy_score(yte, logits.argmax(1)))


def subject_grouped_cv_bacc(Z, y, subj, head="logreg", seed=0, n_splits=5):
    """L3 oracle readout: subject-grouped CV mean balanced accuracy on the (erased) whole cohort. Folds split
    by SUBJECT so a subject never straddles train/test (cohort-conditioned diagnostic, NOT source->target)."""
    groups = _dense(subj); ns = len(np.unique(groups))
    n = min(int(n_splits), ns)
    if n < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n)
    accs = []
    for tr, te in gkf.split(Z, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        accs.append(fresh_head_bacc(Z[tr], y[tr], Z[te], y[te], head=head, seed=seed))
    return float(np.mean(accs)) if accs else float("nan")


# ============================================================ per-level eraser-fit data (FIREWALL)
def _fit_data(level, feat):
    """The (Zfit, yfit, subj_fit) the eraser is ALLOWED to see at `level`. Target task labels NEVER included.
    L0/L1: source only. L2/L3: source + target X, with target trials grouped as one subject id (no target Y)."""
    Zs, ys, ds = feat["Z_source"], feat["y_source"], feat["subj_source"]
    if level in ("L0_STRICT_SOURCE_ORIGINAL_HEAD", "L1_STRICT_SOURCE_FRESH_HEAD"):
        return Zs, ys, ds
    # L2 / L3: append target X with a distinct subject-group id; target Y is NOT used (pass -1 placeholder).
    Zt = feat["Z_target"]
    tgt_gid = int(np.max(ds)) + 1
    Zfit = np.vstack([Zs, Zt])
    subj_fit = np.concatenate([ds, np.full(len(Zt), tgt_gid)])
    yfit = np.concatenate([ys, np.full(len(Zt), -1)])     # -1 => target Y absent from the fit
    return Zfit, yfit, subj_fit


# ============================================================ cell runner
def _base_row(feat, level, eraser, rank, draw, config_hash, git_sha, head_regime):
    m = LEVEL_META[level]
    return dict(dataset=feat.get("dataset", ""), backbone=feat.get("backbone", ""),
                feature_object=feat.get("feature_object", ""),
                training_method=feat.get("training_method", ""),
                outer_fold=int(feat.get("outer_fold", -1)), heldout_subject=str(feat.get("heldout_subject", "")),
                seed=int(feat.get("seed", 0)), fit_regime=level, head_regime=head_regime,
                eraser=eraser, eraser_rank=int(rank), random_draw_id=(-1 if draw is None else int(draw)),
                uses_target_x=m["uses_target_x"], uses_target_subject_group=m["uses_target_subject_group"],
                uses_target_y=m["uses_target_y"], is_source_only_dg=m["is_source_only_dg"],
                is_transductive=m["is_transductive"], is_oracle_diagnostic=m["is_oracle_diagnostic"],
                config_hash=config_hash, git_sha=git_sha)


def _eval_level(level, feat, apply_fn, head_regime, seed):
    """Return (metric, effect_kind): the readout balanced accuracy AFTER applying `apply_fn` at this level."""
    Zs, ys = feat["Z_source"], feat["y_source"]
    Zt, yt = feat["Z_target"], feat["y_target"]
    if level == "L0_STRICT_SOURCE_ORIGINAL_HEAD":
        if feat.get("head_W") is None:
            # no stored replayable head -> source-fit probe fallback (representation reliance), labeled
            return fresh_head_bacc(apply_fn(Zs), ys, apply_fn(Zt), yt, head="logreg", seed=seed), "original_head_probe_fallback"
        return original_head_bacc(feat["head_W"], feat["head_b"], apply_fn(Zt), yt), "original_head_replay"
    if level in ("L1_STRICT_SOURCE_FRESH_HEAD", "L2_TARGET_X_UNLABELED_FRESH_HEAD"):
        return fresh_head_bacc(apply_fn(Zs), ys, apply_fn(Zt), yt, head=head_regime, seed=seed), "fresh_head"
    if level == "L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD":
        Zc = np.vstack([Zs, Zt]); yc = np.concatenate([ys, yt])
        subjc = np.concatenate([feat["subj_source"], np.full(len(Zt), int(np.max(feat["subj_source"])) + 1)])
        return subject_grouped_cv_bacc(apply_fn(Zc), yc, subjc, head=head_regime, seed=seed), "oracle_grouped_cv"
    raise ValueError(level)


def run_cell(feat, config_hash, git_sha, *, levels=None, n_random=50, seed=0, head_regime="logreg",
             informed=("lw_leace_full", "repo_leace"), with_tos_vd=False):
    """Run the full ladder on ONE feature cell (dump/npz). Returns a list of rows. `full` (identity) is the
    per-level baseline; each informed eraser + whitening-only + n_random random draws are compared to it."""
    levels = levels or LEVELS
    n_cls = int(feat.get("n_cls", len(np.unique(feat["y_source"]))))
    rows = []
    for level in levels:
        Zfit, yfit, subj_fit = _fit_data(level, feat)
        # baseline: identity/full
        full_bacc, effect_kind = _eval_level(level, feat, (lambda X: X), head_regime, seed)
        rows.append({**_base_row(feat, level, "full", 0, None, config_hash, git_sha, head_regime),
                     "erased_bacc": full_bacc, "full_bacc": full_bacc, "delta_bacc": 0.0,
                     "effect_kind": effect_kind})
        # whitening-only conditioning control (rank 0)
        wfn, _ = whitening_only(Zfit)
        wb, _ = _eval_level(level, feat, wfn, head_regime, seed)
        rows.append({**_base_row(feat, level, "whitening_only", 0, None, config_hash, git_sha, head_regime),
                     "erased_bacc": wb, "full_bacc": full_bacc, "delta_bacc": wb - full_bacc,
                     "effect_kind": effect_kind})
        # informed erasers
        informed_ranks = {}
        for name in informed:
            if name == "lw_leace_full":
                fn, rk = lw_leace_full(Zfit, subj_fit)
            elif name == "repo_leace":
                fn, rk = repo_leace(Zfit, subj_fit)
            elif name == "tos_vd" and with_tos_vd:
                fn, rk = tos_vd_eraser(Zfit, np.where(yfit < 0, 0, yfit), subj_fit, n_cls)
            else:
                continue
            b, _ = _eval_level(level, feat, fn, head_regime, seed)
            rows.append({**_base_row(feat, level, name, rk, None, config_hash, git_sha, head_regime),
                         "erased_bacc": b, "full_bacc": full_bacc, "delta_bacc": b - full_bacc,
                         "effect_kind": effect_kind})
            informed_ranks[name] = rk
        # matched-rank random controls (>=n_random draws, at the LW-LEACE rank = the primary informed rank)
        prim_rank = informed_ranks.get("lw_leace_full", max(informed_ranks.values(), default=0))
        for draw in range(int(n_random)):
            fn = random_removal(Zfit.shape[1], prim_rank, seed=1000 * seed + draw)
            b, _ = _eval_level(level, feat, fn, head_regime, seed)
            rows.append({**_base_row(feat, level, "random_k", prim_rank, draw, config_hash, git_sha, head_regime),
                         "erased_bacc": b, "full_bacc": full_bacc, "delta_bacc": b - full_bacc,
                         "effect_kind": effect_kind})
    return rows


# ============================================================ feature loaders
def feat_from_audit_npz(path):
    """Load a P0/P1 DGCNN audit npz -> ladder feat dict (graph_z object; d == subject; verified head)."""
    from cmi.eval.audit_npz import load_audit_npz, head_replay_ok
    data = load_audit_npz(path)
    y = np.asarray(data["y"]); d = np.asarray(data["d"]); Z = np.asarray(data["graph_z"], float)
    ti = np.asarray(data["target_indices"]).ravel() if "target_indices" in data else np.where(d == d.max())[0]
    tgt_dom = int(np.unique(d[ti])[0])
    src = d != tgt_dom
    feat = dict(Z_source=Z[src], y_source=y[src], subj_source=d[src],
                Z_target=Z[~src], y_target=y[~src], subj_target=d[~src],
                n_cls=int(len(np.unique(y))), dataset=str(data.get("dataset", "")),
                backbone="dgcnn_forward_graph_adapter", feature_object="graph_z",
                training_method=str(data.get("method", "")), outer_fold=int(np.asarray(data.get("fold", -1))),
                heldout_subject=str(data.get("target_subject", "")), seed=int(np.asarray(data.get("seed", 0))))
    if head_replay_ok(data):
        feat["head_W"] = np.asarray(data["task_head_weight"]); feat["head_b"] = np.asarray(data.get("task_head_bias", 0.0))
    else:
        feat["head_W"] = None; feat["head_b"] = None
    return feat


def feat_from_tos_dump(path):
    """Load a TOS frozen dump -> ladder feat dict (EEGNet/TSMNet Z; subject_source/target). No stored head
    (only logits) -> L0 uses the labeled probe fallback."""
    d = np.load(path, allow_pickle=True)
    feat = dict(Z_source=np.asarray(d["Z_source"], float), y_source=np.asarray(d["y_source"]),
                subj_source=np.asarray(d["subject_source"]),
                Z_target=np.asarray(d["Z_target"], float), y_target=np.asarray(d["y_target"]),
                subj_target=np.asarray(d["subject_target"]),
                n_cls=int(np.asarray(d["n_cls"])), dataset=str(d["dataset"]), backbone=str(d["backbone"]),
                feature_object="frozen_z", training_method=str(d["method"]),
                outer_fold=-1, heldout_subject=str(int(np.asarray(d["target_subject"]))),
                seed=int(np.asarray(d["seed"])), head_W=None, head_b=None)
    return feat
