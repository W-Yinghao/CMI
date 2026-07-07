#!/usr/bin/env python
"""FSR Phase 8B — L1/L4/L5/L6 encoder audit on frozen CodeBrain/CBraMod pooled embeddings (see docs/FSR_46).
Red-team-hardened: L1 within-subject SESSION-held-out subject probe (marginal + class-conditional) with
permutation nulls; L4 task-head vs LABEL-CONDITIONAL subject subspace (k-curve); L5 held-out-SOURCE reliance with
variance-matched + oracle-task nulls + removed-variance fraction; L6 target consequence (final scoring only) with
the conservative-null caveat. Per-dataset only. Target labels never fit/select. Writes the 8B CSVs + verdict.

    python scripts/cb_cbm_8b_audit.py --model codebrain --dataset shu
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as BACC

OUT = Path("results/fsr_codebrain_cbramod_8b")
RNG = np.random.default_rng(0)
KS = [1, 2, 4, 8]
PRIMARY_K = 2
NPERM = 1000


def _std(X):
    mu = X.mean(0); sd = X.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def top_k(M, k):
    """top-k right singular directions of row-matrix M -> basis (k, D)."""
    if M.shape[0] == 0:
        return np.zeros((0, M.shape[1]))
    U, S, Vt = np.linalg.svd(M - M.mean(0, keepdims=True) if M.shape[0] > 1 else M, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])]


def subject_offsets(X, y, d):
    """label-conditional subject offset rows sqrt(n_yd)*(mean(X|y,d)-mean(X|y))."""
    rows = []
    for yy in np.unique(y):
        my = X[y == yy].mean(0)
        for dd in np.unique(d[y == yy]):
            m = (y == yy) & (d == dd); n = m.sum()
            if n > 0:
                rows.append(np.sqrt(n) * (X[m].mean(0) - my))
    return np.array(rows) if rows else np.zeros((0, X.shape[1]))


def task_offsets(X, y):
    """between-class (task) offset rows sqrt(n_y)*(mean(X|y)-mean(X)) -> oracle task subspace."""
    mu = X.mean(0)
    return np.array([np.sqrt((y == yy).sum()) * (X[y == yy].mean(0) - mu) for yy in np.unique(y)])


def erase(X, basis):
    return X - (X @ basis.T) @ basis if basis.shape[0] else X


def var_frac_removed(X, basis):
    if basis.shape[0] == 0:
        return 0.0
    Xc = X - X.mean(0)
    return float(((Xc @ basis.T) ** 2).sum() / (Xc ** 2).sum())


def clu_ci(vals, nb=2000):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return None, [None, None]
    b = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(nb)]
    return round(float(v.mean()), 4), [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


# ---------------- L1: within-subject session-held-out subject probe ----------------
def l1_subject_probe(X, y, d, ses, cond_on_y=False):
    subs = np.unique(d); chance = 1.0 / len(subs)
    Xr = X.copy()
    if cond_on_y:                       # remove per-class means -> subject info beyond the task label
        for yy in np.unique(y):
            Xr[y == yy] -= Xr[y == yy].mean(0)
    Xs, _, _ = _std(Xr)
    sessions = np.unique(ses)
    accs, perm_all = [], []
    for te in sessions:                 # leave-one-session-out (each subject present in train and test)
        tr = ses != te; teM = ses == te
        if len(np.unique(d[tr])) < len(subs) or teM.sum() < len(subs):
            continue
        clf = LDA().fit(Xs[tr], d[tr])
        pred = clf.predict(Xs[teM]); acc = BACC(d[teM], pred)
        accs.append(acc)
        dt = d[teM]
        perm_all.append([BACC(RNG.permutation(dt), pred) for _ in range(NPERM)])
    if not accs:
        return dict(bacc=None, chance=round(chance, 4), null_mean=None, p=None, n_folds=0)
    obs = float(np.mean(accs)); perm = np.array(perm_all).mean(0)
    return dict(bacc=round(obs, 4), chance=round(chance, 4), null_mean=round(float(perm.mean()), 4),
                null_p95=round(float(np.percentile(perm, 95)), 4),
                p=round(float((perm >= obs).mean()), 4), effect=round(obs - float(perm.mean()), 4), n_folds=len(accs))


# ---------------- L4/L5/L6: LOSO over subjects ----------------
def loso_audit(X, y, d):
    subs = np.unique(d); chance = 1.0 / len(np.unique(y))
    l4_rows, l5_rows, l6_rows = [], [], []
    for tgt in subs:
        src = d != tgt; tg = d == tgt
        Xs = X[src]; ys = y[src]; ds = d[src]
        # nested source split for held-out-source L5 (by remaining subjects: 80/20 subject split)
        ssub = np.unique(ds); RNG.shuffle(ssub)
        val = set(ssub[:max(1, len(ssub) // 5)])
        vmask = np.array([s in val for s in ds]); tmask = ~vmask
        Xstd, mu, sd = _std(Xs[tmask])
        def T(Z): return (Z - mu) / sd
        head = LDA().fit(Xstd, ys[tmask])
        W = head.coef_ if head.coef_.ndim == 2 else head.coef_.reshape(1, -1)  # (n_cls-1 or n_cls, D)
        subj_M = subject_offsets(Xstd, ys[tmask], ds[tmask])
        for k in KS:
            Bsub = top_k(subj_M, k)
            # L4 alignment: mean top principal cosine between task-head rows and subject subspace
            if Bsub.shape[0] and W.shape[0]:
                Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
                align = float(np.mean(np.max(np.abs(Wn @ Bsub.T), axis=1)))
            else:
                align = None
            l4_rows.append(dict(target_subject=int(tgt), k=k, task_head_subject_alignment=round(align, 4) if align is not None else None,
                                subj_subspace_rank=int(Bsub.shape[0])))
        # L5 at PRIMARY_K on held-out SOURCE: subject vs variance-matched vs oracle-task
        k = PRIMARY_K
        Bsub = top_k(subj_M, k); Bvar = top_k(Xstd - Xstd.mean(0), k); Btask = top_k(task_offsets(Xstd, ys[tmask]), k)
        Xv = T(Xs[vmask]); yv = ys[vmask]
        base = BACC(yv, head.predict(Xv))
        def drop(B): return round(base - BACC(yv, head.predict(erase(Xv, B))), 4)
        l5_rows.append(dict(target_subject=int(tgt), k=k, base_bacc=round(base, 4),
                            drop_subject=drop(Bsub), drop_variance=drop(Bvar), drop_oracle_task=drop(Btask),
                            var_removed_subject=round(var_frac_removed(Xv, Bsub), 4),
                            var_removed_variance=round(var_frac_removed(Xv, Bvar), 4)))
        # L6 target consequence (final scoring only): erase source-estimated subject subspace on TARGET
        Xt = T(X[tg]); yt = y[tg]
        base_t = BACC(yt, head.predict(Xt))
        l6_rows.append(dict(target_subject=int(tgt), k=k, chance=round(chance, 4), target_base_bacc=round(base_t, 4),
                            target_after_subject=round(BACC(yt, head.predict(erase(Xt, Bsub))), 4),
                            target_after_variance=round(BACC(yt, head.predict(erase(Xt, Bvar))), 4),
                            delta_subject=round(base_t - BACC(yt, head.predict(erase(Xt, Bsub))), 4)))
    return l4_rows, l5_rows, l6_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--dataset", default="shu")
    args = ap.parse_args()
    z = np.load(OUT / "embeddings" / f"{args.model}_{args.dataset}_F0.npz")
    X, y, d, ses = z["X"].astype(np.float64), z["y"].astype(int), z["d"].astype(int), z["ses"].astype(int)
    tag = f"{args.model}_{args.dataset}"

    l1m = l1_subject_probe(X, y, d, ses, cond_on_y=False)
    l1c = l1_subject_probe(X, y, d, ses, cond_on_y=True)
    l4, l5, l6 = loso_audit(X, y, d)

    def w(fn, rows):
        if rows:
            with open(OUT / fn, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["model", "dataset"] + list(rows[0].keys()))
                if (OUT / fn).stat().st_size == 0:
                    wr.writeheader()
                for r in rows:
                    wr.writerow(dict(model=args.model, dataset=args.dataset, **r))
    w("l1_subject_probe.csv", [dict(estimand="marginal", **l1m), dict(estimand="class_conditional", **l1c)])
    w("l4_task_head_alignment.csv", l4)
    w("l5_subject_subspace_replay.csv", l5)
    w("l6_target_consequence.csv", l6)

    # summaries (per-dataset)
    l5p = [r for r in l5 if r["k"] == PRIMARY_K]
    ds_m, ds_ci = clu_ci([r["drop_subject"] for r in l5p]); dv_m, _ = clu_ci([r["drop_variance"] for r in l5p])
    l6d_m, l6d_ci = clu_ci([r["delta_subject"] for r in l6])
    l4p = [r["task_head_subject_alignment"] for r in l4 if r["k"] == PRIMARY_K and r["task_head_subject_alignment"] is not None]
    tbase, _ = clu_ci([r["target_base_bacc"] for r in l6])
    summary = dict(
        model=args.model, dataset=args.dataset, n_subjects=int(np.unique(d).size),
        classes=sorted(int(c) for c in np.unique(y)), task_chance=round(1.0 / len(np.unique(y)), 4),
        L1_marginal=l1m, L1_class_conditional=l1c,
        L4_task_subject_alignment_k2=round(float(np.mean(l4p)), 4) if l4p else None,
        L5_drop_subject_k2=[ds_m, ds_ci], L5_drop_variance_k2=dv_m,
        L5_subject_beats_variance=bool(ds_m is not None and dv_m is not None and ds_ci[0] is not None and ds_ci[0] > dv_m),
        L6_target_base_bacc=tbase, L6_delta_subject_k2=[l6d_m, l6d_ci],
        task_decodable_above_chance=bool(tbase is not None and tbase > 1.0 / len(np.unique(y)) + 0.02),
        note=("L1=subject decodability (session-held-out); L4=task-head/subject-subspace alignment; L5=held-out-"
              "source reliance vs variance-matched null; L6=target consequence (conservative: source-estimated "
              "subspace on novel target -> null L6 is NOT evidence subject info is task-irrelevant)."))
    (OUT / f"audit_summary_{tag}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
