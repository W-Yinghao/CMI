#!/usr/bin/env python
"""FSR Phase 8B — F1 spatial-preserving audit (see docs/FSR_46 + PM revision). F1 = per-channel encoder feature
(mean over patches, keep channels -> C*200) preserves MI C3/C4 lateralization that F0 pooling washes out. PCA to
95%-variance (cap 128) fit on SOURCE-TRAIN only; head selected on source-val only; NO target label in PCA/head/
selection. First a TASK GATE (source-val bAcc >= 0.60 and target bAcc >= 0.58, or target beats F0 by >= +0.04);
only if it passes are L4/L5/L6 promoted to main conclusions. L1 (subject decodability) reported regardless.

    python scripts/cb_cbm_8b_f1_audit.py --model codebrain --dataset shu --f0_target_bacc 0.530
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as BACC
import cb_cbm_8b_audit as A   # reuse top_k, subject_offsets, task_offsets, erase, var_frac_removed, clu_ci

OUT = Path("results/fsr_codebrain_cbramod_8b")
RNG = np.random.default_rng(0)
PCA_VAR, PCA_CAP, PRIMARY_K = 0.95, 128, 2


def pca_fit(Xtr):
    p = PCA(n_components=min(PCA_CAP, Xtr.shape[1], Xtr.shape[0] - 1)).fit(Xtr)
    dim = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), PCA_VAR) + 1)
    return p, min(dim, p.n_components_)


def zfit(Z):
    mu = Z.mean(0); sd = Z.std(0) + 1e-8
    return mu, sd


def l1_f1(X, y, d, ses):
    """within-subject session-held-out subject probe on PCA(F1); marginal + class-conditional; perm null."""
    subs = np.unique(d); chance = 1.0 / len(subs); out = {}
    for cond in (False, True):
        Xr = X.copy()
        if cond:
            for yy in np.unique(y):
                Xr[y == yy] -= Xr[y == yy].mean(0)
        accs, perms = [], []
        for te in np.unique(ses):
            tr = ses != te; teM = ses == te
            if len(np.unique(d[tr])) < len(subs) or teM.sum() < len(subs):
                continue
            p, dim = pca_fit(Xr[tr]); Ztr = p.transform(Xr[tr])[:, :dim]; Zte = p.transform(Xr[teM])[:, :dim]
            mu, sd = zfit(Ztr)
            clf = LDA().fit((Ztr - mu) / sd, d[tr]); pred = clf.predict((Zte - mu) / sd)
            accs.append(BACC(d[teM], pred)); dt = d[teM]
            perms.append([BACC(RNG.permutation(dt), pred) for _ in range(1000)])
        if accs:
            obs = float(np.mean(accs)); perm = np.array(perms).mean(0)
            out["class_conditional" if cond else "marginal"] = dict(
                bacc=round(obs, 4), chance=round(chance, 4), null_mean=round(float(perm.mean()), 4),
                p=round(float((perm >= obs).mean()), 4), effect=round(obs - float(perm.mean()), 4), n_folds=len(accs))
    return out


def loso_f1(X, y, d):
    subs = np.unique(d); tchance = 1.0 / len(np.unique(y))
    gate, l4, l5, l6 = [], [], [], []
    for tgt in subs:
        src = d != tgt; tg = d == tgt
        Xs, ys, ds = X[src], y[src], d[src]
        ssub = np.unique(ds).copy(); RNG.shuffle(ssub); val = set(ssub[:max(1, len(ssub) // 5)])
        vmask = np.array([s in val for s in ds]); tmask = ~vmask
        p, dim = pca_fit(Xs[tmask])
        Ztr = p.transform(Xs[tmask])[:, :dim]; Zval = p.transform(Xs[vmask])[:, :dim]; Ztg = p.transform(X[tg])[:, :dim]
        mu, sd = zfit(Ztr)
        def S(Z): return (Z - mu) / sd
        head = LDA().fit(S(Ztr), ys[tmask])
        sv = BACC(ys[vmask], head.predict(S(Zval))); tb = BACC(y[tg], head.predict(S(Ztg)))
        gate.append(dict(target_subject=int(tgt), pca_dim=int(dim), source_val_bacc=round(sv, 4), target_bacc=round(tb, 4)))
        W = head.coef_ if head.coef_.ndim == 2 else head.coef_.reshape(1, -1)
        Ztr_s = S(Ztr); subjM = A.subject_offsets(Ztr_s, ys[tmask], ds[tmask])
        for k in (1, 2, 4, 8):
            Bs = A.top_k(subjM, k)
            al = float(np.mean(np.max(np.abs((W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)) @ Bs.T), axis=1))) if Bs.shape[0] and W.shape[0] else None
            l4.append(dict(target_subject=int(tgt), k=k, task_head_subject_alignment=round(al, 4) if al is not None else None, subj_rank=int(Bs.shape[0])))
        k = PRIMARY_K
        Bs = A.top_k(subjM, k); Bv = A.top_k(Ztr_s - Ztr_s.mean(0), k); Bt = A.top_k(A.task_offsets(Ztr_s, ys[tmask]), k)
        Zv = S(Zval); yv = ys[vmask]; base = BACC(yv, head.predict(Zv))
        def drop(B): return round(base - BACC(yv, head.predict(A.erase(Zv, B))), 4)
        l5.append(dict(target_subject=int(tgt), k=k, base_bacc=round(base, 4), drop_subject=drop(Bs),
                       drop_variance=drop(Bv), drop_oracle_task=drop(Bt),
                       var_removed_subject=round(A.var_frac_removed(Zv, Bs), 4)))
        Zt = S(Ztg); yt = y[tg]; bt2 = BACC(yt, head.predict(Zt))
        l6.append(dict(target_subject=int(tgt), k=k, chance=round(tchance, 4), target_base_bacc=round(bt2, 4),
                       delta_subject=round(bt2 - BACC(yt, head.predict(A.erase(Zt, Bs))), 4),
                       delta_variance=round(bt2 - BACC(yt, head.predict(A.erase(Zt, Bv))), 4)))
    return gate, l4, l5, l6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--dataset", default="shu")
    ap.add_argument("--f0_target_bacc", type=float, default=None)
    ap.add_argument("--suffix", default="", help="e.g. _bs32 for the determinism-invariance dump")
    args = ap.parse_args()
    z = np.load(OUT / "embeddings" / f"{args.model}_{args.dataset}_F1{args.suffix}.npz")
    X, y, d, ses = z["X"].astype(np.float64), z["y"].astype(int), z["d"].astype(int), z["ses"].astype(int)

    l1 = l1_f1(X, y, d, ses)
    gate, l4, l5, l6 = loso_f1(X, y, d)

    def w(fn, rows):
        if rows:
            new = not (OUT / fn).exists() or (OUT / fn).stat().st_size == 0
            with open(OUT / fn, "a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["model", "dataset"] + list(rows[0].keys()))
                if new:
                    wr.writeheader()
                for r in rows:
                    wr.writerow(dict(model=args.model, dataset=args.dataset, **r))
    if not args.suffix:   # only the main bs64 run populates the shared CSVs
        w("f1_task_gate.csv", gate); w("f1_l1_subject_probe.csv", [dict(estimand=k, **v) for k, v in l1.items()])
        w("f1_l4_task_alignment.csv", l4); w("f1_l5_replay.csv", l5); w("f1_l6_target_consequence.csv", l6)

    sv, sv_ci = A.clu_ci([r["source_val_bacc"] for r in gate]); tb, tb_ci = A.clu_ci([r["target_bacc"] for r in gate])
    l5p = l5  # k=PRIMARY_K only
    ds_m, ds_ci = A.clu_ci([r["drop_subject"] for r in l5p]); dv_m, _ = A.clu_ci([r["drop_variance"] for r in l5p])
    l6d, l6d_ci = A.clu_ci([r["delta_subject"] for r in l6])
    l4p = [r["task_head_subject_alignment"] for r in l4 if r["k"] == PRIMARY_K and r["task_head_subject_alignment"] is not None]
    tchance = round(1.0 / len(np.unique(y)), 4)
    beats_f0 = (args.f0_target_bacc is not None and tb is not None and tb - args.f0_target_bacc >= 0.04 and (tb_ci[0] is not None and tb_ci[0] > tchance))
    task_gate_pass = bool((sv is not None and sv >= 0.60 and tb is not None and tb >= 0.58) or beats_f0)
    summary = dict(
        model=args.model, dataset=args.dataset, feature="F1_spatial_pca95cap128", n_subjects=int(np.unique(d).size),
        task_chance=tchance, source_val_bacc=[sv, sv_ci], target_bacc=[tb, tb_ci],
        f0_target_bacc=args.f0_target_bacc, target_beats_f0_by_0p04=bool(beats_f0), task_gate_pass=task_gate_pass,
        L1_marginal=l1.get("marginal"), L1_class_conditional=l1.get("class_conditional"),
        L4_alignment_k2=round(float(np.mean(l4p)), 4) if l4p else None,
        L5_drop_subject_k2=[ds_m, ds_ci], L5_drop_variance_k2=dv_m,
        L5_subject_beats_variance=bool(ds_m is not None and dv_m is not None and ds_ci[0] is not None and ds_ci[0] > dv_m),
        L6_target_base_bacc=[tb, tb_ci], L6_delta_subject_k2=[l6d, l6d_ci],
        note=("F1 spatial audit. L4/L5/L6 are MAIN conclusions only if task_gate_pass=True; else task still too weak "
              "and only L1 (subject decodability) is reportable. PCA 95%/cap128 on source-train; head source-val; "
              "no target-label in PCA/head/selection."))
    (OUT / f"f1_audit_summary_{args.model}_{args.dataset}{args.suffix}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
