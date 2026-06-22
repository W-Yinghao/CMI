"""V2 smoke: does the H2 encoder LEARN real MI left/right, and does the operator/EA path run?
Trains a source model on BNCI2014_001-LR session 0 (a few subjects) and reports identity / pooled /
EA balanced accuracy on session 1 (cross-session). De-risk only -- not a frozen result."""
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import balanced_accuracy_score

from h2cmi.config import core_config, H2Config
from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.train.trainer import train_h2, reference_prior, H2Model
from h2cmi.eval.harness import _embed, _predict_generative, _predict_transform
from h2cmi.tta.class_conditional import (ClassConditionalTTA, B1A_VARIANTS_BY_NAME,
                                         reference_weighted_source_moments)
from h2cmi.data.real_eeg import load_dataset, contiguous_split, N_TIMES, FS
from h2cmi.eval.ea import reference_cov, ea_transform, apply_ea


def build_site_domains(subject):
    subs = np.unique(subject); smap = {int(s): i for i, s in enumerate(subs)}
    site = np.array([smap[int(s)] for s in subject], np.int64)
    dag = DomainDAG([DomainFactor("site", len(subs), (), "invariant", 0.02)])
    return dag, DomainLabels(dag, site.reshape(-1, 1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    subs = [int(x) for x in args.subjects.split(",")]

    ep = load_dataset("BNCI2014_001", subs)
    print(f"loaded BNCI2014_001-LR: X{ep.X.shape} y_balance={np.bincount(ep.y)} "
          f"sessions={ep.session_names} chans={len(ep.channels)}", flush=True)
    s0 = ep.session == 0
    s1 = ep.session == 1
    Xs, ys, subj_s = ep.X[s0], ep.y[s0], ep.subject[s0]
    Xt, yt, subj_t = ep.X[s1], ep.y[s1], ep.subject[s1]

    cfg = core_config(H2Config(n_classes=2))
    cfg.encoder.n_chans = ep.X.shape[1]; cfg.encoder.n_times = N_TIMES; cfg.encoder.fs = FS
    cfg.train.epochs = args.epochs; cfg.train.device = args.device; cfg.train.seed = 0
    cfg.cmi.enabled = False
    dag, dom = build_site_domains(subj_s)
    pi_star = reference_prior(ys, 2, "uniform")
    model, *_ = train_h2(Xs, ys, dom, dag, cfg, align_factor="site")

    # cross-session identity bAcc (pooled over subjects)
    U_t = _embed(model, Xt, args.device)
    uni = np.full(2, 0.5)
    p_id = _predict_generative(model, U_t, uni)
    bacc_id = balanced_accuracy_score(yt, p_id.argmax(1))
    # in-distribution check: identity on session-0 itself
    U_s = _embed(model, Xs, args.device)
    bacc_src = balanced_accuracy_score(ys, _predict_generative(model, U_s, uni).argmax(1))
    print(f"[learnability] source-session bAcc={bacc_src:.3f}  cross-session identity bAcc={bacc_id:.3f}", flush=True)

    # operator path: per-subject pooled (fit on adapt half of session1, apply to eval half)
    pooled_ref = reference_weighted_source_moments(U_s, ys, pi_star)
    tta = ClassConditionalTTA(model.head.density, pi_star, cfg.tta, 2, args.device)
    d_pool, d_ea = [], []
    R_src = reference_cov(Xs)
    for s in subs:
        a, e = contiguous_split(ep, s, 1)
        if len(a) < cfg.tta.min_target or len(e) < 4:
            continue
        Ua, Ue = _embed(model, ep.X[a], args.device), _embed(model, ep.X[e], args.device)
        ye = ep.y[e]
        bid = balanced_accuracy_score(ye, _predict_generative(model, Ue, uni).argmax(1))
        fit = tta.fit_variant(Ua, B1A_VARIANTS_BY_NAME["pooled_empirical_diag"], pooled_ref=pooled_ref, tta_seed=1)
        bpool = balanced_accuracy_score(ye, _predict_transform(model, Ue, fit.transform, uni).argmax(1))
        d_pool.append(bpool - bid)
        # EA: target ref from adapt block raw trials, apply frozen transform to eval raw trials
        M = ea_transform(R_src, reference_cov(ep.X[a]))
        Ue_ea = _embed(model, apply_ea(ep.X[e], M), args.device)
        bea = balanced_accuracy_score(ye, _predict_generative(model, Ue_ea, uni).argmax(1))
        d_ea.append(bea - bid)
    print(f"[operators] pooled Δ={np.mean(d_pool):+.3f} (n={len(d_pool)})  EA Δ={np.mean(d_ea):+.3f}", flush=True)
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
