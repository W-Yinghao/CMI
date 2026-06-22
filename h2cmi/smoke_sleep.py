"""W2 sleep smoke: does the H2 encoder learn 5-class Sleep-EDF staging from Fpz-Cz+Pz-Oz, and is
there natural per-subject stage-prevalence variation? Train on a subject set, eval on unseen subjects.
De-risk only."""
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from h2cmi.config import core_config, H2Config
from h2cmi.domains import DomainDAG, DomainFactor, DomainLabels
from h2cmi.train.trainer import train_h2, reference_prior
from h2cmi.eval.harness import _embed, _predict_generative
from h2cmi.data.sleep_eeg import load_subjects, subject_list, SLEEP_N_TIMES, SLEEP_FS, STAGE_NAMES


def sleep_cfg(epochs, device):
    cfg = core_config(H2Config(n_classes=5))
    cfg.encoder.n_chans = 2; cfg.encoder.n_times = SLEEP_N_TIMES; cfg.encoder.fs = SLEEP_FS
    cfg.encoder.use_spd = False; cfg.encoder.use_graph = False; cfg.encoder.use_temporal = True
    cfg.train.epochs = epochs; cfg.train.device = device; cfg.train.seed = 0
    cfg.cmi.enabled = False
    return cfg


def _domains(subject):
    subs = np.unique(subject); smap = {int(s): i for i, s in enumerate(subs)}
    site = np.array([smap[int(s)] for s in subject], np.int64)
    dag = DomainDAG([DomainFactor("site", max(1, len(subs)), (), "invariant", 0.02)])
    return dag, DomainLabels(dag, site.reshape(-1, 1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    subs = subject_list()
    tr_subs = subs[:args.n_train]; te_subs = subs[args.n_train:args.n_train + args.n_test]
    print(f"sleep subjects total={len(subs)}; train={tr_subs} test={te_subs}", flush=True)
    tr = load_subjects(tr_subs)
    print(f"train X{tr.X.shape} stage counts={np.bincount(tr.y, minlength=5).tolist()} ({STAGE_NAMES})", flush=True)
    cfg = sleep_cfg(args.epochs, args.device)
    dag, dom = _domains(tr.subject)
    model, *_ = train_h2(tr.X, tr.y, dom, dag, cfg, align_factor="site")
    pi_unif = np.full(5, 0.2)
    # in-distribution check
    U_tr = _embed(model, tr.X, args.device)
    b_tr = balanced_accuracy_score(tr.y, _predict_generative(model, U_tr, pi_unif).argmax(1))
    print(f"[learnability] train bAcc(5-class)={b_tr:.3f}", flush=True)
    # unseen-subject + per-subject prevalence
    te = load_subjects(te_subs)
    for s in te_subs:
        m = te.subject == s
        if m.sum() == 0:
            continue
        U = _embed(model, te.X[m], args.device); yy = te.y[m]
        p = _predict_generative(model, U, pi_unif).argmax(1)
        prev = np.bincount(yy, minlength=5) / len(yy)
        print(f"  subj {s}: n={m.sum()} bAcc={balanced_accuracy_score(yy,p):.3f} "
              f"macroF1={f1_score(yy,p,average='macro'):.3f} prevalence={np.round(prev,2).tolist()}", flush=True)
    print("SLEEP_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
