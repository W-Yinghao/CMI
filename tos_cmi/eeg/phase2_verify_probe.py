"""Phase 2 VERIFICATION probe -- settles whether the exact +0.000 target ΔbAcc of the deployable
task-preserving erasers (tp_leace, cc_leace_predicted) is a REAL mechanism (task-carrier / argmax
invariance) or an artifact (identity application, delta-math bug, masked non-zero). For a few folds it prints:

  * rel_change = ||Z_t - E(Z_t)||_F / ||Z_t||_F              -> >0 proves the eraser is NOT identity
  * src subject decode full->eras (stratified split)          -> confirms subject IS erased on source
  * target bAcc full vs eras with a LogReg head AND an MLP head-> robustness of the exact-zero
  * changed-argmax fraction on target (LogReg)                -> 0 => predictions literally unchanged
  * target NLL full vs eras                                    -> representation change even when argmax fixed
  * oracle (true-label routing) target bAcc                    -> the label-leakage upper bound
  python -m tos_cmi.eeg.phase2_verify_probe
"""
from __future__ import annotations
import glob
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss

from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eeg.erasure_baselines import _ids
from tos_cmi.eeg.source_ood_benefit_gate import build_eraser, _subj_acc
from tos_cmi.eeg.task_preserving_linear_erasure import tp_leace_factory
from tos_cmi.eeg.class_conditional_leace import cc_leace_factory_predicted, cc_leace_apply_oracle

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"


def _bacc_head(Ztr, ytr, Zte, yte, head):
    h = head.fit(Ztr, ytr)
    return float(balanced_accuracy_score(yte, h.predict(Zte))), h


def probe(p, seed=0):
    d = np.load(p, allow_pickle=True)
    Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
    Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
    subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
    tag = p.split("/")[-2].replace("_LOSO", "") + "/" + re.search(r"(sub\d+)_", p).group(1)
    erasers = {
        "leace_baseline": build_eraser(Zs, ys, subj, n_cls, "LEACE", ScoreFisherConfig(), seed),
        "tp_leace": tp_leace_factory(Zs, ys, subj, n_cls, seed),
        "cc_predicted": cc_leace_factory_predicted(Zs, ys, subj, n_cls, seed),
    }
    # source stratified split for subject-decode-after
    rng = np.random.default_rng(seed); perm = rng.permutation(len(ys)); cut = len(ys) // 2
    A = np.zeros(len(ys), bool); A[perm[:cut]] = True; Bm = ~A
    lr = lambda: LogisticRegression(max_iter=200, C=1.0)
    mlp = lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=0)
    bacc_full_lr, hf = _bacc_head(Zs, ys, Zt, yt, lr())
    bacc_full_mlp, _ = _bacc_head(Zs, ys, Zt, yt, mlp())
    pred_full = hf.predict(Zt)
    nll_full = log_loss(yt, _pad(hf, Zt, n_cls), labels=np.arange(n_cls))
    subj_full = _subj_acc(Zs[A], subj[A], Zs[Bm], subj[Bm])
    print("\n[%s] n_cls=%d  |Zs|=%d |Zt|=%d  full: LR bAcc=%.3f MLP bAcc=%.3f subj(src)=%.2f"
          % (tag, n_cls, len(ys), len(yt), bacc_full_lr, bacc_full_mlp, subj_full), flush=True)
    for nm, E in erasers.items():
        rel = float(np.linalg.norm(Zt - E(Zt)) / (np.linalg.norm(Zt) + 1e-12))
        bacc_e_lr, he = _bacc_head(E(Zs), ys, E(Zt), yt, lr())
        bacc_e_mlp, _ = _bacc_head(E(Zs), ys, E(Zt), yt, mlp())
        pred_e = he.predict(E(Zt))
        chg = float(np.mean(pred_e != pred_full))
        nll_e = log_loss(yt, _pad(he, E(Zt), n_cls), labels=np.arange(n_cls))
        subj_e = _subj_acc(E(Zs[A]), subj[A], E(Zs[Bm]), subj[Bm])
        print("   %-14s relchg=%.3f | subj(src) %.2f->%.2f | tgt bAcc LR %.3f (Δ%+.3f) MLP %.3f (Δ%+.3f) "
              "| argmax-changed %.1f%% | tgt NLL %.3f (Δ%+.3f)"
              % (nm, rel, subj_full, subj_e, bacc_e_lr, bacc_e_lr - bacc_full_lr,
                 bacc_e_mlp, bacc_e_mlp - bacc_full_mlp, 100 * chg, nll_e, nll_e - nll_full), flush=True)
    # oracle: route by TRUE labels (label-leakage upper bound)
    apF = cc_leace_apply_oracle(Zs, ys, subj, n_cls)
    bo, _ = _bacc_head(apF(Zs, ys), ys, apF(Zt, yt), yt, lr())
    print("   %-14s tgt bAcc LR %.3f (Δ%+.3f)  [routes target by TRUE yt -> NOT deployable]"
          % ("cc_ORACLE", bo, bo - bacc_full_lr), flush=True)


def _pad(h, X, n_cls):
    p = h.predict_proba(X); P = np.zeros((len(X), n_cls)); P[:, h.classes_] = p; return P


def main():
    for ds in ["Lee2019_MI", "Cho2017"]:
        ps = sorted(glob.glob("%s/%s_EEGNet_LOSO/sub*_erm_lam0_seed0.npz" % (RESULTS, ds)),
                    key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))[:3]
        for p in ps:
            probe(p)
    print("\nPHASE2_VERIFY_DONE")


if __name__ == "__main__":
    main()
