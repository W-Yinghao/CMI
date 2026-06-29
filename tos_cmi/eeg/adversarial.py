"""Phase 2.0 adversarial check #1 -- is the 'domain' leakage SUBJECT identity or SESSION /
recording-condition? Per ERM dump: select the domain-rich/label-light V_D on trial-split A using
the deployed subject-domain, then on disjoint B probe (MLP) {subject, session, subject_session}
from the full Z and the complement RZ=(I-P_N)Z. If SESSION is what is decodable (and from RZ), the
write-up must say 'recording-condition leakage', not 'subject leakage'. Also confirms whether ANY
domain definition is removed by deleting V_D (it is not, per the main ablation)."""
from __future__ import annotations
import glob
import json
import os
import sys
import numpy as np

from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _cross_fit_fisher, _SplitPlan,
                                  candidate_order, _m_proj)


def _ids(arr):
    u = {v: i for i, v in enumerate(sorted(set(map(str, arr))))}
    return np.array([u[str(v)] for v in arr]), len(u)


def _probe(Ztr, ltr, Zte, lte):
    try:
        if len(np.unique(ltr)) < 2:
            return float("nan")
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0).fit(Ztr, ltr)
        return float((clf.predict(Zte) == lte).mean())
    except Exception:
        return float("nan")


def analyze(npz_path, cfg=None, seed=0):
    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z_source"].astype(np.float64); y = d["y_source"]
    subj = d["subject_source"]; sess = d["session_source"]
    n_cls = int(d["n_cls"]); z_dim = Z.shape[1]
    dom_subj, n_subj = _ids(subj); dom_sess, n_sess = _ids(sess)
    dom_ss, n_ss = _ids([f"{a}|{b}" for a, b in zip(subj, sess)])
    cfg = cfg or ScoreFisherConfig()
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y)); cut = len(y) // 2
    A = np.zeros(len(y), bool); A[perm[:cut]] = True; B = ~A
    plan = _SplitPlan(int(A.sum()), cfg.n_folds, 1)
    M = _metric(Z[A], y[A], n_cls, cfg)
    G_Y = _cross_fit_fisher(Z[A], y[A], None, n_cls, z_dim, 0, cfg, plan, 0)
    G_DgY = _cross_fit_fisher(Z[A], dom_subj[A], np.eye(n_cls)[y[A]], n_subj, z_dim, n_cls, cfg, plan, 100)
    V_D = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]
    k = int(V_D.shape[1])
    out = {"tag": npz_path.split("/")[-1], "method": str(d["method"]), "lam": float(d["lam"]),
           "target_subject": int(d["target_subject"]), "nDcand": k,
           "n_subj": n_subj, "n_sess": n_sess, "chance_subj": 1.0 / n_subj, "chance_sess": 1.0 / n_sess}
    RZ = Z if k == 0 else Z - Z @ _m_proj(V_D, M).T
    for nm, dom in [("subj", dom_subj), ("sess", dom_sess), ("ss", dom_ss)]:
        out["%s_Z" % nm] = _probe(Z[A], dom[A], Z[B], dom[B])
        out["%s_RZ" % nm] = _probe(RZ[A], dom[A], RZ[B], dom[B])
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO"
    paths = [p for p in sorted(glob.glob("%s/*.npz" % base)) if "_erm_" in p]
    sd = os.environ.get("TOS_SEED_FILTER")
    if sd:
        paths = [p for p in paths if p.endswith("_seed%s.npz" % sd)]
    rows = []
    for p in paths:
        try:
            r = analyze(p); rows.append(r)
            print("[erm tgt%d] subj Z=%.2f RZ=%.2f (ch %.2f) | sess Z=%.2f RZ=%.2f (ch %.2f) | "
                  "subj|sess Z=%.2f RZ=%.2f" % (r["target_subject"], r["subj_Z"], r["subj_RZ"],
                  r["chance_subj"], r["sess_Z"], r["sess_RZ"], r["chance_sess"], r["ss_Z"], r["ss_RZ"]),
                  flush=True)
        except Exception as e:
            print("[FAIL] %s : %r" % (p.split("/")[-1], e), flush=True)
    if rows:
        m = lambda k: float(np.nanmean([r[k] for r in rows]))
        print("\n===== ADVERSARIAL domain-definition (ERM, pooled) =====")
        print("subject:  Z=%.2f RZ=%.2f (chance %.2f)" % (m("subj_Z"), m("subj_RZ"), m("chance_subj")))
        print("session:  Z=%.2f RZ=%.2f (chance %.2f)" % (m("sess_Z"), m("sess_RZ"), m("chance_sess")))
        print("subj|sess:Z=%.2f RZ=%.2f" % (m("ss_Z"), m("ss_RZ")))
        json.dump(rows, open("%s/adversarial_report.json" % base, "w"), indent=1)
    print("ADVERSARIAL_DONE")


if __name__ == "__main__":
    main()
