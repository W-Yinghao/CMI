"""Phase 2.0 -- PROJECTION ABLATION on frozen features (the removability test; stronger than
cos2=0). Selects the domain-rich/label-light candidate subspace V_D on a DISJOINT source split A
(group-aware by subject), forms P_N = proj onto span(V_D), R = I - P_N, and on the held-out split B
trains+tests task (Y) and domain (subject) heads -- linear AND MLP -- on each representation:

    Z      (full)            RZ = (I-P_N)Z  (kept)        P_N Z (deleted)        random-k (control)

Removability is supported iff:  task(RZ) ~ task(Z)  AND  domain(RZ) << domain(Z)  AND
task(P_N Z) ~ chance  AND  domain(P_N Z) high  AND  random-k does NOT match P_N (i.e. P_N is special).
Selection on A / evaluation on B kills the 'label-light by construction' circularity: V_D is found
on A but its task/domain content is judged on independent B. This is a DIAGNOSTIC (no encoder
training, no certified deletion -- EEG has no exact-scope power certificate).
"""
from __future__ import annotations
import glob
import json
import sys
import numpy as np

from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _cross_fit_fisher, _SplitPlan,
                                  candidate_order, _m_proj, _m_orthonormal)


def _probe(Ztr, ltr, Zte, lte, family="linear", seed=0):
    """Held-out accuracy of a task/domain head; chance-aware caller. linear=multinomial logistic,
    mlp=1-hidden MLP. Returns acc or nan."""
    try:
        if len(np.unique(ltr)) < 2:
            return float("nan")
        if family == "mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=300)
        clf.fit(Ztr, ltr)
        return float((clf.predict(Zte) == lte).mean())
    except Exception:
        return float("nan")


def ablate(npz_path, cfg=None, seed=0):
    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z_source"].astype(np.float64); y = d["y_source"]; dom = d["domain_source"]
    subj = d["subject_source"]; n_cls = int(d["n_cls"]); n_dom = int(d["n_dom_source"]); z_dim = Z.shape[1]
    cfg = cfg or ScoreFisherConfig()
    rng = np.random.default_rng(seed)
    # TRIAL-level A/B split (all source subjects in BOTH): V_D selected on A, task+domain heads
    # trained on A and judged on disjoint trials B. Disjoint TRIALS kill the 'label-light by
    # construction' circularity (V_D's task/domain content is judged out-of-sample); shared
    # subjects keep the domain(=subject) probe well-posed (a group split would make A/B subjects
    # disjoint -> domain probe trained on A cannot predict B's subjects).
    perm = rng.permutation(len(y)); cut = len(y) // 2
    A = np.zeros(len(y), bool); A[perm[:cut]] = True; B = ~A
    planA = _SplitPlan(int(A.sum()), cfg.n_folds, 1)
    M = _metric(Z[A], y[A], n_cls, cfg)
    G_Y = _cross_fit_fisher(Z[A], y[A], None, n_cls, z_dim, 0, cfg, planA, 0)
    G_DgY = _cross_fit_fisher(Z[A], dom[A], np.eye(n_cls)[y[A]],   # conditioner = one-hot Y
                              n_dom, z_dim, n_cls, cfg, planA, 100)
    V_D = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]               # domain-rich, label-light (on A)
    k = int(V_D.shape[1])
    out = {"tag": npz_path.split("/")[-1], "method": str(d["method"]), "lam": float(d["lam"]),
           "target_subject": int(d["target_subject"]), "nDcand": k,
           "label_chance": 1.0 / n_cls, "domain_chance_B": 1.0 / max(len(np.unique(dom[B])), 1)}
    if k == 0:
        out["note"] = "no_candidate"; return out
    PN = _m_proj(V_D, M)                                            # M-oblique proj onto span(V_D)
    Vr = _m_orthonormal(rng.standard_normal((z_dim, k)), M)         # random-k control
    PNr = _m_proj(Vr, M)
    reps = {"Z": Z, "RZ": Z - Z @ PN.T, "PNZ": Z @ PN.T,
            "Rrand": Z - Z @ PNr.T, "PNrand": Z @ PNr.T}
    # probe-train on A-projected, test on B-projected (V_D fixed from A)
    for name, Zr in reps.items():
        for fam in ("linear", "mlp"):
            out["task_%s_%s" % (name, fam)] = _probe(Zr[A], y[A], Zr[B], y[B], fam, seed)
            out["domain_%s_%s" % (name, fam)] = _probe(Zr[A], dom[A], Zr[B], dom[B], fam, seed)
    # headline deltas (MLP heads, the strict test)
    out["task_delta_RZ_mlp"] = out["task_Z_mlp"] - out["task_RZ_mlp"]        # ~0 desired
    out["domain_delta_RZ_mlp"] = out["domain_Z_mlp"] - out["domain_RZ_mlp"]  # large desired
    out["task_PNZ_mlp"] = out["task_PNZ_mlp"]; out["domain_PNZ_mlp"] = out["domain_PNZ_mlp"]
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO"
    only = sys.argv[2] if len(sys.argv) > 2 else "erm"             # default ablate ERM dumps
    paths = [p for p in sorted(glob.glob("%s/*.npz" % base)) if ("_%s_" % only) in p or only == "all"]
    rows = []
    for p in paths:
        try:
            r = ablate(p); rows.append(r)
            if r.get("nDcand", 0) == 0:
                print("[%s] tgt%d lam=%g | nDcand=0 (no candidate)" % (r["method"], r["target_subject"], r["lam"]), flush=True)
            else:
                print("[%s] tgt%d lam=%g k=%d | task Z=%.2f RZ=%.2f PNZ=%.2f (chance %.2f) | "
                      "domain Z=%.2f RZ=%.2f PNZ=%.2f randR=%.2f (chance %.2f) [MLP]"
                      % (r["method"], r["target_subject"], r["lam"], r["nDcand"],
                         r["task_Z_mlp"], r["task_RZ_mlp"], r["task_PNZ_mlp"], r["label_chance"],
                         r["domain_Z_mlp"], r["domain_RZ_mlp"], r["domain_PNZ_mlp"],
                         r["domain_Rrand_mlp"], r["domain_chance_B"]), flush=True)
        except Exception as e:
            print("[FAIL] %s : %r" % (p.split("/")[-1], e), flush=True)
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["lam"])].append(r)
    agg = {}
    for (m, lam), rs in sorted(by.items(), key=lambda kv: (kv[0][0] != "erm", kv[0][1])):
        ok = [r for r in rs if r.get("nDcand", 0) > 0]
        mean = lambda k: (float(np.nanmean([r[k] for r in ok])) if ok else float("nan"))
        agg["%s:%g" % (m, lam)] = {"n": len(rs), "n_with_cand": len(ok),
            "task_Z": mean("task_Z_mlp"), "task_RZ": mean("task_RZ_mlp"), "task_PNZ": mean("task_PNZ_mlp"),
            "domain_Z": mean("domain_Z_mlp"), "domain_RZ": mean("domain_RZ_mlp"),
            "domain_PNZ": mean("domain_PNZ_mlp"), "domain_Rrand": mean("domain_Rrand_mlp")}
    print("\n===== PROJECTION ABLATION pooled (MLP heads) =====")
    for cfg_, a in agg.items():
        print("%-14s n=%d cand=%d | task Z=%.2f RZ=%.2f PNZ=%.2f | dom Z=%.2f RZ=%.2f PNZ=%.2f randR=%.2f"
              % (cfg_, a["n"], a["n_with_cand"], a["task_Z"], a["task_RZ"], a["task_PNZ"],
                 a["domain_Z"], a["domain_RZ"], a["domain_PNZ"], a["domain_Rrand"]))
    json.dump({"rows": rows, "aggregate": agg}, open("%s/ablation_report.json" % base, "w"), indent=1)
    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
