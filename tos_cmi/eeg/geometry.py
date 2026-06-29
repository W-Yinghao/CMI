"""Phase 2.0 Experiment 3 -- ERM-vs-LPC feature GEOMETRY on the frozen dumps (CPU). Tests the
core TOS-CMI hypothesis directly: is the domain-rich (subject-decodable, label-light) subspace
ENTANGLED with the task subspace, and does global LPC compress task-overlapping leakage?

Per dump (source Z), on the whitening metric M = (Sigma_W + eps Sigma_ref)^-1:
  - task carrier  T   = label score-Fisher top eigenspace (estimate_task_basis)
  - domain-rich   V_D = domain|Y score-Fisher candidate directions (candidate_order, label-light)
  - cos2(V_D, T)  = M-metric overlap of domain-rich leakage with the task subspace (HIGH => the
                    leakage the gate would delete sits IN the task subspace => deletion costs task)
  - domain decodability THROUGH the task projection P_T Z  vs the complement (I-P_T) Z: does the
    subject identity leak through the task-carrying directions?
Across lambda this shows whether global LPC removes the SAFE (non-overlapping) domain directions
first (leaving entangled leakage -> collapse) or attacks task-overlapping directions (-> task loss).
"""
from __future__ import annotations
import glob
import json
import sys
import numpy as np

from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _cross_fit_fisher, _SplitPlan,
                                  candidate_order, estimate_task_basis, _m_orthonormal)


def _m_cos2(A, B, M):
    """Mean cos^2 of principal angles between span(A), span(B) in the M-inner-product."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    Qa = _m_orthonormal(A, M); Qb = _m_orthonormal(B, M)
    s = np.linalg.svd(Qa.T @ M @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def _domain_acc(Z, dom, seed=0, frac=0.5):
    try:
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(seed); idx = rng.permutation(len(dom))
        cut = int(frac * len(dom)); tr, te = idx[:cut], idx[cut:]
        if len(np.unique(dom[tr])) < 2:
            return float("nan")
        return float((LogisticRegression(max_iter=300).fit(Z[tr], dom[tr]).predict(Z[te]) == dom[te]).mean())
    except Exception:
        return float("nan")


def analyze_geometry(npz_path, cfg=None):
    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z_source"].astype(np.float64); y = d["y_source"]; dom = d["domain_source"]
    n_cls = int(d["n_cls"]); n_dom = int(d["n_dom_source"]); z_dim = Z.shape[1]
    cfg = cfg or ScoreFisherConfig()
    y_oh = np.eye(n_cls)[y]
    plan = _SplitPlan(len(y), cfg.n_folds, 1)
    M = _metric(Z, y, n_cls, cfg)
    G_Y = _cross_fit_fisher(Z, y, None, n_cls, z_dim, 0, cfg, plan, 0)
    G_DgY = _cross_fit_fisher(Z, dom, y_oh, n_dom, z_dim, n_cls, cfg, plan, 100)
    T = estimate_task_basis(G_Y, M, cfg, energy_frac=0.9)            # task carrier
    V_D = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]                # domain-rich, label-light
    # domain decodability through the task projection vs its complement (Euclidean proj on T span)
    Tb = _m_orthonormal(T, M) if T.shape[1] else T
    P_T = Tb @ Tb.T @ M if Tb.shape[1] else np.zeros((z_dim, z_dim))  # M-oblique proj onto span(T)
    Z_T = Z @ P_T.T; Z_perp = Z - Z_T
    return {"tag": npz_path.split("/")[-1], "method": str(d["method"]), "lam": float(d["lam"]),
            "target_subject": int(d["target_subject"]),
            "task_dim": int(T.shape[1]), "n_domain_candidates": int(V_D.shape[1]),
            "cos2_domainrich_task": _m_cos2(V_D, T, M),       # KEY: leakage<->task overlap
            "domain_acc_full": _domain_acc(Z, dom), "domain_acc_via_task": _domain_acc(Z_T, dom),
            "domain_acc_via_complement": _domain_acc(Z_perp, dom), "domain_chance": 1.0 / n_dom}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO"
    paths = sorted(glob.glob("%s/*.npz" % base))
    rows = []
    for p in paths:
        try:
            r = analyze_geometry(p); rows.append(r)
            print("[%s] tgt%d lam=%g | task_dim=%d nDcand=%d cos2(Dom,Task)=%.3f | "
                  "domAcc full=%.2f viaTask=%.2f viaPerp=%.2f (chance %.2f)"
                  % (r["method"], r["target_subject"], r["lam"], r["task_dim"],
                     r["n_domain_candidates"], r["cos2_domainrich_task"], r["domain_acc_full"],
                     r["domain_acc_via_task"], r["domain_acc_via_complement"], r["domain_chance"]),
                  flush=True)
        except Exception as e:
            print("[FAIL] %s : %r" % (p.split("/")[-1], e), flush=True)
    # pooled over folds by (method, lam)
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["lam"])].append(r)
    agg = {}
    for (m, lam), rs in sorted(by.items(), key=lambda kv: (kv[0][0] != "erm", kv[0][1])):
        mean = lambda k: float(np.nanmean([x[k] for x in rs]))
        agg["%s:%g" % (m, lam)] = {"n": len(rs), "cos2_domainrich_task": mean("cos2_domainrich_task"),
                                   "n_domain_candidates": mean("n_domain_candidates"),
                                   "domain_acc_via_task": mean("domain_acc_via_task"),
                                   "domain_acc_via_complement": mean("domain_acc_via_complement")}
    print("\n===== GEOMETRY pooled (ERM vs LPC) =====")
    for cfg_, a in agg.items():
        print("%-16s n=%d | cos2(Dom,Task)=%.3f nDcand=%.1f domAcc viaTask=%.2f viaPerp=%.2f"
              % (cfg_, a["n"], a["cos2_domainrich_task"], a["n_domain_candidates"],
                 a["domain_acc_via_task"], a["domain_acc_via_complement"]))
    json.dump({"rows": rows, "aggregate": agg}, open("%s/geometry_report.json" % base, "w"), indent=1)
    print("GEOMETRY_DONE")


if __name__ == "__main__":
    main()
