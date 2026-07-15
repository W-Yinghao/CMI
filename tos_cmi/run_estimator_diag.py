"""Phase 1.3 diagnostic -- classify the nested critic's synergy UNDER-DETECTION (the smoke found
the gate ACCEPTS the Bayes-unsafe injected span). For the failing config (synergy +
oracle_nuisance, oracle-T) sweep sample-size x critic-capacity/optimization and report, per prefix
k, probe_task_gain_ucb vs the exact Bayes Delta* (the gap is the under-detection). The pattern
classifies the failure:

  gap -> 0 as n grows               => FINITE-SAMPLE
  gap closes with hidden/epochs     => CAPACITY / OPTIMIZATION
  gap persists under both           => STRUCTURAL (critic class can't represent explaining-away)

  python -m tos_cmi.run_estimator_diag
"""
import json
import numpy as np
import torch

torch.set_num_threads(1)

from dataclasses import replace
from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _m_orthonormal, ucb_rank_gate)
from tos_cmi.data.synthetic import make_partial_synergy
from tos_cmi.eval.bayes_oracle import bayes_conditional_task_delta, classify_safety
from tos_cmi.score_fisher import task_protected_projector


def one(n, hidden, epochs, seed=0, tm=2.0, dm=2.6):
    data = make_partial_synergy(n=n, sep_label=tm, sep_safe=dm, sep_over=0.5 * dm, seed=seed)
    s = data["spec"]; n_cls, n_dom = s.n_cls, s.n_dom
    Z = data["Z"].astype(np.float64); y = data["y"]; d = data["d"]; truth = data["truth"]
    cfg = replace(ScoreFisherConfig(), task_protect=True, hidden=hidden, epochs=epochs,
                  gate_boot=200, n_perm_null=2)
    M = _metric(Z, y, n_cls, cfg)
    V = _m_orthonormal(data["nuisance_basis"].astype(np.float64), M)   # inject Bayes-unsafe span
    T = _m_orthonormal(data["task_overlap_basis"].astype(np.float64), M)
    kstar, grecs, _ = ucb_rank_gate(Z, y, d, V, M, n_cls, n_dom, cfg, seed, T_task=T)
    out = []
    for g in grecs:
        if "task_info_ucb" not in g:
            continue
        P, _ = task_protected_projector(V[:, :g["k"]], T, M)
        b = bayes_conditional_task_delta(truth["mu_yd"], truth["sigma"], truth["py"],
                                         truth["pdy"], P, n_mc=15000, seed=seed)
        verdict = classify_safety(b["ci_lo"], b["ci_hi"], cfg.delta_Y)
        out.append({"n": n, "hidden": hidden, "epochs": epochs, "k": g["k"],
                    "probe_ucb": round(g["probe_task_gain_ucb"], 4),
                    "probe_delta": round(g["task_info_delta_mean"], 4),
                    "bayes": round(b["delta"], 4), "gap": round(b["delta"] - g["probe_task_gain_ucb"], 4),
                    "accepted": bool(g["risk_feasible"]), "verdict": verdict,
                    "unsafe_accept": bool(g["risk_feasible"]) and verdict == "UNSAFE"})
    return out


def main():
    rows = []
    configs = [(64, 200), (256, 600)]                        # capacity / optimization
    for n in [2000, 6000, 18000]:                            # finite-sample
        for hidden, epochs in configs:
            r = one(n, hidden, epochs)
            rows.extend(r)
            for x in r:
                print("n=%-6d h=%-3d ep=%-3d k=%d | probe_ucb=%.4f bayes=%.4f gap=%.4f acc=%s %s%s"
                      % (n, hidden, epochs, x["k"], x["probe_ucb"], x["bayes"], x["gap"],
                         x["accepted"], x["verdict"], "  <<UNSAFE_ACCEPT" if x["unsafe_accept"] else ""),
                      flush=True)
    import os
    os.makedirs("tos_cmi/results", exist_ok=True)
    with open("tos_cmi/results/estimator_diag.json", "w") as f:
        json.dump(rows, f, indent=1)
    nU = sum(r["unsafe_accept"] for r in rows)
    # does the gap shrink with n (at fixed cap) or with cap (at fixed n)?
    print("\nn UNSAFE_ACCEPT total:", nU)
    print("ESTIMATOR_DIAG_DONE")


if __name__ == "__main__":
    main()
