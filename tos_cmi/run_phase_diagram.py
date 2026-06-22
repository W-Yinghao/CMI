"""Phase 1.3 driver -- sweep task-margin x domain-margin x sample-size x seed x generator and
search for UNSAFE_ACCEPT. Forces task_protect=True (the deployment projector). Dumps per-prefix
records to JSON and prints the 5-class summary + any unsafe acceptance.

  python -m tos_cmi.run_phase_diagram --smoke      # tiny grid, fast sanity
  python -m tos_cmi.run_phase_diagram              # full grid (SLURM)
"""
import argparse
import json
import numpy as np
import torch

torch.set_num_threads(1)

from dataclasses import replace
from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.data.synthetic import make_partial_synergy, make_partial_factorized
from tos_cmi.eval.phase_diagram import run_cell, summarize

GENS = {"synergy": make_partial_synergy, "factorized": make_partial_factorized}


def build(gen, task_margin, dom_margin, n, seed):
    return GENS[gen](n=n, sep_label=task_margin, sep_safe=dom_margin,
                     sep_over=0.5 * dom_margin, seed=seed)


def sweep(task_margins, dom_margins, ns, seeds, gens, cfg, n_mc):
    rows = []
    for gen in gens:
        for n in ns:
            for tm in task_margins:
                for dm in dom_margins:
                    for sd in seeds:
                        data = build(gen, tm, dm, n, sd)
                        recs = run_cell(data, cfg, seed=sd, n_mc=n_mc)
                        for r in recs:
                            r.update({"gen": gen, "n": n, "task_margin": tm,
                                      "dom_margin": dm, "seed": sd})
                        rows.extend(recs)
                        s = summarize(recs)
                        print("[%s n=%d tm=%.1f dm=%.1f s=%d] %s unsafe=%d worst_acc=%.4f"
                              % (gen, n, tm, dm, sd, s["counts"], s["n_unsafe_accept"],
                                 s["worst_accepted_bayes_delta"]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--power", default="", help="path to power table -> enable the power floor")
    ap.add_argument("--out", default="tos_cmi/results/phase_diagram.json")
    args = ap.parse_args()
    cfg = replace(ScoreFisherConfig(), task_protect=True,
                  epochs=120 if args.smoke else 200, gate_boot=150 if args.smoke else 250,
                  n_perm_null=2,
                  task_power_floor=bool(args.power), task_power_table=args.power)
    if args.smoke:
        # re-test the previously-failing small-n injection regime under the capacity-bumped gate
        rows = sweep([2.0], [2.6], [2000, 3000, 6000], [0], ["synergy", "factorized"], cfg, n_mc=10000)
    else:
        rows = sweep(task_margins=[1.0, 2.0, 3.5], dom_margins=[1.5, 2.6, 4.0],
                     ns=[2000, 6000, 18000], seeds=[0, 1, 2],
                     gens=["synergy", "factorized"], cfg=cfg, n_mc=15000)

    full = summarize(rows)
    print("\n===== PHASE 1.3 SUMMARY =====")
    print("classes:", full["counts"])
    print("n UNSAFE_ACCEPT:", full["n_unsafe_accept"])
    print("worst accepted Bayes delta:", round(full["worst_accepted_bayes_delta"], 4))
    for r in full["unsafe_accept"]:
        print("  UNSAFE_ACCEPT:", {k: r.get(k) for k in
              ("gen", "candidate_mode", "t_source", "n", "task_margin", "dom_margin",
               "seed", "k", "bayes_delta", "probe_task_ucb")})
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": {k: v for k, v in full.items() if k != "unsafe_accept"},
                   "rows": rows}, f, indent=1)
    print("wrote", args.out)
    print("PHASE_DIAGRAM_DONE" + (" UNSAFE_ACCEPT_FOUND" if full["n_unsafe_accept"] else " CLEAN"))


if __name__ == "__main__":
    main()
