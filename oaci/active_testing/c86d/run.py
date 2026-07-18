"""C86D authorized real execution — runs P0/A1/A2H on the real C86L field and
evaluates against the real C85U held utility. Development only; fail-closed.

Runs ONLY under a direct '授权 C86D' (see pipeline.execute).
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import time

import numpy as np

from . import core
from .core import BUDGET_GRID, METHOD_FREEZE, exact_upper_cvar, verify_c85u_identity
from .policies import load_pool
from .pipeline import HeldEvaluator, run_selection
from .server import start_query_server

FIELD_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c86l-development-field-v1"
C85U_UTILITY_INDEX = ("/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v2/"
                      "c85u-v2-77382c16a593f7c2-91a428488a634268/stage_u1_candidate_utility_v2/"
                      "candidate_utility_index.csv")
C85U_UTILITY_INDEX_SHA = "83bddf56290c4e06a306d64dadfc9611115a177f479d433fe0e4485b0c181509"
TAU = 0.02


class C86DRunError(RuntimeError):
    pass


def _require(c, m):
    if not c:
        raise C86DRunError(m)


def load_held_utility():
    """(target=(ds,subj), context) -> composite_utility[81]; fail-closed identity checks."""
    ident = verify_c85u_identity()                       # opens+hashes the acceptance manifest
    h = hashlib.sha256(open(C85U_UTILITY_INDEX, "rb").read()).hexdigest()
    _require(h == C85U_UTILITY_INDEX_SHA, f"C85U utility index SHA mismatch: {h}")
    canon = ["ERM:0"] + [f"OACI:{t}" for t in range(1, 41)] + [f"SRC:{t}" for t in range(1, 41)]
    held = {}
    order_ok = 0
    tmp = collections.defaultdict(dict)
    n = 0
    for r in csv.DictReader(open(C85U_UTILITY_INDEX)):
        key = (r["dataset"], int(r["target_subject_id"]))
        ctx = f"panel={r['panel']}|seed={r['training_seed']}|level={r['level']}"
        ci = int(r["candidate_index"])
        tmp[(key, ctx)][ci] = (float(r["composite_utility"]), f"{r['regime']}:{r['trajectory_order']}")
        n += 1
    _require(n == 76_464, f"C85U utility rows {n} != 76464")
    _require(len(tmp) == 944, f"C85U contexts {len(tmp)} != 944")
    for (key, ctx), cm in tmp.items():
        util = np.empty(81)
        for i in range(81):
            u, cid = cm[i]
            _require(cid == canon[i], f"C85U candidate_index {i} != canonical at {key}/{ctx}")
            util[i] = u
        held[(key, ctx)] = util
        order_ok += 1
    return held, {"c85u_acceptance": ident, "contexts": order_ok}


def run_c86d(output_root: str) -> dict:
    t0 = time.time()
    held, held_ident = load_held_utility()
    pool = load_pool(os.path.join(FIELD_ROOT, "acquisition_unlabeled_pool"))
    cohort_of = {tgt: tgt[0] for tgt in pool}            # cohort = dataset
    budgets = [b for b in BUDGET_GRID]
    methods = ["P0", "A1", "A2H"]
    seeds = list(METHOD_FREEZE["seed_schedule"])

    client = start_query_server(os.path.join(FIELD_ROOT, "acquisition_label_oracle"),
                                os.path.join(FIELD_ROOT, "query_contribution_store"))
    evaluator = HeldEvaluator(held, verify_identity=False)   # identity already verified above
    results = {}            # (method,budget) -> {cohort: [target_regret,...]}
    try:
        for method in methods:
            for budget in budgets:
                by_cohort = collections.defaultdict(list)
                for tgt, tpool in pool.items():
                    reg_seeds = []
                    for sd in seeds:
                        fr = run_selection(client, tpool, tgt, method, budget, seed=sd)
                        reg_seeds.append(evaluator.evaluate(fr)["target_regret"])
                    by_cohort[cohort_of[tgt]].append(float(np.mean(reg_seeds)))
                results[(method, budget)] = dict(by_cohort)
    finally:
        client.close()

    # endpoints per (method, budget)
    endpoints = {}
    for (method, budget), by_cohort in results.items():
        m = core.compute_endpoints(by_cohort)
        endpoints[f"{method}|{budget}"] = {
            "mean_regret": m.mean_regret, "tail_regret": m.tail_regret,
            "target_near_opt_prob": m.target_near_opt_prob,
            "mean_by_cohort": m.mean_by_cohort, "tail_by_cohort": m.tail_by_cohort,
            "near_opt_by_cohort": m.near_opt_by_cohort,
        }

    # deterministic pre-registered method-freeze rule
    frozen = {"method": "P0", "budget": None, "reason": "no_registered_active_gain"}
    for budget in budgets:
        p0 = endpoints[f"P0|{budget}"]["mean_regret"]
        active = {mth: endpoints[f"{mth}|{budget}"]["mean_regret"] for mth in ("A1", "A2H")}
        best = min(active, key=lambda k: active[k])
        if p0 - active[best] >= TAU:
            frozen = {"method": best, "budget": budget,
                      "reason": "smallest_budget_active_beats_P0_by_tau", "tau": TAU}
            break

    manifest = {
        "gate": "C86D_DEVELOPMENT_ACTIVE_POLICY_FIELD_FROZEN_C86H_REVIEW_REQUIRED",
        "authorization": "授权 C86D",
        "development_only": True,
        "confirmatory_claim": False,
        "methods": methods, "budgets": [str(b) for b in budgets],
        "cohorts_are_datasets": True,
        "n_targets": len(pool),
        "replicates_per_cell": len(seeds),
        "endpoints": endpoints,
        "c86h_method_freeze": frozen,
        "method_freeze_rule": METHOD_FREEZE,
        "identity": {
            "c86l_field_root": FIELD_ROOT,
            "c85u_utility_index": C85U_UTILITY_INDEX,
            "c85u_acceptance": held_ident["c85u_acceptance"],
        },
        "endpoint_definitions": {
            "target_regret": "equal-weight mean over the target's 8 contexts (max composite util - selected)",
            "target_near_opt_prob": "P(target regret <= 0.05)",
            "tail": "exact upper-0.25 CVaR with fractional boundary mass over cohort target regrets",
        },
        "run_seconds": round(time.time() - t0, 1),
    }
    staging = output_root + ".staging"
    os.makedirs(staging, exist_ok=True)
    with open(os.path.join(staging, "C86D_RESULT_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(staging, output_root)
    manifest["output_root"] = output_root
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--authorization", required=True)
    a = ap.parse_args()
    if a.authorization != "授权 C86D":
        raise SystemExit("C86D run requires --authorization '授权 C86D'")
    print(json.dumps(run_c86d(a.output_root), indent=2)[:2000])
