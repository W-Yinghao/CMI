"""C86D Stage D2 — HELD EVALUATION only. Reads the frozen D1 selections, opens and
verifies C85U, and scores with the registered STANDARDIZED regret.

No query-server handle here. Primary risk = held standardized regret; raw utility
gap is used only for the epsilon / near-optimal geometry. Per-cohort mean/tail/
near-opt taxonomy; the C86H registry stays {P0, A1, A2H} regardless of result.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import time

import numpy as np

from .core import (NEAR_OPT_EPS, TAIL_FRACTION, C85U_UTILITY_INDEX,
                   C85U_UTILITY_INDEX_SHA, exact_upper_cvar, verify_c85u_identity)

TAU = 0.02
NEAROPT_MARGIN = 0.05


def _require(c, m):
    if not c:
        raise RuntimeError(m)


def load_c85u_field():
    """(target,context) -> (std_regret[81], composite_utility[81]); fail-closed identity."""
    verify_c85u_identity()                               # opens+hashes acceptance manifest
    h = hashlib.sha256(open(C85U_UTILITY_INDEX, "rb").read()).hexdigest()
    _require(h == C85U_UTILITY_INDEX_SHA, f"C85U utility index SHA mismatch {h}")
    canon = ["ERM:0"] + [f"OACI:{t}" for t in range(1, 41)] + [f"SRC:{t}" for t in range(1, 41)]
    sr, cu = {}, {}
    tmp = collections.defaultdict(dict)
    n = 0
    for r in csv.DictReader(open(C85U_UTILITY_INDEX)):
        key = ((r["dataset"], int(r["target_subject_id"])),
               f"panel={r['panel']}|seed={r['training_seed']}|level={r['level']}")
        tmp[key][int(r["candidate_index"])] = (float(r["standardized_regret"]),
                                               float(r["composite_utility"]),
                                               f"{r['regime']}:{r['trajectory_order']}")
        n += 1
    _require(n == 76_464 and len(tmp) == 944, f"C85U field {n}/{len(tmp)} != 76464/944")
    for key, cm in tmp.items():
        s = np.empty(81); u = np.empty(81)
        for i in range(81):
            a, b, cid = cm[i]
            _require(cid == canon[i], f"C85U candidate order broken at {key}")
            s[i] = a; u[i] = b
        sr[key] = s; cu[key] = u
    return sr, cu


def run_d2(d1_root: str, output_root: str) -> dict:
    t0 = time.time()
    d1 = json.load(open(os.path.join(d1_root, "C86D_D1_MANIFEST.json")))
    _require(d1["c85u_accessed"] is False, "D1 manifest claims C85U access — isolation breach")
    std_regret, comp_util = load_c85u_field()

    # gather selections: (method,budget) -> target -> [per-seed target regret(std), raw-gap]
    prim = collections.defaultdict(lambda: collections.defaultdict(list))
    rawg = collections.defaultdict(lambda: collections.defaultdict(list))
    for entry in d1["freeze_index"]:
        blob = open(os.path.join(d1_root, entry["file"])).read()
        _require(hashlib.sha256(blob.encode()).hexdigest() == entry["sha256"],
                 f"D1 freeze tampered: {entry['file']}")
        rec = json.loads(blob)
        tgt = (rec["target"][0], rec["target"][1]); method = rec["method"]
        for bf in rec["budgets"]:
            budget = bf["budget"]
            sregs, gaps = [], []
            for ctx, sel in bf["selected_by_context"].items():
                s = std_regret[(tgt, ctx)]; u = comp_util[(tgt, ctx)]
                sregs.append(float(s[sel]))
                gaps.append(float(u.max() - u[sel]))
            prim[(method, budget)][tgt].append(float(np.mean(sregs)))   # 8-context equal-weight
            rawg[(method, budget)][tgt].append(float(np.mean(gaps)))

    cohort_of = lambda t: t[0]
    endpoints = {}
    for (method, budget), tmap in prim.items():
        by_cohort = collections.defaultdict(list); nearopt = collections.defaultdict(list)
        mc = []
        for tgt, seedvals in tmap.items():
            by_cohort[cohort_of(tgt)].append(float(np.mean(seedvals)))      # primary: standardized
            mc.append(float(np.std(seedvals)))
            nearopt[cohort_of(tgt)].append(float(np.mean(rawg[(method, budget)][tgt]) <= NEAR_OPT_EPS))
        allr = np.array([v for vs in by_cohort.values() for v in vs])
        endpoints[f"{method}|{budget}"] = {
            "mean_regret_std": float(allr.mean()),
            "tail_regret_std": exact_upper_cvar(allr, TAIL_FRACTION),
            "target_near_opt_prob_rawgap": float(np.mean([v for vs in nearopt.values() for v in vs])),
            "mean_by_cohort": {c: float(np.mean(v)) for c, v in by_cohort.items()},
            "tail_by_cohort": {c: exact_upper_cvar(v, TAIL_FRACTION) for c, v in by_cohort.items()},
            "near_opt_by_cohort": {c: float(np.mean(nearopt[c])) for c in by_cohort},
            "mc_std_over_replicates": float(np.mean(mc)),
        }

    # per-cohort development taxonomy (mean AND tail AND near-opt, every cohort)
    cohorts = sorted({c for k in endpoints for c in endpoints[k]["mean_by_cohort"]})
    disposition = {"label": "NO_REGISTERED_ACTIVE_GAIN", "budget": None, "method": None}
    for budget in d1["budgets"]:
        p0 = endpoints[f"P0|{budget}"]
        for method in ("A1", "A2H"):
            a = endpoints[f"{method}|{budget}"]
            crossed = all(
                (p0["mean_by_cohort"][c] - a["mean_by_cohort"][c]) >= TAU
                and (p0["tail_by_cohort"][c] - a["tail_by_cohort"][c]) >= TAU
                and (a["near_opt_by_cohort"][c] - p0["near_opt_by_cohort"][c]) >= NEAROPT_MARGIN
                for c in cohorts)
            if crossed:
                disposition = {"label": "REGISTERED_ACTIVE_GAIN", "budget": budget, "method": method}
                break
        if disposition["label"] != "NO_REGISTERED_ACTIVE_GAIN":
            break

    manifest = {
        "gate": "C86D_DEVELOPMENT_ACTIVE_POLICY_FIELD_FROZEN_C86H_REVIEW_REQUIRED",
        "stage": "C86D_D2_HELD_EVALUATION",
        "development_only": True, "confirmatory_claim": False,
        "primary_risk": "held_standardized_regret",
        "near_opt_scale": "raw_composite_utility_gap (epsilon geometry only)",
        "endpoints": endpoints,
        "development_disposition": disposition,
        "c86h_method_registry": ["P0", "A1", "A2H"],       # unchanged regardless of result
        "c86h_note": "registry is not pruned by development performance",
        "identity": {"c85u_utility_index_sha256": C85U_UTILITY_INDEX_SHA,
                     "d1_root": d1_root, "n_freeze_files": d1["n_freeze_files"]},
        "d2_seconds": round(time.time() - t0, 1),
    }
    staging = output_root + ".staging"
    os.makedirs(staging, exist_ok=True)
    with open(os.path.join(staging, "C86D_D2_RESULT_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(staging, output_root)
    manifest["output_root"] = output_root
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--authorization", required=True)
    a = ap.parse_args()
    if a.authorization != "授权 C86D":
        raise SystemExit("C86D D2 requires --authorization '授权 C86D'")
    print(json.dumps({k: run_d2(a.d1_root, a.output_root)[k]
                      for k in ("gate", "development_disposition", "c86h_method_registry")}, indent=2))
