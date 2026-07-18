"""C86D Stage D2 — HELD EVALUATION only. No query-server handle.

Order (fixed): read D1 manifest -> FULLY verify every freeze (path/count/SHA/schema)
-> build a selected-actions-only object -> ONLY THEN open + verify C85U. Primary
risk = held standardized regret; raw utility gap only for the epsilon geometry.
Persists the per (target,method,budget,replicate) table + paired MC SE, and a 5-way
development taxonomy. The C86H registry stays {P0,A1,A2H}.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import time

import numpy as np

from .core import NEAR_OPT_EPS, TAIL_FRACTION, exact_upper_cvar, verify_c85u_identity

TAU = 0.02
NEAROPT_MARGIN = 0.05
CEIL_MAX = 0.05
_REQUIRED_FREEZE_KEYS = {"method", "target", "chain", "seed", "budgets"}


def _require(c, m):
    if not c:
        raise RuntimeError(m)


def verify_freezes(d1_root):
    """FULL freeze verification BEFORE any C85U access. Returns selected-actions only."""
    d1 = json.load(open(os.path.join(d1_root, "C86D_D1_MANIFEST.json")))
    _require(d1["c85u_accessed"] is False, "D1 manifest claims C85U access")
    idx = d1["freeze_index"]
    _require(len(idx) == d1["n_freeze_files"], "freeze count mismatch")
    actions = []                                          # (method,target,chain,budget,{ctx:sel})
    seen = set()
    for e in idx:
        path = os.path.join(d1_root, e["file"])
        _require(os.path.exists(path), f"missing freeze {e['file']}")
        blob = open(path).read()
        _require(hashlib.sha256(blob.encode()).hexdigest() == e["sha256"], f"tampered {e['file']}")
        rec = json.loads(blob)
        _require(_REQUIRED_FREEZE_KEYS <= set(rec), f"bad schema {e['file']}")
        key = (rec["method"], tuple(rec["target"]), rec["chain"])
        _require(key not in seen, f"duplicate freeze {key}")
        seen.add(key)
        for bf in rec["budgets"]:
            actions.append((rec["method"], (rec["target"][0], rec["target"][1]),
                            rec["chain"], bf["budget"], bf["selected_by_context"]))
    return d1, actions


def load_c85u_field():
    from .c85u_config import C85U_UTILITY_INDEX, C85U_UTILITY_INDEX_SHA
    verify_c85u_identity()                                # opens+hashes acceptance manifest
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
    _require(n == 76_464 and len(tmp) == 944, f"C85U field {n}/{len(tmp)}")
    for key, cm in tmp.items():
        s = np.empty(81); u = np.empty(81)
        for i in range(81):
            a, b, cid = cm[i]
            _require(cid == canon[i], f"C85U order broken {key}")
            s[i] = a; u[i] = b
        sr[key] = s; cu[key] = u
    return sr, cu


def _classify(endpoints, budgets, cohorts):
    """5-way development taxonomy on per-cohort mean/tail/near-opt + FULL ceiling."""
    # ceiling = best-case FULL over methods, per cohort
    def ceil_ok():
        for c in cohorts:
            best_mean = min(endpoints[f"{m}|FULL"]["mean_by_cohort"][c] for m in ("P0", "A1", "A2H"))
            best_tail = min(endpoints[f"{m}|FULL"]["tail_by_cohort"][c] for m in ("P0", "A1", "A2H"))
            if best_mean > CEIL_MAX or best_tail > CEIL_MAX:
                return False
        return True
    if not ceil_ok():
        return {"label": "ACQUISITION_VIEW_NONTRANSPORTABLE", "budget": "FULL"}
    for budget in budgets:
        p0 = endpoints[f"P0|{budget}"]
        for m in ("A1", "A2H"):
            a = endpoints[f"{m}|{budget}"]
            gain = (p0["mean_regret_std"] - a["mean_regret_std"]) >= TAU
            if not gain:
                continue
            crossed = all(
                (p0["mean_by_cohort"][c] - a["mean_by_cohort"][c]) >= TAU
                and (p0["tail_by_cohort"][c] - a["tail_by_cohort"][c]) >= TAU
                and (a["near_opt_by_cohort"][c] - p0["near_opt_by_cohort"][c]) >= NEAROPT_MARGIN
                for c in cohorts)
            return {"label": "BOUNDARY_OPERATIONALLY_CROSSED" if crossed
                    else "BOUNDARY_WEAKENED_NOT_ROBUST", "budget": budget, "method": m}
    return {"label": "NO_REGISTERED_ACTIVE_GAIN", "budget": None,
            "note": "POLICY_LIMITED requires a separate oracle-acquisition diagnostic (not run in C86D)"}


def run_d2(d1_root: str, output_root: str) -> dict:
    t0 = time.time()
    d1, actions = verify_freezes(d1_root)                 # <-- fully before C85U
    std_regret, comp_util = load_c85u_field()             # <-- only now

    rows = []
    cohort_of = lambda t: t[0]
    # per (method,budget): target -> per-replicate (std target regret, raw-gap near-opt indicator)
    cell = collections.defaultdict(lambda: collections.defaultdict(lambda: {"sr": [], "no": []}))
    for method, tgt, chain, budget, selby in actions:
        sregs, gaps = [], []
        for ctx, sel in selby.items():
            sregs.append(float(std_regret[(tgt, ctx)][sel]))
            gaps.append(float(comp_util[(tgt, ctx)].max() - comp_util[(tgt, ctx)][sel]))
        tgt_sr = float(np.mean(sregs)); tgt_gap = float(np.mean(gaps))
        cell[(method, budget)][tgt]["sr"].append(tgt_sr)
        cell[(method, budget)][tgt]["no"].append(float(tgt_gap <= NEAR_OPT_EPS))   # indicator-first
        rows.append({"method": method, "dataset": tgt[0], "subject": tgt[1], "chain": chain,
                     "budget": budget, "target_std_regret": tgt_sr, "target_raw_gap": tgt_gap,
                     "near_opt_indicator": int(tgt_gap <= NEAR_OPT_EPS)})

    endpoints = {}
    for (method, budget), tmap in cell.items():
        by_cohort = collections.defaultdict(list); no_cohort = collections.defaultdict(list)
        for tgt, d in tmap.items():
            by_cohort[cohort_of(tgt)].append(float(np.mean(d["sr"])))
            no_cohort[cohort_of(tgt)].append(float(np.mean(d["no"])))    # replicate-averaged indicator
        allr = np.array([v for vs in by_cohort.values() for v in vs])
        endpoints[f"{method}|{budget}"] = {
            "mean_regret_std": float(allr.mean()),
            "tail_regret_std": exact_upper_cvar(allr, TAIL_FRACTION),
            "target_near_opt_prob": float(np.mean([v for vs in no_cohort.values() for v in vs])),
            "mean_by_cohort": {c: float(np.mean(v)) for c, v in by_cohort.items()},
            "tail_by_cohort": {c: exact_upper_cvar(v, TAIL_FRACTION) for c, v in by_cohort.items()},
            "near_opt_by_cohort": {c: float(np.mean(no_cohort[c])) for c in by_cohort},
        }
    cohorts = sorted({c for k in endpoints for c in endpoints[k]["mean_by_cohort"]})
    budgets = d1["budgets"]

    # FULL acquisition-invariance positive control: P0/A1/A2H select identically at FULL
    full_sel = collections.defaultdict(dict)
    for method, tgt, chain, budget, selby in actions:
        if budget == "FULL":
            for ctx, sel in selby.items():
                full_sel[(tgt, ctx)][method] = sel
    full_invariant = all(len(set(ms.values())) == 1 for ms in full_sel.values())

    # paired active-minus-P0 MC SE at each budget (over targets)
    mc_se = {}
    for budget in budgets:
        p0t = {t: np.mean(cell[("P0", budget)][t]["sr"]) for t in cell[("P0", budget)]}
        for m in ("A1", "A2H"):
            diffs = [np.mean(cell[(m, budget)][t]["sr"]) - p0t[t] for t in p0t]
            mc_se[f"{m}-P0|{budget}"] = {"mean_diff": float(np.mean(diffs)),
                                         "se": float(np.std(diffs) / np.sqrt(len(diffs)))}

    disposition = _classify(endpoints, budgets, cohorts)
    manifest = {
        "gate": "C86D_DEVELOPMENT_ACTIVE_POLICY_FIELD_FROZEN_C86H_REVIEW_REQUIRED",
        "stage": "C86D_D2_HELD_EVALUATION", "development_only": True, "confirmatory_claim": False,
        "primary_risk": "held_standardized_regret",
        "near_opt_scale": "raw_composite_utility_gap (epsilon geometry; indicator-first)",
        "freeze_verification": "all freezes SHA-verified BEFORE C85U opened",
        "full_acquisition_invariant": bool(full_invariant),
        "endpoints": endpoints, "paired_active_minus_p0_mc_se": mc_se,
        "development_disposition": disposition,
        "c86h_method_registry": ["P0", "A1", "A2H"],
        "c86h_note": "registry not pruned by development performance",
        "n_replicate_rows": len(rows),
        "d2_seconds": round(time.time() - t0, 1),
    }
    staging = output_root + ".staging"
    os.makedirs(staging, exist_ok=True)
    with open(os.path.join(staging, "C86D_D2_RESULT_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(staging, "C86D_REPLICATE_TABLE.json"), "w") as fh:
        json.dump(rows, fh)
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
    m = run_d2(a.d1_root, a.output_root)
    print(json.dumps({k: m[k] for k in ("gate", "development_disposition", "full_acquisition_invariant",
                                        "c86h_method_registry")}, indent=2))
