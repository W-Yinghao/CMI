"""C86D Stage D2 — HELD EVALUATION only. No query-server handle.

Order: read D1 manifest -> FULLY verify every freeze (count, uniqueness, seed
binding, schema, per-freeze context/action/length/nested-prefix, INPUT_UNAVAILABLE
validity) -> ONLY THEN open + verify C85U. Primary risk = held standardized regret;
raw utility gap only for the epsilon (near-opt) geometry, indicator-first. B32
INPUT_UNAVAILABLE cells are described but never enter cross-cohort gates.
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
from .policies import chain_seed

TAU = 0.02
NEAROPT_MARGIN = 0.05
CEIL_MAX = 0.05
CEIL_NEAROPT_MIN = 0.90
BUDGET_SET = {"4", "8", "16", "32", "FULL"}
CANON_CTX = {f"panel={p}|seed={s}|level={l}" for p in ("A", "B") for s in (5, 6) for l in (0, 1)}
_REQ = {"method", "target", "chain", "seed", "budgets"}


def _require(c, m):
    if not c:
        raise RuntimeError(m)


def verify_freezes(d1_root):
    """FULL freeze verification BEFORE any C85U access. Returns validated actions."""
    d1 = json.load(open(os.path.join(d1_root, "C86D_D1_MANIFEST.json")))
    _require(d1["c85u_accessed"] is False, "D1 manifest claims C85U access")
    idx = d1["freeze_index"]
    _require(len(idx) == d1["n_freeze_files"], "freeze count vs manifest mismatch")
    methods = ("P0", "A1", "A2H"); n_chains = d1["chains"]; n_targets = d1["n_targets"]
    _require(len(idx) == len(methods) * n_targets * n_chains,
             f"expected {len(methods)*n_targets*n_chains} freezes, got {len(idx)}")
    seen = set()
    actions = []                     # (method,target,chain,budget,selected|None,status)
    full_by_ctx_chain = collections.defaultdict(dict)   # (target,ctx,chain) -> {method: sel}
    for e in idx:
        path = os.path.join(d1_root, e["file"])
        _require(os.path.exists(path), f"missing freeze {e['file']}")
        blob = open(path).read()
        _require(hashlib.sha256(blob.encode()).hexdigest() == e["sha256"], f"tampered {e['file']}")
        rec = json.loads(blob)
        _require(_REQ <= set(rec), f"bad schema {e['file']}")
        method = rec["method"]; tgt = (rec["target"][0], rec["target"][1]); chain = rec["chain"]
        _require(method == e["method"] and list(tgt) == e["target"] and chain == e["chain"],
                 f"index/internal identity mismatch {e['file']}")
        key = (method, tgt, chain)
        _require(key not in seen, f"duplicate freeze {key}"); seen.add(key)
        _require(rec["seed"] == chain_seed(tgt[0], tgt[1], chain), f"seed not target-bound {key}")
        bmap = {bf["budget"]: bf for bf in rec["budgets"]}
        _require(set(bmap) == BUDGET_SET, f"incomplete budget set {key}: {set(bmap)}")
        avail_seqs = {}
        for budget, bf in bmap.items():
            if bf.get("status") == "INPUT_UNAVAILABLE":
                _require(budget != "FULL" and bf["pool_size"] < int(budget),
                         f"invalid INPUT_UNAVAILABLE {key} {budget}")
                actions.append((method, tgt, chain, budget, None, "INPUT_UNAVAILABLE"))
                continue
            _require(bf.get("status") == "AVAILABLE", f"bad status {key} {budget}")
            sel = bf["selected_by_context"]; comp = bf["composite_by_context"]
            _require(set(sel) == CANON_CTX, f"contexts != 8 canonical {key} {budget}")
            qs = bf["query_sequence"]
            M = len(qs)
            _require(M == (int(budget) if budget != "FULL" else M), f"bad prefix len {key} {budget}")
            _require(len(bf["q_seq"]) == M and len(bf["lure_weights"]) == M
                     and len(bf["receipts"]) == M, f"length mismatch {key} {budget}")
            _require(len(set(qs)) == M, f"duplicate query trial {key} {budget}")
            for ctx in CANON_CTX:
                _require(0 <= sel[ctx] <= 80, f"selected out of range {key} {budget} {ctx}")
                cv = np.asarray(comp[ctx])
                _require(cv.shape == (81,), f"composite len != 81 {key} {budget} {ctx}")
                _require(int(np.argmax(cv)) == sel[ctx], f"selected != first argmax {key} {budget} {ctx}")
            avail_seqs[budget] = qs
            actions.append((method, tgt, chain, budget, sel, "AVAILABLE"))
            if budget == "FULL":
                for ctx in CANON_CTX:
                    full_by_ctx_chain[(tgt, ctx, chain)][method] = sel[ctx]
        # nested-prefix among AVAILABLE budgets (4 ⊂ 8 ⊂ 16 ⊂ 32 ⊂ FULL)
        ordered = [b for b in ("4", "8", "16", "32", "FULL") if b in avail_seqs]
        for a, b in zip(ordered, ordered[1:]):
            _require(avail_seqs[a] == avail_seqs[b][:len(avail_seqs[a])],
                     f"non-nested prefix {key} {a}->{b}")
    # FULL acquisition invariance per (target,context,chain) AND across chains
    per_group = all(len(set(ms.values())) == 1 for ms in full_by_ctx_chain.values())
    by_ctx = collections.defaultdict(set)
    for (tgt, ctx, chain), ms in full_by_ctx_chain.items():
        by_ctx[(tgt, ctx)].update(ms.values())
    cross_chain = all(len(v) == 1 for v in by_ctx.values())
    return d1, actions, {"full_invariant_within_group": bool(per_group),
                         "full_invariant_across_chains": bool(cross_chain)}


def load_c85u_field():
    from .c85u_config import C85U_UTILITY_INDEX, C85U_UTILITY_INDEX_SHA
    verify_c85u_identity()
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


def _classify(endpoints, eligible_budgets, cohorts):
    def ceil_ok():
        for c in cohorts:
            best_mean = min(endpoints[f"{m}|FULL"]["mean_by_cohort"][c] for m in ("P0", "A1", "A2H"))
            best_tail = min(endpoints[f"{m}|FULL"]["tail_by_cohort"][c] for m in ("P0", "A1", "A2H"))
            best_no = max(endpoints[f"{m}|FULL"]["near_opt_by_cohort"][c] for m in ("P0", "A1", "A2H"))
            if best_mean > CEIL_MAX or best_tail > CEIL_MAX or best_no < CEIL_NEAROPT_MIN:
                return False
        return True
    if not ceil_ok():
        return {"label": "ACQUISITION_VIEW_NONTRANSPORTABLE", "budget": "FULL"}
    for budget in eligible_budgets:                     # cross-cohort only where all cohorts eligible
        p0 = endpoints[f"P0|{budget}"]
        for m in ("A1", "A2H"):
            a = endpoints[f"{m}|{budget}"]
            if (p0["mean_regret_std"] - a["mean_regret_std"]) < TAU:
                continue
            crossed = all(
                (p0["mean_by_cohort"][c] - a["mean_by_cohort"][c]) >= TAU
                and (p0["tail_by_cohort"][c] - a["tail_by_cohort"][c]) >= TAU
                and (a["near_opt_by_cohort"][c] - p0["near_opt_by_cohort"][c]) >= NEAROPT_MARGIN
                for c in cohorts)
            return {"label": "BOUNDARY_OPERATIONALLY_CROSSED" if crossed
                    else "BOUNDARY_WEAKENED_NOT_ROBUST", "budget": budget, "method": m}
    return {"label": "NO_REGISTERED_ACTIVE_GAIN", "budget": None,
            "note": "POLICY_LIMITED requires a separate oracle-acquisition diagnostic (not run)"}


def run_d2(d1_root: str, output_root: str) -> dict:
    t0 = time.time()
    d1, actions, full_inv = verify_freezes(d1_root)         # <-- entirely before C85U
    std_regret, comp_util = load_c85u_field()               # <-- only now

    cohort_of = lambda t: t[0]
    # per (method,budget): target -> chain -> (std target regret, near-opt indicator); track availability
    cell = collections.defaultdict(lambda: collections.defaultdict(dict))
    unavailable = collections.defaultdict(set)              # (method,budget) -> {cohort with any unavail}
    rows = []
    for method, tgt, chain, budget, sel, status in actions:
        if status == "INPUT_UNAVAILABLE":
            unavailable[(method, budget)].add(cohort_of(tgt))
            rows.append({"method": method, "dataset": tgt[0], "subject": tgt[1], "chain": chain,
                         "budget": budget, "status": "INPUT_UNAVAILABLE"})
            continue
        sregs, gaps = [], []
        for ctx, s in sel.items():
            sregs.append(float(std_regret[(tgt, ctx)][s]))
            gaps.append(float(comp_util[(tgt, ctx)].max() - comp_util[(tgt, ctx)][s]))
        tgt_sr = float(np.mean(sregs)); tgt_gap = float(np.mean(gaps))
        cell[(method, budget)][tgt][chain] = (tgt_sr, float(tgt_gap <= NEAR_OPT_EPS))
        rows.append({"method": method, "dataset": tgt[0], "subject": tgt[1], "chain": chain,
                     "budget": budget, "status": "AVAILABLE", "target_std_regret": tgt_sr,
                     "target_raw_gap": tgt_gap, "near_opt_indicator": int(tgt_gap <= NEAR_OPT_EPS)})

    endpoints = {}
    for (method, budget), tmap in cell.items():
        by_cohort = collections.defaultdict(list); no_cohort = collections.defaultdict(list)
        for tgt, chains in tmap.items():
            srs = [v[0] for v in chains.values()]; nos = [v[1] for v in chains.values()]
            by_cohort[cohort_of(tgt)].append(float(np.mean(srs)))          # target = mean over chains
            no_cohort[cohort_of(tgt)].append(float(np.mean(nos)))          # replicate-averaged indicator
        allr = np.array([v for vs in by_cohort.values() for v in vs])
        endpoints[f"{method}|{budget}"] = {
            "mean_regret_std": float(allr.mean()),
            "tail_regret_std": exact_upper_cvar(allr, TAIL_FRACTION),
            "target_near_opt_prob": float(np.mean([v for vs in no_cohort.values() for v in vs])),
            "mean_by_cohort": {c: float(np.mean(v)) for c, v in by_cohort.items()},
            "tail_by_cohort": {c: exact_upper_cvar(v, TAIL_FRACTION) for c, v in by_cohort.items()},
            "near_opt_by_cohort": {c: float(np.mean(no_cohort[c])) for c in by_cohort},
            "n_targets_available_by_cohort": {c: len(v) for c, v in by_cohort.items()},
        }
    cohorts = sorted({c for k in endpoints for c in endpoints[k]["mean_by_cohort"]})
    # a budget is cross-cohort eligible only if NO cohort had an INPUT_UNAVAILABLE cell
    eligible = [b for b in d1["budgets"]
                if not any(unavailable[(m, b)] for m in ("P0", "A1", "A2H"))]

    # chain-level paired active−P0 Monte-Carlo SE (distinct from target heterogeneity SE)
    mc_se = {}
    for budget in d1["budgets"]:
        for m in ("A1", "A2H"):
            common = set(cell[("P0", budget)]) & set(cell[(m, budget)])
            chains = sorted({c for t in common for c in cell[("P0", budget)][t]})
            eff = []
            for ch in chains:
                d = [cell[(m, budget)][t][ch][0] - cell[("P0", budget)][t][ch][0]
                     for t in common if ch in cell[(m, budget)][t] and ch in cell[("P0", budget)][t]]
                if d:
                    eff.append(float(np.mean(d)))                          # target-equal within chain
            if len(eff) > 1:
                mc_se[f"{m}-P0|{budget}"] = {"chain_mean_diff": float(np.mean(eff)),
                                             "mc_se": float(np.std(eff, ddof=1) / np.sqrt(len(eff))),
                                             "n_chains": len(eff)}

    disposition = _classify(endpoints, [b for b in eligible if any(f"P0|{b}" == k for k in endpoints)],
                            cohorts)
    manifest = {
        "gate": "C86D_DEVELOPMENT_ACTIVE_POLICY_FIELD_FROZEN_C86H_REVIEW_REQUIRED",
        "stage": "C86D_D2_HELD_EVALUATION", "development_only": True, "confirmatory_claim": False,
        "primary_risk": "held_standardized_regret",
        "near_opt_scale": "raw_composite_utility_gap (epsilon geometry; indicator-first)",
        "freeze_verification": "count/uniqueness/seed-binding/schema/context/action/nested-prefix "
                               "fully verified BEFORE C85U opened",
        "full_acquisition_invariance": full_inv,
        "cross_cohort_eligible_budgets": eligible,
        "input_unavailable_cells": {f"{m}|{b}": sorted(unavailable[(m, b)])
                                    for (m, b) in unavailable},
        "endpoints": endpoints, "chain_level_paired_mc_se": mc_se,
        "development_disposition": disposition,
        "c86h_method_registry": ["P0", "A1", "A2H"],
        "c86h_note": "registry not pruned by development performance",
        "n_replicate_rows": len(rows), "d2_seconds": round(time.time() - t0, 1),
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
    print(json.dumps({k: m[k] for k in ("gate", "development_disposition",
                                        "full_acquisition_invariance", "cross_cohort_eligible_budgets",
                                        "c86h_method_registry")}, indent=2))
