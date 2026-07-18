"""C86D Stage D1 — SELECTION only. No C85U path/import; no held utility.

Runs the process-isolated query client/server, produces nested-prefix selections
for every (method, target, budget, replicate), and persists + hashes all selection
freezes to disk. Held evaluation happens later, in a separate process (run_d2).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from .core import BUDGET_GRID, METHOD_FREEZE          # NOTE: no C85U names imported here
from .policies import acquisition_path, budget_prefix, composite_select
from .server import start_query_server

FIELD_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c86l-development-field-v1"
POOL_ROOT = os.path.join(FIELD_ROOT, "acquisition_unlabeled_pool")
ORACLE_ROOT = os.path.join(FIELD_ROOT, "acquisition_label_oracle")
CONTRIB_ROOT = os.path.join(FIELD_ROOT, "query_contribution_store")
_PLUGIN_FIELDS = ("nll", "correct", "confidence", "conf_bin")


def _freeze_budget(queried, order, q_seq, budget):
    pre, w = budget_prefix(order, q_seq, len(order), budget)
    full = (budget == "FULL")
    per_ctx = {}
    for trial in pre:
        label, contexts = queried[trial]
        for ctx, row in contexts.items():
            d = per_ctx.setdefault(ctx, {"labels": [], **{f: [] for f in _PLUGIN_FIELDS}})
            d["labels"].append(label)
            for f in _PLUGIN_FIELDS:
                d[f].append(np.asarray(row[f]))
    selected, comp = {}, {}
    for ctx, d in per_ctx.items():
        contribs = {f: np.array(d[f]) for f in _PLUGIN_FIELDS}
        sel, metrics = composite_select(d["labels"], contribs, w, full=full)
        selected[ctx] = int(sel)
        comp[ctx] = metrics["composite"].tolist()
    return {"budget": str(budget), "query_sequence": list(pre),
            "q_seq": [q_seq[i] for i in range(len(pre))], "lure_weights": w.tolist(),
            "receipts": [[t, queried[t][0]] for t in pre],
            "selected_by_context": selected, "composite_by_context": comp}


def run_d1(output_root: str) -> dict:
    from .policies import load_pool
    t0 = time.time()
    pool = load_pool(POOL_ROOT)
    methods = list(METHOD_FREEZE["primary_registry"])
    budgets = list(BUDGET_GRID)
    seeds = list(METHOD_FREEZE["seed_schedule"])
    staging = output_root + ".staging"
    fdir = os.path.join(staging, "freezes")
    os.makedirs(fdir, exist_ok=True)

    client = start_query_server(ORACLE_ROOT, CONTRIB_ROOT)
    index = []
    try:
        for method in methods:
            for tgt in sorted(pool):
                for seed in seeds:
                    order, q_seq = acquisition_path(pool[tgt], method, seed)
                    attempt = client.open_attempt(tgt, "FULL")
                    queried = {}
                    for trial in order:
                        label, contexts = client.query(attempt, trial)
                        queried[trial] = (int(label),
                                          {ctx: {f: np.asarray(row[f]).tolist() for f in _PLUGIN_FIELDS}
                                           for ctx, row in contexts.items()})
                    rec = {"method": method, "target": list(tgt), "seed": int(seed),
                           "budgets": [_freeze_budget(queried, order, q_seq, b) for b in budgets]}
                    name = f"{method}__{tgt[0]}__{tgt[1]}__s{seed}.json"
                    blob = json.dumps(rec, sort_keys=True)
                    (open(os.path.join(fdir, name), "w")).write(blob)
                    index.append({"file": f"freezes/{name}", "method": method,
                                  "target": list(tgt), "seed": int(seed),
                                  "sha256": hashlib.sha256(blob.encode()).hexdigest()})
    finally:
        client.close()

    manifest = {
        "stage": "C86D_D1_SELECTION",
        "c85u_accessed": False, "held_utility_opened": False,
        "c86l_field_root": FIELD_ROOT,
        "methods": methods, "budgets": [str(b) for b in budgets],
        "n_targets": len(pool), "replicates": len(seeds),
        "n_freeze_files": len(index), "freeze_index": index,
        "warm_start": METHOD_FREEZE["a1"]["sampling_prob_floor"] and 4,
        "d1_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(staging, "C86D_D1_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(staging, output_root)
    return {k: manifest[k] for k in ("stage", "n_freeze_files", "n_targets", "c85u_accessed")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--authorization", required=True)
    a = ap.parse_args()
    if a.authorization != "授权 C86D":
        raise SystemExit("C86D D1 requires --authorization '授权 C86D'")
    print(json.dumps(run_d1(a.output_root), indent=2))
