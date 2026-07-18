"""C86D Stage-D1 PATH-BLIND selection worker.

This module deliberately defines and imports NO oracle / contribution / field path.
It receives only the pipe connection to the sealed server and the client-visible
pool directory, so the worker PROCESS (spawned by the launcher) cannot reach the
sealed stores even via module globals.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

_PLUGIN_FIELDS = ("nll", "correct", "confidence", "conf_bin")


def freeze_budget(queried, order, q_seq, budget):
    from .policies import budget_available, budget_prefix, composite_select
    if not budget_available(budget, len(order)):
        return {"budget": str(budget), "status": "INPUT_UNAVAILABLE", "pool_size": len(order)}
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
    selected, comp, comp_sha = {}, {}, {}
    for ctx, d in per_ctx.items():
        contribs = {f: np.array(d[f]) for f in _PLUGIN_FIELDS}
        sel, metrics = composite_select(d["labels"], contribs, w, full=full, n_pool=len(order))
        selected[ctx] = int(sel); comp[ctx] = metrics["composite"].tolist()
        blob = (metrics["balanced_accuracy"].tobytes() + metrics["nll"].tobytes()
                + metrics["ece"].tobytes())
        comp_sha[ctx] = hashlib.sha256(blob).hexdigest()
    return {"budget": str(budget), "status": "AVAILABLE", "query_sequence": list(pre),
            "q_seq": [q_seq[i] for i in range(len(pre))], "lure_weights": w.tolist(),
            "receipts": [[t, queried[t][0]] for t in pre],
            "selected_by_context": selected, "composite_by_context": comp,
            "component_sha_by_context": comp_sha}


def run_worker(conn, pool_root, fdir, methods, budgets, chains):
    """Only the pipe + the client-visible pool. No sealed paths in this process."""
    from .policies import acquisition_path, chain_seed, load_pool
    from .server import QueryClientHandle
    client = QueryClientHandle(conn, None)
    pool = load_pool(pool_root)
    index = []
    for tgt in sorted(pool):
        for chain in chains:
            seed = chain_seed(tgt[0], tgt[1], chain)           # target-bound; shared across methods
            for method in methods:
                order, q_seq = acquisition_path(pool[tgt], method, seed)
                attempt = client.open_attempt(tgt, "FULL")
                queried = {}
                for trial in order:
                    label, contexts = client.query(attempt, trial)
                    queried[trial] = (int(label),
                                      {ctx: {f: np.asarray(row[f]).tolist() for f in _PLUGIN_FIELDS}
                                       for ctx, row in contexts.items()})
                rec = {"method": method, "target": list(tgt), "chain": int(chain), "seed": int(seed),
                       "budgets": [freeze_budget(queried, order, q_seq, b) for b in budgets]}
                name = f"{method}__{tgt[0]}__{tgt[1]}__c{chain}.json"
                blob = json.dumps(rec, sort_keys=True)
                open(os.path.join(fdir, name), "w").write(blob)
                index.append({"file": f"freezes/{name}", "method": method, "target": list(tgt),
                              "chain": int(chain), "sha256": hashlib.sha256(blob.encode()).hexdigest()})
    client.close()
    json.dump(index, open(os.path.join(os.path.dirname(fdir), "worker_index.json"), "w"))
