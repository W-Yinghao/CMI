"""C86D Stage D1 — SELECTION only. No C85U import anywhere in this process tree.

Architecture (path separation):
  launcher  : replays the accepted C86L content-addressed manifest, holds the sealed
              oracle/contribution paths, starts the server process, and spawns a
              PATH-BLIND worker with only the pipe handle + the (client-visible) pool.
  server    : separate process, owns the sealed oracle/contribution.
  worker    : separate spawned process, no sealed paths; runs P0/A1/A2H with
              target-bound chain seeds and persists SHA-hashed selection freezes.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np

from .core import BUDGET_GRID, METHOD_FREEZE           # no C85U names here
from .server import start_server_process

FIELD_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c86l-development-field-v1"
POOL_ROOT = os.path.join(FIELD_ROOT, "acquisition_unlabeled_pool")
ORACLE_ROOT = os.path.join(FIELD_ROOT, "acquisition_label_oracle")
CONTRIB_ROOT = os.path.join(FIELD_ROOT, "query_contribution_store")
ACCEPTANCE_MANIFEST = os.path.join(os.path.dirname(__file__),
                                   "../../reports/C86L_ACCEPTANCE_MANIFEST.json")
_PLUGIN_FIELDS = ("nll", "correct", "confidence", "conf_bin")


def replay_c86l_acceptance():
    """Re-hash the C86L field artifacts against the accepted content-addressed manifest."""
    man = json.load(open(ACCEPTANCE_MANIFEST))
    inv = man["output_artifact_hashes"]
    checked = 0
    for a in inv:
        p = os.path.join(FIELD_ROOT, a["path"])
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != a["sha256"]:
            raise RuntimeError(f"C86L acceptance replay mismatch: {a['path']}")
        checked += 1
    return {"c86l_acceptance_gate": man["gate"], "artifacts_replayed": checked}


def _freeze_budget(queried, order, q_seq, budget):
    from .policies import budget_prefix, composite_select
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
        sel, metrics = composite_select(d["labels"], contribs, w, full=full, n_pool=len(order))
        selected[ctx] = int(sel); comp[ctx] = metrics["composite"].tolist()
    return {"budget": str(budget), "query_sequence": list(pre),
            "q_seq": [q_seq[i] for i in range(len(pre))], "lure_weights": w.tolist(),
            "receipts": [[t, queried[t][0]] for t in pre],
            "selected_by_context": selected, "composite_by_context": comp}


def _d1_worker(conn, pool_root, fdir, methods, budgets, chains):
    """PATH-BLIND worker: only the pipe + the client-visible pool. No sealed paths."""
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
                rec = {"method": method, "target": list(tgt), "chain": int(chain),
                       "seed": int(seed),
                       "budgets": [_freeze_budget(queried, order, q_seq, b) for b in budgets]}
                name = f"{method}__{tgt[0]}__{tgt[1]}__c{chain}.json"
                blob = json.dumps(rec, sort_keys=True)
                open(os.path.join(fdir, name), "w").write(blob)
                index.append({"file": f"freezes/{name}", "method": method, "target": list(tgt),
                              "chain": int(chain), "sha256": hashlib.sha256(blob.encode()).hexdigest()})
    client.close()
    json.dump(index, open(os.path.join(os.path.dirname(fdir), "worker_index.json"), "w"))


def run_d1(output_root: str) -> dict:
    t0 = time.time()
    accept = replay_c86l_acceptance()
    methods = list(METHOD_FREEZE["primary_registry"])
    budgets = list(BUDGET_GRID)
    chains = list(METHOD_FREEZE["seed_schedule"])
    staging = output_root + ".staging"
    fdir = os.path.join(staging, "freezes")
    os.makedirs(fdir, exist_ok=True)

    # launcher holds sealed paths, starts server, hands the worker only the pipe + pool
    conn, server_proc = start_server_process(ORACLE_ROOT, CONTRIB_ROOT)
    ctx = mp.get_context("spawn")
    worker = ctx.Process(target=_d1_worker, args=(conn, POOL_ROOT, fdir, methods, budgets, chains))
    worker.start()
    worker.join()
    try:
        server_proc.terminate()
    except Exception:
        pass
    if worker.exitcode != 0:
        raise RuntimeError(f"D1 worker failed exit {worker.exitcode}")

    index = json.load(open(os.path.join(staging, "worker_index.json")))
    manifest = {
        "stage": "C86D_D1_SELECTION",
        "c85u_accessed": False, "held_utility_opened": False,
        "path_blind_worker": True,
        "c86l_field_root": FIELD_ROOT, **accept,
        "methods": methods, "budgets": [str(b) for b in budgets],
        "n_targets": len({(e["target"][0], e["target"][1]) for e in index}),
        "chains": len(chains), "warm_start": 4,
        "seed_binding": "low64(SHA256(C86_ACTIVE_CHAIN_V1|dataset|subject|chain))",
        "n_freeze_files": len(index), "freeze_index": index,
        "d1_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(staging, "C86D_D1_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(staging, output_root)
    return {k: manifest[k] for k in ("stage", "n_freeze_files", "n_targets", "c85u_accessed",
                                     "path_blind_worker", "c86l_acceptance_gate")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--authorization", required=True)
    a = ap.parse_args()
    if a.authorization != "授权 C86D":
        raise SystemExit("C86D D1 requires --authorization '授权 C86D'")
    print(json.dumps(run_d1(a.output_root), indent=2))
