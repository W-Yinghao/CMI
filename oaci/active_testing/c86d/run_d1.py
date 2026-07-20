"""C86D Stage D1 — SELECTION launcher. No C85U import anywhere in this tree.

The launcher replays the accepted C86L content-addressed manifest, derives the
externally-bound target registry + per-target pool sizes from the accepted field,
holds the sealed oracle/contribution paths, starts the server process, and spawns
the PATH-BLIND worker (``selection_worker``) with only the pipe + the client-visible
pool. The worker module defines no sealed path.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import multiprocessing as mp
import os
import time

from .core import BUDGET_GRID, METHOD_FREEZE            # no C85U names here
from .server import start_server_process
from . import selection_worker

FIELD_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-c86l-development-field-v1"
POOL_ROOT = os.path.join(FIELD_ROOT, "acquisition_unlabeled_pool")
ORACLE_ROOT = os.path.join(FIELD_ROOT, "acquisition_label_oracle")
CONTRIB_ROOT = os.path.join(FIELD_ROOT, "query_contribution_store")
ACCEPTANCE_MANIFEST = os.path.join(os.path.dirname(__file__),
                                   "../../reports/C86L_ACCEPTANCE_MANIFEST.json")
_ACCEPTED_GATE = ("C86L_DEVELOPMENT_FIELD_CONTENT_ADDRESSED_AND_FULLY_REPLAYED_"
                  "READY_FOR_C86D_PROTOCOL")
_ACCEPTED_INVENTORY = 1891


def replay_c86l_acceptance():
    man = json.load(open(ACCEPTANCE_MANIFEST))
    if man.get("acceptance_ok") is not True:
        raise RuntimeError("C86L acceptance not acceptance_ok=true")
    if man.get("gate") != _ACCEPTED_GATE:
        raise RuntimeError(f"C86L acceptance gate mismatch: {man.get('gate')}")
    inv = man["output_artifact_hashes"]
    if len(inv) != _ACCEPTED_INVENTORY:
        raise RuntimeError(f"C86L inventory {len(inv)} != {_ACCEPTED_INVENTORY}")
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
    if checked != _ACCEPTED_INVENTORY:
        raise RuntimeError(f"C86L artifacts checked {checked} != {_ACCEPTED_INVENTORY}")
    return {"c86l_acceptance_gate": man["gate"], "artifacts_replayed": checked}


def _expected_targets_and_pools():
    """Target registry (from the accepted context index) + per-target construction pool size."""
    ctx_index = json.load(open(os.path.join(FIELD_ROOT, "C86L_CONTEXT_INDEX.json")))
    targets = sorted({(c["dataset"], int(c["subject"])) for c in ctx_index})
    pools = collections.defaultdict(int)
    for r in csv.DictReader(open(os.path.join(ORACLE_ROOT, "labels.csv"))):
        pools[(r["dataset"], int(r["target_subject_id"]))] += 1
    return [list(t) for t in targets], {f"{t[0]}|{t[1]}": pools[t] for t in targets}


def run_d1(output_root: str) -> dict:
    t0 = time.time()
    accept = replay_c86l_acceptance()
    targets, pool_sizes = _expected_targets_and_pools()
    methods = list(METHOD_FREEZE["primary_registry"])
    budgets = list(BUDGET_GRID)
    chains = list(METHOD_FREEZE["seed_schedule"])
    staging = output_root + ".staging"
    fdir = os.path.join(staging, "freezes")
    os.makedirs(fdir, exist_ok=True)

    conn, server_proc = start_server_process(ORACLE_ROOT, CONTRIB_ROOT)
    ctx = mp.get_context("spawn")
    worker = ctx.Process(target=selection_worker.run_worker,
                         args=(conn, POOL_ROOT, fdir, methods, budgets, chains))
    worker.start(); worker.join()
    try:
        server_proc.terminate()
    except Exception:
        pass
    if worker.exitcode != 0:
        raise RuntimeError(f"D1 worker failed exit {worker.exitcode}")

    index = json.load(open(os.path.join(staging, "worker_index.json")))
    manifest = {
        "stage": "C86D_D1_SELECTION",
        "c85u_accessed": False, "held_utility_opened": False, "path_blind_worker": True,
        "c86l_field_root": FIELD_ROOT, **accept,
        "methods": methods, "budgets": [str(b) for b in budgets], "chain_ids": [int(c) for c in chains],
        "expected_targets": targets, "n_targets": len(targets),
        "target_pool_sizes": pool_sizes,
        "expected_freeze_count": len(methods) * len(targets) * len(chains),
        "seed_binding": "low64(SHA256(C86_ACTIVE_CHAIN_V1|dataset|subject|chain))",
        "n_freeze_files": len(index), "freeze_index": index,
        "d1_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(staging, "C86D_D1_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(staging, output_root)
    return {k: manifest[k] for k in ("stage", "n_freeze_files", "n_targets", "c85u_accessed",
                                     "path_blind_worker", "c86l_acceptance_gate",
                                     "expected_freeze_count")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--authorization", required=True)
    a = ap.parse_args()
    if a.authorization != "授权 C86D":
        raise SystemExit("C86D D1 requires --authorization '授权 C86D'")
    print(json.dumps(run_d1(a.output_root), indent=2))
