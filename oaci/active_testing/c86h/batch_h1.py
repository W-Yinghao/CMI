"""C86H batch H1 — label-independent order generation (H1a) + sealed batch contribution
evaluation (H1b).

The acquisition order is label-INDEPENDENT (P0 uniform; A1/A2H scores are precomputed from
unlabeled probabilities; querying never updates the score), so the ~93M-RPC per-query server
is unnecessary. H1a generates every (target, method, chain) order from unlabeled probabilities
alone; a SEALED H1b process then reads each target's acquisition labels/contributions ONCE and
batch-evaluates the composite selection. Both stages REUSE the frozen dispatcher kernels
(``acquisition_path``, ``freeze_budget``), so selections are byte-identical to the per-RPC
C86D worker — only the IPC and serialization change.

Compact freeze: one NPZ per (target, method) holding all chains + a content-addressed H1
manifest, i.e. ~53*3 = 159 files rather than 53*3*2048 JSONs.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from ..c86d.policies import acquisition_path, chain_seed
from ..c86d.selection_worker import freeze_budget
from ..c86d.server import _load_sealed
from . import contract as K, field_spec

CONTEXTS = field_spec.field_context_keys()
BUDGETS = tuple(str(b) for b in K.BUDGET_GRID)          # ("4","8","16","32","FULL")
_FULL_BI = BUDGETS.index("FULL")
WARM_START = 4


def _budget_arg(b: str):
    return "FULL" if b == "FULL" else int(b)


# ------------------------------------------------------------------- H1a (label-free)
def run_h1a(pool_root: str, orders_dir: str, methods, chains) -> dict:
    """Generate every (target, method, chain) order + q_seq from UNLABELED probabilities and
    write one order NPZ per target. Reads NO labels; the seed (chain_seed) is method-independent,
    so the first WARM_START picks are shared across methods (paired CRN)."""
    pool = field_spec.load_pool(pool_root)
    os.makedirs(orders_dir, exist_ok=True)
    pool_sizes = {}
    for tgt in sorted(pool):
        ds, subj = tgt
        ptr = sorted(pool[tgt]); ps = len(ptr)
        idxmap = {t: i for i, t in enumerate(ptr)}
        seeds = [chain_seed(ds, subj, c) for c in chains]
        O = np.empty((len(methods), len(chains), ps), dtype=np.int64)
        Q = np.empty((len(methods), len(chains), ps), dtype=np.float64)
        for mi, method in enumerate(methods):
            for ci in range(len(chains)):
                order, q = acquisition_path(pool[tgt], method, seeds[ci])
                O[mi, ci] = [idxmap[t] for t in order]
                Q[mi, ci] = q
        np.savez(os.path.join(orders_dir, f"{ds}__{subj}.npz"),
                 dataset=ds, subject=subj, pool_trials=np.array(ptr),
                 methods=np.array(list(methods)),
                 chains=np.array(list(chains), dtype=np.int64),
                 seeds=np.array(seeds, dtype=np.uint64), orders=O, q_seq=Q, pool_size=ps)
        pool_sizes[tgt] = ps
    return pool_sizes


# ------------------------------------------------------------------- H1b (sealed batch)
def run_h1b_sealed(orders_dir, oracle_root, contrib_root, out_dir, methods, chains) -> dict:
    """SEALED: read the label-free order matrices + the acquisition labels/contributions once,
    batch-evaluate composite selections via the frozen ``freeze_budget``, and write compact
    per-(target,method) freezes + a content-addressed manifest. Emits ONLY selections + content
    hashes (never bulk labels)."""
    import glob
    _, trial_label, contrib = _load_sealed(oracle_root, contrib_root)
    os.makedirs(out_dir, exist_ok=True)
    file_index = {}
    for ofile in sorted(glob.glob(os.path.join(orders_dir, "*.npz"))):
        z = np.load(ofile, allow_pickle=True)
        ds = str(z["dataset"]); subj = int(z["subject"])
        ptr = [str(t) for t in z["pool_trials"]]
        ps = int(z["pool_size"]); O = z["orders"]; Q = z["q_seq"]
        seeds = z["seeds"]; ofile_methods = [str(m) for m in z["methods"]]
        queried_full = {t: (int(trial_label[(ds, subj, t)]),
                            {cx: {f: np.asarray(r[f]).tolist()
                                  for f in ("nll", "correct", "confidence", "conf_bin")}
                             for cx, r in contrib[(ds, subj, t)].items()})
                        for t in ptr}
        avail = np.array([1 if (b == "FULL" or int(b) <= ps) else 0 for b in BUDGETS],
                         dtype=np.int64)
        for mi, method in enumerate(methods):
            omi = ofile_methods.index(method)
            selected = np.full((len(chains), len(BUDGETS), len(CONTEXTS)), -1, dtype=np.int64)
            comp_parts = []
            for ci in range(len(chains)):
                order = [ptr[i] for i in O[omi, ci]]
                q = list(Q[omi, ci])
                for bi, b in enumerate(BUDGETS):
                    fb = freeze_budget(queried_full, order, q, _budget_arg(b))
                    if fb["status"] != "AVAILABLE":
                        continue
                    for cxi, cx in enumerate(CONTEXTS):
                        selected[ci, bi, cxi] = int(fb["selected_by_context"][cx])
                        comp_parts.append(fb["component_sha_by_context"][cx].encode())
            content_sha = hashlib.sha256(b"".join(comp_parts) + selected.tobytes()).hexdigest()
            fname = f"{method}__{ds}__{subj}.npz"
            path = os.path.join(out_dir, fname)
            np.savez(path, method=method, dataset=ds, subject=subj,
                     pool_trials=np.array(ptr), seeds=np.asarray(seeds, dtype=np.uint64),
                     orders=O[omi], q_seq=Q[omi], selected=selected, availability=avail,
                     chains=np.array(list(chains), dtype=np.int64),
                     content_sha=content_sha, pool_size=ps)
            with open(path, "rb") as fh:
                file_index[fname] = hashlib.sha256(fh.read()).hexdigest()
    manifest = {"stage": "C86H_H1_BATCH", "methods": list(methods),
                "n_chains": len(chains), "contexts": CONTEXTS, "budgets": list(BUDGETS),
                "n_files": len(file_index), "file_sha256": file_index}
    with open(os.path.join(out_dir, "C86H_H1_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# ------------------------------------------------------------------- verify + load
def _load_h1a_orders(orders_dir, ds, subj):
    """Load the LABEL-FREE H1a orders for a target (per method): (orders[m], q_seq[m], seeds)."""
    z = np.load(os.path.join(orders_dir, f"{ds}__{subj}.npz"), allow_pickle=True)
    ofm = [str(m) for m in z["methods"]]
    return z["orders"], z["q_seq"], z["seeds"], ofm, int(z["pool_size"])


def verify_h1(freeze_dir, orders_dir, expected_targets, methods, chains) -> None:
    """Verify every H1 freeze BEFORE held is opened, and RECONCILE it against the label-free
    H1a orders (the label-independence audit). Loads the H1 manifest from disk (not from the
    in-memory object the label-holding H1b returned)."""
    def req(cond, msg):
        if not cond:
            raise RuntimeError(f"C86H H1 verification failed: {msg}")

    exp = {tuple(t) for t in expected_targets}
    max_cand = K.CANDIDATES_PER_CONTEXT - 1
    manifest = json.load(open(os.path.join(freeze_dir, "C86H_H1_MANIFEST.json")))
    req(set(manifest["methods"]) == set(methods), "method set")
    req(manifest["n_chains"] == len(chains), "chain count")
    req(manifest["contexts"] == CONTEXTS, "contexts")
    seen, warm, full_ctx = {}, {}, {}
    for method in methods:
        for tgt in exp:
            ds, subj = tgt
            fname = f"{method}__{ds}__{subj}.npz"
            path = os.path.join(freeze_dir, fname)
            req(os.path.isfile(path), f"missing freeze {fname}")
            with open(path, "rb") as fh:
                req(hashlib.sha256(fh.read()).hexdigest() == manifest["file_sha256"].get(fname),
                    f"tamper {fname}")
            z = np.load(path, allow_pickle=True)
            req(str(z["method"]) == method and str(z["dataset"]) == ds
                and int(z["subject"]) == subj, f"identity {fname}")
            ps = int(z["pool_size"]); nc = len(chains)
            req(list(z["chains"]) == list(chains), f"chains {fname}")
            req([int(s) for s in z["seeds"]]
                == [chain_seed(ds, subj, c) for c in chains], f"seed binding {fname}")
            avail = list(z["availability"])
            req(avail == [1 if (b == "FULL" or int(b) <= ps) else 0 for b in BUDGETS],
                f"availability {fname}")
            orders = z["orders"]; q_seq = z["q_seq"]; selected = z["selected"]
            req(orders.shape == (nc, ps), f"orders shape {fname}")
            for ci in range(nc):
                req(sorted(orders[ci].tolist()) == list(range(ps)),
                    f"order not a permutation {fname} c{ci}")
            req(bool(np.all((q_seq > 0.0) & (q_seq <= 1.0))), f"q_seq not in (0,1] {fname}")
            # RECONCILE against the label-free H1a orders (audit label-independence)
            hO, hQ, hSeeds, hMethods, hps = _load_h1a_orders(orders_dir, ds, subj)
            req(hps == ps, f"pool_size drift vs orders {fname}")
            omi = hMethods.index(method)
            req(np.array_equal(orders, hO[omi]), f"freeze order != label-free H1a order {fname}")
            req(np.array_equal(q_seq, hQ[omi]), f"freeze q_seq != label-free H1a q_seq {fname}")
            req([int(s) for s in z["seeds"]] == [int(s) for s in hSeeds],
                f"seed != H1a seed {fname}")
            for bi, b in enumerate(BUDGETS):
                if avail[bi] == 0:
                    req(bool(np.all(selected[:, bi, :] == -1)), f"unavail selected {fname} {b}")
                else:
                    req(bool(np.all((selected[:, bi, :] >= 0) & (selected[:, bi, :] <= max_cand))),
                        f"selected range {fname} {b}")
            seen[(method, tgt)] = True
            warm[(method, tgt)] = orders[:, :WARM_START]
            for ci in range(nc):
                for cxi in range(len(CONTEXTS)):
                    full_ctx.setdefault((tgt, cxi), set()).add(int(selected[ci, _FULL_BI, cxi]))
    for method in methods:
        for tgt in exp:
            req((method, tgt) in seen, f"missing cell {method} {tgt}")
    for tgt in exp:
        base = warm[(methods[0], tgt)]
        for method in methods[1:]:
            req(np.array_equal(warm[(method, tgt)], base), f"warm-start CRN drift {tgt} {method}")
    for (tgt, cxi), sels in full_ctx.items():
        req(len(sels) == 1, f"FULL not invariant {tgt} ctx{cxi}: {sels}")


def load_selections(freeze_dir, methods, expected_targets, chains) -> dict:
    """Reconstruct freezes[(method,tgt,chain)] = {budget: {status, selected_by_context}}
    for the H2 held evaluator (same interface held_eval.evaluate expects)."""
    freezes = {}
    for method in methods:
        for tgt in {tuple(t) for t in expected_targets}:
            ds, subj = tgt
            z = np.load(os.path.join(freeze_dir, f"{method}__{ds}__{subj}.npz"),
                        allow_pickle=True)
            selected = z["selected"]; avail = list(z["availability"]); ps = int(z["pool_size"])
            for ci, chain in enumerate(chains):
                bud = {}
                for bi, b in enumerate(BUDGETS):
                    if avail[bi] == 0:
                        bud[b] = {"status": "INPUT_UNAVAILABLE", "pool_size": ps}
                    else:
                        bud[b] = {"status": "AVAILABLE",
                                  "selected_by_context": {cx: int(selected[ci, bi, cxi])
                                                          for cxi, cx in enumerate(CONTEXTS)}}
                freezes[(method, tgt, int(chain))] = bud
    return freezes
