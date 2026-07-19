"""C86H integrated terminal runner — F0..H4, ONE gated entrypoint.

Physical barriers, not extra authorization rounds, provide the safety:
  F0  preflight            : verify §12 content-addressed bindings + resource envelope (outcome-free)
  F1  zoo train + freeze    : train the fresh 11-ch 648-model zoo (real: gated + authorized modules)
  F2  predictions + split   : target-unlabeled predictions, label-blind split (real: gated)
  H1  path-blind selection  : reuse the frozen C86D SPAWN server + path-blind worker (2,048 chains)
  H2  verify-then-open-held : fully verify every freeze BEFORE opening the held split, then endpoints
  H3  inference + taxonomy   : within-cohort max-T / tail / LOTO + two-level C86-A..E / L1..L4 + descriptor
  H4  immutable result       : write one result manifest, stop

``execute`` refuses without ``授权 C86H`` AND refuses again until the untouched field
exists (a separately authorized step). ``run_synthetic`` exercises the full F0->H4 chain
on a synthetic production-format field for the e2e test/benchmark, touching no real data.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np

from ..c86d.server import start_server_process
from ..c86d import selection_worker
from ..c86d.policies import chain_seed
from . import contract as K
from . import batch_h1, field_spec, held_eval

AUTHORIZATION_TOKEN = "授权 C86H"
REAL_FIELD_ROOT = ("/projects/EEG-foundation-model/yinghao/"
                   "oaci-c86h-untouched-confirmation-field-v1")
_CANON_CTX = field_spec.field_context_keys()


def _registered_target_cohort() -> dict:
    """The registered 53-target mapping (interface_name, int_subject) -> cohort."""
    tc = {}
    for cohort, spec in K.COHORTS.items():
        for subj in spec["subjects"]:
            n = int(subj) if str(subj).isdigit() else int(str(subj).split("-")[-1])
            tc[(cohort, n)] = cohort
    return tc


def _registered_cohort_dataset() -> dict:
    return {cohort: cohort for cohort in K.COHORTS}


# ---------------------------------------------------------------------------- F0
def f0_preflight(field_root: str | None = None) -> dict:
    bindings = K.verify_bindings()
    return {"stage": "C86H_F0_PREFLIGHT", "bindings": bindings,
            "active_chains": K.ACTIVE_CHAINS, "maxt_draws": K.MAXT_DRAWS,
            "field_present": bool(field_root) and os.path.isdir(field_root),
            "ready": bindings["ok"]}


# ---------------------------------------------------------------------------- H1
def _run_h1_selection(field_root, out_dir, methods, budgets, chains) -> tuple:
    """Reuse the frozen C86D SPAWN server + path-blind worker. Sealed roots stay in this
    launcher; the worker is spawned with only the pipe + client-visible pool."""
    pool_root = os.path.join(field_root, "acquisition_unlabeled_pool")
    oracle_root = os.path.join(field_root, "acquisition_label_oracle")
    contrib_root = os.path.join(field_root, "query_contribution_store")
    fdir = os.path.join(out_dir, "freezes")
    os.makedirs(fdir, exist_ok=True)
    conn, server_proc = start_server_process(oracle_root, contrib_root)
    ctx = mp.get_context("spawn")
    worker = ctx.Process(target=selection_worker.run_worker,
                         args=(conn, pool_root, fdir, methods, budgets, chains))
    worker.start()
    worker.join()
    try:
        server_proc.terminate()
    except Exception:
        pass
    if worker.exitcode != 0:
        raise RuntimeError(f"H1 selection worker failed (exit {worker.exitcode})")
    index = json.load(open(os.path.join(out_dir, "worker_index.json")))
    return fdir, index


# ---------------------------------------------------------------------------- H2 verify
def _load_and_verify_freezes(fdir, index, chains, methods, expected_targets) -> dict:
    """Fully verify every freeze BEFORE any held label is opened. Returns freezes dict.

    Enforces (C86D-parity): per-blob sha256 tamper + index/blob identity; method/chain
    membership; per-(method,target,chain) UNIQUENESS; target-bound seed; exactly 5 budgets
    {4,8,16,32,FULL} with FULL AVAILABLE; INPUT_UNAVAILABLE only for a finite budget above
    its pool; per-AVAILABLE 8 canonical contexts + composite shape (81,) + selected==first
    argmax; nested prefix; expected target registry + Cartesian completeness; and BLOCKING
    FULL acquisition invariance (all methods present AND one candidate per (target,ctx,chain)).
    """
    def req(cond, msg):
        if not cond:
            raise RuntimeError(f"C86H freeze verification failed: {msg}")

    req(len(index) > 0, "empty freeze index")
    order5 = ("4", "8", "16", "32", "FULL")
    freezes, full_by, full_ctx, full_len, input_pool, seen_methods = {}, {}, {}, {}, {}, set()
    for entry in index:
        path = os.path.join(os.path.dirname(fdir), entry["file"])
        blob = open(path, "rb").read()
        req(hashlib.sha256(blob).hexdigest() == entry["sha256"], f"tamper {entry['file']}")
        rec = json.loads(blob)
        method = rec["method"]; tgt = tuple(rec["target"]); chain = int(rec["chain"])
        req(method == entry["method"] and tgt == tuple(entry["target"])
            and chain == int(entry["chain"]), f"index/blob identity {entry['file']}")
        seen_methods.add(method)
        req(method in methods, f"unexpected method {method}")
        req(chain in chains, f"unexpected chain {chain}")
        req((method, tgt, chain) not in freezes, f"duplicate freeze {(method, tgt, chain)}")
        req(rec["seed"] == chain_seed(tgt[0], tgt[1], chain), f"seed binding {tgt} {chain}")
        blist = rec["budgets"]
        req(len(blist) == 5, f"expected 5 budget rows, got {len(blist)}")
        budgets = {b["budget"]: b for b in blist}
        req(set(budgets) == set(order5), f"budget set {set(budgets)}")
        req(budgets["FULL"]["status"] == "AVAILABLE", "FULL must be AVAILABLE")
        for b in order5:
            fb = budgets[b]; st = fb.get("status")
            req(st in ("AVAILABLE", "INPUT_UNAVAILABLE"), f"bad status {st} @ {b}")
            if st == "INPUT_UNAVAILABLE":
                req(b != "FULL" and int(b) > int(fb["pool_size"]),
                    f"invalid INPUT_UNAVAILABLE @ {b} pool={fb.get('pool_size')}")
                input_pool.setdefault(tgt, set()).add(int(fb["pool_size"]))
                continue
            req(set(fb["selected_by_context"]) == set(_CANON_CTX), f"contexts @ {b}")
            req(set(fb.get("component_sha_by_context", {})) == set(_CANON_CTX),
                f"component_sha contexts @ {b}")
            for cx, sel in fb["selected_by_context"].items():
                req(0 <= int(sel) <= 80, f"selected range {sel}")
                comp = np.asarray(fb["composite_by_context"][cx], dtype=np.float64)
                req(comp.shape == (81,), f"composite shape @ {b} {cx}")
                req(int(sel) == int(np.argmax(comp)), f"selected==first argmax @ {b} {cx}")
            qs = fb["query_sequence"]
            if b != "FULL":
                req(len(qs) == int(b), f"query_sequence len @ {b}")
            req(len(set(qs)) == len(qs), f"duplicate query trials @ {b}")
            req(len(fb["q_seq"]) == len(qs) and len(fb["lure_weights"]) == len(qs)
                and len(fb["receipts"]) == len(qs), f"parallel-array lengths @ {b}")
            req(all(0.0 < q <= 1.0 for q in fb["q_seq"]), f"q_seq in (0,1] @ {b}")
            req(all(np.isfinite(w) and w >= 0.0 for w in fb["lure_weights"]),
                f"lure weights finite>=0 @ {b}")
            req([r[0] for r in fb["receipts"]] == qs
                and all(int(r[1]) in (0, 1) for r in fb["receipts"]), f"receipts @ {b}")
        avail = [b for b in order5 if budgets[b]["status"] == "AVAILABLE"]
        seqs = {b: budgets[b]["query_sequence"] for b in avail}
        for i in range(len(avail) - 1):
            s, t = seqs[avail[i]], seqs[avail[i + 1]]
            req(s == t[:len(s)], f"nested prefix {avail[i]}<{avail[i+1]}")
        freezes[(method, tgt, chain)] = budgets
        full_len.setdefault(tgt, set()).add(len(budgets["FULL"]["query_sequence"]))
        for cx, sel in budgets["FULL"]["selected_by_context"].items():
            full_by.setdefault((tgt, cx, chain), {})[method] = int(sel)
            full_ctx.setdefault((tgt, cx), set()).add(int(sel))
    req(seen_methods == set(methods), f"method coverage {seen_methods}")
    # expected target registry + Cartesian completeness (no cherry-picking / missing cells)
    exp = {tuple(t) for t in expected_targets}
    seen_tgts = {t for (_m, t, _c) in freezes}
    req(seen_tgts == exp, f"target registry mismatch: {seen_tgts ^ exp}")
    for t in exp:
        for m in methods:
            for c in chains:
                req((m, t, c) in freezes, f"missing freeze cell {m} {t} c{c}")
    # BLOCKING FULL acquisition invariance: all methods present per (tgt,ctx,chain) AND one
    # candidate across all methods AND chains (FULL queries the whole pool -> chain-invariant)
    for (tgt, cx, chain), msel in full_by.items():
        req(set(msel) == set(methods), f"FULL missing method {(tgt, cx, chain)}: {set(msel)}")
    for (tgt, cx), sels in full_ctx.items():
        req(len(sels) == 1, f"FULL not invariant across methods+chains {(tgt, cx)}: {sels}")
    # pool-size consistency: one FULL length per target across chains/methods, and any
    # INPUT_UNAVAILABLE pool_size equals it
    for tgt, lens in full_len.items():
        req(len(lens) == 1, f"FULL length inconsistent across chains {tgt}: {lens}")
        psize = next(iter(lens))
        req(all(p == psize for p in input_pool.get(tgt, set())),
            f"INPUT_UNAVAILABLE pool_size != FULL pool length {tgt}")
    return freezes


# ------------------------------------------------------- batch H1 (label-independent) / H2
def _h1a_worker(conn, pool_root, orders_dir, methods, chains):
    """LABEL-FREE process: capability-isolated order generation (only the unlabeled pool +
    orders_dir; NO oracle/contribution path is passed to this process)."""
    try:
        pool_sizes = batch_h1.run_h1a(pool_root, orders_dir, methods, list(chains))
        conn.send(("ok", {str(k): v for k, v in pool_sizes.items()}))
    except Exception:
        import traceback
        conn.send(("err", traceback.format_exc()))
    finally:
        conn.close()


def _h1b_worker(conn, orders_dir, oracle_root, contrib_root, out_dir, methods, chains):
    """SEALED process: read the label-free orders + acquisition labels/contributions once and
    batch-evaluate the composite selections; emit selections + content hashes only."""
    try:
        man = batch_h1.run_h1b_sealed(orders_dir, oracle_root, contrib_root, out_dir,
                                      methods, list(chains))
        conn.send(("ok", man))
    except Exception:
        import traceback
        conn.send(("err", traceback.format_exc()))
    finally:
        conn.close()


def _h2_batch_worker(conn, freeze_dir, orders_dir, held_root, expected_targets, methods,
                     chains, target_cohort, cohort_dataset):
    """H2 process (no server/oracle capability): verify every H1 freeze AND reconcile it
    against the label-free H1a orders BEFORE opening held; then evaluate."""
    try:
        batch_h1.verify_h1(freeze_dir, orders_dir, expected_targets, methods, list(chains))
        freezes = batch_h1.load_selections(freeze_dir, methods, expected_targets, list(chains))
        held = field_spec.load_held_field(held_root)
        result = held_eval.evaluate(freezes, held, target_cohort, cohort_dataset)
        conn.send(("ok", result))
    except Exception:
        import traceback
        conn.send(("err", traceback.format_exc()))
    finally:
        conn.close()


def _spawn_recv(target, args):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=target, args=(child,) + tuple(args))
    proc.start()
    child.close()                       # so parent.recv() EOFs if the worker dies before send()
    try:
        status, payload = parent.recv()
    except EOFError:
        proc.join()
        raise RuntimeError(f"C86H worker died before sending a result (exit {proc.exitcode})")
    proc.join()
    if status == "err":
        raise RuntimeError(payload)
    return payload


# --------------------------------------------------------------------- confirmation
def run_confirmation(field_root, out_dir, target_cohort, cohort_dataset,
                     authorization: str = "", chains=None, synthetic: bool = False) -> dict:
    """F0 -> H1 -> verify -> H2/H3 -> H4 over an existing field. This is the code path that
    opens real EEG/label data, so the '授权 C86H' token is enforced HERE, not only on
    execute(). A synthetic run must present a synthetic field and must never target the real
    field root."""
    t0 = time.time()
    real_root = os.path.realpath(REAL_FIELD_ROOT)
    this = os.path.realpath(field_root)
    syn_present = os.path.isfile(os.path.join(field_root, "C86H_SYNTHETIC_FIELD_MANIFEST.json"))
    if synthetic:
        if this == real_root or not syn_present:
            raise RuntimeError("synthetic run requires a synthetic field and must not target "
                               "the real field root")
    else:
        if authorization != AUTHORIZATION_TOKEN:
            raise SystemExit("C86H real confirmation requires authorization '授权 C86H'")
        if this != real_root:
            raise RuntimeError("real confirmation must target the registered REAL_FIELD_ROOT")
        if not os.path.isdir(field_root):
            raise RuntimeError("C86H untouched field not generated (separately authorized step)")
        # bind the confirmed population to the registered cohorts (no truncation/cherry-picking)
        if set(cohort_dataset) != set(K.COHORTS):
            raise RuntimeError(f"real cohort set {set(cohort_dataset)} != registered {set(K.COHORTS)}")
        if len(target_cohort) != K.N_TARGETS:
            raise RuntimeError(f"real target count {len(target_cohort)} != registered {K.N_TARGETS}")
    if chains is None:
        chains = list(range(K.ACTIVE_CHAINS))
    elif not synthetic:
        raise RuntimeError("real confirmation must use the full registered chain schedule "
                           "range(2048); caller-reduced chains are forbidden")
    else:
        chains = list(chains)
    os.makedirs(out_dir, exist_ok=True)
    pre = f0_preflight(field_root)
    if not pre["bindings"]["ok"]:
        raise RuntimeError(f"C86H F0 binding failure: {pre['bindings']['mismatches']}")
    methods = list(K.METHOD_REGISTRY)
    pool_root = os.path.join(field_root, "acquisition_unlabeled_pool")
    oracle_root = os.path.join(field_root, "acquisition_label_oracle")
    contrib_root = os.path.join(field_root, "query_contribution_store")
    held_root = os.path.join(field_root, "held_evaluation_field")
    orders_dir = os.path.join(out_dir, "orders")
    h1_dir = os.path.join(out_dir, "h1")
    exp = sorted(target_cohort)

    # H1a: label-free batch order generation in a CAPABILITY-ISOLATED spawned process
    # (only the unlabeled pool + orders_dir; no sealed roots reach it)
    _spawn_recv(_h1a_worker, (pool_root, orders_dir, methods, chains))
    # H1b: SEALED spawned process (labels in, selections out)
    h1_manifest = _spawn_recv(_h1b_worker,
                              (orders_dir, oracle_root, contrib_root, h1_dir, methods, chains))
    # H2: separate spawned process (no server/oracle capability); verify + RECONCILE against the
    # label-free H1a orders BEFORE opening held
    result = _spawn_recv(_h2_batch_worker,
                         (h1_dir, orders_dir, held_root, exp, methods, chains,
                          target_cohort, cohort_dataset))

    # H4: complete + atomic immutable terminal result (fresh attempt root, staging + rename)
    manifest = {
        "stage": "C86H_H4_TERMINAL_RESULT", "confirmatory": True,
        "held_opened_after_freeze_verification": True,
        "method_registry": methods, "n_chains": len(chains),
        "h1_files": h1_manifest["n_files"],
        "classification": result["classification"],
        "endpoints": result["endpoints"],
        "full_ceiling": result["full_ceiling"],
        "active_gain": result["active_gain"],
        "inference_detail": result.get("inference_detail"),
        "materiality_margin": K.MATERIALITY_MARGIN,
        "maxt_draws": K.MAXT_DRAWS,
        "pooled_dataset_pvalue": K.POOLED_DATASET_PVALUE,
        "h1_manifest_files": h1_manifest["file_sha256"],
        "bindings": pre["bindings"],
        "stop_rule": "terminal: one field, one confirmation, one audit, stop; no auto-C87",
        "seconds": round(time.time() - t0, 2),
    }
    result_path = os.path.join(out_dir, "C86H_TERMINAL_RESULT.json")
    if os.path.exists(result_path):                        # immutability: C86H is one-shot
        raise RuntimeError(f"C86H terminal result already exists at {result_path}; refusing to "
                           "overwrite (terminal: one field, one confirmation, one audit, stop)")
    blob = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    staging = result_path + ".staging"
    with open(staging, "w") as fh:
        fh.write(blob)
    os.replace(staging, result_path)                       # atomic finalize
    with open(result_path + ".sha256", "w") as fh:         # persisted integrity digest (sidecar)
        fh.write(digest + "  C86H_TERMINAL_RESULT.json\n")
    manifest["result_sha256"] = digest
    return manifest


# --------------------------------------------------------------------- gated real path
def execute(authorization: str, output_root: str | None = None):
    if authorization != AUTHORIZATION_TOKEN:
        raise SystemExit("C86H requires authorization '授权 C86H'; this build is prep only")
    bindings = K.verify_bindings()
    if not bindings["ok"]:
        raise RuntimeError(f"C86H binding verification failed: {bindings['mismatches']}")
    from . import f1f2
    if not os.path.isdir(REAL_FIELD_ROOT):
        f1f2.f1_train_zoo(authorization, REAL_FIELD_ROOT)     # F1 -> raises (authorized step)
        f1f2.f2_generate_predictions(authorization, REAL_FIELD_ROOT)  # F2 -> raises
    # Under authorization the flow is: F1 train zoo -> F2 predictions+split -> run_confirmation.
    return run_confirmation(REAL_FIELD_ROOT, output_root or (REAL_FIELD_ROOT + ".run"),
                            _registered_target_cohort(), _registered_cohort_dataset(),
                            authorization=authorization, synthetic=False)


# --------------------------------------------------------------- synthetic e2e (no real data)
def run_synthetic(root: str, seed: int = 12345, chains=(0, 1), n_trials: int = 88) -> dict:
    """Full F0->H4 on a synthetic production-format field. Touches NO real EEG/label."""
    cohorts = {
        "SYN_COHORT_A": {"dataset": "SYN_A", "subjects": [1, 2, 3], "n_trials": n_trials},
        "SYN_COHORT_B": {"dataset": "SYN_B", "subjects": [1, 2, 3, 4], "n_trials": n_trials},
    }
    field_root = os.path.join(root, "field")
    field_spec.synthesize_field(field_root, cohorts, seed=seed)
    target_cohort, cohort_dataset = {}, {}
    for cohort, spec in cohorts.items():
        cohort_dataset[cohort] = cohort
        for subj in spec["subjects"]:
            target_cohort[(spec["dataset"], int(subj))] = cohort
    return run_confirmation(field_root, os.path.join(root, "run"),
                            target_cohort, cohort_dataset, chains=list(chains), synthetic=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="C86H integrated runner (prep only)")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--authorization", default="")
    args = ap.parse_args()
    if args.preflight:
        print(json.dumps(f0_preflight(REAL_FIELD_ROOT), indent=2, ensure_ascii=False))
    else:
        execute(args.authorization)
