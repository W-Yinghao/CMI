"""Process-isolated C86D query server.

The server runs in a SEPARATE process that exclusively owns the sealed label
oracle and contribution store. The active client receives only a
``QueryClientHandle`` (a pipe) — it never learns the oracle/contribution paths and
can only ``query`` one physical trial at a time (one label + that trial's 8
context-specific contribution rows). This is the process/access-path separation
C86D adds on top of C86L's directory separation.
"""
from __future__ import annotations

import csv
import glob
import json
import multiprocessing as mp
import os

import numpy as np

_CONTRIB_FIELDS = ("nll", "correct", "confidence", "conf_bin", "signed_calibration")


def _load_sealed(oracle_root: str, contrib_root: str):
    """Build (target=(ds,subj)) trial -> label and -> {context: contribution row}."""
    labels = {(r["dataset"], int(r["target_subject_id"]), r["target_trial_id"]):
              int(r["canonical_class_label"])
              for r in csv.DictReader(open(os.path.join(oracle_root, "labels.csv")))}
    contrib = {}   # (ds,subj,trial) -> {context_key: {field: vec}}
    trial_label = {}
    for cf in sorted(glob.glob(os.path.join(contrib_root, "*.npz"))):
        z = np.load(cf, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        ds, subj = meta["dataset"], meta["subject"]
        ctx = f"panel={meta['panel']}|seed={meta['seed']}|level={meta['level']}"
        tids = list(z["trial_ids"]); y = z["true_label"]
        for j, t in enumerate(tids):
            key = (ds, subj, t)
            contrib.setdefault(key, {})[ctx] = {f: z[f][j] for f in _CONTRIB_FIELDS}
            trial_label[key] = int(y[j])
    return labels, trial_label, contrib


def _server_loop(conn, oracle_root: str, contrib_root: str):
    labels, trial_label, contrib = _load_sealed(oracle_root, contrib_root)
    trials_by_target = {}
    for (ds, subj, t) in contrib:
        trials_by_target.setdefault((ds, subj), []).append(t)
    attempts = {}
    n_attempts = 0
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        op = msg[0]
        if op == "shutdown":
            break
        elif op == "open":
            _, target, budget = msg
            pool = trials_by_target.get(tuple(target), [])
            if not pool:
                conn.send(("err", f"unknown target {target}")); continue
            n_attempts += 1
            aid = f"a{n_attempts}"
            cap = len(pool) if budget == "FULL" else int(budget)
            attempts[aid] = {"target": tuple(target), "cap": cap, "queried": set(), "n": 0}
            conn.send(("ok", aid))
        elif op == "query":
            _, aid, trial = msg
            at = attempts.get(aid)
            if at is None:
                conn.send(("err", "unknown attempt")); continue
            key = (at["target"][0], at["target"][1], trial)
            if key not in contrib:
                conn.send(("err", f"unknown trial {trial}")); continue
            if trial in at["queried"]:
                conn.send(("err", "duplicate query")); continue
            if at["n"] >= at["cap"]:
                conn.send(("err", "budget exhausted")); continue
            at["queried"].add(trial); at["n"] += 1
            conn.send(("ok", trial_label[key], contrib[key]))   # one label + 8 context rows
        elif op == "remaining":
            at = attempts[msg[1]]
            conn.send(("ok", at["cap"] - at["n"]))
        else:
            conn.send(("err", f"bad op {op}"))
    conn.close()


class C86DServerError(RuntimeError):
    pass


class QueryClientHandle:
    """Client-side handle: a pipe only. No oracle/contribution paths are held here."""

    def __init__(self, conn, proc):
        self._conn = conn          # ONLY the pipe; deliberately no sealed paths
        self._proc = proc

    def _rpc(self, msg):
        self._conn.send(msg)
        r = self._conn.recv()
        if r[0] == "err":
            raise C86DServerError(r[1])
        return r[1:]

    def open_attempt(self, target, budget):
        return self._rpc(("open", tuple(target), budget))[0]

    def query(self, attempt, trial):
        label, contexts = self._rpc(("query", attempt, trial))
        return label, contexts

    def remaining(self, attempt):
        return self._rpc(("remaining", attempt))[0]

    def close(self):
        try:
            self._conn.send(("shutdown",))
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()


def start_query_server(oracle_root: str, contrib_root: str) -> QueryClientHandle:
    """Launcher: spawns the sealed server process; returns a path-free client handle."""
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_server_loop, args=(child, oracle_root, contrib_root), daemon=True)
    proc.start()
    child.close()
    return QueryClientHandle(parent, proc)
