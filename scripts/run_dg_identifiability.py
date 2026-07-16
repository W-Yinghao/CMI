#!/usr/bin/env python
"""HARDENED DG-identifiability rescue on REAL EEG (Phase-0 protocol + Phase-1 bounded basis rescue).

For each EEGNet frozen dump (one LOSO fold) and each candidate basis family x {full, contested} x objective:
  * crossfit_target_oracle    : honest target-hindsight UPPER BOUND (select on T_select, report on T_query).
  * nested_source_meta        : SOURCE-ONLY refittable rule (k*), pseudo-target excluded from basis/head/sel.
  * apply_rule_to_target_full : the rule's true gain on T_query + matched-rank random control.
Writes one row per (cell, family, contested, objective). Aggregation computes RecoveryRatio + the 4-state
verdict with subject/fold-cluster CIs; CMI certification (posterior-KL ruler) is a SEPARATE step on winners.

  python scripts/run_dg_identifiability.py --dataset BNCI2014_001 --backbone EEGNet --seeds 0 1 2
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eval.dg_identifiability import (get_candidate_basis, crossfit_target_oracle,
    nested_source_meta_multi, apply_rule_to_target_full)

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
CFG = REPO / "configs" / "cmi_trace_relaxation_ladder.yaml"
FAMILIES = ["marg", "cond", "rule", "grad"]
OBJECTIVES = ["mean_1se", "cvar25"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--max_rank", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--oracle_mode", default="greedy", choices=["greedy", "prefix"],
                    help="existence oracle: greedy=arbitrary-coordinate (honest upper bound); prefix=top-k only (weaker)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg_hash = hashlib.sha256(CFG.read_bytes()).hexdigest()[:12] if CFG.exists() else "no_config"
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    dd = REPO / "tos_cmi/results/tos_cmi_eeg_frozen" / f"{a.dataset}_{a.backbone}_LOSO"
    cells = [p for p in sorted(glob.glob(str(dd / "sub*_erm_lam0_seed*.npz")))
             if any(p.endswith(f"_seed{s}.npz") for s in a.seeds)]
    if a.limit:
        cells = cells[: a.limit]
    if not cells:
        raise SystemExit(f"[dg-id] no dumps for {a.dataset}/{a.backbone}")
    outdir = OUT / f"{a.dataset}_{a.backbone}"; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / f"raw_rows_seed{'-'.join(a.seeds)}.jsonl"    # per-invocation file (no concurrent-append races)
    done = set()
    if raw.exists():
        for line in open(raw):
            try:
                r = json.loads(line); done.add((r["heldout_subject"], r["seed"], r["family"], r["contested"], r["objective"]))
            except Exception:
                pass
    print(f"[dg-id] {a.dataset}/{a.backbone} cells={len(cells)} families={FAMILIES}", flush=True)
    t0 = time.time()
    with open(raw, "a") as fh:
        for i, cp in enumerate(cells):
            f = feat_from_tos_dump(cp)
            hs, sd = str(f["heldout_subject"]), int(f["seed"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
            ds = _dense(f["subj_source"]); Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
            for fam in FAMILIES:
                for contested in (False, True):
                    if all((hs, sd, fam, contested, obj) in done for obj in OBJECTIVES):
                        continue
                    B = get_candidate_basis(fam, contested, Zs, ys, ds, max_rank=a.max_rank, seed=sd)
                    orc = crossfit_target_oracle(Zs, ys, Zt, yt, B, seed=sd, max_k=a.max_rank, mode=a.oracle_mode)
                    sms = nested_source_meta_multi(Zs, ys, ds, fam, contested, max_rank=a.max_rank,
                                                   seed=sd, objectives=tuple(OBJECTIVES), eps=a.eps)
                    ev_cache = {}
                    for obj in OBJECTIVES:
                        if (hs, sd, fam, contested, obj) in done:
                            continue
                        sm = sms[obj]; ks = sm["k_star"]
                        if ks not in ev_cache:                # same k* -> same target eval; cache across objectives
                            ev_cache[ks] = apply_rule_to_target_full(Zs, ys, ds, Zt, yt, fam, contested, ks, seed=sd)
                        ev = ev_cache[ks]
                        row = dict(dataset=a.dataset, backbone=a.backbone, heldout_subject=hs, seed=sd,
                                   family=fam, contested=bool(contested), objective=obj, rank=int(B.shape[0]),
                                   oracle_mode=a.oracle_mode,
                                   oracle_delta_query=float(orc["delta_query"]),
                                   oracle_delta_query_random=float(orc["delta_query_random"]),
                                   oracle_k=int(orc["k_selected"]),
                                   meta_delta_query=float(ev["delta_query"]),
                                   meta_delta_query_random=float(ev.get("delta_query_random", 0.0)),
                                   meta_k_star=int(ks), meta_no_harm=bool(sm["no_harm_ok"]),
                                   meta_sign_consistency=float(sm["sign_consistency"]) if sm["sign_consistency"] == sm["sign_consistency"] else None,
                                   meta_subspace_stability=float(sm["subspace_stability"]) if sm["subspace_stability"] == sm["subspace_stability"] else None,
                                   n_inner=int(sm["n_inner"]), config_hash=cfg_hash, git_sha=sha)
                        fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  [{i+1}/{len(cells)}] sub{hs} seed{sd} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[dg-id] wrote -> {raw}", flush=True)


if __name__ == "__main__":
    main()
