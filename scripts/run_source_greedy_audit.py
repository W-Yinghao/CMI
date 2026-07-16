#!/usr/bin/env python
"""GREEDY source-only identifiability audit (the adversarial-verification gate before any P2 GPU spend).

The nested source-meta selector is prefix-only, so it could never express the arbitrary-coordinate greedy
ticket the cross-fit oracle confirms exists. This runner tests identifiability GREEDY-vs-GREEDY: a source-only
greedy selector (maximize source-LOSO held-out bAcc, arbitrary coordinates = what a differentiable supermask
implements) is applied to the TRUE target; we ask whether its target gain is positive, beats matched-rank
random, and whether its subspace ALIGNS with the greedy target-hindsight ticket. Full (contested=False) bases.

  python scripts/run_source_greedy_audit.py --dataset BNCI2014_001 --backbone EEGNet --seeds 0 1 2
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eval.dg_identifiability import get_candidate_basis, source_greedy_audit, crossfit_target_oracle

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
FAMILIES = ["marg", "cond", "rule", "grad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--max_rank", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
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
        raise SystemExit(f"[src-audit] no dumps for {a.dataset}/{a.backbone}")
    outdir = OUT / f"{a.dataset}_{a.backbone}"; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / f"audit_rows_seed{'-'.join(a.seeds)}.jsonl"
    print(f"[src-audit] {a.dataset}/{a.backbone} cells={len(cells)} families={FAMILIES}", flush=True)
    t0 = time.time()
    with open(raw, "w") as fh:
        for i, cp in enumerate(cells):
            f = feat_from_tos_dump(cp)
            hs, sd = str(f["heldout_subject"]), int(f["seed"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
            ds = _dense(f["subj_source"]); Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
            for fam in FAMILIES:
                B = get_candidate_basis(fam, False, Zs, ys, ds, max_rank=a.max_rank, seed=sd)
                au = source_greedy_audit(Zs, ys, ds, Zt, yt, B, seed=sd, max_k=a.max_rank)
                orc = crossfit_target_oracle(Zs, ys, Zt, yt, B, seed=sd, max_k=a.max_rank, mode="greedy")
                row = dict(dataset=a.dataset, backbone=a.backbone, heldout_subject=hs, seed=sd, family=fam,
                           rank=int(B.shape[0]),
                           src_delta_target=float(au["delta_src"]),
                           src_delta_target_random=float(au["delta_src_random"]),
                           src_tgt_alignment=float(au["alignment"]) if au["alignment"] == au["alignment"] else None,
                           k_src=int(au["k_src"]), k_tgt=int(au["k_tgt"]),
                           oracle_delta_query=float(orc["delta_query"]),
                           oracle_delta_query_random=float(orc["delta_query_random"]),
                           git_sha=sha)
                fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  [{i+1}/{len(cells)}] sub{hs} seed{sd} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[src-audit] wrote -> {raw}", flush=True)


if __name__ == "__main__":
    main()
