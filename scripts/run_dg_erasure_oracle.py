#!/usr/bin/env python
"""DG-erasure oracle on REAL EEG (the P1 go/no-go). For each EEGNet frozen dump (one LOSO fold), build the
source subject subspace as the candidate basis B and run:
  * target_dg_oracle        (target-label UPPER BOUND: does ANY deletion help target?)
  * source_meta_subset_oracle (SOURCE-LOSO: is a target-beneficial subset identifiable from source alone?)
  * cmi_only_selector       (old objective, contrast)
plus best-prefix + matched-rank random. Writes per-fold d_target for each selector. CPU.

  python scripts/run_dg_erasure_oracle.py --dataset BNCI2014_001 --backbone EEGNet --max_rank 10
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eeg.erasure_oracle import marginal_subject_basis
from tos_cmi.eval.dg_erasure_oracle import (target_dg_greedy, source_meta_greedy,
                                            cmi_only_selector, evaluate_on_target)

OUT = REPO / "results" / "cmi_trace_dg_oracle"
CFG = REPO / "configs" / "cmi_trace_relaxation_ladder.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--max_rank", type=int, default=10)
    ap.add_argument("--gamma_cmi", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg_hash = hashlib.sha256(CFG.read_bytes()).hexdigest() if CFG.exists() else "no_config"
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
        raise SystemExit(f"[dg-oracle] no dumps for {a.dataset}/{a.backbone}")
    outdir = OUT / f"{a.dataset}_{a.backbone}"; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "raw_rows.jsonl"
    done = set()
    if raw.exists():
        for line in open(raw):
            try:
                r = json.loads(line); done.add((r["heldout_subject"], r["seed"]))
            except Exception:
                pass
    print(f"[dg-oracle] {a.dataset}/{a.backbone} cells={len(cells)} max_rank={a.max_rank}", flush=True)
    t0 = time.time()
    with open(raw, "a") as fh:
        for i, cp in enumerate(cells):
            f = feat_from_tos_dump(cp)
            key = (str(f["heldout_subject"]), int(f["seed"]))
            if key in done:
                continue
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
            ds = _dense(f["subj_source"]); Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
            tgt_dom = int(ds.max()) + 1
            Z = np.vstack([Zs, Zt]); y = np.concatenate([ys, yt])
            d = np.concatenate([ds, np.full(len(Zt), tgt_dom)])
            B = marginal_subject_basis(Zs, ds, max_rank=a.max_rank)       # candidate = source subject subspace
            if B.shape[0] == 0:
                continue
            tdo = target_dg_greedy(Z, y, d, tgt_dom, B, seed=int(f["seed"]))
            smo = source_meta_greedy(Z, y, d, tgt_dom, B, seed=int(f["seed"]), gamma_cmi=a.gamma_cmi)
            ev_meta = evaluate_on_target(Z, y, d, tgt_dom, B, smo["S_star"], seed=int(f["seed"]))
            cmi = cmi_only_selector(Z, y, d, tgt_dom, B, seed=int(f["seed"]))
            ev_cmi = evaluate_on_target(Z, y, d, tgt_dom, B, cmi["S_cmi"], seed=int(f["seed"]))
            # matched-rank random baseline (same |S*|): informed must beat this for a DG claim
            k_star = len(smo["S_star"]); rr = B.shape[0]; rng = np.random.default_rng(11 + int(f["seed"]))
            d_rand = float(np.mean([evaluate_on_target(Z, y, d, tgt_dom, B,
                                    list(rng.choice(rr, min(k_star, rr), replace=False)), seed=int(f["seed"]))["d_target"]
                                    for _ in range(10)])) if k_star > 0 else 0.0
            row = dict(dataset=a.dataset, backbone=a.backbone, heldout_subject=str(f["heldout_subject"]),
                       seed=int(f["seed"]), rank=int(B.shape[0]),
                       d_target_upper_bound=float(tdo["d_target_best"]),
                       d_target_source_meta=float(ev_meta["d_target"]),
                       d_target_random=d_rand,
                       source_meta_bacc=float(smo["star"].get("source_meta_bacc", float("nan"))),
                       source_meta_k=int(len(smo["S_star"])),
                       d_target_cmi_only=float(ev_cmi["d_target"]),
                       identity_target_bacc=float(ev_meta["identity_target_bacc"]),
                       config_hash=cfg_hash, git_sha=sha)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  [{i+1}/{len(cells)}] sub{f['heldout_subject']} seed{f['seed']}: "
                  f"UB={row['d_target_upper_bound']:+.3f} SOURCE-META={row['d_target_source_meta']:+.3f} "
                  f"rand={row['d_target_random']:+.3f} cmi-only={row['d_target_cmi_only']:+.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[dg-oracle] wrote -> {raw}", flush=True)


if __name__ == "__main__":
    main()
