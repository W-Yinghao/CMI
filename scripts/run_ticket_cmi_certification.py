#!/usr/bin/env python
"""F2.0b: CORRECTED posterior-KL CMI certification of the DG deletion ticket (supersedes the flawed pass).

PM code review flagged three defects in the first pass: (1) it certified a re-selected FULL-target ticket, not
the split-specific cross-fitted one; (2) it violated the three-way cross-fitting contract (the deletion basis
was fit on all source incl. the posterior-eval trials); (3) it used an unpaired ticket-vs-random rule with a
single random draw. This runner fixes all three:
  * cond basis is fit on the SOURCE ERASER-FIT partition only (disjoint from posterior train/eval);
  * the ticket is the SPLIT-SPECIFIC greedy ticket selected on each target T_select (source head fit on the
    eraser partition), and that SAME ticket is applied to the source posterior partitions;
  * >=20 matched-rank random subsets; the certified statistic is the PAIRED
        dI_specific(hd) = mean_i kl(random_i) - kl(ticket)      [full-representation KL cancels]
    at capacities hd in {8 (near-linear), 32 (small), 128 (large)}, reported per capacity + max;
    kl is the raw held-out posterior-KL (ticket & random measured identically -> the null baseline cancels
    in the paired difference). Certification (aggregator) = cluster LCB95(dI_specific) > 0 at the PRIMARY
    (large) capacity, robustness reported across capacities. cond/full basis, source Z.

  python scripts/run_ticket_cmi_certification.py --dataset BNCI2014_001 --seeds 0 --device cpu
"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eval.dg_identifiability import (get_candidate_basis, source_greedy_select, _select_subset,
                                             _target_splits)
from cmi.eval.conditional_subject_leakage import three_way_support_split
from cmi.eval.graph_leakage import fit_conditional_domain_probe

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
CAPS = [("linear", 8), ("small", 32), ("large", 128)]     # probe capacities (PM: linear / MLP-small / MLP-large)


def _del(Z, B, S):
    if not S:
        return Z
    Bs = B[list(S)]
    return Z - (Z @ Bs.T) @ Bs


def _kl(Z, ys, ds, n_cls, n_dom, pt, pe, hd, epochs, seed, device):
    g = fit_conditional_domain_probe(Z, ys, ds, n_cls, n_dom, train_idx=pt, val_idx=pe,
                                     hidden_dim=hd, epochs=epochs, seed=seed, device=device)
    return float(g["kl_mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001"); ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0"]); ap.add_argument("--family", default="cond")
    ap.add_argument("--max_rank", type=int, default=10); ap.add_argument("--n_target_splits", type=int, default=3)
    ap.add_argument("--n_random", type=int, default=20); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--limit", type=int, default=0)
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
        raise SystemExit(f"[cmi-cert2] no dumps for {a.dataset}")
    outdir = OUT / f"{a.dataset}_{a.backbone}"; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / f"cmi_cert2_seed{'-'.join(a.seeds)}.jsonl"
    print(f"[cmi-cert2] {a.dataset} cells={len(cells)} family={a.family} caps={[c[0] for c in CAPS]}", flush=True)
    t0 = time.time()
    with open(raw, "w") as fh:
        for i, cp in enumerate(cells):
            f = feat_from_tos_dump(cp); hs, sd = str(f["heldout_subject"]), int(f["seed"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
            ds = _dense(f["subj_source"]); Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
            n_cls = int(f["n_cls"]); n_dom = int(len(np.unique(ds)))
            er, pt, pe, diag = three_way_support_split(ys, ds, seed=sd)          # SOURCE 3-way disjoint split
            if pt.size < 6 or pe.size < 6 or er.size < 6:
                continue
            # eraser-fit basis (disjoint from posterior pt/pe): fit cond basis on the ERASER partition only
            B = get_candidate_basis(a.family, False, Zs[er], ys[er], ds[er], max_rank=a.max_rank, seed=sd)
            if B.shape[0] == 0:
                continue
            r = B.shape[0]
            # split-specific greedy tickets: select on target T_select, source head on the ERASER partition
            splits = _target_splits(yt, a.n_target_splits, 0.5, sd)
            S_splits = [_select_subset(Zs[er], ys[er], Zt[sel], yt[sel], B, "greedy", a.max_rank, sd) for sel in splits]
            S_splits = [S for S in S_splits if S] or [[0]]                        # ensure non-empty
            S_src = source_greedy_select(Zs, ys, ds, B, seed=sd, max_k=a.max_rank) or [0]
            kbar = max(1, int(round(np.mean([len(S) for S in S_splits]))))
            rng = np.random.default_rng(1000 + sd)
            S_rands = [list(rng.choice(r, min(kbar, r), replace=False)) for _ in range(a.n_random)]
            row = dict(dataset=a.dataset, backbone=a.backbone, heldout_subject=hs, seed=sd, family=a.family,
                       k_ticket=kbar, k_src=len(S_src), rank=r, n_ptrain=int(pt.size), n_peval=int(pe.size),
                       git_sha=sha)
            for tag, hd in CAPS:
                kl_full = _kl(Zs, ys, ds, n_cls, n_dom, pt, pe, hd, a.epochs, sd, a.device)
                kl_tkt = float(np.mean([_kl(_del(Zs, B, S), ys, ds, n_cls, n_dom, pt, pe, hd, a.epochs, sd, a.device)
                                        for S in S_splits]))
                kl_src = _kl(_del(Zs, B, S_src), ys, ds, n_cls, n_dom, pt, pe, hd, a.epochs, sd, a.device)
                kl_rnds = [_kl(_del(Zs, B, S), ys, ds, n_cls, n_dom, pt, pe, hd, a.epochs, sd, a.device) for S in S_rands]
                kl_rnd = float(np.mean(kl_rnds))
                row[f"kl_{tag}_full"] = kl_full; row[f"kl_{tag}_ticket"] = kl_tkt
                row[f"kl_{tag}_source"] = kl_src; row[f"kl_{tag}_random"] = kl_rnd
                row[f"dI_ticket_{tag}"] = float(kl_full - kl_tkt)                 # leakage removed by ticket
                row[f"dI_random_{tag}"] = float(kl_full - kl_rnd)                 # by random (matched rank)
                row[f"dI_specific_{tag}"] = float(kl_rnd - kl_tkt)               # PAIRED: ticket beyond random
            row["dI_specific_max"] = float(max(row[f"dI_specific_{t}"] for t, _ in CAPS))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  [{i+1}/{len(cells)}] sub{hs} seed{sd} k={kbar} "
                  f"dI_spec(lin/sm/lg)={row['dI_specific_linear']:+.4f}/{row['dI_specific_small']:+.4f}/"
                  f"{row['dI_specific_large']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[cmi-cert2] wrote -> {raw}", flush=True)


if __name__ == "__main__":
    main()
