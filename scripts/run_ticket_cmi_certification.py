#!/usr/bin/env python
"""P0.3 closeout: posterior-KL CMI certification of the DG deletion tickets (the load-bearing gap).

Prior runs proved a target-hindsight deletion improves target bAcc, but NOT that it actually reduces the
VALIDATED conditional subject leakage I(Z;D|Y) more than a matched-rank random deletion. This runner applies
the SAME posterior-KL ruler (cmi_ruler_across_transforms, shared ptrain/peval split) on SOURCE Z to:
  full (identity) | target-greedy ticket | source-greedy subset | matched-rank random
at MLP-small (hidden_dim=16) and MLP-large (hidden_dim=128), with the retrained within-label permutation null.
Reports delta_I = excess_over_null(full) - excess_over_null(deleted) per fold; a ticket is a certified
subject-leakage deletion iff delta_I(ticket) > 0 and > delta_I(random). cond/full basis.

  python scripts/run_ticket_cmi_certification.py --dataset BNCI2014_001 --seeds 0 --device cpu
"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eval.dg_identifiability import get_candidate_basis, source_greedy_select, _select_subset
from cmi.eval.conditional_subject_leakage import three_way_support_split, cmi_ruler_across_transforms

OUT = REPO / "results" / "cmi_trace_dg_identifiability"


def _del(Z, B, S):
    if not S:
        return Z
    Bs = B[list(S)]
    return Z - (Z @ Bs.T) @ Bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--family", default="cond")
    ap.add_argument("--max_rank", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cpu")
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
        raise SystemExit(f"[cmi-cert] no dumps for {a.dataset}")
    outdir = OUT / f"{a.dataset}_{a.backbone}"; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / f"cmi_cert_seed{'-'.join(a.seeds)}.jsonl"
    print(f"[cmi-cert] {a.dataset} cells={len(cells)} family={a.family} device={a.device}", flush=True)
    t0 = time.time()
    with open(raw, "w") as fh:
        for i, cp in enumerate(cells):
            f = feat_from_tos_dump(cp)
            hs, sd = str(f["heldout_subject"]), int(f["seed"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int)
            ds = _dense(f["subj_source"]); Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
            n_cls = int(f["n_cls"]); n_dom = int(len(np.unique(ds)))
            B = get_candidate_basis(a.family, False, Zs, ys, ds, max_rank=a.max_rank, seed=sd)
            if B.shape[0] == 0:
                continue
            S_tkt = _select_subset(Zs, ys, Zt, yt, B, "greedy", a.max_rank, sd)     # target-hindsight ticket
            S_src = source_greedy_select(Zs, ys, ds, B, seed=sd, max_k=a.max_rank)  # source-greedy
            k = max(len(S_tkt), 1)
            rng = np.random.default_rng(1000 + sd); S_rnd = list(rng.choice(B.shape[0], min(k, B.shape[0]), replace=False))
            transforms = {"full": Zs, "ticket": _del(Zs, B, S_tkt), "source_greedy": _del(Zs, B, S_src),
                          "random_k": _del(Zs, B, S_rnd)}
            er, pt, pe, diag = three_way_support_split(ys, ds, seed=sd)
            if pt.size < 4 or pe.size < 4:
                continue
            row = dict(dataset=a.dataset, backbone=a.backbone, heldout_subject=hs, seed=sd, family=a.family,
                       k_ticket=len(S_tkt), k_src=len(S_src), k_rand=len(S_rnd), git_sha=sha)
            for hd, tag in [(16, "small"), (128, "large")]:
                rep = cmi_ruler_across_transforms(transforms, ys, ds, n_cls, n_dom, pt, pe,
                                                  n_perm=a.n_perm, seed=sd, device=a.device,
                                                  hidden_dim=hd, epochs=a.epochs, with_residual=False)
                base = rep["full"]["excess_over_null"]
                for name in ("ticket", "source_greedy", "random_k"):
                    row[f"excess_{tag}_{name}"] = float(rep[name]["excess_over_null"])
                    row[f"dI_{tag}_{name}"] = float(base - rep[name]["excess_over_null"])   # leakage removed
                    row[f"permp_{tag}_{name}"] = float(rep[name]["perm_p"])
                row[f"excess_{tag}_full"] = float(base)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  [{i+1}/{len(cells)}] sub{hs} seed{sd} k_tkt={len(S_tkt)} "
                  f"dI_small_ticket={row.get('dI_small_ticket', float('nan')):+.4f} "
                  f"dI_small_random={row.get('dI_small_random_k', float('nan')):+.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[cmi-cert] wrote -> {raw}", flush=True)


if __name__ == "__main__":
    main()
