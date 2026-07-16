#!/usr/bin/env python
"""Track B — subspace candidate-span GEOMETRY audit (B1). Read-only characterization; NO mask learning, NO
selector, NO TTE. In the SOURCE Ledoit-Wolf-whitened metric, build B_cond / B_rule / B_grad + their union and
report span rank, full/free/contested rank, pairwise principal angles, task-head overlap, effective rank, and
whitening diagnostics. Stability across folds/seeds is computed by the aggregator (principal-angle / chordal /
projector-overlap; NOT Jaccard). Firewall: source-only geometry; no target labels used here.

  python scripts/run_trackb_geometry.py --backbone EEGNet --seeds 0 1 2
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, feat_from_audit_npz, _dense
from tos_cmi.eval import targetx_metric as M
from tos_cmi.eval.dg_identifiability import basis_rule, basis_grad

OUT = REPO / "results" / "cmi_trace_trackb_geometry"
DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _eff_rank(B):
    if B.shape[0] == 0:
        return 0.0
    s = np.linalg.svd(B, compute_uv=False); p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def _overlap_rowspace(B, row):
    if B.shape[0] == 0 or row.shape[0] == 0:
        return 0.0
    return float(np.sum((B @ row.T) ** 2) / B.shape[0])       # fraction of energy inside row(W_c)


def _cells(dataset, backbone, seeds):
    if backbone == "EEGNet":
        return [(p, "tos") for p in sorted(glob.glob(str(REPO / "tos_cmi/results/tos_cmi_eeg_frozen" /
                f"{dataset}_EEGNet_LOSO" / "sub*_erm_lam0_seed*.npz")))
                if any(p.endswith(f"_seed{s}.npz") for s in seeds)]
    pats = [str(REPO / "results/cmi_trace_relaxation_ladder" / f"dgcnn_graph_z_{dataset}" / "*.npz"),
            str(REPO / "results/cmi_trace_p0p1/objective_comparison" / dataset / "audit" / "*erm*seed*.audit.npz")]
    out = []
    for pat in pats:
        out += [(p, "audit") for p in sorted(glob.glob(pat)) if any(f"seed{s}" in Path(p).name for s in seeds)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="EEGNet", choices=["EEGNet", "DGCNN"])
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"]); ap.add_argument("--max_rank", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        cells = _cells(ds, a.backbone, a.seeds)
        if a.limit:
            cells = cells[: a.limit]
        for cp, kind in cells:
            f = feat_from_tos_dump(cp) if kind == "tos" else feat_from_audit_npz(cp)
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int); ds_ = _dense(f["subj_source"])
            if len(np.unique(ys)) < 2 or len(np.unique(ds_)) < 2:
                continue
            W = M.source_whitener(Zs); Zs_w = M.to_whitened(Zs, W)
            row_w, null_w = M.whitened_head_rowspace(Zs_w, ys, 0,
                W_stored=f.get("head_W"), A_inv=W["A_inv"] if f.get("head_W") is not None else None)
            B = {"cond": M.whitened_cond_basis(Zs_w, ys, ds_, max_rank=a.max_rank),
                 "rule": basis_rule(Zs_w, ys, ds_, max_rank=a.max_rank, seed=0),
                 "grad": basis_grad(Zs_w, ys, ds_, max_rank=a.max_rank, seed=0)}
            union = M._orthonormal(np.vstack([b for b in B.values() if b.shape[0]]) if any(b.shape[0] for b in B.values()) else np.zeros((1, Zs.shape[1])))
            row = dict(dataset=ds, backbone=a.backbone, heldout_subject=str(f["heldout_subject"]), seed=int(f.get("seed", 0)),
                       latent_dim=int(Zs.shape[1]), n_source_subjects=int(len(np.unique(ds_))),
                       whitening_method="LedoitWolf", cov_condition_number=W["cond"], whitening_hash=W["whitening_hash"],
                       head_rowspace_rank=int(row_w.shape[0]), union_rank=int(union.shape[0]),
                       union_effective_rank=_eff_rank(union),
                       uses_target_cal_y=False, uses_target_query_y_for_selection=False, oracle_non_deployable=False,
                       firewall="source_only_geometry_no_target_labels", git_sha=sha)
            for name, b in B.items():
                bc = M.project_basis(b, row_w); bf = M.project_basis(b, null_w)
                row[f"{name}_rank"] = int(b.shape[0]); row[f"{name}_contested_rank"] = int(bc.shape[0])
                row[f"{name}_free_rank"] = int(bf.shape[0]); row[f"{name}_eff_rank"] = _eff_rank(b)
                row[f"{name}_head_overlap"] = _overlap_rowspace(b, row_w)
            for x, y in (("cond", "rule"), ("cond", "grad"), ("rule", "grad")):
                pa = M.principal_angles_cos(B[x][:3], B[y][:3])
                row[f"angle_{x}_{y}_meancos"] = float(np.mean(pa)) if pa.size else float("nan")
                row[f"overlap_{x}_{y}"] = M.normalized_projector_overlap(B[x][:3], B[y][:3])
            rows.append(row)
        print(f"[trackb-geom] {a.backbone}/{ds}: {sum(r['dataset']==ds and r['backbone']==a.backbone for r in rows)} cells", flush=True)
    fp = OUT / f"geometry_rows_{a.backbone}_{'-'.join(a.seeds)}.jsonl"
    with open(fp, "w") as fh:
        [fh.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n") for r in rows]
    print(f"[trackb-geom] wrote {len(rows)} rows -> {fp}")


if __name__ == "__main__":
    main()
