#!/usr/bin/env python
"""CMI-Trace Relaxation Ladder runner (Stage 2). Runs the L0-L3 ladder over a feature family and writes
raw_rows.jsonl (one row per feature-cell x level x eraser [x random draw]). Artifact-driven, resumable.

Feature families:
  dgcnn_graph_z : existing P0/P1 DGCNN audit npz (CPU; graph_z + verified head). No regeneration.
  tos_frozen    : regenerated EEGNet/TSMNet dumps (Stage 7).

  # DGCNN (CPU), primary training methods erm + cigl_graph_node:
  python scripts/run_cmi_trace_relaxation_ladder.py --family dgcnn_graph_z --dataset BNCI2014_001 \
      --methods erm cigl_graph_node --n_random 50
  # TOS (after dumps exist):
  python scripts/run_cmi_trace_relaxation_ladder.py --family tos_frozen --dataset BNCI2014_001 --backbone EEGNet
"""
from __future__ import annotations
import argparse, glob, json, hashlib, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import (run_cell, feat_from_audit_npz, feat_from_tos_dump, LEVELS)

OUT = REPO / "results" / "cmi_trace_relaxation_ladder"
CFG = REPO / "configs" / "cmi_trace_relaxation_ladder.yaml"


def _cfg_hash():
    return hashlib.sha256(CFG.read_bytes()).hexdigest() if CFG.exists() else "no_config"


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _dgcnn_cells(dataset, methods, seeds):
    d = REPO / "results" / "cmi_trace_p0p1" / "objective_comparison" / dataset / "audit"
    cells = []
    for m in methods:
        for p in sorted(glob.glob(str(d / f"{dataset}_fold*_sub*_{m}_seed*.audit.npz"))):
            if seeds and not any(p.endswith(f"_seed{s}.audit.npz") for s in seeds):
                continue
            cells.append(p)
    return cells


def _tos_cells(dataset, backbone, seeds):
    d = REPO / "tos_cmi" / "results" / "tos_cmi_eeg_frozen" / f"{dataset}_{backbone}_LOSO"
    cells = [p for p in sorted(glob.glob(str(d / "sub*_erm_lam0_seed*.npz")))
             if (not seeds or any(p.endswith(f"_seed{s}.npz") for s in seeds))]
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["dgcnn_graph_z", "tos_frozen"])
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet", help="tos_frozen only")
    ap.add_argument("--methods", nargs="+", default=["erm", "cigl_graph_node"], help="dgcnn_graph_z only")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--n_random", type=int, default=50)
    ap.add_argument("--head_regime", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--with_tos_vd", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg_hash, git_sha = _cfg_hash(), _git_sha()

    if a.family == "dgcnn_graph_z":
        cells = _dgcnn_cells(a.dataset, a.methods, a.seeds); loader = feat_from_audit_npz
        tag = a.out_tag or f"{a.family}_{a.dataset}"
    else:
        cells = _tos_cells(a.dataset, a.backbone, a.seeds); loader = feat_from_tos_dump
        tag = a.out_tag or f"{a.family}_{a.dataset}_{a.backbone}"
    if a.limit:
        cells = cells[: a.limit]
    if not cells:
        raise SystemExit(f"[ladder] no feature cells found for {a.family}/{a.dataset}. "
                         f"(dgcnn: check audit npz; tos: regenerate dumps via feature_dump.py first.)")

    outdir = OUT / tag; outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw_rows.jsonl"
    done = set()
    if raw_path.exists() and not a.overwrite:
        for line in open(raw_path):
            try:
                r = json.loads(line); done.add((r["dataset"], r["backbone"], r["training_method"],
                                                r["outer_fold"], r["heldout_subject"], r["seed"]))
            except Exception:
                pass
    informed = ("lw_leace_full", "repo_leace") + (("tos_vd",) if a.with_tos_vd else ())
    print(f"[ladder] family={a.family} tag={tag} cells={len(cells)} informed={informed} "
          f"cfg={cfg_hash[:12]} sha={git_sha}", flush=True)
    t0 = time.time(); n_rows = 0
    with open(raw_path, "a") as fh:
        for i, cp in enumerate(cells):
            try:
                feat = loader(cp)
            except Exception as e:
                print(f"  [skip] {Path(cp).name}: load error {e!r}", flush=True); continue
            key = (feat.get("dataset", ""), feat.get("backbone", ""), feat.get("training_method", ""),
                   int(feat.get("outer_fold", -1)), str(feat.get("heldout_subject", "")), int(feat.get("seed", 0)))
            if key in done and not a.overwrite:
                continue
            rows = run_cell(feat, cfg_hash, git_sha, n_random=a.n_random, seed=int(feat.get("seed", 0)),
                            head_regime=a.head_regime, informed=informed, with_tos_vd=a.with_tos_vd)
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush(); n_rows += len(rows)
            print(f"  [{i+1}/{len(cells)}] {Path(cp).name} -> {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
    print(f"[ladder] wrote {n_rows} rows -> {raw_path}", flush=True)


if __name__ == "__main__":
    main()
