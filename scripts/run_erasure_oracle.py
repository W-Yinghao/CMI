#!/usr/bin/env python
"""CMI-Trace erasure-oracle runner — Priority 1: exact-head nullspace oracle over DGCNN audit npzs.

For every DGCNN audit cell (which stores a VERIFIED linear head), fit the label-conditional subject subspace
INSIDE ker(W_c) and remove it: measure the CMI removed (delta_D), the matched-rank random-null control, the
softmax replay error (algebraic task-safety), and that the stored head's predictions are unchanged. Writes
raw_rows.jsonl. CPU-only.

  python scripts/run_erasure_oracle.py --dataset BNCI2014_001 --methods erm cigl_graph_node --n_perm 30
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_audit_npz
from tos_cmi.eeg.erasure_oracle import run_exact_head_null_oracle

OUT = REPO / "results" / "cmi_trace_erasure_oracle"
CFG = REPO / "configs" / "cmi_trace_relaxation_ladder.yaml"


def _cfg_hash():
    return hashlib.sha256(CFG.read_bytes()).hexdigest() if CFG.exists() else "no_config"


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--methods", nargs="+", default=["erm", "cigl_graph_node"])
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--n_perm", type=int, default=30)
    ap.add_argument("--k", type=int, default=0, help="oracle rank; 0 = full subject-subspace rank in ker(W_c)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg_hash, git_sha = _cfg_hash(), _git_sha()
    d = REPO / "results/cmi_trace_p0p1/objective_comparison" / a.dataset / "audit"
    cells = []
    for m in a.methods:
        cells += [p for p in sorted(glob.glob(str(d / f"{a.dataset}_fold*_sub*_{m}_seed*.audit.npz")))
                  if any(p.endswith(f"_seed{s}.audit.npz") for s in a.seeds)]
    if a.limit:
        cells = cells[: a.limit]
    if not cells:
        raise SystemExit(f"[oracle] no DGCNN audit cells for {a.dataset}")
    outdir = OUT / f"dgcnn_{a.dataset}"; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "raw_rows.jsonl"
    done = set()
    if raw.exists():
        for line in open(raw):
            try:
                r = json.loads(line); done.add((r["dataset"], r["training_method"], r["heldout_subject"], r["seed"]))
            except Exception:
                pass
    print(f"[oracle] {a.dataset} cells={len(cells)} cfg={cfg_hash[:12]} sha={git_sha}", flush=True)
    t0 = time.time(); n = 0
    with open(raw, "a") as fh:
        for i, cp in enumerate(cells):
            feat = feat_from_audit_npz(cp)
            if feat.get("head_W") is None:
                print(f"  [skip] {Path(cp).name}: no stored head", flush=True); continue
            key = (feat["dataset"], feat["training_method"], str(feat["heldout_subject"]), int(feat["seed"]))
            if key in done:
                continue
            r = run_exact_head_null_oracle(feat, k=(a.k or None), n_perm=a.n_perm, seed=int(feat["seed"]))
            r["config_hash"] = cfg_hash; r["git_sha"] = git_sha
            fh.write(json.dumps(r) + "\n"); fh.flush(); n += 1
            print(f"  [{i+1}/{len(cells)}] {Path(cp).name} dD_hn={r['delta_D_headnull']:+.3f} "
                  f"dD_rand={r['delta_D_randomnull']:+.3f} softmax_err={r['softmax_replay_err_headnull']:.1e} "
                  f"task_unchanged={r['task_bacc_unchanged']} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[oracle] wrote {n} rows -> {raw}", flush=True)


if __name__ == "__main__":
    main()
