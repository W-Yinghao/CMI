#!/usr/bin/env python
"""FFW-EEG runner: learn the Fantastic-Weights neuron mask on frozen DGCNN graph_z, evaluate the
task-retention vs subject-CMI frontier, and contrast with the exact-head-null SUBSPACE oracle. CPU.

  python scripts/run_ffw.py --dataset BNCI2014_001 --methods erm --seeds 0 --gamma 2.0
"""
from __future__ import annotations
import argparse, glob, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_audit_npz, _dense
from tos_cmi.eeg.ffw import find_fantastic_weights, prune_frontier
from tos_cmi.eeg.erasure_oracle import run_exact_head_null_oracle
from cmi.eval.conditional_subject_leakage import three_way_support_split, flat_conditional_cmi
from sklearn.metrics import balanced_accuracy_score

OUT = REPO / "results" / "cmi_trace_ffw"
CFG = REPO / "configs" / "cmi_trace_relaxation_ladder.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--methods", nargs="+", default=["erm"])
    ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--n_perm", type=int, default=15)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg_hash = hashlib.sha256(CFG.read_bytes()).hexdigest() if CFG.exists() else "no_config"
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    ad = REPO / "results/cmi_trace_p0p1/objective_comparison" / a.dataset / "audit"
    cells = []
    for m in a.methods:
        cells += [p for p in sorted(glob.glob(str(ad / f"{a.dataset}_fold*_sub*_{m}_seed*.audit.npz")))
                  if any(p.endswith(f"_seed{s}.audit.npz") for s in a.seeds)]
    if a.limit:
        cells = cells[: a.limit]
    if not cells:
        raise SystemExit(f"[ffw] no audit cells for {a.dataset}")
    outdir = OUT / a.dataset; outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "raw_rows.jsonl"
    done = set()
    if raw.exists():
        for line in open(raw):
            try:
                r = json.loads(line); done.add((r["training_method"], r["heldout_subject"], r["seed"]))
            except Exception:
                pass
    print(f"[ffw] {a.dataset} cells={len(cells)} gamma={a.gamma}", flush=True)
    t0 = time.time()
    with open(raw, "a") as fh:
        for i, cp in enumerate(cells):
            f = feat_from_audit_npz(cp)
            if f.get("head_W") is None:
                continue
            key = (f["training_method"], str(f["heldout_subject"]), int(f["seed"]))
            if key in done:
                continue
            Z = f["Z_source"]; y = f["y_source"].astype(int); d = _dense(f["subj_source"])
            W = f["head_W"]; b = f["head_b"]; ncls = int(f["n_cls"]); ndom = len(np.unique(d))
            er, pt, pe, _ = three_way_support_split(y, d, seed=int(f["seed"]))
            tb = lambda Zx: balanced_accuracy_score(y, (Zx @ W.T + b).argmax(1))
            cm = lambda Zx: flat_conditional_cmi(Zx, y, d, ncls, ndom, pt, pe, n_perm=a.n_perm,
                                                 seed=int(f["seed"]), epochs=50, with_residual=False)["posterior_kl_nats"]
            scores, mask, diag = find_fantastic_weights(Z, y, d, W, b, ncls, ndom, gamma=a.gamma,
                                                        n_temps=3, inner_epochs=40, p_epochs=25, seed=int(f["seed"]))
            fr = prune_frontier(Z, y, d, W, b, scores, cm, tb, ks=a.ks, seed=int(f["seed"]))
            orc = run_exact_head_null_oracle(f, n_perm=a.n_perm, seed=int(f["seed"]), epochs=50)
            best = fr["task_safe_best"]
            row = dict(dataset=a.dataset, training_method=f["training_method"], heldout_subject=str(f["heldout_subject"]),
                       seed=int(f["seed"]), full_task=fr["full_task"], full_cmi=fr["full_cmi"],
                       ffw_task_safe_k=int(best["k"]), ffw_task_safe_cmi=float(best["ffw_cmi"]),
                       ffw_task_safe_task=float(best["ffw_task"]),
                       ffw_task_safe_cmi_reduction=float(fr["full_cmi"] - best["ffw_cmi"]),
                       ffw_task_safe_exists=bool(fr["task_safe_exists"]),
                       oracle_subspace_delta_D=float(orc["delta_D_headnull"]),
                       oracle_task_unchanged=bool(orc["task_bacc_unchanged"]),
                       frontier=fr["frontier"], config_hash=cfg_hash, git_sha=sha)
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  [{i+1}/{len(cells)}] {Path(cp).name} FFW_safe_k={best['k']} dCMI={row['ffw_task_safe_cmi_reduction']:+.3f} "
                  f"| oracle_dD={orc['delta_D_headnull']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[ffw] wrote -> {raw}", flush=True)


if __name__ == "__main__":
    main()
