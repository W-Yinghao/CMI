#!/usr/bin/env python
"""FSR Phase 4B — branch-local L1-L6 verification from the ERM-refit dumps (CPU-only, no training).

For each refit fold (dataset, target subject, seed) it reads the branch-latent dumps + the output JSON
and produces the four-part evidence chain:
  L1 branch_leakage_probe.csv        source-only per-branch subject/domain probe (bAcc, null ratio)
  L4 branch_task_coupling.csv        per-branch ablation drop + fusion-gate weight (from the run JSON)
  L5 branch_reliance_replay.csv      erase the source-fit subject subspace from a branch's TARGET latent,
                                     recompose head3(_fuse3(...)), measure task-drop + logit SymKL vs a
                                     random-subspace control (the branch-local analogue of CIGL R3)
  L6 branch_target_consequence.csv   target bAcc/NLL/ECE (from the run JSON)
Target labels are used ONLY to score L5/L6; probes/subspaces are fit on source only.

    <icml python> scripts/analyze_rq4_branch_local.py [--seed_tag seed0] [--out results/fsr_rq4_refit]
"""
from __future__ import annotations
import argparse, csv, glob, json, os
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from cmi.run_loso import _infer_ch_names
from cmi.models.fb_lgg_dualcmi import central_strip_groups
from cmi.models.backbones import build_backbone

NCH = {"BNCI2014_001": 22, "BNCI2015_001": 13}
BRANCHES = ["graph_z", "temporal_z", "spatial_z", "fused_z"]
FUSION = ["graph", "temporal", "spatial"]
K = 2                       # subspace rank (matches CIGL R3 k2)
RNG = np.random.default_rng(0)


def subject_subspace(Z, d, k=K):
    """Top-k between-subject mean-scatter directions (first-moment subject subspace)."""
    doms = np.unique(d)
    if len(doms) < 2:
        return np.zeros((0, Z.shape[1]))
    mu = Z.mean(0)
    M = np.stack([Z[d == s].mean(0) - mu for s in doms])       # [n_dom, dim]
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])]


def erase(Z, basis):
    if basis.shape[0] == 0:
        return Z.copy()
    Q, _ = np.linalg.qr(basis.T)                                # [dim, k] orthonormal
    return Z - (Z @ Q) @ Q.T


def probe_bacc(Z, d):
    """5-fold source-only balanced-accuracy of a linear domain probe + label-permutation null."""
    d = np.asarray(d)
    if len(np.unique(d)) < 2 or np.min(np.bincount(d)) < 5:
        return dict(probe_bacc=float("nan"), null_bacc=float("nan"), null_ratio=float("nan"),
                    chance=1.0 / max(1, len(np.unique(d))))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    def cv(dd):
        pr = np.zeros_like(dd)
        for tr, te in skf.split(Z, dd):
            clf = LogisticRegression(max_iter=200, C=1.0, multi_class="auto")
            clf.fit(Z[tr], dd[tr])
            pr[te] = clf.predict(Z[te])
        return balanced_accuracy_score(dd, pr)

    real = cv(d)
    null = np.mean([cv(RNG.permutation(d)) for _ in range(3)])
    chance = 1.0 / len(np.unique(d))
    return dict(probe_bacc=float(real), null_bacc=float(null),
                null_ratio=float((real - chance) / max(1e-6, null - chance)), chance=float(chance))


def symkl(a, b):
    pa = torch.softmax(torch.tensor(a), 1).numpy() + 1e-9
    pb = torch.softmax(torch.tensor(b), 1).numpy() + 1e-9
    return float(np.mean(np.sum(pa * np.log(pa / pb) + pb * np.log(pb / pa), 1)))


def load_model(ds, cfg, ckpt, ncls):
    nch, nt = NCH[ds], int(cfg["resample"] * (cfg["tmax"] - cfg["tmin"])) + 1
    ch_names, _ = _infer_ch_names(ds, nch)
    groups, named, _ = central_strip_groups(ds, ch_names)
    bb = build_backbone("FBCSPLGGGraph", nch, nt, ncls, device="cpu", ch_names=ch_names,
                        groups=groups, group_names=named, grouping_scheme="central_strip_v1", fusion_floor=0.0)
    bb.load_state_dict(ckpt["state_dict"], strict=True)
    bb.eval()
    return bb


def recompose(bb, gz, tz, sz):
    with torch.no_grad():
        return bb.head3(bb._fuse3(torch.tensor(gz, dtype=torch.float32),
                                  torch.tensor(tz, dtype=torch.float32),
                                  torch.tensor(sz, dtype=torch.float32))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", default="results/fsr_rq4_refit/latents")
    ap.add_argument("--ckpt_dir", default="results/fsr_rq4_refit/ckpt")
    ap.add_argument("--json_glob", default="results/fsr_rq4_refit/4b*_*.json")
    ap.add_argument("--out", default="results/fsr_rq4_refit")
    args = ap.parse_args()

    # index the run JSONs -> per (dataset, target_subject) rec
    recs = {}
    for jp in glob.glob(args.json_glob):
        d = json.load(open(jp))
        ds = d["config"]["dataset"]
        for r in d.get("summary", {}).get("erm:0", {}).get("per_target", []):
            recs[(ds, str(r["target"]))] = r

    l1, l4, l5, l6 = [], [], [], []
    manifests = sorted(glob.glob(os.path.join(args.latent_dir, "*_latent_dump_manifest.json")))
    for mp in manifests:
        man = json.load(open(mp))
        ds, tag, tsub = man["dataset"], man["tag"], man["target_subject"]
        seed = man["seed"]
        src = np.load(os.path.join(args.latent_dir, f"{tag}_source_latents.npz"))
        tgt = np.load(os.path.join(args.latent_dir, f"{tag}_target_latents.npz"))
        sd = src["d"].astype(int)
        rec = recs.get((ds, tsub), {})
        ckpt = torch.load(os.path.join(args.ckpt_dir, f"{tag}_ckpt_best.pt"), map_location="cpu", weights_only=False)
        ncls = int(src["src_logits"].shape[1])
        bb = load_model(ds, ckpt["config"], ckpt, ncls)

        # ---- L1 per-branch source-only leakage probe ----
        for b in BRANCHES:
            m = probe_bacc(src[f"src_{b}"], sd)
            l1.append(dict(dataset=ds, target_subject=tsub, seed=seed, branch=b, **{k: round(v, 4) for k, v in m.items()}))

        # ---- L4 per-branch task coupling (from run JSON) ----
        base = rec.get("balanced_acc")
        for fb in FUSION:
            abl = rec.get(f"ablate_zero_{fb}_target_bacc")
            l4.append(dict(dataset=ds, target_subject=tsub, seed=seed, branch=f"{fb}_z",
                           base_target_bacc=_r(base), ablate_zero_target_bacc=_r(abl),
                           ablation_drop=_r((base - abl) if (base is not None and abl is not None) else None),
                           gate_weight=_r(rec.get(f"gate_{fb}_mean"))))

        # ---- L5 per-branch reliance replay (erase source-fit subject subspace from TARGET branch) ----
        gz, tz, sz = tgt["tgt_graph_z"], tgt["tgt_temporal_z"], tgt["tgt_spatial_z"]
        ty = tgt["y"].astype(int)
        logit_orig = recompose(bb, gz, tz, sz)
        bacc_orig = balanced_accuracy_score(ty, logit_orig.argmax(1))
        for b, arr in (("graph_z", gz), ("temporal_z", tz), ("spatial_z", sz)):
            basis = subject_subspace(src[f"src_{b}"], sd)
            erased = erase(arr, basis)
            rk = min(K, arr.shape[1])
            rbasis = RNG.standard_normal((rk, arr.shape[1]))
            rand = erase(arr, rbasis)
            packs = {"graph_z": (erased if b == "graph_z" else gz),
                     "temporal_z": (erased if b == "temporal_z" else tz),
                     "spatial_z": (erased if b == "spatial_z" else sz)}
            rpacks = {"graph_z": (rand if b == "graph_z" else gz),
                      "temporal_z": (rand if b == "temporal_z" else tz),
                      "spatial_z": (rand if b == "spatial_z" else sz)}
            lg_e = recompose(bb, packs["graph_z"], packs["temporal_z"], packs["spatial_z"])
            lg_r = recompose(bb, rpacks["graph_z"], rpacks["temporal_z"], rpacks["spatial_z"])
            l5.append(dict(dataset=ds, target_subject=tsub, seed=seed, branch=b,
                           bacc_orig=_r(bacc_orig),
                           task_drop=_r(bacc_orig - balanced_accuracy_score(ty, lg_e.argmax(1))),
                           task_drop_random=_r(bacc_orig - balanced_accuracy_score(ty, lg_r.argmax(1))),
                           logit_symkl=_r(symkl(logit_orig, lg_e)),
                           logit_symkl_random=_r(symkl(logit_orig, lg_r)), subspace_k=rk))

        # ---- L6 target consequence (from run JSON) ----
        l6.append(dict(dataset=ds, target_subject=tsub, seed=seed,
                       target_bacc=_r(rec.get("balanced_acc")), target_nll=_r(rec.get("nll")),
                       target_ece=_r(rec.get("ece")), source_bacc=_r(rec.get("source_bacc"))))
        print(f"[rq4-audit] {tag}: L1/L4/L5/L6 done (bacc_orig={bacc_orig:.3f})", flush=True)

    _wcsv(Path(args.out) / "branch_leakage_probe.csv", l1)
    _wcsv(Path(args.out) / "branch_task_coupling.csv", l4)
    _wcsv(Path(args.out) / "branch_reliance_replay.csv", l5)
    _wcsv(Path(args.out) / "branch_target_consequence.csv", l6)
    print(f"wrote 4 CSVs to {args.out} over {len(manifests)} folds")


def _r(x):
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return ""


def _wcsv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
