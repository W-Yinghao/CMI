#!/usr/bin/env python
"""CMI-Trace Relaxation Ladder diagnostics runner (Stages 4/5/6, source-only + CMI ruler).

Per feature cell (DGCNN audit npz or TOS dump) computes, all from SOURCE features:
  * Stage 5 gate decisions G0-G3 (accept/refuse + refusal_reason + all source-only diagnostics);
  * Stage 4 task-direction consistency (binary or 4-class macro) + subspace overlap + geometry;
  * Stage 6 same CMI ruler (conditional_subject_leakage + multicapacity_probe familywise) on
    {full, whitening_only, lw_leace_full, random_k} with 3-way cross-fitting.

Writes gate_decisions.csv, direction_consistency.csv, subspace_overlap.csv, cmi_ruler.csv. CPU-only.

  python scripts/run_cmi_trace_ladder_diagnostics.py --family dgcnn_graph_z --dataset BNCI2014_001 \
      --methods erm cigl_graph_node --cmi_ruler --cmi_nperm 20
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import (feat_from_audit_npz, feat_from_tos_dump, lw_leace_full,
                                           whitening_only, random_removal, _dense)
from tos_cmi.eeg import selective_erasure as SE
from cmi.eval.task_direction_consistency import (direction_consistency_binary,
                                                 direction_consistency_multiclass, task_subject_overlap,
                                                 representation_geometry)

OUT = REPO / "results" / "cmi_trace_relaxation_ladder"
CFG = REPO / "configs" / "cmi_trace_relaxation_ladder.yaml"
GATES = ["G0", "G1", "G2", "G3"]


def _cfg_hash():
    return hashlib.sha256(CFG.read_bytes()).hexdigest() if CFG.exists() else "no_config"


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _meta(feat):
    return dict(dataset=feat.get("dataset", ""), backbone=feat.get("backbone", ""),
                feature_object=feat.get("feature_object", ""), training_method=feat.get("training_method", ""),
                heldout_subject=str(feat.get("heldout_subject", "")), seed=int(feat.get("seed", 0)))


def _gate_rows(feat, cfg_hash, git_sha):
    diag = SE.source_diagnostics(feat, seed=int(feat.get("seed", 0)))
    rows = []
    for pol in GATES:
        acc, reason = SE.gate_decision(diag, pol)
        rows.append({**_meta(feat), "policy": pol, "accepted": bool(acc), "refusal_reason": reason,
                     "policy_positive": "",  # filled later against target (H5) — decision itself is source-only
                     **{k: diag[k] for k in diag}, "config_hash": cfg_hash, "git_sha": git_sha})
    # H5 gated-vs-identity target effect (target Y in scoring only), per policy
    for r in rows:
        eff = SE.gated_target_effect(feat, r["policy"], diag, seed=int(feat.get("seed", 0)))
        r.update(gated_target_bacc=eff["gated_target_bacc"], identity_target_bacc=eff["identity_target_bacc"],
                 gated_minus_identity=eff["gated_minus_identity"])
    return rows


def _direction_rows(feat, cfg_hash, git_sha):
    Zs, ys, ds = feat["Z_source"], feat["y_source"], feat["subj_source"]
    classes = sorted(int(c) for c in np.unique(ys))
    base = {**_meta(feat), "config_hash": cfg_hash, "git_sha": git_sha}
    out = []
    if len(classes) == 2:
        r = direction_consistency_binary(Zs, ys, ds, classes[1], classes[0])
        out.append({**base, "kind": "binary", "pair": f"{classes[1]}v{classes[0]}",
                    "consistency": r["mean_pairwise_cosine"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                    "perm_p": r["perm_p"], "snr": r["contrast_snr"], "n_clusters": r["n_clusters"]})
    else:
        r = direction_consistency_multiclass(Zs, ys, ds, classes)
        out.append({**base, "kind": "multiclass_macro", "pair": "macro",
                    "consistency": r["macro_avg_consistency"], "ci_lo": r["macro_avg_ci_lo"],
                    "ci_hi": r["macro_avg_ci_hi"], "perm_p": "", "snr": "", "n_clusters": r["n_valid_pairs"]})
        for pr, v in r["per_pair"].items():
            out.append({**base, "kind": "multiclass_pair", "pair": f"{pr[0]}v{pr[1]}",
                        "consistency": v["mean_pairwise_cosine"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                        "perm_p": v["perm_p"], "snr": v["contrast_snr"], "n_clusters": v["n_used"]})
    return out


def _overlap_row(feat, cfg_hash, git_sha):
    Zs, ys, ds = feat["Z_source"], feat["y_source"], feat["subj_source"]
    ov = task_subject_overlap(Zs, ys, ds)
    geo = representation_geometry(Zs)
    return {**_meta(feat), "normalized_overlap": ov["normalized_overlap"], "raw_overlap": ov["raw_overlap"],
            "rank_Y": ov["rank_Y"], "rank_D": ov["rank_D"], "null_mean": ov["null_mean"],
            "null_ci_hi": ov["null_ci_hi"], "feature_norm": geo["feature_norm"],
            "cov_condition_number": geo["cov_condition_number"], "effective_rank": geo["effective_rank"],
            "top_singular_value": geo["top_singular_value"], "config_hash": cfg_hash, "git_sha": git_sha}


def _cmi_ruler_rows(feat, cfg_hash, git_sha, n_perm=20, epochs=60):
    from cmi.eval.conditional_subject_leakage import three_way_support_split
    from cmi.eval.multicapacity_probe import multicapacity_cmi
    Zs, ys = feat["Z_source"], feat["y_source"]
    ds = _dense(feat["subj_source"]); n_cls = int(feat.get("n_cls", len(np.unique(ys))))
    n_dom = int(len(np.unique(ds)))
    er, pt, pe, _ = three_way_support_split(ys.astype(int), ds, seed=int(feat.get("seed", 0)))
    Zfit = Zs[er]
    fn_lw, rk = lw_leace_full(Zfit, ds[er])
    fn_w, _ = whitening_only(Zfit)
    fn_r = random_removal(Zs.shape[1], rk, seed=int(feat.get("seed", 0)))
    transforms = {"full": Zs, "whitening_only": fn_w(Zs), "lw_leace_full": fn_lw(Zs), "random_k": fn_r(Zs)}
    rows = []
    for name, Z in transforms.items():
        m = multicapacity_cmi(Z, ys.astype(int), ds, n_cls, n_dom, pt, pe, n_perm=n_perm, seed=int(feat.get("seed", 0)), epochs=epochs)
        pc = m["per_capacity"]
        rows.append({**_meta(feat), "eraser": name, "eraser_rank": (0 if name in ("full", "whitening_only") else rk),
                     "primary_mlp_small_kl": pc["mlp_small"]["kl"], "linear_kl": pc["linear"]["kl"],
                     "mlp_large_kl": pc["mlp_large"]["kl"], "familywise_max_kl": m["familywise_max_kl"],
                     "familywise_max_perm_p": m["familywise_max_perm_p"], "primary_perm_p": m["primary_perm_p"],
                     "config_hash": cfg_hash, "git_sha": git_sha})
    return rows


def _cells(a):
    if a.family == "dgcnn_graph_z":
        d = REPO / "results/cmi_trace_p0p1/objective_comparison" / a.dataset / "audit"
        cs = []
        for m in a.methods:
            cs += [p for p in sorted(glob.glob(str(d / f"{a.dataset}_fold*_sub*_{m}_seed*.audit.npz")))
                   if any(p.endswith(f"_seed{s}.audit.npz") for s in a.seeds)]
        return cs, feat_from_audit_npz
    d = REPO / "tos_cmi/results/tos_cmi_eeg_frozen" / f"{a.dataset}_{a.backbone}_LOSO"
    return [p for p in sorted(glob.glob(str(d / "sub*_erm_lam0_seed*.npz")))
            if any(p.endswith(f"_seed{s}.npz") for s in a.seeds)], feat_from_tos_dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="dgcnn_graph_z", choices=["dgcnn_graph_z", "tos_frozen"])
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--methods", nargs="+", default=["erm", "cigl_graph_node"])
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--cmi_ruler", action="store_true")
    ap.add_argument("--cmi_nperm", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_tag", default=None)
    a = ap.parse_args()
    cfg_hash, git_sha = _cfg_hash(), _git_sha()
    cells, loader = _cells(a)
    if a.limit:
        cells = cells[: a.limit]
    if not cells:
        raise SystemExit(f"[diag] no cells for {a.family}/{a.dataset}")
    tag = a.out_tag or (f"{a.family}_{a.dataset}" + (f"_{a.backbone}" if a.family == "tos_frozen" else ""))
    outdir = OUT / tag; outdir.mkdir(parents=True, exist_ok=True)
    gate_rows, dir_rows, ov_rows, cmi_rows = [], [], [], []
    t0 = time.time()
    for i, cp in enumerate(cells):
        try:
            feat = loader(cp)
        except Exception as e:
            print(f"  [skip] {Path(cp).name}: {e!r}", flush=True); continue
        gate_rows += _gate_rows(feat, cfg_hash, git_sha)
        dir_rows += _direction_rows(feat, cfg_hash, git_sha)
        ov_rows.append(_overlap_row(feat, cfg_hash, git_sha))
        if a.cmi_ruler:
            cmi_rows += _cmi_ruler_rows(feat, cfg_hash, git_sha, n_perm=a.cmi_nperm)
        print(f"  [{i+1}/{len(cells)}] {Path(cp).name} ({time.time()-t0:.0f}s)", flush=True)
    _csv(outdir / "gate_decisions.csv", gate_rows)
    _csv(outdir / "direction_consistency.csv", dir_rows)
    _csv(outdir / "subspace_overlap.csv", ov_rows)
    if cmi_rows:
        _csv(outdir / "cmi_ruler.csv", cmi_rows)
    print(f"[diag] wrote gate/direction/overlap{'/cmi_ruler' if cmi_rows else ''} -> {outdir}", flush=True)


def _csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
