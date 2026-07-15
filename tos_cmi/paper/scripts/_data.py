"""Pure data loaders for the paper figures (no plotting). Guarantees Fig 4 / Fig 5 read identical
metrics so their shared panels are directly comparable."""
from __future__ import annotations
import glob
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EEG = os.path.join(ROOT, "tos_cmi", "results", "tos_cmi_eeg_frozen")
LPC = os.path.join(EEG, "lpc_collapse_curves")
Z_DIM = {"TSMNet": 210, "EEGNet": 16}


def load_ablation(backbone):
    """Pool all per-fold rows across the 3 seeds; return mean/sd/list per metric + chances + nDcand."""
    base = os.path.join(EEG, "BNCI2014_001_%s_LOSO" % backbone)
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "ablation_report_seed*.json"))):
        d = json.load(open(p)); rows += (d["rows"] if isinstance(d, dict) else d)
    def col(k):
        v = np.array([r[k] for r in rows if k in r], float)
        return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "vals": v}
    metrics = {}
    for base_m in ["task", "domain"]:
        for rep in ["Z", "RZ", "PNZ", "Rrand", "PNrand"]:
            for fam in ["linear", "mlp"]:
                k = "%s_%s_%s" % (base_m, rep, fam)
                if any(k in r for r in rows):
                    metrics[k] = col(k)
    return {
        "backbone": backbone, "z_dim": Z_DIM.get(backbone), "n": len(rows),
        "n_seeds": len(glob.glob(os.path.join(base, "ablation_report_seed*.json"))),
        "metrics": metrics, "nDcand": col("nDcand"),
        "domain_chance": float(np.nanmean([r["domain_chance_B"] for r in rows])),
        "label_chance": float(np.nanmean([r["label_chance"] for r in rows])),
    }


def load_lpc_sweep(backbone, lams=(0.0, 0.3, 1.0, 3.0)):
    """Per-lambda arrays from the raw_lpc sweep jsons (handles TSMNet flat tags + EEGNet variant tags)."""
    d = os.path.join(LPC, backbone)
    # RAW global-LPC sweep only: Phase-2.1 flat tags (sub*_lam*_seed*) + Phase-2.2/3 raw_lpc_* tags;
    # EXCLUDE the scale_invariant / warm_ramp variants. Dedup by (subject, lam, seed).
    paths = sorted(p for p in set(glob.glob(os.path.join(d, "*sub*_lam*_seed*.json")))
                   if "scale_invariant" not in os.path.basename(p) and "warm_ramp" not in os.path.basename(p))
    by = {}
    chance = None
    seen = set()
    for p in paths:
        r = json.load(open(p)); lam = round(r["lam"], 3)
        key = (r.get("target_subject"), lam, r.get("seed"))
        if key in seen:
            continue
        seen.add(key)
        c = r.get("curves") or [{}]
        rec = {"src": r.get("final_source_bAcc"), "tgt": r.get("final_target_bAcc"),
               "subj": r.get("final_subject_decode"), "task_dec": r.get("final_task_decode"),
               "feat_norm": c[-1].get("feat_norm_mean")}
        by.setdefault(lam, []).append(rec)
        chance = r.get("chance_subj", chance)
    out = {"backbone": backbone, "chance_subj": chance, "lams": [], "per_lam": {}}
    for lam in lams:
        rs = by.get(round(lam, 3), [])
        if not rs:
            continue
        out["lams"].append(lam)
        agg = {}
        for k in ["src", "tgt", "subj", "task_dec", "feat_norm"]:
            v = np.array([x[k] for x in rs if x.get(k) is not None], float)
            agg[k] = {"median": float(np.nanmedian(v)) if len(v) else float("nan"),
                      "mean": float(np.nanmean(v)) if len(v) else float("nan"),
                      "sd": float(np.nanstd(v)) if len(v) else float("nan"), "vals": v}
        agg["n"] = len(rs)
        out["per_lam"][lam] = agg
    return out
