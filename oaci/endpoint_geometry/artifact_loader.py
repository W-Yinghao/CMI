"""C31 — read-only loader + join + generalized AUC primitives. Joins per-candidate target endpoint metrics
(bAcc/NLL/ECE + per-fold ERM reference, from the C10 replay) to the C22 source-rank sidecar, and exposes
within-target / pooled AUC + sign-consistency for ANY factor->ANY endpoint label. Nothing is refit or tuned."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from ..identifiability.multivariate_probe import _auc
from ..information_ladder import artifact_loader as il
from . import schema

load_scores = il.load
_finite = il._finite


def load_endpoints(c10_dir=None):
    """Per-candidate target endpoint metrics + the per-(seed,target,level) ERM reference (read-only)."""
    root = c10_dir or schema.C10_REPLAY_DIR
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "seed-*-target-*.json"))):
        m = re.search(r"seed-(\d+)-target-(\d+)", f)
        seed, target = int(m.group(1)), int(m.group(2))
        d = json.load(open(f))
        for L, lv in d["levels"].items():
            cs = lv["candidates"]
            erm = next(c for c in cs if c.get("is_erm"))
            eb, en, ee = erm.get("target_worst_bacc"), erm.get("target_worst_nll"), erm.get("target_worst_ece")
            for c in cs:
                if c.get("is_erm") or not c.get("feasible"):
                    continue
                rows.append({"seed": seed, "target": target, "level": int(L), "model_hash": c["model_hash"],
                             "bacc": c.get("target_worst_bacc"), "nll": c.get("target_worst_nll"),
                             "ece": c.get("target_worst_ece"), "erm_bacc": eb, "erm_nll": en, "erm_ece": ee})
    return rows


def merge(score_rows, endpoint_rows, mode="in_regime"):
    """Merge source-rank sidecar rows (mode) with endpoint metrics by (seed,target,level,model_hash)."""
    emap = {(r["seed"], r["target"], r["level"], r["model_hash"]): r for r in endpoint_rows}
    out = []
    for s in score_rows:
        if s["mode"] != mode:
            continue
        k = (s["seed"], s["target"], s["level"], s["model_hash"])
        if k in emap:
            out.append({**s, **{kk: emap[k][kk] for kk in ("bacc", "nll", "ece", "erm_bacc", "erm_nll", "erm_ece")}})
    return out


# ---- generalized AUC primitives (explicit factor + label key) ----
def _v(r, key):
    v = r.get(key)
    return float(v) if _finite(v) else None


def within_target_auc(rows, factor_key, label_key) -> float:
    per = []
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t]
        y = np.array([_v(r, label_key) for r in g], dtype=float)
        x = np.array([_v(r, factor_key) for r in g], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        yb = y[ok]
        if ok.sum() > 2 and yb.std() > 1e-9 and 0 < yb.sum() < ok.sum() and x[ok].std() > 1e-9:
            per.append(_auc(yb.astype(int), x[ok]))
    return float(np.mean(per)) if per else None


def pooled_auc(rows, factor_key, label_key) -> float:
    y = np.array([_v(r, label_key) for r in rows], dtype=float)
    x = np.array([_v(r, factor_key) for r in rows], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    yb = y[ok]
    return _auc(yb.astype(int), x[ok]) if (ok.sum() > 2 and 0 < yb.sum() < ok.sum()) else None


def rank_strength(auc):
    return None if auc is None else abs(auc - 0.5)


def sign_consistency(rows, factor_key, label_key) -> dict:
    per = {}
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t]
        y = np.array([_v(r, label_key) for r in g], dtype=float)
        x = np.array([_v(r, factor_key) for r in g], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        yb = y[ok]
        if ok.sum() > 2 and yb.std() > 1e-9 and 0 < yb.sum() < ok.sum() and x[ok].std() > 1e-9:
            per[t] = _auc(yb.astype(int), x[ok])
    if not per:
        return {"sign_consistency": None, "transfers": None, "n_targets": 0, "n_above_half": None}
    above = sum(1 for a in per.values() if a > 0.5)
    frac = max(above, len(per) - above) / len(per)
    return {"sign_consistency": float(frac), "transfers": bool(frac >= 0.8), "n_targets": len(per), "n_above_half": above}
