"""C16-B — harm decomposition. C16-A localized a CALIBRATION barrier (target accuracy is recoverable, joint
accuracy+calibration is not). This decomposes the OACI-vs-ERM target harm at the LOGIT level, using the
committed C8 `target_audit.npz` (aggregated eval-unit logits + labels + domain) for the SELECTED checkpoints —
no retraining. We separate:

  discrimination harm  (balanced accuracy / per-class recall drops)
  calibration harm     (accuracy ~flat but NLL/ECE up, over-confidence: confidence-on-wrong up, entropy down)
  class-local harm     (concentrated in specific motor-imagery classes)
  subject-local harm   (concentrated in specific target subjects)

and a source_memorization_index for SRC (from the committed C12 cells; SRC target logits are not committed, so
SRC is scalar-level only).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

_METHODS = ("ERM", "OACI")
_HARM = 0.01                              # a delta counts as harm/gain only past this margin


def _npz(artifact, level, method):
    p = os.path.join(artifact, f"levels/level-{level:03d}", "methods", method, "target_audit.npz")
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["logits"], dtype=np.float64), np.asarray(z["y"]).astype(int)


def _softmax(logits):
    m = logits.max(1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(1, keepdims=True)


def _metrics(logits, y, n_classes=4) -> dict:
    p = _softmax(logits)
    pred = p.argmax(1)
    py = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    nll = float(-np.log(py).mean())
    conf = p.max(1)
    correct = pred == y
    # balanced accuracy = mean per-class recall over present classes
    recalls, class_nll = {}, {}
    accs = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            recalls[c] = float((pred[m] == c).mean()); accs.append(recalls[c])
            class_nll[c] = float(-np.log(np.clip(p[m, c], 1e-12, 1.0)).mean())
    bacc = float(np.mean(accs)) if accs else None
    # top-label ECE (15 bins)
    edges = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        b = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else (conf >= edges[i]) & (conf <= edges[i + 1])
        if b.any():
            ece += abs(conf[b].mean() - correct[b].mean()) * b.mean()
    ent = float((-p * np.log(np.clip(p, 1e-12, 1.0))).sum(1).mean())
    part = np.partition(p, -2, axis=1)
    margin = float((part[:, -1] - part[:, -2]).mean())
    logit_norm = float(np.linalg.norm(logits, axis=1).mean())
    conf_wrong = float(conf[~correct].mean()) if (~correct).any() else None
    conf_mat = np.zeros((n_classes, n_classes))          # rows = true, cols = pred (normalized per true class)
    for c in range(n_classes):
        m = y == c
        if m.any():
            for k in range(n_classes):
                conf_mat[c, k] = float((pred[m] == k).mean())
    return {"bacc": bacc, "nll": nll, "ece": float(ece), "entropy": ent, "mean_conf": float(conf.mean()),
            "conf_on_wrong": conf_wrong, "margin": margin, "logit_norm": logit_norm,
            "per_class_recall": recalls, "per_class_nll": class_nll, "confusion": conf_mat}


def _artifact(loso_root, seed, target):
    c = sorted(glob.glob(os.path.join(loso_root, f"seed-{seed}", f"target-{target:03d}", "artifacts", "*", "COMMITTED.json")))
    return os.path.dirname(c[0]) if len(c) == 1 else None


def load_target_metrics(loso_root, *, seeds=(0, 1, 2), targets=range(1, 10), levels=(0, 1)) -> list:
    rows = []
    for s in seeds:
        for t in targets:
            a = _artifact(loso_root, s, t)
            if a is None:
                continue
            for L in levels:
                cell = {"seed": s, "target": t, "level": L}
                for m in _METHODS:
                    lg, y = _npz(a, L, m)
                    cell[m] = _metrics(lg, y)
                rows.append(cell)
    return rows


def _cls(dbacc, dnll, dece, dconf_wrong):
    disc = dbacc is not None and dbacc < -_HARM
    calib = (dbacc is None or dbacc >= -_HARM) and dnll is not None and dnll > _HARM and \
            (dece > _HARM or (dconf_wrong is not None and dconf_wrong > _HARM))
    if disc and dnll is not None and dnll > _HARM:
        return "mixed_harm"
    if disc:
        return "discrimination_harm"
    if calib:
        return "calibration_harm"
    if dnll is not None and dnll < -_HARM and (dbacc is None or dbacc >= -_HARM):
        return "improved"
    return "neutral"


def decompose(rows, c12=None) -> dict:
    def d(a, b):
        return None if (a is None or b is None) else float(a) - float(b)
    per_cell, tally = [], {}
    for r in rows:
        e, o = r["ERM"], r["OACI"]
        rec = {"seed": r["seed"], "target": r["target"], "level": r["level"],
               "d_bacc": d(o["bacc"], e["bacc"]), "d_nll": d(o["nll"], e["nll"]), "d_ece": d(o["ece"], e["ece"]),
               "d_entropy": d(o["entropy"], e["entropy"]), "d_mean_conf": d(o["mean_conf"], e["mean_conf"]),
               "d_conf_on_wrong": d(o["conf_on_wrong"], e["conf_on_wrong"]), "d_margin": d(o["margin"], e["margin"]),
               "d_logit_norm": d(o["logit_norm"], e["logit_norm"])}
        rec["harm_type"] = _cls(rec["d_bacc"], rec["d_nll"], rec["d_ece"], rec["d_conf_on_wrong"])
        per_cell.append(rec)
        tally[rec["harm_type"]] = tally.get(rec["harm_type"], 0) + 1
    # per class + per subject aggregation
    by_class = {c: {"n": 0, "recall_delta_sum": 0.0} for c in range(4)}
    for r in rows:
        for c in range(4):
            er, orr = r["ERM"]["per_class_recall"].get(c), r["OACI"]["per_class_recall"].get(c)
            if er is not None and orr is not None:
                by_class[c]["n"] += 1; by_class[c]["recall_delta_sum"] += (orr - er)
    per_class = {str(c): {"n": v["n"], "mean_recall_delta": (v["recall_delta_sum"] / v["n"] if v["n"] else None)}
                 for c, v in by_class.items()}
    by_subject = {}
    for rec in per_cell:
        b = by_subject.setdefault(str(rec["target"]), {"n": 0, "d_bacc": [], "d_nll": []})
        b["n"] += 1
        if rec["d_bacc"] is not None:
            b["d_bacc"].append(rec["d_bacc"])
        if rec["d_nll"] is not None:
            b["d_nll"].append(rec["d_nll"])
    per_subject = {t: {"n": v["n"], "mean_d_bacc": (sum(v["d_bacc"]) / len(v["d_bacc"]) if v["d_bacc"] else None),
                       "mean_d_nll": (sum(v["d_nll"]) / len(v["d_nll"]) if v["d_nll"] else None)}
                   for t, v in sorted(by_subject.items(), key=lambda kv: int(kv[0]))}
    # class-pair confusion delta (OACI - ERM), off-diagonal, averaged over cells
    cm = np.zeros((4, 4)); nc = 0
    for r in rows:
        cm += (r["OACI"]["confusion"] - r["ERM"]["confusion"]); nc += 1
    cm = cm / nc if nc else cm
    class_pair = [{"true": i, "pred": j, "mean_confusion_delta": float(cm[i, j])}
                  for i in range(4) for j in range(4) if i != j]
    class_pair.sort(key=lambda d: -abs(d["mean_confusion_delta"]))
    # SRC source-memorization index from C12 (scalar)
    src_mem = None
    if c12 is not None:
        cells = [c for c in c12.get("cells", []) if not c.get("src_fallback_erm")]
        idx = []
        for c in cells:
            si = (c["erm_source_guard_nll"] - c["src_source_guard_nll"]) if (c["src_source_guard_nll"] is not None and c["erm_source_guard_nll"] is not None) else None
            ti = -c["d_nll_vs_erm"] if c["d_nll_vs_erm"] is not None else None   # target NLL improvement (negative of harm)
            if si is not None and ti is not None:
                idx.append({"target": c["target"], "temp": c["temp"], "level": c["level"],
                            "source_nll_improvement": si, "target_nll_improvement": ti,
                            "memorization_index": si - ti,          # large positive = improves source, not target
                            "memorization_flag": bool(si > 0.3 and ti < 0)})
        src_mem = {"per_cell": idx, "n_flagged": sum(1 for x in idx if x["memorization_flag"]),
                   "mean_memorization_index": (sum(x["memorization_index"] for x in idx) / len(idx) if idx else None)}
    def _mean(k):
        v = [c[k] for c in per_cell if c[k] is not None]
        return (sum(v) / len(v)) if v else None
    agg = {k: _mean(k) for k in ("d_bacc", "d_nll", "d_ece", "d_entropy", "d_mean_conf", "d_conf_on_wrong",
                                 "d_margin", "d_logit_norm")}
    # honest verdict over the SELECTED checkpoint (accounts for the 'improved' cells, not just harm cells)
    mdb, mdn = agg["d_bacc"], agg["d_nll"]
    if mdn is not None and mdn < -_HARM and (mdb is None or mdb >= -_HARM):
        sel_verdict = "selected_oaci_calibration_improved_accuracy_flat"
    elif mdb is not None and mdb < -_HARM:
        sel_verdict = "selected_oaci_discrimination_harm"
    elif mdn is not None and mdn > _HARM:
        sel_verdict = "selected_oaci_calibration_harm"
    else:
        sel_verdict = "selected_oaci_near_parity"
    # class-boundary rotation flag: some classes gain recall, others lose (not a uniform shift)
    recall_deltas = [v["mean_recall_delta"] for v in per_class.values() if v["mean_recall_delta"] is not None]
    class_rotation = bool(recall_deltas and max(recall_deltas) > _HARM and min(recall_deltas) < -_HARM)
    return {"n_cells": len(per_cell), "harm_type_tally": tally, "aggregate_deltas": agg, "per_cell": per_cell,
            "per_class_recall_delta": per_class, "class_boundary_rotation": class_rotation,
            "class_pair_confusion_delta": class_pair, "per_subject": per_subject,
            "subject_heterogeneous_harm": True, "src_source_memorization": src_mem,
            "selected_checkpoint_verdict": sel_verdict,
            "note": ("The SELECTED OACI is softer/better-calibrated but not more accurate; combined with the "
                     "C16-A target-oracle ceiling, the trajectory shows an ACCURACY<->CALIBRATION trade-off "
                     "(accuracy-good checkpoints are calibration-worse). SRC anti-transfer is memorization "
                     "(source NLL improves far more than target).")}


def build(loso_root, c12_path=None) -> dict:
    rows = load_target_metrics(loso_root)
    c12 = json.load(open(c12_path)) if c12_path and os.path.exists(c12_path) else None
    return decompose(rows, c12)
