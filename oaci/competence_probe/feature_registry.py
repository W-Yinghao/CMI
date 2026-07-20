"""C19 — frozen feature registry + extraction. Computes the pre-registered ROBUST-CORE source observables
(confidence geometry + calibration + leakage) and the SECONDARY fragile accuracy-endpoint features per
candidate, from the committed C18 replay artifacts (per-unit source logits + extracted/recomputed leakage).
Confidence geometry is aggregated over whatever source units survive a mask -> deletion-robust (never NaN);
worst-domain reference bAcc is the fragile endpoint (NaN when a domain loses a class)."""
from __future__ import annotations

import numpy as np

from ..eval.calibration import fixed_bin_edges, top_label_ece
from ..support_stress import masks
from ..support_stress import source_signal_recompute as ssr
from . import schema


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def _conf_geom(logits, y, edges) -> dict:
    """Deletion-robust confidence geometry + calibration over the (masked) source units."""
    if len(y) == 0:
        return {s: float("nan") for s in ("nll", "ece", "entropy", "confidence", "margin", "logit_norm", "conf_on_wrong")}
    z = np.asarray(logits, dtype=np.float64); y = np.asarray(y).astype(int); p = _softmax(z)
    py = np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)
    srt = np.sort(p, 1); pred = p.argmax(1); wrong = pred != y
    return {"nll": float(-np.log(py).mean()),
            "ece": float(top_label_ece(z, y, bin_edges=edges)),
            "entropy": float(-(p * np.log(np.clip(p, 1e-9, 1.0))).sum(1).mean()),
            "confidence": float(p.max(1).mean()),
            "margin": float((srt[:, -1] - srt[:, -2]).mean()),
            "logit_norm": float(np.linalg.norm(z, axis=1).mean()),
            "conf_on_wrong": float(p.max(1)[wrong].mean()) if wrong.any() else float("nan")}


def candidate_features(fld, ci, source_na, audit_na, *, edges, leakage) -> dict:
    """Robust-core (confidence geometry per role + leakage) + fragile endpoint (worst-domain bAcc) for one
    candidate under a regime's dual-side mask. `leakage` = (selection_leakage_point, audit_leakage_point)."""
    seed, target, level = fld["seed"], fld["target"], fld["level"]
    out = {}
    for role, na in (("source_guard", source_na), ("source_audit", audit_na)):
        u = fld["units"][role]
        keep, _ = masks.unit_keep_weight(na, u["domain_raw"], u["y"], seed=seed, target=target, level=level)
        idx = np.where(keep)[0]
        cg = _conf_geom(fld["logits"][role][ci][idx], np.asarray(u["y"])[idx], edges)
        for s, v in cg.items():
            out[f"{role}_{s}"] = v
        # fragile endpoint: worst-domain reference bAcc (NaN if a domain misses a class)
        b, _, _ = ssr._worst_domain(fld["logits"][role][ci], u["y"], u["domain_raw"], keep,
                                    list(range(fld["logits"][role].shape[2])), edges)
        out[f"{role}_worst_bacc"] = b
    sel, aud = leakage
    out["selection_leakage_point"], out["audit_leakage_point"] = sel, aud
    return out


def build_atlas(extract_dir, c10_dir, regime, *, boundary_classes, leakage_lookup, folds=None) -> list:
    """One row per feasible-OACI candidate (the C17/C18 population): robust-core + endpoint features +
    diagnostic target label. `leakage_lookup(seed,target,level,model_hash) -> (sel, aud)`."""
    labels = ssr._target_labels(c10_dir)
    fold_dirs = folds if folds is not None else ssr._list_folds(extract_dir)
    edges = fixed_bin_edges(15)
    rows = []
    for (seed, target) in fold_dirs:
        for level in ssr._levels(extract_dir, seed, target):
            fld = ssr.load_fold_level(extract_dir, seed, target, level)
            source_na, _ = ssr._regime_name_actions(regime, fld["support_source"], boundary_classes=boundary_classes,
                                                    seed=seed, target=target, level=level, n_perturb=2)
            audit_na, _ = ssr._regime_name_actions(regime, fld["support_audit"], boundary_classes=boundary_classes,
                                                   seed=seed, target=target, level=level, n_perturb=2)
            for cm in fld["cand_meta"]:
                if cm["is_erm"] or not cm["feasible"]:
                    continue
                key = (seed, target, level, cm["model_hash"])
                if key not in labels:
                    continue
                lk = leakage_lookup(seed, target, level, cm["model_hash"])
                feats = candidate_features(fld, cm["index"], source_na, audit_na, edges=edges, leakage=lk)
                row = {"seed": seed, "target": target, "level": level, "model_hash": cm["model_hash"],
                       "regime": regime, "diagnostic_only_non_deployable": True}
                row.update(feats)
                row.update(labels[key])
                rows.append(row)
    return rows
