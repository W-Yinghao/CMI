"""C18-R (H4) — class-boundary source-visibility under support stress. C17 found the OACI-ERM class-boundary
rotation is MIRRORED on the held-out SOURCE audit split (source<->target per-class recall-delta corr +0.547).
H4 asks whether that source-visible structure collapses SPECIFICALLY when the perturbed cells touch the
boundary-rotation classes (S6 boundary-aligned) vs an equally-severe random mask (S7). We recompute the SOURCE
per-class recall delta under each regime's audit-side mask (from the persisted per-unit source_audit logits of
the selected ERM/OACI checkpoints) and correlate it with the UNMASKED target per-class recall delta (committed
C8 target_audit.npz; target is never degraded). The S6-vs-S7 gap is the key contrast."""
from __future__ import annotations

import json
import os

import numpy as np

from ..eval.metrics import per_class_recall
from ..mechanism.harm_decomposition import _artifact
from . import masks, schema
from . import source_signal_recompute as ssr
from . import stress_plan as sp


def _c10_selected(c10_dir, seed, target) -> dict:
    d = json.load(open(os.path.join(c10_dir, f"seed-{seed}-target-{target:03d}.json")))
    return {int(L): lv["selected"] for L, lv in d["levels"].items()}


def _target_recall(loso_root, seed, target, level, method, classes):
    a = _artifact(loso_root, seed, target)
    if a is None:
        return None
    p = os.path.join(a, f"levels/level-{level:03d}", "methods", method, "target_audit.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    logits = np.asarray(z["logits"], dtype=np.float64); y = np.asarray(z["y"]).astype(int)
    return per_class_recall(y, logits.argmax(1), classes)


def _source_recall_masked(fld, model_hash, name_actions, classes):
    idx_by_hash = {cm["model_hash"]: cm["index"] for cm in fld["cand_meta"]}
    if model_hash not in idx_by_hash:
        return None
    ci = idx_by_hash[model_hash]
    u = fld["units"]["source_audit"]
    keep, _ = masks.unit_keep_weight(name_actions, u["domain_raw"], u["y"], seed=fld["seed"],
                                     target=fld["target"], level=fld["level"])
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return None
    logits = np.asarray(fld["logits"]["source_audit"][ci])[idx]; y = np.asarray(u["y"])[idx]
    return per_class_recall(y, logits.argmax(1), classes)


def _pearson(x, y):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def boundary_visibility_stress(extract_dir, loso_root, c10_dir, *, boundary_classes, n_perturb=2, folds=None) -> dict:
    fold_dirs = folds if folds is not None else ssr._list_folds(extract_dir)
    per_regime = {}
    for r in schema.REGIME_ORDER:
        src_deltas, tgt_deltas = [], []
        n_points = 0
        for (seed, target) in fold_dirs:
            for level in ssr._levels(extract_dir, seed, target):
                fld = ssr.load_fold_level(extract_dir, seed, target, level)
                classes = list(range(fld["logits"]["source_audit"].shape[2]))
                sel = _c10_selected(c10_dir, seed, target).get(level)
                if not sel:
                    continue
                audit_na, _ = ssr._regime_name_actions(r, fld["support_audit"], boundary_classes=boundary_classes,
                                                       seed=seed, target=target, level=level, n_perturb=n_perturb)
                se = _source_recall_masked(fld, sel["ERM"], audit_na, classes)
                so = _source_recall_masked(fld, sel["OACI"], audit_na, classes)
                te = _target_recall(loso_root, seed, target, level, "ERM", classes)
                to = _target_recall(loso_root, seed, target, level, "OACI", classes)
                if None in (se, so, te, to):
                    continue
                for c in classes:
                    sd = (so.get(c) - se.get(c)) if (c in so and c in se) else None
                    td = (to.get(c) - te.get(c)) if (c in to and c in te) else None
                    if sd is not None and td is not None and np.isfinite(sd) and np.isfinite(td):
                        src_deltas.append(sd); tgt_deltas.append(td); n_points += 1
        corr = _pearson(src_deltas, tgt_deltas)
        per_regime[r] = {"n_class_fold_points": n_points, "source_target_recall_delta_corr": corr,
                         "boundary_source_visible": bool(corr is not None and corr > 0.3)}
    s6 = per_regime["S6_boundary_aligned_mask"]["source_target_recall_delta_corr"]
    s7 = per_regime["S7_random_matched_mask"]["source_target_recall_delta_corr"]
    s0 = per_regime["S0_full_support"]["source_target_recall_delta_corr"]
    boundary_specific = bool(s6 is not None and s7 is not None
                             and abs(s6) < 0.3 and (abs(s7) - abs(s6)) > 0.15)
    return {"per_regime": per_regime, "s0_corr": s0, "s6_corr": s6, "s7_corr": s7,
            "boundary_aligned_destroys_mirror_vs_random": boundary_specific,
            "scope": "SELECTED ERM/OACI checkpoints; source masked, target unmasked (support degradation is source-side)",
            "note": ("if S6 (boundary-aligned) collapses the source<->target mirror while S7 (random, severity-"
                     "matched) preserves it, source-visible boundary structure depends on support coverage of "
                     "the boundary-rotation classes.")}
