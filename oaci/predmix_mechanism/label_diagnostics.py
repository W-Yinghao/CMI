"""C26 Q5 — LABEL-DIAGNOSTIC-ONLY (labels join ONLY here, never in feature construction). Does the predicted-
class mix correspond to the target's class-error geometry (true class prior / per-class recall / class-pair
confusion)? Needs per-sample target labels + predictions per candidate, which the C24 aggregate sidecar does not
store -> availability-gated (REQUIRES_REPERSISTENCE_REINFERENCE), NOT proxied. If it aligns with error geometry,
predmix reflects the frozen model's target decision-boundary occupancy, not a pure identity fingerprint."""
from __future__ import annotations

import os

import numpy as np

from . import schema


def availability(split_sidecar=None) -> dict:
    path = split_sidecar or schema.C26_SPLIT_SIDECAR
    ready = os.path.exists(path)
    return {"path": path, "label_diag_ready": ready,
            "status": schema.STATUS_OK if ready else schema.STATUS_REQUIRES_REINFERENCE,
            "reason": ("" if ready else "per-sample target labels not persisted; label diagnostics need the "
                       "scoped re-persistence re-inference (label diagnostics QUARANTINED, never features).")}


def label_diagnostics(rows, mode, split_sidecar=None) -> dict:
    av = availability(split_sidecar)
    if not av["label_diag_ready"]:
        return {"status": schema.STATUS_REQUIRES_REINFERENCE, "reason": av["reason"], "alignment": None}
    import json
    d = json.load(open(av["path"]))
    ld = d.get("label_diagnostics")
    if ld is None:
        return {"status": schema.STATUS_REQUIRES_REINFERENCE, "reason": "split sidecar lacks label_diagnostics block", "alignment": None}
    # ld: per-candidate {predmix vector, true_prior, per_class_recall, offset}. Correlate mix-vs-error summaries.
    percand = ld["per_candidate"]
    predmix = np.array([[c["predmix"][k] for k in schema.PRED_PROP] for c in percand])
    true_prior = np.array([c["true_prior"] for c in percand])
    recall = np.array([c["per_class_recall"] for c in percand])
    # class-wise correlation of predicted mix vs (true prior, recall) across candidates
    prior_corr = float(np.mean([np.corrcoef(predmix[:, k], true_prior[:, k])[0, 1] for k in range(schema.N_CLASSES)]))
    recall_corr = float(np.mean([np.corrcoef(predmix[:, k], recall[:, k])[0, 1] for k in range(schema.N_CLASSES)]))
    mix_dist_from_prior = float(np.mean(np.linalg.norm(predmix - true_prior, axis=1)))
    tracks_error = bool(abs(recall_corr) >= 0.30 or mix_dist_from_prior >= 0.10)
    return {"status": schema.STATUS_OK, "predmix_vs_true_prior_corr": prior_corr,
            "predmix_vs_per_class_recall_corr": recall_corr, "mix_distance_from_true_prior": mix_dist_from_prior,
            "tracks_target_error_geometry": tracks_error,
            "note": ("predmix deviates from the true (balanced) class prior and tracks per-class recall -> reflects "
                     "the frozen model's target decision-boundary occupancy / error geometry, not the true label "
                     "prior itself" if tracks_error else "predmix tracks the true class prior; no distinct error-"
                     "geometry signal")}
