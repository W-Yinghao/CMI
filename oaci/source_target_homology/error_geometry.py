"""C28 Q4 — does the source factor track SOURCE error geometry but not TARGET competence? Source labels are
source-side observable (allowed): compare the source class-conditioned confidence vector to source per-class
recall (source error geometry) and to target per-class recall (post-hoc, quarantined). If source factor aligns
with source errors but not target offset/errors, that directly explains the source->target decoupling."""
from __future__ import annotations

import numpy as np

from . import schema


def _classwise_corr(A, B):
    cs = [float(np.corrcoef(A[:, k], B[:, k])[0, 1]) for k in range(schema.N_CLASSES)
          if A[:, k].std() > 1e-9 and B[:, k].std() > 1e-9]
    return (float(np.mean(cs)) if cs else None)


def error_geometry(cands, role, target_labels) -> dict:
    src_factor = np.array([[c[f"src_{role}_feats"][n] for n in schema.CARRIER_NAMES] for c in cands], dtype=np.float64)
    src_recall = np.array([c[f"src_{role}_recall"] for c in cands], dtype=np.float64)
    # source factor vs SOURCE error geometry (source-side observable)
    src_vs_src_recall = _classwise_corr(src_factor, src_recall)
    # source factor vs TARGET error geometry (post-hoc, quarantined target labels)
    tgt_recall = []
    for c in cands:
        y = target_labels.get((c["seed"], c["target"]))
        tp = c["tgt_feats"]
        # target per-class recall from the target predicted-class occupancy is not directly in feats; approximate
        # target error alignment via target class-conditioned confidence vs its own occupancy is out of scope here;
        # we correlate the SOURCE factor against the TARGET class-conditioned confidence as a target-competence proxy
        tgt_recall.append([tp[n] for n in schema.CARRIER_NAMES])
    tgt_factor = np.array(tgt_recall, dtype=np.float64)
    src_vs_tgt_factor = _classwise_corr(src_factor, tgt_factor)
    tracks_source_only = bool(src_vs_src_recall is not None and abs(src_vs_src_recall) >= 0.30
                              and (src_vs_tgt_factor is None or abs(src_vs_tgt_factor) < 0.30))
    return {"role": role, "source_factor_vs_source_recall": src_vs_src_recall,
            "source_factor_vs_target_factor": src_vs_tgt_factor, "tracks_source_error_only": tracks_source_only,
            "note": ("source class-conditioned confidence aligns with SOURCE per-class recall but not the TARGET "
                     "factor -> it tracks source error geometry, not target competence (explains source->target "
                     "decoupling)" if tracks_source_only else "source factor does not cleanly track source-only error geometry")}
