"""C27-D — label alignment under interventions (LABEL-DIAGNOSTIC-ONLY; quarantined labels joined post-hoc). C26
P7 showed predicted-class mix aligns with target per-class recall (error geometry). C27 asks: do the logit
interventions that DESTROY the offset recovery ALSO destroy this error-geometry alignment? If the same logit
structure carries both, offset recovery and error-geometry alignment are coupled (L6)."""
from __future__ import annotations

import numpy as np

from . import factor_registry, logit_counterfactuals, schema


def _predmix_recall_corr(logit_cands, labels, transform=None):
    predmix, recall = [], []
    for c in logit_cands:
        L = c["L"] if transform is None else transform(c["L"])
        pred = factor_registry._softmax(L).argmax(1)
        y = labels.get((c["seed"], c["target"]))
        if y is None or len(y) != len(pred):
            continue
        predmix.append([float((pred == k).mean()) for k in range(schema.N_CLASSES)])
        recall.append([float((pred[y == k] == k).mean()) if np.any(y == k) else 0.0 for k in range(schema.N_CLASSES)])
    A, B = np.array(predmix), np.array(recall)
    cs = [float(np.corrcoef(A[:, k], B[:, k])[0, 1]) for k in range(schema.N_CLASSES)
          if A[:, k].std() > 1e-9 and B[:, k].std() > 1e-9]
    return (float(np.mean(cs)) if cs else None)


def label_alignment(logit_cands, labels, destroyers) -> dict:
    transforms = {"raw": None, **logit_counterfactuals._TRANSFORMS}
    per = {}
    for name, tf in transforms.items():
        per[name] = _predmix_recall_corr(logit_cands, labels, tf)
    base = per.get("raw")
    # a transform "destroys alignment" if the |corr| drops >= 50% vs raw
    align_destroyers = [n for n, v in per.items() if n != "raw" and base and v is not None and abs(v) < 0.5 * abs(base)]
    # only the per-SAMPLE transforms are alignment-testable (gauge-level shuffles have no per-sample analogue)
    testable = [d for d in destroyers if d in transforms and d != "raw"]
    coupled = sorted(set(testable) & set(align_destroyers))            # destroy BOTH offset + alignment
    decouplers = sorted(set(testable) - set(align_destroyers))         # destroy offset but PRESERVE alignment
    return {"predmix_recall_corr_by_intervention": per, "raw_alignment": base,
            "alignment_destroyers": align_destroyers, "offset_and_alignment_coupled": bool(coupled),
            "coupled_interventions": coupled, "decoupling_interventions": decouplers,
            "coupling_partial": bool(coupled and decouplers),
            "note": (("%s destroy(s) BOTH offset recovery and error-geometry alignment (coupled)" % ", ".join(coupled)
                      + ("; BUT %s destroy(s) offset recovery while PRESERVING alignment -> coupling is PARTIAL, "
                         "not clean (occupancy magnitude carries offset separably from error geometry)"
                         % ", ".join(decouplers) if decouplers else "")) if coupled else
                     "no per-sample intervention destroys both offset recovery and error alignment -> not cleanly coupled")}
