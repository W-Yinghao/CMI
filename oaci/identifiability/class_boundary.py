"""C17 — class-boundary rotation identifiability (SELECTED-checkpoint level). C16-B found OACI rotates target
class boundaries (some MI classes gain recall, others lose). This asks whether that rotation is MIRRORED on
the held-out SOURCE audit split — i.e. is the class-boundary change source-identifiable? Uses the committed
C8 source_audit.npz and target_audit.npz for the SELECTED ERM/OACI checkpoints (per-candidate class recall is
not committed, so this is a selected-checkpoint diagnostic; stated as a scope limit)."""
from __future__ import annotations

import numpy as np

from ..mechanism.harm_decomposition import _artifact, _metrics, _npz


def _npz_role(artifact, level, method, role):
    import os
    import numpy as _np
    z = _np.load(os.path.join(artifact, f"levels/level-{level:03d}", "methods", method, f"{role}.npz"),
                 allow_pickle=True)
    return _np.asarray(z["logits"], dtype=_np.float64), _np.asarray(z["y"]).astype(int)


def class_boundary_identifiability(loso_root, *, seeds=(0, 1, 2), targets=range(1, 10), levels=(0, 1)) -> dict:
    src_deltas, tgt_deltas = [], []                       # per (class, fold-level) OACI-ERM recall deltas
    per_class = {c: {"src": [], "tgt": []} for c in range(4)}
    for s in seeds:
        for t in targets:
            a = _artifact(loso_root, s, t)
            if a is None:
                continue
            for L in levels:
                try:
                    se, ye = _npz_role(a, L, "ERM", "source_audit"); so, yo = _npz_role(a, L, "OACI", "source_audit")
                    te, yte = _npz_role(a, L, "ERM", "target_audit"); to, yto = _npz_role(a, L, "OACI", "target_audit")
                except Exception:
                    continue
                me, mo = _metrics(se, ye), _metrics(so, yo)
                mte, mto = _metrics(te, yte), _metrics(to, yto)
                for c in range(4):
                    sd = (mo["per_class_recall"].get(c, None) - me["per_class_recall"].get(c, None)) \
                        if (c in mo["per_class_recall"] and c in me["per_class_recall"]) else None
                    td = (mto["per_class_recall"].get(c, None) - mte["per_class_recall"].get(c, None)) \
                        if (c in mto["per_class_recall"] and c in mte["per_class_recall"]) else None
                    if sd is not None and td is not None:
                        src_deltas.append(sd); tgt_deltas.append(td)
                        per_class[c]["src"].append(sd); per_class[c]["tgt"].append(td)
    # correlation between source and target per-class recall deltas
    def pear(x, y):
        if len(x) < 3:
            return None
        x, y = np.array(x), np.array(y)
        if x.std() < 1e-9 or y.std() < 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])
    r = pear(src_deltas, tgt_deltas)
    pc = {str(c): {"n": len(v["src"]), "mean_src_recall_delta": (float(np.mean(v["src"])) if v["src"] else None),
                   "mean_tgt_recall_delta": (float(np.mean(v["tgt"])) if v["tgt"] else None),
                   "src_tgt_corr": pear(v["src"], v["tgt"])} for c, v in per_class.items()}
    identifiable = r is not None and r > 0.3
    return {"n_class_fold_points": len(src_deltas), "source_target_recall_delta_corr": r,
            "class_boundary_source_identifiable": bool(identifiable), "per_class": pc,
            "scope": "SELECTED checkpoints only (per-candidate class recall is not committed)",
            "note": ("If source and target per-class recall deltas correlate, the class-boundary rotation is "
                     "source-identifiable; if not, the boundary change that helps target is source-invisible.")}
