"""C23 — TARGET-ANONYMOUS source-only gauge feature registry. A gauge vector for a target is built from MOMENTS
(mean, std) of that target's candidates' SOURCE observables (the 16 robust-core features) + finite-availability
summaries. It deliberately EXCLUDES: the probe score itself (would be circular with the offset), target labels,
target ID, seed, regime, source subject / domain IDs, and any target-wise centering/rank transform. The gauge
is source-only + target-anonymous; whether it nonetheless encodes target identity is tested separately (the
identity-leakage HARD GATE)."""
from __future__ import annotations

import numpy as np

from ..competence_probe import schema as c19
from . import schema
from .artifact_loader import _finite, by_target


def gauge_feature_names() -> list:
    names = []
    for s in c19.ROBUST_CORE_FEATURES:
        for m in schema.GAUGE_MOMENTS:
            names.append(f"{s}__{m}")
    names += list(schema.GAUGE_EXTRA)
    return names


def assert_target_anonymous(feature_names) -> None:
    bad = [f for f in feature_names for tok in schema.FORBIDDEN_GAUGE_INPUTS if tok in f.lower()]
    if bad:
        raise ValueError(f"gauge feature names contain forbidden target-identity/answer tokens: {bad}")


def _vector(cands) -> dict:
    v = {}
    for s in c19.ROBUST_CORE_FEATURES:
        col = "feat__" + s
        vals = [c[col] for c in cands if _finite(c.get(col))]
        v[f"{s}__mean"] = float(np.mean(vals)) if vals else 0.0
        v[f"{s}__std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    n = len(cands)
    finite = sum(1 for c in cands if all(_finite(c.get("feat__" + s)) for s in c19.ROBUST_CORE_FEATURES))
    v["finite_feature_rate"] = finite / n if n else 0.0
    v["n_candidates"] = float(n)
    return v


def build_gauge_table(rows, mode) -> dict:
    """{target: {gauge_features..., offset, R_src_mean}}. gauge_features are TARGET-ANONYMOUS source summaries."""
    import statistics as st
    assert_target_anonymous(gauge_feature_names())
    out = {}
    for t, cands in by_target(rows, mode).items():
        gv = _vector(cands)
        out[t] = {"gauge": gv, "offset": st.mean([c["score"] for c in cands]),
                  "R_src_mean": st.mean([c["R_src"] for c in cands if c.get("R_src") is not None]),
                  "n": len(cands)}
    return out
