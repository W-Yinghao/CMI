"""C20 — SECONDARY endpoint-augmented cross-regime check. Adds the fragile accuracy endpoints to the frozen
robust-core, scoring only candidates whose endpoints are estimable. This is SECONDARY: it can never rescue a
failed robust-core primary; it only characterizes whether accuracy endpoints add anything when available
(C19 found they do not). Under cell DELETION many endpoints are non-estimable, so this is expected to abstain."""
from __future__ import annotations

from ..competence_probe import estimability_gate
from ..competence_probe import schema as c19
from . import cross_regime_validation as crv
from . import schema


def endpoint_augmented_cross_regime(dev_by_regime, val_rows, *, n_perm=schema.N_PERM) -> dict:
    cols = list(c19.ROBUST_CORE_FEATURES) + list(c19.ENDPOINT_FEATURES)
    # score ONLY candidates whose endpoints are estimable (abstain otherwise); dev + val both filtered
    robust = list(c19.ROBUST_CORE_FEATURES); endp = list(c19.ENDPOINT_FEATURES)

    def est(rows):
        return [r for r in rows if estimability_gate.score_status(r, robust, endp) == "scored"]

    dev_est = {r: est(dev_by_regime[r]) for r in schema.DEVELOPMENT_REGIMES}
    val_est = est(val_rows)
    res = crv.cross_regime_loto(dev_est, val_est, cols, n_perm=n_perm)
    res["n_endpoint_estimable_val"] = len(val_est)
    res["is_secondary"] = True
    res["note"] = "SECONDARY: cannot rescue a failed robust-core primary."
    return res
