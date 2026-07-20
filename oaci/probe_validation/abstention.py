"""C20 — feature availability / abstention per held-out regime. Reuses the C19 estimability gate. Under cell
DELETION the fragile accuracy endpoints go non-estimable; the robust-core features may also abstain where
leakage becomes non-estimable. C20 reports this as a first-class output so a low held-out AUC can be attributed
to feature availability vs source-target relationship (C18 endpoint-estimability framing)."""
from __future__ import annotations

from ..competence_probe import estimability_gate
from ..competence_probe import schema as c19


def availability(val_rows) -> dict:
    robust = list(c19.ROBUST_CORE_FEATURES)
    # robust-ONLY availability (what the PRIMARY probe needs) -- must NOT include the endpoint check, else it
    # reads 0 under deletion because the fragile endpoints are NaN while the robust core is fully finite.
    g_robust = estimability_gate.gate_summary(val_rows, robust)
    # endpoint availability (robust AND endpoint) -- for the SECONDARY endpoint-augmented probe only.
    g_endpoint = estimability_gate.gate_summary(val_rows, robust, list(c19.ENDPOINT_FEATURES))
    return {"n_candidates": g_robust["n_candidates"],
            "robust_core_scored_rate": g_robust["scored_rate"],
            "robust_core_insufficient_rate": g_robust["insufficient_finite_rate"],
            "endpoint_available_rate": g_endpoint["scored_rate"],
            "endpoint_nonestimable_rate": g_endpoint["endpoint_nonestimable_rate"],
            "counts": g_robust["counts"]}
