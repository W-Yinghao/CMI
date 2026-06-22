"""Single end-to-end deployment API (notes/ACAR_FROZEN_v2.md A2). This IS the deployed scoring path; the
metamorphic guard is applied directly to it.

`route_batch(state, routers, z) -> (chosen_action, U_by_action, phi_by_action)`:
  - reads ONLY the serialized source state + frozen routers (per-action ĝ_a + shared conformal q + delta);
  - NO y argument;
  - len(z) < MIN_BATCH -> forced identity (label-blind), batch RETAINED upstream;
  - act argmin_a U_a over non-identity actions with U_a < -delta, else identity.

`route_fvec(routers, fvec_by_action)` applies the SAME decision rule to precomputed feature vectors (used by the
CV replay). route_batch is defined in terms of it, so the two cannot diverge.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .config import MIN_BATCH, NON_IDENTITY
from .scoring import score_actions


@dataclass
class Routers:
    regs: dict          # {action: ActionRegressor} for non-identity actions
    q: float            # shared (simultaneous) one-sided conformal quantile; may be +inf
    delta: float = 0.0
    actions: tuple = tuple(NON_IDENTITY)


def _U(routers: "Routers", fvec_by_action: dict) -> dict:
    return {a: float(routers.regs[a].predict(np.asarray(fvec_by_action[a])[None])[0] + routers.q)
            for a in routers.actions}


def _decide(U: dict, delta: float) -> str:
    eligible = {a: u for a, u in U.items() if u < -delta}
    return min(eligible, key=eligible.get) if eligible else "identity"


def route_fvec(routers: "Routers", fvec_by_action: dict):
    """Decision on precomputed (label-free) feature vectors. Returns (chosen_action, U_by_action)."""
    U = _U(routers, fvec_by_action)
    return _decide(U, routers.delta), U


def route_batch(state, routers: "Routers", z):
    """Returns (chosen_action, U_by_action, phi_by_action). Deterministic in (state, z, routers)."""
    z = np.asarray(z, float)
    if len(z) < MIN_BATCH:                                    # forced identity; do not run adapters on tiny batches
        return "identity", {a: float("inf") for a in routers.actions}, {a: None for a in routers.actions}
    s = score_actions(state, z, list(routers.actions))
    phi_by = {a: s[a]["fvec"] for a in routers.actions}
    chosen, U = route_fvec(routers, phi_by)
    return chosen, U, phi_by
