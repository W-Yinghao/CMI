"""K2 decision (pre-registered): reproducible OACI gain on the worst-domain endpoints across seeds/levels.

Endpoints (higher/lower better is a property of the metric, not a paper choice): ``worst_domain_bacc`` ↑,
``worst_domain_nll`` ↓. Per unit ``Δ = OACI - ERM``. With ``level_policy='both_levels'`` a gain must hold at
EVERY (seed, level) unit; an endpoint reproduces iff every unit is a gain beyond its margin. States:
``reproducible_gain`` (≥1 endpoint reproduces), ``stop_no_reproducible_gain`` (neither), or an abstain when
there are too few seeds or a required endpoint is missing. All thresholds come from the manifest — this
function REQUIRES them (no code-level defaults)."""
from __future__ import annotations

K2_REPRODUCIBLE = "reproducible_gain"
K2_STOP = "stop_no_reproducible_gain"
K2_ABSTAIN_SEEDS = "abstain_insufficient_seeds"
K2_ABSTAIN_ENDPOINT = "abstain_missing_endpoint"

# higher-is-better lives in the metric definition, not the manifest.
_HIGHER_IS_BETTER = {"worst_domain_bacc": True, "worst_domain_nll": False}


def _is_gain(endpoint: str, delta: float, margin: float) -> bool:
    if _HIGHER_IS_BETTER[endpoint]:
        return float(delta) > float(margin)                     # ↑ metric: OACI-ERM must exceed +margin
    return float(delta) < -float(margin)                        # ↓ metric: OACI-ERM must be below -margin


def k2_decision(units, *, endpoints, min_seeds, level_policy, margins) -> dict:
    """``units`` = list of ``{seed, level, deltas: {endpoint: Δ|None}}`` (Δ = OACI - ERM). ``margins`` =
    ``{endpoint: margin}``. Order-invariant (decision depends only on the unit SET)."""
    for e in endpoints:
        if e not in _HIGHER_IS_BETTER:
            raise ValueError(f"unknown K2 endpoint {e!r}; direction undefined")
        if e not in margins:
            raise ValueError(f"no margin supplied for endpoint {e!r} (manifest must provide it)")
    if str(level_policy) != "both_levels":
        raise ValueError(f"unsupported level_policy {level_policy!r} (only 'both_levels' is wired)")

    seeds = sorted({int(u["seed"]) for u in units})
    levels = sorted({int(u["level"]) for u in units})
    base = {"n_seeds": len(seeds), "seeds": seeds, "levels": levels, "min_seeds": int(min_seeds),
            "level_policy": str(level_policy), "endpoints": list(endpoints)}

    if len(seeds) < int(min_seeds):
        return {"k2_status": K2_ABSTAIN_SEEDS, "continue": False,
                "reason": f"{len(seeds)} seed(s) < min_seeds {int(min_seeds)}", **base}

    per_endpoint = {}
    any_evaluable = False
    reproduced = []
    for e in endpoints:
        vals = [u["deltas"].get(e) for u in units]
        evaluable = len(vals) > 0 and all(v is not None for v in vals)
        rep = bool(evaluable and all(_is_gain(e, v, margins[e]) for v in vals))
        per_endpoint[e] = {"evaluable": evaluable, "reproducible": rep, "n_units": len(vals),
                           "n_gain": sum(1 for v in vals if v is not None and _is_gain(e, v, margins[e])),
                           "margin": float(margins[e])}
        any_evaluable = any_evaluable or evaluable
        if rep:
            reproduced.append(e)

    if not any_evaluable:
        return {"k2_status": K2_ABSTAIN_ENDPOINT, "continue": False,
                "reason": "no required endpoint has complete values across all units",
                "per_endpoint": per_endpoint, **base}
    status = K2_REPRODUCIBLE if reproduced else K2_STOP
    return {"k2_status": status, "continue": bool(reproduced), "reproduced_endpoints": reproduced,
            "per_endpoint": per_endpoint, **base}
