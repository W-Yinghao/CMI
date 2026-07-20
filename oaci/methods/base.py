"""Method activity status. A single ``SplitPlan.method_inactive`` cannot speak for four methods:
OACI is active iff the support graph has >=1 COMPARABLE class (an estimable cross-domain
comparison); the full-domain alignment baselines (global_lpc, uniform) only need >=2 source
domains; ERM is always active. The runner records per-method status after the missing-cell mask.
"""
from __future__ import annotations

METHODS = ("ERM", "OACI", "global_lpc", "uniform")


def method_activity(support_graph, n_source_domains: int) -> dict:
    """Per-method ``{name: {active, reason}}`` for the CURRENT (post-deletion) support state."""
    n_comparable = len(support_graph.comparable_classes)
    out = {"ERM": {"active": True, "reason": "baseline always runs"}}
    out["OACI"] = {
        "active": n_comparable >= 1,
        "reason": "ok" if n_comparable >= 1 else "no comparable class -> OACI is a byte-exact ERM no-op",
    }
    for m in ("global_lpc", "uniform"):
        out[m] = {
            "active": n_source_domains >= 2,
            "reason": "ok" if n_source_domains >= 2 else f"{n_source_domains} source domain(s) < 2 -> inactive",
        }
    return out
