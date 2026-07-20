"""Formal per-method activity over the CURRENT (post-deletion) support state.

    ERM         — always available after a successful Stage-1
    OACI        — active iff >=1 comparable class
    global_lpc  — active iff the level-0 universe has >=2 domains AND source_train has >=2 observed
    uniform     —   "      "
"""
from __future__ import annotations

from dataclasses import dataclass

METHODS = ("ERM", "OACI", "global_lpc", "uniform")


@dataclass(frozen=True)
class MethodStatus:
    active: bool
    reason: str | None = None


def method_status(name: str, support_graph, level0_universe_size: int,
                  n_observed_source_domains: int) -> MethodStatus:
    if name == "ERM":
        return MethodStatus(True, None)
    if name == "OACI":
        ok = len(support_graph.comparable_classes) >= 1
        return MethodStatus(ok, None if ok else "no comparable class -> byte-exact ERM no-op")
    if name in ("global_lpc", "uniform"):
        ok = level0_universe_size >= 2 and n_observed_source_domains >= 2
        if ok:
            return MethodStatus(True, None)
        why = (f"level-0 universe {level0_universe_size} < 2" if level0_universe_size < 2
               else f"observed source domains {n_observed_source_domains} < 2")
        return MethodStatus(False, why)
    raise ValueError(f"unknown method {name!r}")


def all_method_status(support_graph, level0_universe_size, n_observed_source_domains) -> dict:
    return {m: method_status(m, support_graph, level0_universe_size, n_observed_source_domains)
            for m in METHODS}
