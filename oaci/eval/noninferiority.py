"""Noninferiority / superiority decisions on a paired Δ = OACI − ERM.

higher-is-better metric (bAcc):   NI  ⟺  LCL_{1−α}(Δ) > −δ ;   superiority ⟺ LCL > 0.
lower-is-better metric (risk,NLL): NI ⟺  UCL_{1−α}(Δ) < +δ ;   superiority ⟺ UCL < 0.

``epsilon`` (the training risk slack) may only be reused for the SOURCE-RISK decision, and only
when the audit risk metric equals the training constraint metric. It must NOT be used as a
balanced-accuracy margin; ``delta_bacc`` is supplied explicitly by the experiment config (no
library default that could be mistaken for a paper threshold).
"""
from __future__ import annotations


def noninferiority(ci: dict, margin: float, higher_is_better: bool, limit: str = "basic"):
    if not ci.get("estimable", True):
        return None
    if higher_is_better:
        return bool(ci[f"{limit}_lcl"] > -margin)
    return bool(ci[f"{limit}_ucl"] < margin)


def superiority(ci: dict, higher_is_better: bool, limit: str = "basic"):
    if not ci.get("estimable", True):
        return None
    return bool(ci[f"{limit}_lcl"] > 0) if higher_is_better else bool(ci[f"{limit}_ucl"] < 0)


def source_risk_noninferiority(ci: dict, epsilon: float, audit_metric: str, train_metric: str,
                               limit: str = "basic"):
    """UCL(R_OACI − R_ERM) ≤ ε — VALID ONLY when the audit metric matches the training metric."""
    if audit_metric != train_metric:
        raise ValueError(
            f"epsilon reuse requires the audit risk metric ({audit_metric!r}) to equal the "
            f"training constraint metric ({train_metric!r}); refusing to reuse ε across metrics."
        )
    if not ci.get("estimable", True):
        return None
    return bool(ci[f"{limit}_ucl"] <= epsilon)
