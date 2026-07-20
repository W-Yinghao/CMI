"""K1 decision rule (pre-registered): stop unless the held-out audit leakage reduction escapes the null.

``Δ = OACI - ERM`` (lower better). ``p_lower`` = lower-tail permutation p for ``Δ < 0``. ``p_lower < alpha``
=> ``leakage_reduction_detected`` (continue to K2); else ``stop_no_detectable_heldout_leakage_reduction``.
This is the manifest's ``decision_rule: stop_if_within_null_band``.
"""
from __future__ import annotations

K1_DETECTED = "leakage_reduction_detected"
K1_STOP = "stop_no_detectable_heldout_leakage_reduction"


def k1_decision(perm_result: dict, *, alpha: float | None = None) -> dict:
    a = float(perm_result["alpha"] if alpha is None else alpha)
    p_lower = float(perm_result["p_lower"])
    detected = p_lower < a
    return {
        "k1_status": K1_DETECTED if detected else K1_STOP,
        "continue_to_k2": bool(detected),
        "observed_delta": float(perm_result["observed_delta"]),
        "p_lower": p_lower,
        "p_two_sided": float(perm_result["p_two_sided"]),
        "alpha": a,
        "statistic": perm_result["statistic"],
        "n_permutations": int(perm_result["n_permutations"]),
        "permutation_plan_hash": perm_result["permutation_plan_hash"],
    }
