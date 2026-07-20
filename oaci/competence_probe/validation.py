"""C19 — validation wrapper. Runs the frozen probe on the ROBUST-CORE feature set (primary) and the ENDPOINT-
AUGMENTED set (secondary, only where endpoints are estimable), attaches the estimability gate, and evaluates
the pre-registered success criteria. Target labels enter only here, post-hoc, for LOTO meta-evaluation."""
from __future__ import annotations

from . import estimability_gate, labels, schema
from .probe import run_probe


def evaluate_regime(rows, *, n_perm=schema.N_PERM) -> dict:
    labels.assert_diagnostic_only(rows)
    robust = list(schema.ROBUST_CORE_FEATURES)
    endpoint = robust + list(schema.ENDPOINT_FEATURES)
    labels.assert_no_target_in_features(robust)
    labels.assert_no_target_in_features(endpoint)
    robust_res = run_probe(rows, robust, n_perm=n_perm)
    robust_res["gate"] = estimability_gate.gate_summary(rows, robust)
    # endpoint-augmented: score ONLY candidates whose endpoints are estimable (abstain otherwise)
    est_rows = [r for r in rows if estimability_gate.score_status(r, robust, schema.ENDPOINT_FEATURES) == "scored"]
    aug_res = run_probe(est_rows, endpoint, n_perm=n_perm)
    aug_res["gate"] = estimability_gate.gate_summary(rows, robust, schema.ENDPOINT_FEATURES)
    aug_res["n_endpoint_estimable"] = len(est_rows)
    return {"base_rate": labels.label_base_rate(rows), "robust_core": robust_res, "endpoint_augmented": aug_res}


def primary_success(robust_by_regime) -> dict:
    """Pre-registered primary criterion: robust-core passes (beats perm p<0.05 AND margin>=0.03) on EVERY
    success regime (S0 + cell-present S2/S3)."""
    per = {r: bool(robust_by_regime.get(r, {}).get("passes")) for r in schema.SUCCESS_REGIMES}
    return {"per_regime_pass": per, "primary_success": all(per.values())}
