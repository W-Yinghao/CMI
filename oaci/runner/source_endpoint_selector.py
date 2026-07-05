"""C11b — source-train-only endpoint selector for SRC. Selects a checkpoint from the SRC trajectory using
ONLY source-side endpoint signal (never source_audit, never target, never leakage/K1 — K1 stays a pure
MEASUREMENT). Machine-checkable role gating: reading a forbidden field raises + is logged.

Rule: minimize source_guard worst-domain NLL, subject to
  R_src <= tau (= R_ERM + eps)            [risk feasibility]
  source_guard worst bAcc >= ERM - d_bacc
  source_guard worst NLL  <= ERM + d_nll
  source_guard worst ECE  <= ERM + d_ece
If no feasible candidate passes the guards, fall back to ERM.

Operates on a per-candidate endpoint TABLE (model_hash, is_erm, feasible, R_src, source_guard_worst_*); the
pilot computes that table by forwarding each candidate on the source_guard role (reusing the runner's exact
predict/evaluate path). This module is the pure, deterministic (hash-tie-broken) selection LOGIC.
"""
from __future__ import annotations

_FIELD_ROLE = {
    "model_hash": "meta", "is_erm": "meta", "feasible": "meta", "epoch": "meta", "lambda": "meta",
    "R_src": "source_risk",
    "source_guard_worst_bacc": "source_guard", "source_guard_worst_nll": "source_guard",
    "source_guard_worst_ece": "source_guard",
    # forbidden for a DEPLOYABLE source-only selector:
    "source_audit_worst_bacc": "source_audit", "source_audit_worst_nll": "source_audit",
    "audit_leakage_point": "leakage_measurement", "selection_leakage_point": "leakage_measurement",
    "k1_p_lower": "leakage_measurement", "k1_status": "leakage_measurement",
    "target_worst_bacc": "target", "target_worst_nll": "target", "target_worst_ece": "target",
}
_ALLOWED = frozenset({"source_risk", "source_guard"})
DEFAULT_MARGINS = {"bacc": 0.02, "nll": 0.05, "ece": 0.02}


class _AccessLog:
    def __init__(self):
        self.roles_read = set()
        self.forbidden = []

    def read(self, field):
        role = _FIELD_ROLE.get(field, "meta")
        if role == "meta" or role in _ALLOWED:
            self.roles_read.add(role)
            return True
        self.forbidden.append(field)
        return False


class _Gated:
    def __init__(self, row, log):
        self._row, self._log = row, log

    def __getitem__(self, field):
        if not self._log.read(field):
            raise PermissionError(f"source-endpoint selector read forbidden field {field!r} "
                                  f"(role {_FIELD_ROLE.get(field)}); K1/audit/target are not selection signals")
        return self._row[field]

    @property
    def hash(self):
        return self._row["model_hash"]


def select_source_endpoint(candidates, tau, *, margins=None, tol=0.0):
    """candidates: list of endpoint rows (one is is_erm=True). Returns the chosen model_hash + provenance +
    the access record (roles_actually_read, target_read, forbidden)."""
    m = {**DEFAULT_MARGINS, **(margins or {})}
    log = _AccessLog()
    gated = [_Gated(c, log) for c in candidates]
    erm = next((g for g in gated if g["is_erm"]), None)
    if erm is None:
        raise ValueError("candidate set must include the ERM checkpoint")

    def feasible(g):
        return (not g["is_erm"]) and g["feasible"] and g["R_src"] is not None and g["R_src"] <= tau + tol

    def guard(g):
        eb, en, ee = erm["source_guard_worst_bacc"], erm["source_guard_worst_nll"], erm["source_guard_worst_ece"]
        gb, gn, ge = g["source_guard_worst_bacc"], g["source_guard_worst_nll"], g["source_guard_worst_ece"]
        return (None not in (eb, en, ee, gb, gn, ge)
                and gb >= eb - m["bacc"] and gn <= en + m["nll"] and ge <= ee + m["ece"])

    cands = [g for g in gated if feasible(g)]
    passing = [g for g in cands if guard(g)]
    best = min(passing, key=lambda g: (g["source_guard_worst_nll"], g.hash)) if passing else None
    chosen = best.hash if best is not None else erm.hash
    return {"chosen_model_hash": chosen, "fallback_erm": best is None,
            "selection_reason": "source_endpoint_best" if best is not None else "erm_fallback",
            "n_feasible": len(cands), "n_guard_pass": len(passing),
            "access": {"allowed_roles": sorted(_ALLOWED), "roles_actually_read": sorted(log.roles_read),
                       "target_read": ("target" in log.roles_read) or bool(log.forbidden),
                       "read_source_audit": "source_audit" in log.roles_read,
                       "read_leakage": "leakage_measurement" in log.roles_read,
                       "forbidden_fields": list(log.forbidden)}}
