"""C10b — counterfactual source-only + oracle selector replay over OACI's own risk-feasible trajectory.

Each selector picks ONE candidate per (seed, target, level) using a ROLE-GATED view of the candidate table
(candidate_replay.py output). Enforcement is machine-checkable: a selector may only read fields of its
allowed roles; touching a forbidden role (any selector reading `target`; S1–S4 reading `source_audit`) raises
and is logged. Target metrics are EVALUATION-ONLY — read AFTER selection, never by a selector.

Selectors:
  S0_current                     the artifact-selected OACI checkpoint (no guard)
  S1_leakage_worst_source_bacc   min selection leakage s.t. source_guard worst bAcc >= ERM - δ_bacc
  S2_leakage_worst_source_nll    min selection leakage s.t. source_guard worst NLL  <= ERM + δ_nll
  S3_leakage_calibration         min selection leakage s.t. source_guard NLL & ECE noninferior to ERM
  S4_conservative_source_only    OACI only if leakage beats ERM by margin AND all source guards pass, else ERM
  S5_source_audit_oracle         pick by source_audit worst-domain proxy (NON-DEPLOYABLE oracle diagnostic)
"""
from __future__ import annotations

_FIELD_ROLE = {}
for _r, _fs in {
    "meta": ("origin", "model_hash", "epoch", "lambda", "R_src", "feasible", "is_erm", "balanced_err",
             "train_surrogate"),
    "source_train": ("selection_leakage_point",),
    "source_guard": ("source_guard_worst_bacc", "source_guard_worst_nll", "source_guard_worst_ece"),
    "source_audit": ("source_audit_worst_bacc", "source_audit_worst_nll", "source_audit_worst_ece",
                     "audit_leakage_point"),
    "target": ("target_worst_bacc", "target_worst_nll", "target_worst_ece"),
    "identity": ("target_pred_hash", "source_guard_pred_hash", "source_audit_pred_hash"),
}.items():
    for _f in _fs:
        _FIELD_ROLE[_f] = _r

DEFAULT_MARGINS = {"bacc": 0.02, "nll": 0.05, "ece": 0.02, "leakage": 0.05}


class AccessLog:
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.roles_read = set()
        self.forbidden = []

    def read(self, field):
        role = _FIELD_ROLE.get(field, "meta")
        if role == "meta" or role in self.allowed:
            self.roles_read.add(role)
            return True
        self.forbidden.append(field)
        return False


class Gated:
    """A candidate row exposing only allowed-role fields; forbidden reads raise (and are logged)."""
    def __init__(self, row, log):
        self._row, self._log = row, log

    def __getitem__(self, field):
        if not self._log.read(field):
            raise PermissionError(f"selector read forbidden field {field!r} (role {_FIELD_ROLE.get(field)})")
        return self._row[field]

    @property
    def hash(self):
        return self._row["model_hash"]


def _gated(rows, allowed):
    log = AccessLog(allowed)
    return [Gated(r, log) for r in rows], log


def _min_by(cands, key):
    """Argmin with a DETERMINISTIC tie-break on model_hash (so the choice is order-invariant)."""
    best, bk = None, None
    for c in cands:
        v = key(c)
        if v is None:
            continue
        k = (v, c.hash)
        if bk is None or k < bk:
            best, bk = c, k
    return best


def _erm(gated):
    for g in gated:
        if g["is_erm"]:
            return g
    return None


# ---- selectors: (rows, ctx) -> (chosen_model_hash, AccessLog) ----
def s0_current(rows, ctx):
    log = AccessLog({"source_train"})
    log.roles_read.add("meta")
    return ctx["selected_oaci_hash"], log


def _guarded_min_leakage(rows, allowed, guard):
    gated, log = _gated(rows, allowed)
    erm = _erm(gated)
    pool = [erm] + [g for g in gated if not g["is_erm"] and g["feasible"] and guard(g, erm)]
    chosen = _min_by([g for g in pool if g is not None], lambda g: g["selection_leakage_point"])
    return (chosen or erm).hash, log


def s1_leakage_worst_source_bacc(rows, ctx):
    m = ctx["margins"]
    return _guarded_min_leakage(
        rows, {"source_train", "source_guard"},
        lambda g, e: g["source_guard_worst_bacc"] is not None and e["source_guard_worst_bacc"] is not None
        and g["source_guard_worst_bacc"] >= e["source_guard_worst_bacc"] - m["bacc"])


def s2_leakage_worst_source_nll(rows, ctx):
    m = ctx["margins"]
    return _guarded_min_leakage(
        rows, {"source_train", "source_guard"},
        lambda g, e: g["source_guard_worst_nll"] is not None and e["source_guard_worst_nll"] is not None
        and g["source_guard_worst_nll"] <= e["source_guard_worst_nll"] + m["nll"])


def s3_leakage_calibration(rows, ctx):
    m = ctx["margins"]
    return _guarded_min_leakage(
        rows, {"source_train", "source_guard"},
        lambda g, e: (g["source_guard_worst_nll"] is not None and e["source_guard_worst_nll"] is not None
                      and g["source_guard_worst_nll"] <= e["source_guard_worst_nll"] + m["nll"]
                      and g["source_guard_worst_ece"] is not None and e["source_guard_worst_ece"] is not None
                      and g["source_guard_worst_ece"] <= e["source_guard_worst_ece"] + m["ece"]))


def s4_conservative_source_only(rows, ctx):
    m = ctx["margins"]
    gated, log = _gated(rows, {"source_train", "source_guard"})
    erm = _erm(gated)

    def guards_pass(g):
        return (g["source_guard_worst_bacc"] is not None and erm["source_guard_worst_bacc"] is not None
                and g["source_guard_worst_bacc"] >= erm["source_guard_worst_bacc"] - m["bacc"]
                and g["source_guard_worst_nll"] <= erm["source_guard_worst_nll"] + m["nll"]
                and g["source_guard_worst_ece"] <= erm["source_guard_worst_ece"] + m["ece"])
    pool = [g for g in gated if not g["is_erm"] and g["feasible"] and guards_pass(g)]
    best = _min_by(pool, lambda g: g["selection_leakage_point"])
    if best is None or best["selection_leakage_point"] is None or erm["selection_leakage_point"] is None:
        return erm.hash, log
    improves = best["selection_leakage_point"] <= erm["selection_leakage_point"] - m["leakage"]
    return (best.hash if improves else erm.hash), log


def s5_source_audit_oracle(rows, ctx):
    """NON-DEPLOYABLE oracle: pick by held-out source_audit worst-domain bAcc (K2 proxy), tie-break lower NLL
    then model_hash (deterministic / order-invariant)."""
    gated, log = _gated(rows, {"source_audit"})
    elig = [g for g in gated if (g["is_erm"] or g["feasible"]) and g["source_audit_worst_bacc"] is not None]
    elig.sort(key=lambda g: (-g["source_audit_worst_bacc"],
                             g["source_audit_worst_nll"] if g["source_audit_worst_nll"] is not None else 1e9,
                             g.hash))
    return (elig[0] if elig else _erm(gated)).hash, log


SELECTORS = {"S0_current": s0_current, "S1_leakage_worst_source_bacc": s1_leakage_worst_source_bacc,
             "S2_leakage_worst_source_nll": s2_leakage_worst_source_nll, "S3_leakage_calibration": s3_leakage_calibration,
             "S4_conservative_source_only": s4_conservative_source_only, "S5_source_audit_oracle": s5_source_audit_oracle}
_ORACLE = {"S5_source_audit_oracle"}


def run_selectors_on_level(rows, *, selected_oaci_hash, margins=None):
    """Run every selector on one (seed,target,level) candidate list. Returns per-selector: chosen model_hash,
    the chosen row (for target evaluation), and the access record (roles_read, target_read, forbidden)."""
    margins = {**DEFAULT_MARGINS, **(margins or {})}
    ctx = {"selected_oaci_hash": selected_oaci_hash, "margins": margins}
    by_hash = {r["model_hash"]: r for r in rows}
    out = {}
    for name, fn in SELECTORS.items():
        chosen, log = fn(rows, ctx)
        row = by_hash.get(chosen)
        out[name] = {"chosen_model_hash": chosen, "is_erm": bool(row and row.get("is_erm")),
                     "chosen_epoch": (row.get("epoch") if row else None),
                     "target_worst_bacc": (row.get("target_worst_bacc") if row else None),   # EVAL-ONLY
                     "target_worst_nll": (row.get("target_worst_nll") if row else None),
                     "access": {"allowed_roles": sorted(log.allowed),
                                "roles_actually_read": sorted(log.roles_read),
                                "target_read": ("target" in log.roles_read) or bool(log.forbidden),
                                "forbidden_fields": list(log.forbidden),
                                "is_oracle": name in _ORACLE}}
    return out
