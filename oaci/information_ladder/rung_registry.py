"""C24 — the information ladder as an explicit, honest table. Each rung carries its information class,
deployability, whether it needs target inputs/labels, and (for R3/R4) a machine-readable status + reason. The
registry NEVER promotes a blocked rung to 'computed' via a proxy."""
from __future__ import annotations

from . import schema


def ladder(target_unlabeled_availability) -> list:
    """Return the ladder rung metadata, stamping R3/R4 with the live re-inference availability status."""
    r3r4_status = target_unlabeled_availability["r3r4_status"]
    out = []
    for r in schema.RUNGS:
        row = dict(r)
        if r["feasible_readonly"]:
            row["status"] = schema.STATUS_OK
        else:
            row["status"] = r3r4_status
        out.append(row)
    return out


def readonly_rungs() -> tuple:
    return tuple(r["rung"] for r in schema.RUNGS if r["feasible_readonly"])


def reinference_rungs() -> tuple:
    return tuple(r["rung"] for r in schema.RUNGS if not r["feasible_readonly"])
