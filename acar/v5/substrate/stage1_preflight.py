"""ACAR V5 Stage-1A substrate-build PREFLIGHT (dry-run / plan-only; FAIL-CLOSED). Pure/stdlib; reads NO real data, trains NOTHING.

It validates the deterministic build plan (schema + ref types + seed roles + external-site exclusion) and PROVES the real-data
door is shut: any plan entry that declares a `source_path` requires an explicit, tag-bound Stage-1B authorization (never issued in
Stage-1A), and any path targeting real DEV / v4 artifacts / caches / external sites is forbidden outright. With the default
plan-only spec (all source_path=None) it returns STAGE1A_PREFLIGHT_OK having touched no data.
"""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import build_manifest_schema as SCH

PROTOCOL_TAG = "acar-v5-protocol"
REQUIRED_STAGE1B_STATEMENT = (
    "Authorize ACAR-V5 Stage-1B substrate build (real DEV read) exactly under ACAR_V5_STAGE1_PREFLIGHT.md"
)


class Stage1BuildNotAuthorizedError(RuntimeError):
    """Raised when a plan declares a real-data source_path without a valid, tag-bound Stage-1B authorization."""


class Stage1ForbiddenTargetError(RuntimeError):
    """Raised when a source_path targets real DEV / v4-artifact / cache / external-site data (forbidden regardless of auth)."""


def _require_stage1b(auth):
    if not isinstance(auth, dict):
        raise Stage1BuildNotAuthorizedError("real-data source_path present but no Stage-1B authorization supplied")
    if auth.get("protocol_tag") != PROTOCOL_TAG or auth.get("statement") != REQUIRED_STAGE1B_STATEMENT:
        raise Stage1BuildNotAuthorizedError("Stage-1B authorization not tag-bound / statement mismatch")
    return True


def run_preflight(plan=None, *, stage1b_authorization=None):
    """Run the Stage-1A preflight. Returns a report dict. Raises fail-closed on any real-data/forbidden target."""
    plan = plan if plan is not None else PLAN.build_substrate_plan()
    SCH.validate_build_manifest(plan)                          # schema + ref types + seed roles + no external-site token

    entries = list(plan.get("fold_contained_refs", [])) + list(plan.get("final_external_refs", []))
    real_data_entries = 0
    for e in entries:
        sp = e.get("source_path")
        if SCH.path_is_forbidden(sp):                         # forbidden regardless of authorization
            raise Stage1ForbiddenTargetError(f"{e.get('ref')}: forbidden source_path {sp!r} (real DEV/artifact/cache/external)")
        if sp:                                                # any declared real path needs Stage-1B auth (never in Stage-1A)
            _require_stage1b(stage1b_authorization)
            real_data_entries += 1

    return {
        "status": "STAGE1A_PREFLIGHT_OK",
        "protocol_tag": plan["protocol_tag"],
        "counts": plan["counts"],
        "selection_seed": plan["selection_seed"],
        "s1_seeds": plan["s1_seeds"],
        "n_fold_refs": len(plan["fold_contained_refs"]),
        "n_final_external_refs": len(plan["final_external_refs"]),
        "real_data_entries": real_data_entries,               # 0 in Stage-1A (plan-only)
        "note": "plan-only dry-run; no DEV/real/external read, no substrate training, no embedding dump, no selection",
    }


def main(argv=None):
    import json
    rep = run_preflight()                                     # default plan-only; no authorization → must stay 0 real reads
    assert rep["real_data_entries"] == 0, "Stage-1A must read no real data"
    print(json.dumps(rep, indent=2, sort_keys=True))
    return rep


if __name__ == "__main__":
    main()
