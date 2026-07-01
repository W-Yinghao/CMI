"""ACAR V5 Stage-1A substrate-build PREFLIGHT (dry-run / plan-only; FAIL-CLOSED). Pure/stdlib; reads NO real data, trains NOTHING.

It validates the deterministic build plan (schema + ref types + seed roles + external-site exclusion) and PROVES the real-data
door is shut: any plan entry that declares a `source_path` requires an explicit, tag-bound Stage-1B authorization (never issued in
Stage-1A), and any path targeting real DEV / v4 artifacts / caches / external sites is forbidden outright. With the default
plan-only spec (all source_path=None) it returns STAGE1A_PREFLIGHT_OK having touched no data.
"""
from __future__ import annotations
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import build_manifest_schema as SCH
from acar.v5.substrate import stage1b_manifest as MAN
from acar.v5.substrate import stage1b_authorization as SA
# Stage1BuildNotAuthorizedError is defined centrally in stage1b_authorization; re-exported here so PF.<name> keeps working.
from acar.v5.substrate.stage1b_authorization import Stage1BuildNotAuthorizedError  # noqa: F401

# back-compat symbols (the structured contract in stage1b_authorization is now the real gate)
PROTOCOL_TAG = SA.PROTOCOL_TAG
REQUIRED_STAGE1B_STATEMENT = SA.REQUIRED_STAGE1B_STATEMENT


class Stage1ForbiddenTargetError(RuntimeError):
    """Raised when a source_path targets real DEV / v4-artifact / cache / external-site data (forbidden regardless of auth)."""


def run_preflight(plan=None, *, stage1b_authorization=None):
    """Run the Stage-1A preflight (dry-run). Returns a report dict; raises fail-closed on any real-data/forbidden target.
    Final-external refs must be schema-only ALWAYS (even with a valid auth). Any fold-ref source_path requires the STRUCTURED
    Stage-1B authorization contract (never issued in Stage-1A)."""
    plan = plan if plan is not None else PLAN.build_substrate_plan()
    SCH.validate_build_manifest(plan)                          # schema + ref types + seed roles + no external-site token
    MAN.assert_final_external_schema_only(plan.get("final_external_refs", []))   # #1: schema-only ALWAYS (auth-independent)

    real_data_entries = 0
    for e in plan.get("fold_contained_refs", []):
        sp = e.get("source_path")
        if SCH.path_is_forbidden(sp):                         # dry-run blanket forbidden (incl /projects/): forbidden regardless of auth
            raise Stage1ForbiddenTargetError(f"{e.get('ref')}: forbidden source_path {sp!r} (real DEV/artifact/cache/external)")
        if sp:                                                # any declared real path needs the STRUCTURED Stage-1B contract
            SA.validate_stage1b_authorization(stage1b_authorization)   # raises Stage1BuildNotAuthorizedError if missing/malformed
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
