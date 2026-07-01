"""Guard (Stage-1A): the Stage-5 final all-source external-execution substrate ref is a DISTINCT type — it cannot be treated as a
fold ref and the fold registry refuses it (so a final external substrate can never be slipped into the fold registry). Synthetic."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import build_manifest_schema as SCH
from acar.v5.substrate.registry import SubstrateRegistry
from acar.v5.tests._util import expect_raises, ok


def test_final_ref_shape_distinct():
    for r in PLAN.final_external_refs():
        assert SCH.is_final_external_ref(r["ref"]) and not SCH.is_fold_ref(r["ref"])
        assert r["role"] == PLAN.FINAL_EXTERNAL_ROLE and r["source_path"] is None
    for r in PLAN.fold_refs():
        assert SCH.is_fold_ref(r["ref"]) and not SCH.is_final_external_ref(r["ref"])
    ok("final-external refs (external_exec/<disease>/all_source_dev) are a DISTINCT type from fold refs")


def test_fold_registry_refuses_final_ref():
    # the fold registry is keyed by (disease, fold, seed); a final-external substrate has no fold/seed → cannot be registered
    reg = SubstrateRegistry()
    # there is no API to register a final ref into the fold registry; substrate_ref() only accepts fold coordinates
    from acar.v5.substrate.registry import substrate_ref
    expect_raises(ValueError, lambda: substrate_ref("PD", 0, 999), "final/foreign seed rejected by fold key")
    # and the schema validator rejects a plan that lists a final ref among fold_contained_refs
    bad = PLAN.build_substrate_plan()
    bad["fold_contained_refs"] = list(bad["fold_contained_refs"]) + [
        {"ref": "external_exec/PD/all_source_dev", "disease": "PD", "fold": 0, "seed": P.SELECTION_SEED,
         "roles": [PLAN.S1_ROLE], "source_path": None}]
    expect_raises(ValueError, lambda: SCH.validate_build_manifest(bad), "final ref among fold refs")
    ok("fold registry / schema refuse a final-external ref in the fold set (no slipping it in)")


def test_final_refs_not_built_in_stage1():
    # Stage-1A declares final refs as schema only; none carries a real source_path
    for r in PLAN.final_external_refs():
        assert r["source_path"] is None
    ok("final-external refs are schema-only in Stage-1A (source_path=None; built only post-Stage-4)")


def main():
    print("ACAR v5 Stage-1A guard: final external ref separate")
    test_final_ref_shape_distinct()
    test_fold_registry_refuses_final_ref()
    test_final_refs_not_built_in_stage1()
    print("ALL V5 STAGE1-FINAL-EXTERNAL-REF-SEPARATE GUARDS PASS")


if __name__ == "__main__":
    main()
