"""Guard (Stage-1A): the deterministic substrate plan has the exact ref counts/shapes and reads no data. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import build_manifest_schema as SCH
from acar.v5.substrate import stage1_preflight as PF
from acar.v5.tests._util import ok


def test_counts():
    pl = PLAN.build_substrate_plan()
    exp_fold = len(P.DEV_COHORTS) * P.OUTER_K * len(P.S1_SEEDS)           # 2 × 5 × 3 = 30
    exp_sel = len(P.DEV_COHORTS) * P.OUTER_K                              # 2 × 5 × 1 = 10 (seed 20260711 only)
    assert len(pl["fold_contained_refs"]) == exp_fold == 30
    assert len(pl["selection_refs"]) == exp_sel == 10
    assert len(pl["final_external_refs"]) == len(P.DEV_COHORTS) == 2
    assert pl["counts"] == {"diseases": 2, "folds": 5, "seeds": 3, "fold_refs": 30, "selection_refs": 10,
                            "final_external_refs": 2}
    ok("plan counts: 30 fold refs, 10 selection refs (seed 20260711), 2 final-external refs")


def test_ref_shapes():
    pl = PLAN.build_substrate_plan()
    for r in pl["fold_contained_refs"]:
        assert SCH.is_fold_ref(r["ref"]) and not SCH.is_final_external_ref(r["ref"])
        assert r["disease"] in P.DEV_COHORTS and 0 <= r["fold"] < P.OUTER_K and r["seed"] in P.S1_SEEDS
    for r in pl["final_external_refs"]:
        assert SCH.is_final_external_ref(r["ref"]) and not SCH.is_fold_ref(r["ref"])
    ok("every fold ref matches <disease>/fold<n>/seed<seed>; final-external refs are a distinct shape")


def test_selection_refs_are_seed_20260711():
    pl = PLAN.build_substrate_plan()
    for ref in pl["selection_refs"]:
        assert ref.endswith(f"seed{P.SELECTION_SEED}"), ref
    ok("all selection refs carry the canonical selection seed 20260711")


def test_preflight_ok_reads_nothing():
    rep = PF.run_preflight()
    assert rep["status"] == "STAGE1A_PREFLIGHT_OK" and rep["real_data_entries"] == 0
    ok("default preflight = STAGE1A_PREFLIGHT_OK with 0 real-data reads")


def main():
    print("ACAR v5 Stage-1A guard: plan counts + refs")
    test_counts()
    test_ref_shapes()
    test_selection_refs_are_seed_20260711()
    test_preflight_ok_reads_nothing()
    print("ALL V5 STAGE1-PLAN GUARDS PASS")


if __name__ == "__main__":
    main()
