"""Guard (Stage-1A): seed roles — the Stage-2 SELECTION role is allowed ONLY for seed 20260711; seeds 20260712/20260713 are
S1-robustness ONLY. Synthetic only."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.substrate import plan as PLAN
from acar.v5.tests._util import expect_raises, ok


def test_assert_seed_role():
    assert PLAN.assert_seed_role(P.SELECTION_SEED, PLAN.SELECTION_ROLE) is True
    assert PLAN.assert_seed_role(P.SELECTION_SEED, PLAN.S1_ROLE) is True
    for s in (20260712, 20260713):
        assert PLAN.assert_seed_role(s, PLAN.S1_ROLE) is True
        expect_raises(ValueError, lambda s=s: PLAN.assert_seed_role(s, PLAN.SELECTION_ROLE), "12/13 cannot select")
    expect_raises(ValueError, lambda: PLAN.assert_seed_role(99999999, PLAN.S1_ROLE), "unknown seed")
    ok("selection role only for seed 20260711; seeds 20260712/13 are S1-only")


def test_plan_roles_consistent():
    for r in PLAN.fold_refs():
        if r["seed"] == P.SELECTION_SEED:
            assert set(r["roles"]) == {PLAN.SELECTION_ROLE, PLAN.S1_ROLE}
        else:
            assert r["roles"] == [PLAN.S1_ROLE]
    # every selection ref is seed 20260711
    assert all(r["seed"] == P.SELECTION_SEED for r in PLAN.selection_refs())
    ok("plan roles: seed 20260711 → {selection, s1}; seeds 20260712/13 → {s1} only")


def test_selection_ref_count():
    assert len(PLAN.selection_refs()) == len(P.DEV_COHORTS) * P.OUTER_K == 10
    ok("exactly 10 selection refs (2 diseases × 5 folds × seed 20260711)")


def main():
    print("ACAR v5 Stage-1A guard: S1 seed roles")
    test_assert_seed_role()
    test_plan_roles_consistent()
    test_selection_ref_count()
    print("ALL V5 STAGE1-S1-SEED-ROLES GUARDS PASS")


if __name__ == "__main__":
    main()
