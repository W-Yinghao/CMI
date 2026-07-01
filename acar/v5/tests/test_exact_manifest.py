"""Guard (Step 3b): the 22-row candidate manifest matches the frozen table EXACTLY, row by row (id, family, params) — closes the
'counts match but a row drifted' hole. Synthetic (data-free)."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5.tests._util import ok

EXPECTED_MANIFEST = (
    ("V5-P1-001", "P1", {"benefit_q": "q60", "veto_q": "q80"}),
    ("V5-P1-002", "P1", {"benefit_q": "q60", "veto_q": "q90"}),
    ("V5-P1-003", "P1", {"benefit_q": "q80", "veto_q": "q80"}),
    ("V5-P1-004", "P1", {"benefit_q": "q80", "veto_q": "q90"}),
    ("V5-P2-001", "P2", {"conf_q": "q60", "veto_q": "q80"}),
    ("V5-P2-002", "P2", {"conf_q": "q60", "veto_q": "q90"}),
    ("V5-P2-003", "P2", {"conf_q": "q80", "veto_q": "q80"}),
    ("V5-P2-004", "P2", {"conf_q": "q80", "veto_q": "q90"}),
    ("V5-P3-001", "P3", {"action": "matched_coral", "veto_q": "q80"}),
    ("V5-P3-002", "P3", {"action": "matched_coral", "veto_q": "q90"}),
    ("V5-P3-003", "P3", {"action": "spdim", "veto_q": "q80"}),
    ("V5-P3-004", "P3", {"action": "spdim", "veto_q": "q90"}),
    ("V5-P3-005", "P3", {"action": "t3a", "veto_q": "q80"}),
    ("V5-P3-006", "P3", {"action": "t3a", "veto_q": "q90"}),
    ("V5-P4-001", "P4", {"k": 2, "veto_q": "q90"}),
    ("V5-P4-002", "P4", {"k": 3, "veto_q": "q90"}),
    ("V5-P5-001", "P5", {"lambda_q": "q50", "veto_q": "q90"}),
    ("V5-P5-002", "P5", {"lambda_q": "q60", "veto_q": "q90"}),
    ("V5-P5-003", "P5", {"lambda_q": "q70", "veto_q": "q90"}),
    ("V5-P5-004", "P5", {"lambda_q": "q80", "veto_q": "q90"}),
    ("V5-P5-005", "P5", {"lambda_q": "q85", "veto_q": "q90"}),
    ("V5-P5-006", "P5", {"lambda_q": "q90", "veto_q": "q90"}),
)


def test_exact_rows():
    got = tuple((c["id"], c["family"], dict(c["params"])) for c in P.CANDIDATE_MANIFEST)
    assert got == EXPECTED_MANIFEST, "candidate manifest drifted from the frozen 22-row table"
    ok("22-row manifest matches the frozen table EXACTLY (id, family, params), in order")


def test_p3_comparator_role():
    p3 = [c for c in P.CANDIDATE_MANIFEST if c["family"] == "P3"]
    assert len(p3) == 6 and all(c.get("comparator_role") == "candidate+g5_best_fixed" for c in p3)
    ok("P3 rows carry comparator_role=candidate+g5_best_fixed (G5 best-fixed pool)")


def test_every_scoped_both():
    assert all(c["disease_scope"] == "both" for c in P.CANDIDATE_MANIFEST)
    ok("every candidate disease_scope='both' (joint selection, per-disease FIT quantiles only)")


def main():
    print("ACAR v5 guard: exact 22-row manifest (Step 3b)")
    test_exact_rows()
    test_p3_comparator_role()
    test_every_scoped_both()
    print("ALL V5 EXACT-MANIFEST GUARDS PASS")


if __name__ == "__main__":
    main()
