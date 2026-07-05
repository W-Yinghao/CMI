"""Guard (Stage-2A, PROP 6): candidate identity is selected JOINTLY across PD and SCZ — every candidate has disease_scope='both'
(per-disease FIT quantiles only). A per-disease candidate id is rejected. Synthetic (data-free)."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_selection_manifest as MANIFEST
from acar.v5.tests._util import expect_raises, ok


def test_all_candidates_joint():
    assert MANIFEST.assert_joint_disease_scope() is True
    assert all(c["disease_scope"] == "both" for c in MANIFEST.selection_manifest())
    ok("every candidate disease_scope='both' → joint PD/SCZ selection; no per-disease manifest (PROP 6)")


def test_per_disease_candidate_rejected():
    per_disease = tuple(dict(c) for c in P.CANDIDATE_MANIFEST)
    per_disease[0]["disease_scope"] = "PD"                          # a disease-specific candidate id — forbidden
    expect_raises(MANIFEST.Stage2ManifestError, lambda: MANIFEST.assert_joint_disease_scope(per_disease))
    # selection_manifest also refuses a non-joint manifest
    orig = P.CANDIDATE_MANIFEST
    try:
        P.CANDIDATE_MANIFEST = per_disease
        expect_raises(MANIFEST.Stage2ManifestError, MANIFEST.selection_manifest)
    finally:
        P.CANDIDATE_MANIFEST = orig
    ok("a disease_scope='PD' candidate → Stage2ManifestError (no per-disease selection) (PROP 6)")


def main():
    print("ACAR v5 Stage-2A guard: joint PD/SCZ candidate selection only (PROP 6)")
    test_all_candidates_joint()
    test_per_disease_candidate_rejected()
    print("ALL V5 STAGE2A-JOINT-DISEASE GUARDS PASS")


if __name__ == "__main__":
    main()
