"""Guard (Stage-2A, PROP 10): the Stage-2A runner is DRY-RUN ONLY — it admits + validates readiness but selects NO candidate,
and binding selection fails closed (Stage-2B is a separate authorization). Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5 import protocol as P
from acar.v5 import stage2_selection_runner as RUN2
from acar.v5.substrate import plan as PLAN
from acar.v5.substrate import stage1b_output_layout as LO
from acar.v5.tests._util import expect_raises, ok, stage1b_finalized_package, write_synthetic_feat_dump

RUN = "run-syn-0001"


def test_dry_run_selects_nothing():
    with tempfile.TemporaryDirectory() as d:
        stage1b_finalized_package(d, RUN)
        rep = RUN2.dry_run_selection_readiness(d, RUN)
        assert rep["admitted"] is True
        assert rep["n_refs"] == 30
        assert len(rep["selection_refs"]) == 10
        assert len(rep["robustness_excluded_refs"]) == 20
        assert rep["n_candidates"] == 22
        assert rep["family_counts"] == {"P1": 4, "P2": 4, "P3": 6, "P4": 2, "P5": 6}
        assert rep["joint_disease_scope"] is True
        assert rep["label_free_routing"] is True
        assert rep["gate_certifier"].endswith("gate_disease")
        assert rep["selected_candidate"] is None                   # nothing selected
        assert rep["stage2b_authorized"] is False
    ok("dry-run readiness admits + validates but selects NO candidate; stage2b_authorized=False (PROP 10)")


def test_feature_dump_headers_checked():
    with tempfile.TemporaryDirectory() as d:
        stage1b_finalized_package(d, RUN)
        for ref in sorted(r["ref"] for r in PLAN.selection_refs()):
            dd = LO.ref_output_dir(d, RUN, ref)
            os.makedirs(dd, exist_ok=True)
            fold = int(ref.split("fold")[1].split("/")[0])
            write_synthetic_feat_dump(os.path.join(dd, "feat_dump.npz"), ref=ref, disease=ref.split("/")[0],
                                      fold=fold, seed=P.SELECTION_SEED)
        rep = RUN2.dry_run_selection_readiness(d, RUN, check_feature_dumps=True)
        assert rep["feature_dump_headers"] is not None and len(rep["feature_dump_headers"]) == 10
        assert rep["selected_candidate"] is None
    ok("dry-run with feature-dump header validation reads all 10 selection headers (no embeddings), still selects nothing (PROP 10)")


def test_binding_selection_fails_closed():
    expect_raises(RUN2.Stage2BNotAuthorizedError, lambda: RUN2.run_binding_selection())
    expect_raises(RUN2.Stage2BNotAuthorizedError, lambda: RUN2.run_binding_selection("anything", stage2b_authorization={}))
    assert RUN2._STAGE2B_ENABLED is False
    ok("run_binding_selection ALWAYS raises Stage2BNotAuthorizedError in Stage-2A (Stage-2B is separate) (PROP 10)")


def main():
    print("ACAR v5 Stage-2A guard: dry-run selects nothing; binding selection fails closed (PROP 10)")
    test_dry_run_selects_nothing()
    test_feature_dump_headers_checked()
    test_binding_selection_fails_closed()
    print("ALL V5 STAGE2A-DRYRUN-NO-CANDIDATE GUARDS PASS")


if __name__ == "__main__":
    main()
