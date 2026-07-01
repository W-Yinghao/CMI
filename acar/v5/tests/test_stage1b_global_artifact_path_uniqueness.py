"""Guard (Stage-1B6, req6): NO artifact file path may be reused within a ref or across refs. The layout helper enforces it, and the
finalize barrier applies it across all 30 refs BEFORE any registration. Synthetic only."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import stage1b_output_layout as LO
from acar.v5.substrate import stage1b_finalize as FIN
from acar.v5.substrate.registry import SubstrateRegistry
from acar.v5.tests._util import expect_raises, ok, synthetic_canonical_artifacts, synthetic_canonical_paths
import os

_META = dict(git_commit="0" * 40, env_lock_sha256="a" * 64, channel_montage="10-20-19", sampling_rate=128,
             windowing_config="4s/512")


def test_layout_uniqueness_helper():
    assert LO.assert_global_artifact_paths_unique({"r1": ["/a/x", "/a/y"], "r2": ["/b/z"]}) == 3
    expect_raises(LO.Stage1bLayoutError, lambda: LO.assert_global_artifact_paths_unique({"r1": ["/a/x", "/a/x"]}))     # within-ref
    expect_raises(LO.Stage1bLayoutError, lambda: LO.assert_global_artifact_paths_unique({"r1": ["/a/x"], "r2": ["/a/x"]}))  # cross-ref
    ok("assert_global_artifact_paths_unique: unique OK; within-ref dup and cross-ref dup both rejected")


def test_finalize_rejects_cross_ref_path_collision():
    with tempfile.TemporaryDirectory() as root:
        reg = SubstrateRegistry()
        artifacts = synthetic_canonical_artifacts()
        paths = synthetic_canonical_paths(collide=True)       # ref[1] reuses ref[0]'s feat_dump path
        expect_raises(FIN.Stage1bFinalizeError,
                      lambda: FIN.finalize_and_populate(reg, artifacts, paths_by_ref=paths, output_root=root,
                                                        run_id="run-syn-0001", **_META))
        assert len(reg._entries) == 0                          # registry untouched
        assert not os.path.exists(FIN.marker_path(root, "run-syn-0001"))   # no finalized marker
    ok("a cross-ref artifact path collision → finalize raises; registry stays empty; no FINALIZED marker")


def main():
    print("ACAR v5 Stage-1B6 guard: global artifact path uniqueness")
    test_layout_uniqueness_helper()
    test_finalize_rejects_cross_ref_path_collision()
    print("ALL V5 STAGE1B-GLOBAL-PATH-UNIQUENESS GUARDS PASS")


if __name__ == "__main__":
    main()
