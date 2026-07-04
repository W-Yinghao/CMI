"""Guard (Stage-1B6, req8): the finalize BARRIER is all-or-none. A pre-populate failure (fewer than 30 refs, or any barrier check)
leaves the registry EMPTY and writes NO finalized marker; only a fully-validated file-backed build writes the FINALIZED marker.
Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.tests._util import stage1b_repair_staging_root as _RSR
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_finalize as FIN
from acar.v5.substrate.registry import SubstrateRegistry
from acar.v5.tests._util import (expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader,
                                 FakeFileTrainer, FakeFileDumper, synthetic_canonical_artifacts, synthetic_canonical_paths)

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
RUN = "run-syn-0001"
_META = dict(git_commit="0" * 40, env_lock_sha256="a" * 64, channel_montage="10-20-19", sampling_rate=128,
             windowing_config="4s/512")


def test_partial_leaves_registry_empty_no_marker():
    with tempfile.TemporaryDirectory() as root:
        reg = SubstrateRegistry()
        artifacts = dict(list(synthetic_canonical_artifacts().items())[:29])
        paths = {r: synthetic_canonical_paths()[r] for r in artifacts}
        expect_raises(FIN.Stage1bFinalizeError,
                      lambda: FIN.finalize_and_populate(reg, artifacts, paths_by_ref=paths, output_root=root, run_id=RUN, **_META))
        assert len(reg._entries) == 0 and not os.path.exists(FIN.marker_path(root, RUN))
    ok("29/30 refs → finalize raises BEFORE any register; registry empty; no FINALIZED marker")


def test_bytes_success_registers_all_30_no_marker():
    reg = SubstrateRegistry()
    n = FIN.finalize_and_populate(reg, synthetic_canonical_artifacts(), paths_by_ref=None, **_META)
    assert n == 30 and len(reg._entries) == 30
    ok("bytes-backed build (no on-disk layout) → 30 registered, no marker (marker is only for file-backed real builds)")


def test_file_backed_success_writes_marker():
    with tempfile.TemporaryDirectory() as d:
        rep = B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                       stage1b_lock(protocol_tag_target_sha=FULL), output_root=d, repair_staging_root=_RSR(),
                                       dev_reader_factory=lambda ctx: FakeDevReader(),
                                       trainer_factory=lambda ctx: FakeFileTrainer(ctx.output_root, ctx.run_id),
                                       dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id))
        assert rep["n_registered"] == 30 and os.path.exists(FIN.marker_path(d, RUN))
    ok("a fully-validated file-backed build → 30 registered AND a FINALIZED marker written (barrier passed in full)")


def main():
    print("ACAR v5 Stage-1B6 guard: finalize barrier before registry")
    test_partial_leaves_registry_empty_no_marker()
    test_bytes_success_registers_all_30_no_marker()
    test_file_backed_success_writes_marker()
    print("ALL V5 STAGE1B-FINALIZE-BARRIER GUARDS PASS")


if __name__ == "__main__":
    main()
