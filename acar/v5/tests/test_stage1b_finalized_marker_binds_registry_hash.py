"""Guard (Stage-1B9): a file-backed build persists registry.json AND writes FINALIZED.json whose registry_sha256 EXACTLY matches
sha256(registry.json) with n_refs==30 — so downstream admission is bound to the persisted registry, not an in-memory object.
Synthetic only."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_io as RIO
from acar.v5.tests._util import ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeFileTrainer, FakeFileDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
RUN = "run-syn-0001"


def test_marker_binds_registry_hash():
    with tempfile.TemporaryDirectory() as d:
        rep = B.run_stage1b_real_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                                       stage1b_lock(protocol_tag_target_sha=FULL), output_root=d,
                                       dev_reader_factory=lambda ctx: FakeDevReader(),
                                       trainer_factory=lambda ctx: FakeFileTrainer(ctx.output_root, ctx.run_id),
                                       dumper_factory=lambda ctx: FakeFileDumper(ctx.output_root, ctx.run_id))
        assert rep["n_registered"] == 30
        run_root = os.path.join(d, RUN)
        reg_path = os.path.join(run_root, RIO.REGISTRY_FILE)
        marker_path = os.path.join(run_root, RIO.MARKER_FILE)
        assert os.path.isfile(reg_path) and os.path.isfile(marker_path)
        marker = json.load(open(marker_path))
        reg_sha = hashlib.sha256(open(reg_path, "rb").read()).hexdigest()
        assert marker["status"] == "FINALIZED" and marker["n_refs"] == 30
        assert marker["registry_sha256"] == reg_sha           # marker bound to the persisted registry bytes
        admitted = RIO.admit_run(d, RUN)                      # downstream admission succeeds
        assert len(admitted._entries) == 30
    ok("file-backed build → registry.json + FINALIZED.json with registry_sha256==sha256(registry.json), n_refs==30; admit_run OK")


def main():
    print("ACAR v5 Stage-1B9 guard: finalized marker binds registry hash")
    test_marker_binds_registry_hash()
    print("ALL V5 STAGE1B-MARKER-BINDS-REGISTRY-HASH GUARDS PASS")


if __name__ == "__main__":
    main()
