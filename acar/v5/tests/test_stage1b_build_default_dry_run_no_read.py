"""Guard (Stage-1B2): the build defaults to DRY-RUN and reads NOTHING. Synthetic only."""
from __future__ import annotations
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.tests._util import ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL


def test_default_dry_run_reads_nothing():
    reader = FakeDevReader()
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=False, dev_reader=reader, trainer=FakeTrainer())
    assert rep["status"] == "STAGE1B_BUILD_DRYRUN" and rep["n_would_build"] == 30 and rep["reads"] == 0
    assert reader.listed == [] and reader.read_calls == []
    ok("execute=False → STAGE1B_BUILD_DRYRUN, 30 would-build, and the reader is NEVER called (no read/list)")


def main():
    print("ACAR v5 Stage-1B2 guard: build default dry-run no read")
    test_default_dry_run_reads_nothing()
    print("ALL V5 STAGE1B-BUILD-DRYRUN GUARDS PASS")


if __name__ == "__main__":
    main()
