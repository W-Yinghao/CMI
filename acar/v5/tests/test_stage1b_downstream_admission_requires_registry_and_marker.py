"""Guard (Stage-1B9): downstream (Stage-2) admission requires BOTH registry.json and FINALIZED.json with a MATCHING registry_sha256
and n_refs==30 — missing either, a hash mismatch, or a bad n_refs is inadmissible. Synthetic only."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from acar.v5.substrate import stage1b_build as B
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_io as RIO
from acar.v5.tests._util import expect_raises, ok, stage1b_auth, stage1b_lock, stage1b_full_plan, FakeDevReader, FakeTrainer, FakeDumper

FULL = SA.PROTOCOL_TAG_TARGET_SHA_FULL
RUN = "run-syn-0001"


def _registry():
    rep = B.run_stage1b_build(stage1b_full_plan(), stage1b_auth(protocol_tag_target_sha=FULL),
                              stage1b_lock(protocol_tag_target_sha=FULL), execute=True,
                              dev_reader=FakeDevReader(), trainer=FakeTrainer(), dumper=FakeDumper())
    return rep["registry"]


def _layout(root, *, reg=True, marker=True, sha=None, n_refs=30):
    run_root = os.path.join(root, RUN)
    os.makedirs(run_root, exist_ok=True)
    registry = _registry()
    reg_sha = RIO.registry_sha256(registry)
    if reg:
        reg_sha = RIO.write_registry(registry, os.path.join(run_root, RIO.REGISTRY_FILE))
    if marker:
        payload = {"status": "FINALIZED", "n_registered": 30, "n_refs": n_refs,
                   "registry_sha256": (sha if sha is not None else reg_sha), "git_commit": "0" * 40, "env_lock_sha256": "a" * 64}
        with open(os.path.join(run_root, RIO.MARKER_FILE), "w") as f:
            json.dump(payload, f, sort_keys=True)
    return root


def test_admits_complete_run():
    with tempfile.TemporaryDirectory() as d:
        _layout(d)
        admitted = RIO.admit_run(d, RUN)
        assert len(admitted._entries) == 30
        # the returned registry is parsed from the SAME hash-checked bytes (no TOCTOU): its canonical sha == the marker's
        marker = json.load(open(os.path.join(d, RUN, RIO.MARKER_FILE)))
        assert RIO.registry_sha256(admitted) == marker["registry_sha256"]
    ok("complete run → admitted (30 entries) and the returned registry matches the hash-checked bytes (no TOCTOU re-read)")


def test_malformed_json_and_bad_types_fail_closed():
    with tempfile.TemporaryDirectory() as d:
        _layout(d)                                            # then corrupt the marker to invalid JSON
        with open(os.path.join(d, RUN, RIO.MARKER_FILE), "w") as f:
            f.write("{not valid json")
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    with tempfile.TemporaryDirectory() as d:
        _layout(d, sha=None)                                  # rewrite marker with n_refs=null
        run_root = os.path.join(d, RUN)
        marker = json.load(open(os.path.join(run_root, RIO.MARKER_FILE)))
        marker["n_refs"] = None
        with open(os.path.join(run_root, RIO.MARKER_FILE), "w") as f:
            json.dump(marker, f)
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    with tempfile.TemporaryDirectory() as d:
        _layout(d)                                            # corrupt registry.json to invalid JSON (sha will also mismatch)
        with open(os.path.join(d, RUN, RIO.REGISTRY_FILE), "w") as f:
            f.write("{bad")
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    ok("malformed marker/registry JSON or a null n_refs → RegistryIoError (fail-closed, not a raw JSON/Type error)")


def test_missing_registry_or_marker_rejected():
    with tempfile.TemporaryDirectory() as d:
        _layout(d, reg=False)
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    with tempfile.TemporaryDirectory() as d:
        _layout(d, marker=False)
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    ok("missing registry.json OR missing FINALIZED.json → inadmissible")


def test_hash_mismatch_and_bad_nrefs_rejected():
    with tempfile.TemporaryDirectory() as d:
        _layout(d, sha="0" * 64)                              # marker registry_sha256 disagrees with the file
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    with tempfile.TemporaryDirectory() as d:
        _layout(d, n_refs=29)                                 # marker claims != 30
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    with tempfile.TemporaryDirectory() as d:                  # tamper registry.json AFTER writing → sha no longer matches
        _layout(d)
        with open(os.path.join(d, RUN, RIO.REGISTRY_FILE), "ab") as f:
            f.write(b" ")
        expect_raises(RIO.RegistryIoError, lambda: RIO.admit_run(d, RUN))
    ok("registry_sha256 mismatch / n_refs != 30 / tampered registry.json → inadmissible")


def main():
    print("ACAR v5 Stage-1B9 guard: downstream admission requires registry and marker")
    test_admits_complete_run()
    test_missing_registry_or_marker_rejected()
    test_hash_mismatch_and_bad_nrefs_rejected()
    test_malformed_json_and_bad_types_fail_closed()
    print("ALL V5 STAGE1B-DOWNSTREAM-ADMISSION GUARDS PASS")


if __name__ == "__main__":
    main()
