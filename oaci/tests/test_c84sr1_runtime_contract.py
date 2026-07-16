from pathlib import Path

import pytest

from oaci.multidataset.c84s_common import read_json, sha256_file
from oaci.multidataset.c84sr1_common import AUTHORIZATION_PATH, LOCK_PATH
from oaci.multidataset.c84sr1_runtime_guard import (
    EXPECTED, PROTOCOL_INPUTS, external_artifact_rows,
    static_process_isolation_audit, verify_protocol_inputs,
)
from oaci.multidataset.c84sr1_synthetic import (
    synthetic_contexts, synthetic_label_rows,
)


def test_frozen_protocol_and_historical_lock_identities_replay():
    observed = verify_protocol_inputs()
    assert observed == EXPECTED
    assert all(PROTOCOL_INPUTS[name].is_file() for name in EXPECTED)


def test_static_stage_process_isolation_passes():
    checks = static_process_isolation_audit()
    assert len(checks) >= 24
    assert all(row["pass"] == 1 for row in checks)


def test_manifest_identity_external_registry_has_exact_7776_objects():
    rows, summary = external_artifact_rows(verify_bytes=False)
    assert len(rows) == summary["files"] == 7776
    assert summary["counts"] == {
        "target_artifact": 1944, "target_sidecar": 1944,
        "source_audit": 1944, "training_sidecar": 1944,
    }


def test_synthetic_complete_arithmetic_without_real_field_values():
    registry, labels, _ = synthetic_label_rows()
    contexts = synthetic_contexts()
    assert len(registry) == len(labels) == 9621
    assert len(contexts) == 944
    assert sum(len(row.candidates) for row in contexts) == 76464


def test_v3_authorization_was_absent_at_lock_and_is_now_preserved():
    lock = read_json(LOCK_PATH)
    assert lock["authorization"]["record_present_at_lock"] is False
    assert AUTHORIZATION_PATH.is_file()
    assert sha256_file(AUTHORIZATION_PATH) == "441de7edc9a40da6bdf66faad2677d0abca2ca6119a1a9f599bc8727087371ee"
