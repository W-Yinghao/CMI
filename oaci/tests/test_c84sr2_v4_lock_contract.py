from pathlib import Path

from oaci.multidataset.c84sr2_common import AUTHORIZATION_PATH, PROTOCOL_SHA_PATH
from oaci.multidataset.c84sr2_runtime_guard import (
    static_process_isolation_audit, verify_protocol_inputs,
)


def test_repair_protocol_and_historical_v3_evidence_replay():
    observed = verify_protocol_inputs()
    assert observed["repair"] == PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    assert observed["historical_V3_lock"] == "15d8a84b7870021a90b1f0f103a8a4d733523b249321b143a89091978c0aa9fc"
    assert observed["historical_V3_authorization"] == "441de7edc9a40da6bdf66faad2677d0abca2ca6119a1a9f599bc8727087371ee"


def test_v4_authorization_absent_during_repair_readiness():
    assert not AUTHORIZATION_PATH.exists()


def test_static_isolation_includes_label_loader_free_stage_a_replay():
    checks = static_process_isolation_audit()
    assert all(row["pass"] == 1 for row in checks)
    assert any(row["check"] == "immutable_replay_has_no_label_loader" for row in checks)


def test_v4_coordinator_does_not_import_label_provisioner():
    source = Path("oaci/multidataset/c84sr2_execute.py").read_text(encoding="utf-8")
    assert "c84sr1_stage_a_labels" not in source
    assert "c84sr2_stage_a_replay" in source
