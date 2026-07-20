from pathlib import Path

from oaci.multidataset.c84s_common import read_json, sha256_file
from oaci.multidataset.c84sr2_common import (
    AUTHORIZATION_PATH, LOCK_READY_STATUS, PROTOCOL_SHA_PATH,
)
from oaci.multidataset.c84sr2_runtime_guard import (
    static_process_isolation_audit, verify_bound_repository_objects,
    verify_lock_bound_readiness, verify_lock_self, verify_protocol_inputs,
)


def test_repair_protocol_and_historical_v3_evidence_replay():
    observed = verify_protocol_inputs()
    assert observed["repair"] == PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    assert observed["historical_V3_lock"] == "15d8a84b7870021a90b1f0f103a8a4d733523b249321b143a89091978c0aa9fc"
    assert observed["historical_V3_authorization"] == "441de7edc9a40da6bdf66faad2677d0abca2ca6119a1a9f599bc8727087371ee"


def test_v4_authorization_is_preserved_as_consumed_historical_evidence():
    assert AUTHORIZATION_PATH.is_file()
    assert sha256_file(AUTHORIZATION_PATH) == (
        "4419303f8282ab132d2f95a5b76993bfb73191b29e7257e8220cefdde408ff5a"
    )
    record = read_json(AUTHORIZATION_PATH)
    assert record["direct_explicit_PI_authorization"] is True
    assert record["authorized_stage"] == "C84S"


def test_static_isolation_includes_label_loader_free_stage_a_replay():
    checks = static_process_isolation_audit()
    assert all(row["pass"] == 1 for row in checks)
    assert any(row["check"] == "immutable_replay_has_no_label_loader" for row in checks)


def test_v4_coordinator_does_not_import_label_provisioner():
    source = Path("oaci/multidataset/c84sr2_execute.py").read_text(encoding="utf-8")
    assert "c84sr1_stage_a_labels" not in source
    assert "c84sr2_stage_a_replay" in source


def test_v4_lock_preserves_readiness_evidence_but_is_nonoperative_after_v5():
    lock, digest = verify_lock_self()
    assert digest == "582e5074b4b17d62ff1e5fbfd992f037dd3082b7763b22d707630aa19db81c3d"
    assert lock["status"] == LOCK_READY_STATUS
    assert lock["chronology"]["historical_V3_authorization_consumed"] is True
    assert lock["chronology"]["historical_V3_authorization_reusable"] is False
    assert lock["field_descriptor_compatibility"] == {
        "units": 1944,
        "native_sidecars": 1701,
        "historical_C84C_omissions": 243,
        "contexts": 944,
        "table_sha256": lock["field_descriptor_compatibility"]["table_sha256"],
    }
    drift = [
        row["path"] for row in lock["runtime_bound_repository_objects"]
        if sha256_file(Path(row["path"])) != row["sha256"]
    ]
    assert drift == [
        "oaci/multidataset/c84s_common.py",
        "oaci/multidataset/c84s_analysis.py",
        "oaci/multidataset/c84sr1_q0_store.py",
        "oaci/multidataset/c84sr1_method_context_materialization.py",
        "oaci/multidataset/c84sr1_analysis.py",
        "oaci/multidataset/c84sr1_stage_b_selection.py",
    ]
    replay = verify_lock_bound_readiness(lock)
    assert replay["tables"] >= 10
    assert replay["synthetic"] == "7b88c30f0f623894a33cfcfa6aea56149500a8df7a1f5cb202d765e169384749"
