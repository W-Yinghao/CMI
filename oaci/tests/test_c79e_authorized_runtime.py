from __future__ import annotations

from oaci.conditioned_ceiling_coverage import c79e_seed4_replication as c79e
from oaci.conditioned_ceiling_coverage import c79p_post_seed3_protocol as c79p


def test_direct_authorization_record_binds_all_operative_objects():
    record = c79p.require_c79e_authorization()
    protocol, protocol_sha = c79p.load_protocol()
    field_lock, field_sha = c79p.load_field_lock()
    analysis_lock, analysis_sha = c79p.load_analysis_lock()

    assert protocol_sha == "e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587"
    assert record["protocol_sha256"] == protocol_sha
    assert record["protocol_commit"].startswith("ec4834c")
    assert record["field_lock_sha256"] == field_sha
    assert record["field_lock_commit"].startswith("35d0c65")
    assert record["analysis_lock_sha256"] == analysis_sha
    assert record["analysis_lock_commit"].startswith("7cebf2e")
    assert field_lock["protocol_sha256"] == analysis_lock["protocol_sha256"] == protocol_sha
    assert protocol["epistemic_status"]["prospective_to_seed4_checkpoint_outcomes"] is True


def test_authorized_scope_remains_seed4_only_without_oracle_or_scope_expansion():
    record = c79p.require_c79e_authorization()
    assert record["seed"] == 4
    assert record["seed5"] is False
    assert record["BNCI2014_004"] is False
    assert record["same_label_oracle"] is False
    assert record["new_targets"] is False
    assert record["new_feature_kernel_model_search"] is False
    assert record["C80"] is False
    assert record["manuscript"] is False

    field = c79e.field_binding_contract()
    analysis = c79e.analysis_binding_contract()
    assert field["same_label_oracle_created"] is False
    assert analysis["same_label_oracle_reachable"] is False
    assert analysis["target4_primary"] is False
    assert analysis["all_registered_paths_unconditional"] is True
    assert analysis["active_after_Holm_runtime_selection"] is False


def test_authorization_evidence_does_not_mutate_parent_execution_locks():
    field_lock, _ = c79p.load_field_lock()
    analysis_lock, _ = c79p.load_analysis_lock()
    assert field_lock["authorization"] == {
        "C79E_required": True,
        "handoff_or_protocol_text_is_authorization": False,
        "received": False,
        "record_path": "oaci/reports/C79E_PI_AUTHORIZATION_RECORD.json",
    }
    assert analysis_lock["authorization"] == {
        "C79E_required": True,
        "received": False,
        "record_path": "oaci/reports/C79E_PI_AUTHORIZATION_RECORD.json",
    }

