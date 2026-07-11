from __future__ import annotations

import copy

from oaci.conditioned_ceiling_coverage import c79e_analysis_authorization_bridge as bridge
from oaci.conditioned_ceiling_coverage import c79p_post_seed3_protocol as c79p


def test_runtime_bridge_changes_only_authorization_evidence():
    original, original_sha = c79p.load_analysis_lock()
    before = copy.deepcopy(original)
    runtime, replayed_sha, replay = bridge.build_runtime_analysis_lock()

    assert replayed_sha == original_sha
    assert original == before
    assert runtime.keys() == original.keys()
    assert {key: value for key, value in runtime.items() if key != "authorization"} == {
        key: value for key, value in original.items() if key != "authorization"
    }
    assert runtime["authorization"]["received"] is True
    assert runtime["authorization"]["mode"] == bridge.AUTHORIZATION_MODE
    assert replay["scientific_registry_changed"] is False
    assert replay["scientific_degrees_of_freedom_changed"] is False
    assert replay["seed4_outcome_dependent_decision_introduced"] is False


def test_bridge_parent_is_the_authorized_analysis_lock():
    lock, lock_sha = bridge._load_bridge_lock()
    _, parent_sha = c79p.load_analysis_lock()
    assert lock["parent_analysis_lock_sha256"] == parent_sha
    assert lock_sha == c79p.sha256_file(bridge.BRIDGE_LOCK_PATH)
    assert lock["authorization_record_sha256"] == c79p.sha256_file(c79p.AUTHORIZATION_RECORD_PATH)
    assert lock["scientific_registry_changed"] is False
