"""Mechanical C79E authorization-schema bridge for the locked C78S engine.

The C79P analysis lock records the authorization evidence path but intentionally
remains unreceived before C79E.  The historical C78S numerical engine expects a
runtime ``authorization.mode`` field.  This module overlays only that runtime
evidence after replaying the committed C79E authorization record and the
unchanged C79P analysis lock.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from . import c79e_seed4_replication as c79e
from . import c79p_post_seed3_protocol as c79p


BRIDGE_LOCK_PATH = c79p.REPORT_DIR / "C79E_ANALYSIS_AUTHORIZATION_BRIDGE_LOCK.json"
BRIDGE_LOCK_SHA_PATH = c79p.REPORT_DIR / "C79E_ANALYSIS_AUTHORIZATION_BRIDGE_LOCK.sha256"
BRIDGE_REPLAY_PATH = c79p.REPORT_DIR / "C79E_ANALYSIS_AUTHORIZATION_BRIDGE_REPLAY.json"
AUTHORIZATION_MODE = "direct_explicit_PI_authorization_current_execution_conversation"


def _load_bridge_lock() -> tuple[dict[str, Any], str]:
    expected = BRIDGE_LOCK_SHA_PATH.read_text().strip()
    observed = c79p.sha256_file(BRIDGE_LOCK_PATH)
    if observed != expected:
        raise RuntimeError("C79E analysis authorization bridge lock hash drift")
    lock = json.loads(BRIDGE_LOCK_PATH.read_text())
    if lock["scientific_registry_changed"] or lock["scientific_degrees_of_freedom_changed"]:
        raise RuntimeError("C79E authorization bridge attempted a scientific change")
    return lock, observed


def build_runtime_analysis_lock() -> tuple[dict[str, Any], str, dict[str, Any]]:
    record = c79p.require_c79e_authorization()
    analysis_lock, analysis_sha = c79p.load_analysis_lock()
    bridge_lock, bridge_sha = _load_bridge_lock()

    if bridge_lock["parent_analysis_lock_sha256"] != analysis_sha:
        raise RuntimeError("C79E authorization bridge parent lock drift")
    if bridge_lock["authorization_record_sha256"] != c79p.sha256_file(c79p.AUTHORIZATION_RECORD_PATH):
        raise RuntimeError("C79E authorization bridge record hash drift")
    if record["analysis_lock_sha256"] != analysis_sha or not record["direct_explicit_PI_authorization"]:
        raise PermissionError("C79E analysis authorization record does not bind the locked analysis")

    expected_unreceived = {
        "C79E_required": True,
        "received": False,
        "record_path": str(c79p.AUTHORIZATION_RECORD_PATH),
    }
    if analysis_lock["authorization"] != expected_unreceived:
        raise RuntimeError("C79E parent analysis authorization schema drift")

    runtime_lock = copy.deepcopy(analysis_lock)
    runtime_lock["authorization"] = {
        **expected_unreceived,
        "received": True,
        "mode": AUTHORIZATION_MODE,
        "record_sha256": bridge_lock["authorization_record_sha256"],
        "bridge_lock_sha256": bridge_sha,
    }
    replay = {
        "schema_version": "c79e_analysis_authorization_bridge_replay_v1",
        "parent_analysis_lock_sha256": analysis_sha,
        "bridge_lock_sha256": bridge_sha,
        "authorization_record_sha256": bridge_lock["authorization_record_sha256"],
        "changed_paths": [
            "authorization.received",
            "authorization.mode",
            "authorization.record_sha256",
            "authorization.bridge_lock_sha256",
        ],
        "scientific_registry_changed": False,
        "scientific_degrees_of_freedom_changed": False,
        "seed4_outcome_dependent_decision_introduced": False,
    }
    return runtime_lock, analysis_sha, replay


def run() -> dict[str, Any]:
    _, protocol, _, _, analysis = c79e._bind_analysis_engine()
    runtime_lock, analysis_sha, replay = build_runtime_analysis_lock()
    protocol.load_execution_lock = lambda: (runtime_lock, analysis_sha)
    c79p.write_json(BRIDGE_REPLAY_PATH, replay)

    result = analysis.run()
    result["C79P_replacement_protocol_sha256"] = c79p.sha256_file(c79p.PROTOCOL_PATH)
    result["seed"] = 4
    result["seed4_only_primary"] = True
    result["target4_primary"] = False
    result["same_label_oracle_accessed"] = False
    result["all_registered_paths_unconditional"] = True
    result["analysis_authorization_bridge"] = replay
    c79p.write_json(c79e.ANALYSIS_WORK_RESULT, result)
    result["C79_registered_decisions"] = c79e._synthesize_registered_decisions()
    c79p.write_json(c79e.ANALYSIS_WORK_RESULT, result)
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
