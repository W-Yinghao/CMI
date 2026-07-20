from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from oaci.multidataset.c84s_common import sha256_file
from oaci.theory.c85u_runtime_guard import replay_execution_lock
from oaci.theory.c85urp_readiness import LOCK_STATUS, PROTOCOL_COMMIT, SUCCESS_GATE


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "oaci/reports"


def test_c85u_lock_self_hash_schema_and_bound_objects() -> None:
    lock_path = REPORTS / "C85U_EXECUTION_LOCK.json"
    lock, digest, repo_root, lock_commit = replay_execution_lock(lock_path)
    assert repo_root == REPO_ROOT
    assert digest == (REPORTS / "C85U_EXECUTION_LOCK.sha256").read_text().split()[0]
    assert lock["status"] == LOCK_STATUS
    assert lock["authorized"] is False
    assert lock["readiness"]["success_gate"] == SUCCESS_GATE
    assert len(lock["bound_repository_objects"]) == lock["runtime_bound_object_count"]
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, lock["implementation_commit"]],
        cwd=REPO_ROOT,
    ).returncode == 0
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["implementation_commit"], lock_commit],
        cwd=REPO_ROOT,
    ).returncode == 0


def test_c85u_lock_binds_scope_formulas_and_stage_isolation() -> None:
    lock = json.loads((REPORTS / "C85U_EXECUTION_LOCK.json").read_text())
    assert lock["frozen_inputs"]["contexts"] == 944
    assert lock["frozen_inputs"]["target_artifact_registry"]["units"] == 1944
    assert lock["frozen_inputs"]["target_artifact_registry"]["opened_C85URP"] == 0
    assert lock["stages"]["U1"]["candidate_rows"] == 76_464
    assert lock["stages"]["U1"]["selection_inputs"] == 0
    assert lock["stages"]["U2"]["method_context_rows"] == 18_432
    assert lock["stages"]["U2"]["label_or_logit_inputs"] == 0
    assert lock["stages"]["U2"]["Q0_chains"] == 2048
    assert lock["stages"]["U2"]["Q0_resampling"] == 0
    assert lock["stages"]["subprocess_isolation"] is True
    assert lock["numerical_contract"] == {
        "float_dtype": "<f8",
        "metric_and_utility_max_abs": 1e-12,
        "identity_digest_order_midrank": "EXACT",
        "runtime_widening": False,
    }


def test_c85u_target_registry_and_metadata_access_are_exact() -> None:
    path = REPORTS / "c85urp_tables/target_artifact_registry.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1944
    assert len({row["unit_id"] for row in rows}) == 1944
    assert all(row["target_artifact_opened"] == "0" for row in rows)
    lock = json.loads((REPORTS / "C85U_EXECUTION_LOCK.json").read_text())
    assert lock["frozen_inputs"]["target_artifact_registry"]["sha256"] == sha256_file(path)
    assert lock["readiness"]["real_evaluation_label_rows_opened"] == 0
    assert lock["readiness"]["real_target_artifact_payloads_opened"] == 0
    assert lock["readiness"]["real_candidate_utilities_computed"] == 0


def test_c85urp_has_no_authorization_execution_or_c85e_lock() -> None:
    assert not (REPORTS / "C85U_PI_AUTHORIZATION_RECORD.json").exists()
    assert not (REPORTS / "C85E_EXECUTION_LOCK.json").exists()
    assert not Path(
        "/projects/EEG-foundation-model/yinghao/oaci-c85u-candidate-utility-v1"
    ).exists()
    lock = json.loads((REPORTS / "C85U_EXECUTION_LOCK.json").read_text())
    assert lock["forbidden"]["C85E_execution"] is True
    assert lock["forbidden"]["C86"] is True
    assert lock["forbidden"]["active_acquisition"] is True
    assert lock["forbidden"]["manuscript_work"] is True
