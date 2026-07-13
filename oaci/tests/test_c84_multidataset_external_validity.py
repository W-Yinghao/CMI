from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from oaci.multidataset import c84_dataset_registry as registry
from oaci.multidataset import c84_fixed_zoo_protocol as protocol
from oaci.multidataset import c84_synthetic_external_validity as synthetic


def _rows(name: str) -> list[dict[str, str]]:
    with (protocol.TABLE_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_C83_identity_is_locked():
    assert protocol.PARENT_HEAD == "2ecc8efd49d6b9d18b50eae3811be8f2ac4cfa25"
    assert protocol.C83_GATE == "C83_AAAI_EVIDENCE_CLAIM_FIGURE_TABLE_FREEZE_READY_FOR_MANUSCRIPT_AUTHORIZATION"


def test_metadata_registry_has_exact_primary_populations():
    assert {code: spec.subject_count for code, spec in registry.DATASETS.items()} == {
        "Lee2019_MI": 54,
        "Cho2017": 52,
        "PhysionetMI": 108,
    }
    assert registry.DATASETS["PhysionetMI"].excluded_subjects == (88,)


def test_subject_hash_partitions_have_exact_counts_and_no_overlap():
    targets = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
    for dataset, spec in registry.DATASETS.items():
        partition = registry.partition_subjects(spec)
        assert len(partition["source_panel_A"]) == 16
        assert len(partition["source_panel_B"]) == 16
        assert len(partition["targets"]) == targets[dataset]
        values = [set(value) for value in partition.values()]
        assert all(not left & right for index, left in enumerate(values) for right in values[index + 1 :])


def test_subject_partition_is_stable_and_has_predeclared_canaries():
    assert registry.partition_subjects(registry.DATASETS["Lee2019_MI"])["targets"][0] == 19
    assert registry.partition_subjects(registry.DATASETS["Cho2017"])["targets"][0] == 24
    assert registry.partition_subjects(registry.DATASETS["PhysionetMI"])["targets"][0] == 106


def test_source_panels_have_12_training_and_4_audit_subjects():
    for dataset, spec in registry.DATASETS.items():
        partition = registry.partition_subjects(spec)
        for panel in ("A", "B"):
            split = registry.source_train_audit_split(dataset, panel, partition[f"source_panel_{panel}"])
            assert len(split["source_training"]) == 12
            assert len(split["source_audit"]) == 4
            assert not set(split["source_training"]) & set(split["source_audit"])


def test_trial_hash_split_is_deterministic_disjoint_and_class_scoped():
    trial_ids = [f"session=0|run=1|trial={index}" for index in range(23)]
    first = registry.target_trial_split("PhysionetMI", 106, "left_hand", trial_ids)
    second = registry.target_trial_split("PhysionetMI", 106, "left_hand", reversed(trial_ids))
    assert first == second
    assert len(first["construction"]) == 11
    assert len(first["evaluation"]) == 12
    assert not set(first["construction"]) & set(first["evaluation"])


def test_primary_channel_audit_has_one_exact_blocker():
    assert len(registry.PRIMARY_CHANNELS) == 21
    assert registry.DATASETS["Lee2019_MI"].missing_primary_channels == ("FCz",)
    assert registry.DATASETS["Cho2017"].missing_primary_channels == ()
    assert registry.DATASETS["PhysionetMI"].missing_primary_channels == ()
    blockers = [row for row in _rows("channel_allowlist_registry.csv") if row["status"] == "BLOCKER"]
    assert len(blockers) == 1
    assert blockers[0]["dataset"] == "Lee2019_MI"
    assert blockers[0]["canonical_channel"] == "FCz"
    assert blockers[0]["substitution_allowed"] == "0"
    assert blockers[0]["interpolation_allowed"] == "0"


def test_protocol_hashes_replay_exactly():
    paths = (
        "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL",
        "C84_CANARY_PROTOCOL",
        "C84_FIELD_GENERATION_PROTOCOL",
        "C84_SCIENTIFIC_ANALYSIS_PROTOCOL",
    )
    for stem in paths:
        json_path = protocol.REPORT_DIR / f"{stem}.json"
        expected = (protocol.REPORT_DIR / f"{stem}.sha256").read_text().split()[0]
        assert _digest(json_path) == expected


def test_protocol_generation_is_byte_replayable():
    paths = sorted(protocol.REPORT_DIR.glob("C84_*PROTOCOL*"))
    before = {path: _digest(path) for path in paths}
    result = protocol.generate()
    after = {path: _digest(path) for path in paths}
    assert before == after
    assert result["candidate_units_created"] == 0
    assert result["execution_locks_created"] == 0


def test_all_execution_stage_protocols_are_blocked_not_authorizable():
    for stem in ("C84_CANARY_PROTOCOL", "C84_FIELD_GENERATION_PROTOCOL", "C84_SCIENTIFIC_ANALYSIS_PROTOCOL"):
        payload = json.loads((protocol.REPORT_DIR / f"{stem}.json").read_text())
        assert payload["status"] == "BLOCKED_NOT_READY_FOR_AUTHORIZATION"
        assert payload["future_execution_lock_required"] is True
        assert payload["open_blocker"] == protocol.CHANNEL_BLOCKER


def test_PI_broad_intent_is_recorded_but_not_consumable():
    payload = json.loads((protocol.REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL.json").read_text())
    authorization = payload["authorization"]
    assert authorization["PI_message"] == "authorizes C84P C84C C84F C84S in intent"
    assert authorization["C84C_C84F_C84S_authorization_consumable_now"] is False
    assert authorization["magic_token_required"] is False


def test_future_stage_guard_fails_before_missing_lock_or_authorization(tmp_path):
    with pytest.raises(protocol.C84ProtocolError, match="not execution-ready"):
        protocol.require_scope_execution_lock(
            stage_protocol_path=protocol.REPORT_DIR / "C84_CANARY_PROTOCOL.json",
            execution_lock_path=tmp_path / "missing-lock.json",
            authorization_path=tmp_path / "missing-auth.json",
        )


def test_protocol_source_has_no_real_scientific_stack_imports():
    forbidden = {"mne", "moabb", "torch", "braindecode", "skorch"}
    for module in (registry, protocol, synthetic):
        tree = ast.parse(Path(module.__file__).read_text())
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not imported & forbidden


def test_event_contract_is_half_open_480_samples_for_all_datasets():
    rows = _rows("event_window_registry.csv")
    assert len(rows) == 3
    assert {row["endpoint_semantics"] for row in rows} == {"half_open_[0.0,3.0)"}
    assert {row["resample_sfreq_hz"] for row in rows} == {"160"}
    assert {row["expected_n_times"] for row in rows} == {"480"}


def test_fixed_zoo_arithmetic_is_exact():
    units = protocol.candidate_units()
    assert len(units) == 1944
    assert len({row["unit_id"] for row in units}) == 1944
    assert sum(row["canary_subset"] for row in units) == 243
    assert sum(row["regime"] == "ERM" for row in units) == 24
    assert sum(row["regime"] == "OACI" for row in units) == 960
    assert sum(row["regime"] == "SRC" for row in units) == 960


def test_target_context_arithmetic_is_exact():
    rows = _rows("target_context_arithmetic.csv")
    assert sum(int(row["contexts"]) for row in rows) == 944
    assert sum(int(row["candidate_context_evaluations"]) for row in rows) == 76464
    assert {row["candidates_per_context"] for row in rows} == {"81"}


def test_selector_registry_replays_14_locked_controls_and_methods():
    rows = _rows("selector_registry_replay.csv")
    assert len(rows) == 14
    assert {row["method_id"] for row in rows} == {
        "B0", "B1", "B2", "B3", "B4O", "B4S", "B5", "S1", *protocol.PRIMARY_ZERO_METHODS,
    }
    assert all(row["formula_retuned_for_C84"] == "0" for row in rows)


def test_budget_registry_keeps_physionet_on_common_grid_only():
    rows = _rows("common_and_extended_budget_registry.csv")
    physio = [row for row in rows if row["dataset"] == "PhysionetMI"]
    assert [row["budget"] for row in physio] == ["1", "2", "4", "8", "FULL"]
    extended = [row for row in rows if row["grid"] == "Lee_Cho_secondary"]
    assert {(row["dataset"], row["budget"]) for row in extended} == {
        ("Lee2019_MI", "16"), ("Lee2019_MI", "32"),
        ("Cho2017", "16"), ("Cho2017", "32"),
    }


def test_same_method_cross_dataset_taxonomy():
    empty = {dataset: set() for dataset in synthetic.DATASETS}
    same = {dataset: {"U13"} for dataset in synthetic.DATASETS}
    assert synthetic.classify_cross_dataset(q1_methods=same, q2_methods=same)["gate"] == synthetic.GATE_A
    assert synthetic.classify_cross_dataset(q1_methods=same, q2_methods=empty)["gate"] == synthetic.GATE_B
    assert synthetic.classify_cross_dataset(q1_methods=empty, q2_methods=empty)["gate"] == synthetic.GATE_C
    different = {"Lee2019_MI": {"U13"}, "Cho2017": {"U7"}, "PhysionetMI": {"U14"}}
    assert synthetic.classify_cross_dataset(q1_methods=different, q2_methods=empty)["gate"] == synthetic.GATE_D
    assert synthetic.classify_cross_dataset(q1_methods=same, q2_methods=same, blocker=True)["gate"] == synthetic.GATE_E


def test_label_budget_closure_and_cross_dataset_tags():
    assert synthetic.budget_star([False, True, True, True, True]) == 2
    assert synthetic.budget_star([True, False, True, True, True]) == 4
    assert synthetic.budget_star([False, False, False, False, False]) is None
    assert synthetic.classify_label_frontier({dataset: 1 for dataset in synthetic.DATASETS}) == synthetic.LABEL_L1
    assert synthetic.classify_label_frontier({dataset: "FULL" for dataset in synthetic.DATASETS}) == synthetic.LABEL_L2
    assert synthetic.classify_label_frontier({"Lee2019_MI": 1, "Cho2017": 4, "PhysionetMI": "FULL"}) == synthetic.LABEL_L3
    assert synthetic.classify_label_frontier({"Lee2019_MI": 1, "Cho2017": None, "PhysionetMI": 1}) == synthetic.LABEL_L4


def test_target_cluster_maxT_uses_target_count_not_trial_count():
    effects = np.full((20, 2), 0.20)
    result = synthetic.target_cluster_maxT(effects, null_margin=0.05, label="unit-test")
    assert result["scientific_clusters"].tolist() == [20]
    assert result["draws"].tolist() == [65536]
    assert np.all(result["pvalue"] <= 0.05)


def test_panel_seed_aggregation_and_q1_q2_gates():
    q1_cells = np.full((22, 2, 4), 0.20)
    q2_cells = np.zeros((22, 2, 4))
    assert synthetic.q1_decision(q1_cells, family_p=0.01)["pass"] is True
    assert synthetic.q2_decision(q2_cells, family_p=0.01, simultaneous_upper=0.01)["pass"] is True
    q1_cells[:, :, 2:] = -0.01
    assert synthetic.q1_decision(q1_cells, family_p=0.01)["pass"] is False


def test_atomic_failure_never_exposes_final_directory(tmp_path):
    final = tmp_path / "final"
    with pytest.raises(RuntimeError, match="injected"):
        synthetic.atomic_manifest_write(final, {"one.csv": b"x\n", "two.json": b"{}\n"}, inject_failure=True)
    assert not final.exists()


def test_synthetic_S0_through_S14_and_auxiliary_checks_all_pass():
    rows = _rows("synthetic_calibration.csv")
    assert len(rows) == 20
    assert {f"S{index}" for index in range(15)} <= {row["scenario"] for row in rows}
    assert all(row["passed"] == "1" for row in rows)
    assert all(row["real_EEG_arrays_loaded"] == "0" and row["real_labels_read"] == "0" for row in rows)


def test_resource_estimates_are_below_hard_envelopes():
    rows = _rows("resource_estimate.csv")
    assert all(row["within_envelope"] == "1" for row in rows)
    by_resource = {row["resource"]: row for row in rows}
    assert float(by_resource["GPU_phase_hours_safety"]["estimate"]) <= 250.0
    assert float(by_resource["combined_external_payload"]["estimate"]) <= 2048.0


def test_risk_and_failure_ledgers_expose_channel_blocker():
    risks = _rows("risk_register.csv")
    open_risks = [row for row in risks if row["status"] == "OPEN_BLOCKER"]
    assert len(open_risks) == 1
    assert open_risks[0]["risk"] == "missing_common_channel_silently_dropped"
    failures = _rows("failure_reason_ledger.csv")
    assert len(failures) == 1
    assert failures[0]["status"] == "OPEN"
    assert failures[0]["real_EEG_or_label_access"] == "0"


def test_C84P_historical_commit_had_no_lock_and_C84R_adds_only_C84C():
    historical = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "df95f1375f1883dd706a63f65ee9b6313fa1a779", "oaci/reports"],
        cwd=protocol.REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert not [path for path in historical if "C84" in path and "EXECUTION_LOCK" in path]
    assert {path.name for path in protocol.REPORT_DIR.glob("C84*EXECUTION_LOCK*.json")} == {
        "C84C_EXECUTION_LOCK.json"
    }
    forbidden = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".pkl", ".fif", ".edf", ".gdf", ".mat"}
    c84_paths = [path for path in (protocol.REPO_ROOT / "oaci").rglob("*C84*") if path.is_file()]
    assert not any(path.suffix.lower() in forbidden for path in c84_paths)


def test_final_gate_is_reconciliation_required():
    generated = protocol.generate()
    assert generated["gate"] == "C84_DATASET_CHANNEL_EVENT_RESOURCE_OR_PROTOCOL_RECONCILIATION_REQUIRED"
    assert generated["real_EEG_arrays_loaded"] == 0
    assert generated["real_labels_read"] == 0
