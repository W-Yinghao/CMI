from __future__ import annotations

import ast
from pathlib import Path

import pytest

from oaci.conditioned_ceiling_coverage import c78f_full_seed3_field as c78f
from oaci.conditioned_ceiling_coverage import c78f_instrument
from oaci.conditioned_ceiling_coverage import c78f_runtime


def test_wave_assignment_is_deterministic_disjoint_and_balanced():
    assert c78f.wave_targets() == {"A": (8, 9, 3, 6), "B": (5, 2, 7, 1)}
    assert set(c78f.wave_targets()["A"]).isdisjoint(c78f.wave_targets()["B"])
    assert set(c78f.wave_targets()["A"] + c78f.wave_targets()["B"]) == set(c78f.TARGETS)


def test_remaining_unit_manifest_exact_counts():
    rows = c78f.remaining_unit_manifest()
    assert len(rows) == 1296
    assert len({row["unit_id"] for row in rows}) == 1296
    assert {regime: sum(row["regime"] == regime for row in rows) for regime in c78f.REGIMES} == {
        "ERM": 16,
        "OACI": 640,
        "SRC": 640,
    }


def test_full_unit_manifest_includes_exact_parent_canary():
    rows = c78f.full_unit_manifest()
    assert len(rows) == 1458
    assert len({row["unit_id"] for row in rows}) == 1458
    assert sum(row["target"] == 4 for row in rows) == 162


def test_training_phase_manifest_is_48_phases():
    rows = c78f.training_phase_manifest()
    assert len(rows) == 48
    assert all(row["target_outcome_blind"] == 1 for row in rows)
    assert sum(row["regime"] == "ERM" for row in rows) == 16
    assert sum(row["regime"] == "OACI" for row in rows) == 16
    assert sum(row["regime"] == "SRC" for row in rows) == 16


def test_row_arithmetic():
    assert c78f.EXPECTED_SOURCE_ROWS == 5_971_968
    assert c78f.EXPECTED_TARGET_ROWS == 746_496
    assert c78f.FULL_SOURCE_ROWS == 6_718_464
    assert c78f.FULL_TARGET_ROWS == 839_808


def test_direct_authorization_replaces_magic_token():
    assert c78f.AUTHORIZATION_MODE == "direct_explicit_user_authorization"
    assert len(c78f.AUTHORIZATION_EVIDENCE_SHA256) == 64
    source = Path(c78f.__file__).read_text()
    assert "authorization_token_exact" not in source


def test_slurm_execution_has_no_token_argument():
    for name in (
        "oaci/slurm_c78f_train_oaci.sh",
        "oaci/slurm_c78f_train_src.sh",
        "oaci/slurm_c78f_instrument_target.sh",
    ):
        source = Path(name).read_text()
        assert "authorization-token" not in source
        assert "TOKEN" not in source


def test_workers_are_scope_guarded_before_heavy_imports():
    source = Path("oaci/conditioned_ceiling_coverage/c78f_train.py").read_text()
    tree = ast.parse(source)
    top_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in top_imports
    assert "mne" not in top_imports
    assert "moabb" not in top_imports
    assert "runtime.require_authorization()" in source


def test_instrumentation_target_fields_exclude_labels():
    assert not (c78f_instrument.TARGET_INPUT_FIELDS & c78f_instrument.FORBIDDEN_TARGET_FIELDS)
    assert not (c78f_instrument.TARGET_OUTPUT_FIELDS & c78f_instrument.FORBIDDEN_TARGET_FIELDS)
    assert "target_class_label" in c78f_instrument.TARGET_LABEL_FIELDS


def test_c78s_protocol_excludes_target4_and_seed4():
    protocol = c78f.build_c78s_protocol()
    assert protocol["data_roles"]["primary_targets"] == list(c78f.TARGETS)
    assert 4 not in protocol["data_roles"]["primary_targets"]
    assert protocol["data_roles"]["seed4"] == "untouched_locked_confirmation_field"
    assert protocol["inference"]["ERM_role"] == "anchor_not_symmetric_trajectory"


def test_c78s_materiality_is_stricter_than_c77_direction_only_effect():
    materiality = c78f.build_c78s_protocol()["materiality"]
    assert materiality["absolute_topk_hit_improvement_min"] > 0.0075
    assert materiality["incremental_R2_min"] == 0.02
    assert materiality["positive_primary_targets_min"] == 6


def test_c78s_multiplicity_and_nulls_are_locked():
    protocol = c78f.build_c78s_protocol()
    assert protocol["multiplicity"]["primary_family"] == "H1_H6_Holm"
    assert protocol["multiplicity"]["association_p_alone_qualifies_control"] is False
    assert "trajectory_preserving_permutation" in protocol["nulls"]


def test_initial_risk_register_covers_required_risks():
    rows = c78f.initial_risk_register()
    names = {row["risk"] for row in rows}
    assert len(rows) == 25
    assert {
        "authorization_bypass",
        "remaining_target_outcome_peeking",
        "wave_A_outcome_based_wave_B_decision",
        "target_label_training_leakage",
        "seed4_contamination",
        "raw_weights_or_cache_in_git",
    } <= names
    assert not any(row["blocking_open"] for row in rows)


def test_implementation_manifest_is_complete():
    rows = c78f.implementation_manifest()
    assert len(rows) == len(c78f.IMPLEMENTATION_FILES)
    assert all(len(row["sha256"]) == 64 and row["size_bytes"] > 0 for row in rows)


def test_historical_files_exist_and_hash():
    rows = c78f.historical_manifest()
    assert len(rows) == len(c78f.HISTORICAL_PATHS)
    assert all(len(row["current_sha256"]) == 64 for row in rows)


@pytest.mark.parametrize("target", [1, 2, 3, 5, 6, 7, 8, 9])
def test_target_wave_round_trip(target):
    assert c78f.wave_for_target(target) in {"A", "B"}
    assert target in c78f.wave_targets()[c78f.wave_for_target(target)]


def test_target4_is_not_runtime_execution_scope():
    with pytest.raises(ValueError):
        c78f_runtime.require_target(4)


def test_external_payload_root_is_not_repo():
    assert c78f.EXTERNAL_ROOT.is_absolute()
    assert "/projects/EEG-foundation-model/" in str(c78f.EXTERNAL_ROOT)


def test_fixed_cadence_epochs():
    assert len(c78f.OACI_EPOCHS) == 40
    assert c78f.OACI_EPOCHS[0] == 4
    assert c78f.OACI_EPOCHS[-1] == 199
    assert c78f.OACI_EPOCHS == c78f.SRC_EPOCHS


def test_src_contract_remains_negative_control():
    assert c78f.SRC_SMOOTH_TEMPERATURE == 0.1
    assert c78f.SRC_HISTORICAL_COMMIT.startswith("2555b36")


def test_no_final_report_exists_at_protocol_stage():
    # This is the prospective-stage expectation. The test is skipped naturally
    # after finalization so historical regression remains rerunnable.
    report = c78f.REPORT_DIR / "C78F_FULL_SEED3_FIELD.md"
    if report.exists():
        pytest.skip("C78F has already passed red-team and finalized")
    assert not report.exists()
