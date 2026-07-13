from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c81_baseline_comparison as baseline
from oaci.conditioned_ceiling_coverage import c81_synthetic_baseline_benchmark as synthetic


def _rows(name: str):
    with (baseline.TABLE_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_protocol_hash_replays_exactly():
    assert baseline.sha256_file(baseline.PROTOCOL_PATH) == baseline.PROTOCOL_SHA_PATH.read_text().strip()
    assert baseline.PROTOCOL_SHA_PATH.read_text().strip() == "cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee"


def test_protocol_declares_outcome_informed_existing_field_status():
    protocol, _ = baseline.load_protocol()
    status = protocol["epistemic_status"]
    assert status["designed_after_C79_and_C80_outcomes"] is True
    assert status["prospective_to_new_C81_baseline_computations"] is True
    assert status["independent_confirmation"] is False
    assert status["external_validation"] is False


def test_direct_authorization_policy_needs_no_magic_token_or_hash_recital():
    protocol, _ = baseline.load_protocol()
    authorization = protocol["authorization"]
    assert authorization["C81E_requires_direct_PI_authorization"] is True
    assert authorization["magic_token_required"] is False
    assert authorization["PI_hash_recital_required"] is False
    assert authorization["binding_policy_commit"] == "3d9dd76"


def test_C80_historical_and_operative_locks_are_both_preserved():
    protocol, _ = baseline.load_protocol()
    objects = protocol["accepted_C80_operating_objects"]
    assert objects["historical_complete_lock_commit"].startswith("f19acd8")
    assert objects["operative_lock_commit"].startswith("0797599")
    assert objects["repair_changed_science"] is False
    assert objects["repair_evaluation_label_reads"] == 0


def test_candidate_universe_arithmetic_is_exact():
    protocol, _ = baseline.load_protocol()
    universe = protocol["candidate_universe"]
    assert universe["primary_targets"] == [1, 2, 3, 5, 6, 7, 8, 9]
    assert universe["candidates_per_context"] == 81
    assert universe["contexts"] == 32
    assert universe["primary_candidates"] == 2592


def test_method_registry_has_34_unique_locked_methods():
    registry = baseline.load_method_registry()
    ids = [row["id"] for row in registry["methods"]]
    assert len(ids) == len(set(ids)) == 34


def test_unavailable_training_dependent_methods_are_excluded_before_hash():
    methods = {row["id"]: row for row in baseline.load_method_registry()["methods"]}
    for method in ("S3", "S4", "U8", "U9", "U10"):
        assert methods[method]["status"] == "EXCLUDED_INPUT_UNAVAILABLE"
    assert methods["U8"]["retraining"] is True


def test_primary_family_representatives_are_frozen_and_R9_is_unavailable():
    protocol, _ = baseline.load_protocol()
    representatives = protocol["primary_family_representatives"]
    assert representatives["R2"] == "S1_source_validation_balanced_accuracy"
    assert representatives["R9"].startswith("UNAVAILABLE_")
    assert representatives["R10"] == "L1_frozen_Q0_B1"


def test_view_matrix_never_reaches_same_label_oracle():
    rows = _rows("baseline_input_view_matrix.csv")
    assert all(row["same_label_oracle_view"] == "0" for row in rows)
    assert all(row["oracle_reachable"] == "0" for row in rows)


def test_target4_is_mechanically_outside_primary_contract():
    protocol, _ = baseline.load_protocol()
    assert 4 not in protocol["candidate_universe"]["primary_targets"]
    assert protocol["scope"]["target4_primary"] is False
    assert protocol["physical_view_contract"]["target4_excluded_from_primary"] is True


def test_score_directions_cannot_flip_after_outcomes():
    rows = _rows("score_direction_registry.csv")
    assert all(row["may_flip_after_outcome"] == "0" for row in rows)


def test_source_summary_is_perfect_for_perfect_predictions():
    labels = np.tile(np.arange(4), 4)
    domains = np.repeat([10, 11], 8)
    probabilities = np.eye(4)[labels] * 0.97 + 0.03 / 4
    summary = baseline.source_summary(probabilities, labels, domains)
    assert summary["bacc"] == 1.0
    assert summary["negative_nll"] < 0.0


def test_ATC_orders_high_target_confidence_above_low_confidence():
    labels = np.tile(np.arange(4), 4)
    domains = np.repeat([1, 2], 8)
    source = np.eye(4)[labels] * 0.8 + 0.2 / 4
    source[[0, 4, 8, 12]] = np.roll(source[[0, 4, 8, 12]], 1, axis=1)
    high = np.eye(4)[labels[:8]] * 0.9 + 0.1 / 4
    low = np.full((8, 4), 0.25)
    assert baseline.score_atc(source, labels, domains, high) > baseline.score_atc(source, labels, domains, low)


def test_DoC_orders_high_target_confidence_above_low_confidence():
    labels = np.tile(np.arange(4), 4)
    domains = np.repeat([1, 2], 8)
    source = np.eye(4)[labels] * 0.8 + 0.2 / 4
    high = np.eye(4)[labels[:8]] * 0.9 + 0.1 / 4
    low = np.full((8, 4), 0.25)
    assert baseline.score_doc(source, labels, domains, high) > baseline.score_doc(source, labels, domains, low)


def test_nuclear_norm_rewards_confident_dispersed_predictions():
    confident = np.eye(4)[np.tile(np.arange(4), 8)]
    uniform = np.full((32, 4), 0.25)
    assert baseline.score_nuclear_norm(confident) > baseline.score_nuclear_norm(uniform)


def test_MaNo_is_deterministic_and_bounded():
    logits = np.arange(48, dtype=float).reshape(12, 4) / 10.0
    score = baseline.score_mano(logits)
    assert score == baseline.score_mano(logits)
    assert 0.0 < score <= 1.0


def test_COT_enforces_registered_prior_and_penalizes_imbalance():
    probability = np.tile([0.90, 0.05, 0.03, 0.02], (12, 1))
    source_prior = np.full(4, 0.25)
    score = baseline.score_cot(probability, source_prior)
    assert score < np.mean(np.max(probability, axis=1))
    assert len(baseline.cot_matched_costs(probability, source_prior)) == 12


def test_COTT_score_is_a_fraction():
    labels = np.tile(np.arange(4), 4)
    domains = np.repeat([1, 2], 8)
    source = np.eye(4)[labels] * 0.8 + 0.2 / 4
    target = np.eye(4)[np.tile(np.arange(4), 3)] * 0.7 + 0.3 / 4
    score = baseline.score_cott(source, labels, domains, target, np.full(4, 0.25))
    assert 0.0 <= score <= 1.0


def test_SND_identical_predictions_have_dense_maximal_neighborhoods():
    dense = np.tile([0.7, 0.1, 0.1, 0.1], (12, 1))
    sparse = np.eye(4)[np.tile(np.arange(4), 3)]
    assert baseline.score_snd(dense) > baseline.score_snd(sparse)


def test_ALine_recovers_shared_linear_agreement_order():
    rng = np.random.default_rng(91)
    latent = rng.integers(0, 4, size=(81, 64))
    target = latent.copy()
    source_bacc = np.linspace(0.2, 0.9, 81)
    score, diagnostics = baseline.score_aline(latent, target, source_bacc)
    assert np.corrcoef(score, source_bacc)[0, 1] > 0.99
    assert diagnostics["pair_count"] == 3240


def test_all_selection_paths_execute_unconditionally_on_fixture():
    rng = np.random.default_rng(7)
    labels = np.tile(np.arange(4), 4)
    domains = np.repeat([1, 2], 8)
    source = rng.dirichlet(np.ones(4), size=(81, 16))
    target_logits = rng.normal(size=(81, 12, 4))
    regimes = np.asarray(["ERM"] + ["OACI"] * 40 + ["SRC"] * 40)
    orders = np.asarray([0] + list(range(1, 41)) + list(range(1, 41)))
    scores, diagnostics = baseline.score_context(source, labels, domains, target_logits, regimes, orders)
    assert set(scores) == set(baseline.SELECTION_METHODS)
    assert all(value.shape == (81,) for value in scores.values())
    assert diagnostics["pair_count"] == 3240


def test_candidate_tie_rule_is_smallest_canonical_index():
    assert baseline.descending_order(np.zeros(81))[0] == 0


def test_standardized_regret_has_zero_best_and_one_worst():
    utility = np.linspace(0.0, 1.0, 81)
    assert baseline.standardized_regret(utility, 80) == 0.0
    assert baseline.standardized_regret(utility, 0) == 1.0


def test_exact_maxT_rejects_large_consistent_family():
    effects = np.full((8, 12), 0.20)
    result = baseline.exact_signflip_maxT(effects, margin=0.05)
    assert np.all(result["pvalue"] <= 0.05)


def test_taxonomy_blocker_has_highest_priority():
    assert baseline.classify_taxonomy(
        blocker=True, seed3_category="A", seed4_category="A", loto_preserved=16,
    ).startswith("C81-E")


def test_taxonomy_heterogeneity_has_second_priority():
    assert baseline.classify_taxonomy(
        blocker=False, seed3_category="A", seed4_category="B", loto_preserved=16,
    ).startswith("C81-D")
    assert baseline.classify_taxonomy(
        blocker=False, seed3_category="A", seed4_category="A", loto_preserved=11,
    ).startswith("C81-D")


@pytest.mark.parametrize("category,label", [("A", "C81-A"), ("B", "C81-B"), ("C", "C81-C")])
def test_taxonomy_stable_categories_are_exhaustive(category, label):
    assert baseline.classify_taxonomy(
        blocker=False, seed3_category=category, seed4_category=category, loto_preserved=12,
    ).startswith(label)


def test_C80_frontier_replay_has_all_14_seed_budget_rows():
    rows = _rows("c80e_frontier_replay.csv")
    assert len(rows) == 14
    b1 = {(int(row["seed"]), row["budget"]): row for row in rows}
    assert float(b1[(3, "1")]["expected_standardized_regret"]) == pytest.approx(0.35338318633180843)
    assert float(b1[(4, "1")]["expected_standardized_regret"]) == pytest.approx(0.3737050812998979)


def test_C80_LOTO_replay_retains_all_16_sensitive_panels():
    rows = _rows("c80e_loto_stability_replay.csv")
    assert len(rows) == 16
    assert all(row["classification_changed"] == "1" for row in rows)


def test_synthetic_baseline_scenarios_all_pass():
    rows = _rows("synthetic_baseline_calibration.csv")
    assert len(rows) == 13
    assert all(row["passed"] == "1" for row in rows)


def test_synthetic_familywise_and_dependence_calibration_pass():
    assert all(row["passed"] == "1" for row in _rows("synthetic_familywise_error.csv"))
    assert all(row["passed"] == "1" for row in _rows("synthetic_pair_dependence_calibration.csv"))


def test_synthetic_noninferiority_calibration_passes():
    rows = _rows("synthetic_noninferiority_calibration.csv")
    assert len(rows) == 3
    assert all(row["passed"] == "1" for row in rows)


def test_protocol_audit_records_zero_real_statistics_and_label_reads():
    audit = baseline.protocol_audit()
    assert audit["real_baseline_statistics"] == 0
    assert audit["evaluation_label_reads"] == 0
    assert audit["same_label_oracle_accesses"] == 0


def test_run_real_fails_closed_without_direct_C81E_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "AUTHORIZATION_PATH", tmp_path / "absent_authorization.json")
    with pytest.raises(RuntimeError):
        baseline.run_real()


def test_scope_specific_lock_replays_and_binds_real_adapter():
    lock, lock_sha = baseline.load_execution_lock()
    assert lock_sha == baseline.LOCK_SHA_PATH.read_text().split()[0]
    assert lock["repair_protocol"]["sha256"] == "ba0434b4ea7965691dafaf506547af64f851c57bdca330a0a5c88e4fa7ba1b15"
    assert lock["selection_descriptor_repair_protocol"]["sha256"] == (
        "2acf6ecc179c739f73845d430f9eac9e9e83a83015370b1125dbe447b8b59272"
    )
    assert lock["runtime"]["entrypoint"].endswith("c81_baseline_comparison run-real")
    assert lock["runtime"]["selection_manifest_freeze_required"] is True
    assert lock["runtime"]["evaluation_requires_selection_hash_replay"] is True


def test_repaired_lock_implementation_and_correction_commits_are_reachable():
    lock, _ = baseline.load_execution_lock()
    for item in [*lock["implementation"], lock["provenance_correction"]]:
        baseline._git("cat-file", "-e", f"{item['commit']}^{{commit}}")


def test_source_loader_accepts_registered_superset_schema(tmp_path):
    arrays = {
        "probabilities": np.full((3, 4), 0.25),
        "source_class_label": np.asarray([0, 1, 2], dtype=np.int16),
        "source_domain_id": np.asarray([10, 10, 11], dtype=np.int16),
        "logits": np.zeros((3, 4), dtype=np.float64),
    }
    shard = baseline.c74_cache.write_content_addressed_npz(
        tmp_path / "payload", "strict_source_trial", arrays,
    )
    unit_manifest = {
        "unit_id": "unit-1",
        "target": 1,
        "shards": [shard],
    }
    unit_path = tmp_path / "unit.json"
    unit_path.write_text(json.dumps(unit_manifest, sort_keys=True))
    instrumentation_path = tmp_path / "instrumentation.json"
    instrumentation_path.write_text(json.dumps({
        "target": 1,
        "all_gates_passed": True,
        "units": [{
            "unit_id": "unit-1",
            "path": str(unit_path),
            "sha256": baseline.sha256_file(unit_path),
        }],
    }, sort_keys=True))
    route = {"views": {"1": {"instrumentation_manifest": str(instrumentation_path)}}}

    probabilities, labels, domains = baseline._load_source_context(
        route, 1, np.asarray(["unit-1"]),
    )

    assert probabilities.shape == (1, 3, 4)
    assert labels.tolist() == [0, 1, 2]
    assert domains.tolist() == [10, 10, 11]


def test_selection_descriptor_accepts_registered_mixed_dimension_schema(tmp_path):
    arrays = {
        "cell_seed": np.full(32, 3, dtype=np.int16),
        "cell_target": np.tile(np.asarray([1, 2, 3, 5, 6, 7, 8, 9]), 4),
        "cell_level": np.tile(np.asarray([0, 1]), 16),
        "candidate_global_indices": np.tile(np.arange(81), (32, 1)),
        "method_ids": np.asarray(baseline.SELECTION_METHODS),
        "scores": np.zeros((32, 19, 81)),
        "selected_top10": np.zeros((32, 19, 10), dtype=np.int16),
        "aline_slope": np.zeros(32),
        "aline_intercept": np.zeros(32),
        "aline_pair_R2": np.zeros(32),
    }
    descriptor = baseline.c74_cache.write_content_addressed_npz(
        tmp_path / "payload", "c81_locked_baseline_selection", arrays,
    )
    assert baseline._verify_selection_descriptor(descriptor) == Path(descriptor["path"])


def test_selection_descriptor_rejects_unregistered_shape(tmp_path):
    arrays = {
        name: np.zeros(shape, dtype="<U8" if name == "method_ids" else np.float64)
        for name, shape in baseline.SELECTION_ARRAY_SHAPES.items()
    }
    arrays["method_ids"] = np.zeros(18, dtype="<U8")
    descriptor = baseline.c74_cache.write_content_addressed_npz(
        tmp_path / "payload", "c81_locked_baseline_selection", arrays,
    )
    with pytest.raises(RuntimeError, match="shape drift"):
        baseline._verify_selection_descriptor(descriptor)


def test_pre_execution_red_team_passes_all_checks():
    rows = _rows("pre_execution_red_team.csv")
    assert len(rows) == 43
    assert all(row["passed"] == "1" and row["blocking"] == "0" for row in rows)


def test_C80_field_view_and_result_artifacts_replay_exactly():
    assert len(_rows("c80e_field_view_manifest_replay.csv")) == 11
    assert all(row["pass"] == "1" for row in _rows("c80e_field_view_manifest_replay.csv"))
    assert len(_rows("c80e_result_artifact_hash_replay.csv")) == 22
    assert all(row["pass"] == "1" for row in _rows("c80e_result_artifact_hash_replay.csv"))


@pytest.mark.parametrize("module", [baseline, synthetic])
def test_C81P_modules_import_no_EEG_GPU_or_training_packages(module):
    tree = ast.parse(Path(module.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not imported.intersection({"torch", "mne", "moabb"})


def test_risk_register_has_no_open_blocker():
    rows = _rows("risk_register.csv")
    assert len(rows) >= 27
    assert all(row["blocking"] == "0" for row in rows)


def test_C81P_readiness_precedes_direct_C81E_authorization():
    readiness = json.loads((baseline.REPORT_DIR / "C81P_PROTOCOL_READINESS.json").read_text())
    authorization = json.loads(baseline.AUTHORIZATION_PATH.read_text())
    assert readiness["protected_state"]["C81E_authorized"] is False
    assert readiness["protected_state"]["real_baseline_statistics"] == 0
    assert authorization["authorization_received"] is True
    assert authorization["protocol_sha256"] == readiness["protocol"]["sha256"]
    assert authorization["binding_history"][0]["analysis_lock_sha256"] == readiness["analysis_lock"]["sha256"]
    assert authorization["analysis_lock_sha256"] != baseline.LOCK_SHA_PATH.read_text().split()[0]
    assert authorization["binding_history"][-1]["status"] == "operative_repaired_lock_directly_reauthorized"
    assert not (baseline.REPORT_DIR / "C81_FROZEN_FIELD_BASELINE_COMPARISON.json").exists()
