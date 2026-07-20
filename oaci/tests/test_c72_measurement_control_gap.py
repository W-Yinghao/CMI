"""C72 measurement-to-control mechanism and artifact tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

from oaci.conditioned_ceiling_coverage import c72_measurement_control_gap as c72
from oaci.conditioned_ceiling_coverage import synthetic_rank_gauge_generator as synth
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C72_MEASUREMENT_CONTROL_GAP.json"
TABLE_DIR = "oaci/reports/c72_tables"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(name: str) -> list[dict]:
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary() -> dict:
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c72_protocol_taxonomy_and_static_boundary():
    assert c72._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    protocol = json.load(open(c72.PROTOCOL_JSON))
    assert protocol["schema_version"] == "c72_measurement_control_gap_protocol_v1"
    assert protocol["protocol_lock_source_commit"] == "4c6081d88ed314c90a060a5a415262e504a95459"
    assert protocol["parent_c71_protocol_sha256"] == "984f4ca5a2ab57e679cbb07ab42c6ac0ccb1d937655e6fdab035b670ff19e800"
    assert open(c72.PROTOCOL_SHA).read().strip() == _sha256(c72.PROTOCOL_JSON)
    assert protocol["data_roles"]["T2"][0] == "exploratory mechanism estimation"
    assert "no tuning from T3-HO intervention outcomes" in protocol["data_roles"]["T3_HO"]
    assert protocol["execution_boundary"]["new_forward_passes"] is False
    assert protocol["execution_boundary"]["training_or_parameter_updates"] is False
    assert protocol["known_before_lock"]["target_population_claim_allowed"] is False
    assert set(c72.PRIMARY_DECISIONS) == {
        "C72-A_extreme_order_geometry_explains_measurement_control_gap",
        "C72-B_residual_candidate_specific_gauge_dominates_gap",
        "C72-C_finite_label_noise_dominates_gap",
        "C72-D_construction_utility_mismatch_dominates_gap",
        "C72-E_mixed_noise_margin_gauge_mechanism",
        "C72-F_C71_measurement_control_gap_not_mechanistically_resolved",
        "C72-G_rank_gauge_model_contradicted_by_intervention",
    }
    assert set(c72.FINAL_GATES) == {
        "MEASUREMENT_CONTROL_GAP_MECHANISM_RESOLVED",
        "MEASUREMENT_CONTROL_GAP_PARTIALLY_RESOLVED",
        "RANK_GAUGE_INTERVENTION_CONTRADICTED",
        "INTERVENTION_ANALYSIS_BLOCKED_BY_CACHE_FIELDS",
        "PROTOCOL_OR_MASKING_REPAIR_REQUIRED",
        "INDEPENDENT_TARGET_REPLICATION_NOW_JUSTIFIED",
    }


def test_c72_exact_finite_population_pair_distribution_and_identities():
    labels = np.repeat(np.arange(2), 4)
    pool = np.arange(8)
    classes = np.asarray([0, 1])
    err, disagreement, n = c72.exact_stratified_pair_order_error(
        np.ones(8, dtype=int), labels, pool, classes, 2
    )
    assert err == 0.0
    assert disagreement == 1.0
    assert n == 8
    err_bad, _, _ = c72.exact_stratified_pair_order_error(
        -np.ones(8, dtype=int), labels, pool, classes, 2
    )
    assert err_bad == 1.0
    logits = np.asarray([[1.0, -1.0, 0.5], [-2.0, 3.0, 1.0]])
    base = c72._softmax(logits)
    assert np.allclose(base, c72._softmax(logits + 7.25), atol=1e-12, rtol=0.0)
    utility = np.asarray([0.1, 0.4, 0.2])
    assert np.array_equal(np.argsort(utility), np.argsort(utility - 12.0))
    registry = c72._source_score_registry()
    assert registry
    by_checkpoint = {}
    for (checkpoint_id, regime), values in registry.items():
        by_checkpoint.setdefault(checkpoint_id, {})[regime] = values["score"]
    assert all(len(regimes) == 7 for regimes in by_checkpoint.values())
    assert any(len(set(regimes.values())) > 1 for regimes in by_checkpoint.values())


def test_c72_synthetic_generator_has_common_offset_identity_and_candidate_crossings():
    rng = np.random.default_rng(72)
    instance = synth.generate_trial_instance(
        candidate_count=32,
        rank_snr=2.0,
        gauge_sd=0.5,
        gauge_shape="gaussian",
        label_budget=16,
        outcome_type="multiclass4",
        rng=rng,
    )
    assert instance.paired_correctness.shape == (32, 1024)
    assert len(instance.labels) == 1024
    assert np.array_equal(
        np.argsort(instance.latent_utility),
        np.argsort(instance.latent_utility + instance.common_offset),
    )
    draws = [
        synth.aggregate_instance_metrics(
            candidate_count=64,
            rank_snr=2.0,
            gauge_sd=0.5,
            gauge_shape="skewed",
            label_budget=64,
            outcome_type="multiclass4",
            rng=rng,
        )
        for _ in range(32)
    ]
    assert {d["common_offset_rank_flip"] for d in draws} == {0.0}
    assert np.mean([d["candidate_specific_rank_flip"] for d in draws]) > 0


def test_c72_required_artifacts_provenance_and_no_execution():
    d = _summary()
    assert d["milestone"] == "C72"
    assert d["diagnostic_only_non_deployable"] is True
    assert d["protocol_commit"] == "11534dc"
    assert d["protocol_sha256"] == _sha256(c72.PROTOCOL_JSON)
    assert d["forward_passes"] == 0
    assert d["reinference_runs"] == 0
    assert d["training_attempted"] == 0
    assert d["gpu_used"] == 0
    assert d["bnci004_used"] == 0
    assert d["reserved_seeds_used"] == 0
    assert d["selector_artifact_emitted"] == 0
    assert d["checkpoint_recommendation_artifact_emitted"] == 0
    assert d["selected_checkpoint_ids_emitted"] == 0
    assert d["key_numbers"]["T2_units"] == 216
    assert d["key_numbers"]["T3_HO_units"] == 1052
    assert d["key_numbers"]["T3_HO_rows"] == 605952
    for name in c72.TABLE_SPECS:
        assert os.path.exists(os.path.join(TABLE_DIR, name)), name
    assert os.path.exists(c72.MAIN_REPORT)
    assert os.path.exists(c72.THEORY_NOTE)
    assert os.path.exists(c72.RED_REPORT)

    provenance = {r["mode"]: r for r in _rows("c71_authorization_provenance.csv")}
    assert provenance["no_auth_readiness"]["forward_or_reinference_executed"] == "0"
    assert provenance["authorized_T3_HO"]["cache_rows"] == "605952"
    assert provenance["C72"]["forward_or_reinference_executed"] == "0"
    assert {r["passed"] for r in _rows("c71_protocol_hash_replay.csv")} == {"1"}
    cache = {r["stage"]: r for r in _rows("c71_cache_identity_replay.csv")}
    assert cache["T2"]["expected_units"] == "216"
    assert cache["T3-HO"]["expected_units"] == "1052"
    assert {r["passed"] for r in cache.values()} == {"1"}
    assert {r["passed"] for r in _rows("c71_physical_view_replay.csv")} == {"1"}


def test_c72_scientific_contracts_and_information_availability():
    identities = {r["intervention"]: r for r in _rows("intervention_identity.csv")}
    assert identities["I1_utility_common_offset"]["rank_flips"] == "0"
    assert identities["I1_utility_common_offset"]["passed"] == "1"
    assert identities["I2_all_class_logit_scalar"]["rank_flips"] == "0"
    assert identities["I2_all_class_logit_scalar"]["passed"] == "1"
    calibration = _rows("intervention_calibration.csv")
    assert calibration
    assert {r["stage"] for r in calibration} == {"T2_calibration"}
    assert {r["T3_outcomes_used_for_selection"] for r in calibration} == {"0"}
    for family in {r["intervention"] for r in calibration}:
        assert sum(int(r["selected_by_T2"]) for r in calibration if r["intervention"] == family) == 1
    locked = _rows("common_vs_candidate_specific_intervention.csv")
    assert {r["T3_tuned"] for r in locked} == {"0"}
    assert all(float(r["random_top1_base_rate"]) > 0 for r in locked)

    inventory = {r["intervention"]: r for r in _rows("intervention_inventory.csv")}
    assert all(inventory[f"I{i}"]["evaluation_labels_fit"] == "0" for i in range(6))
    assert inventory["I6"]["evaluation_labels_fit"] == "1"
    assert inventory["I6"]["status"] == "executed_after_primary_freeze"
    assert inventory["I7"]["status"] == "unsupported_cache_fields"
    features = {r["feature_family"]: r for r in _rows("feature_availability_ledger.csv")}
    assert features["strict_source_domain_trial_logits_probabilities"]["available"] == "0"
    assert features["frozen_source_checkpoint_score"]["strict_source_trial_feature"] == "0"
    assert features["representation_or_Wdotz"]["available"] == "0"
    assert features["representation_or_Wdotz"]["status"] == "representation_intervention_supported=false"

    decomposition = _rows("measurement_control_gap_decomposition.csv")
    assert {r["component"] for r in decomposition} == {
        "finite_label_noise",
        "construction_utility_mismatch",
        "extreme_order_localization",
        "residual_candidate_specific_gauge",
    }
    assert {r["budget"] for r in decomposition} == {"8", "64", c72.FULL_BUDGET}
    for budget in {"8", "64", c72.FULL_BUDGET}:
        rows = [r for r in decomposition if r["budget"] == budget]
        assert abs(sum(float(r["shapley_top1_gain"]) for r in rows) - float(rows[0]["observed_control_gap"])) < 1e-8
    assert len(_rows("decomposition_order_sensitivity.csv")) == 3 * 24 * 4
    assert _rows("finite_population_best_arm_bound.csv")
    assert {r["class_stratified_without_replacement"] for r in _rows("finite_population_best_arm_bound.csv")} == {"1"}
    assert {r["row_iid_used"] for r in _rows("hierarchical_inference_summary.csv")} == {"0"}


def test_c72_synthetic_hierarchical_red_team_and_artifact_hygiene():
    synthetic = _rows("synthetic_phase_diagram.csv")
    assert len(synthetic) == 2592
    assert any(r["high_reliability_poor_top1"] == "1" for r in synthetic)
    assert {r["common_offset_rank_flip_rate"] for r in synthetic} == {"0.0"}
    assert {r["identity_false_positive_passed"] for r in _rows("synthetic_false_positive_control.csv")} == {"1"}

    risks = {r["risk_id"]: r for r in _rows("risk_register.csv")}
    assert set(risks) == set(c72.RISK_ROWS)
    assert {r["blocking"] for r in risks.values()} == {"0"}
    hypotheses = _rows("primary_hypothesis_summary.csv")
    assert len(hypotheses) == 6
    assert all(r["holm_p"] != "" for r in hypotheses)
    assert {r["permutations"] for r in hypotheses[2:]} == {"4999"}

    red = {r["gate"]: r for r in _rows("red_team_failure_ledger.csv")}
    assert red
    for gate, row in red.items():
        if gate != "tests_green":
            assert row["failed"] == "0", gate
    test_statuses = {r["status"] for r in _rows("test_command_manifest.csv")}
    if test_statuses == {"green"}:
        assert red["tests_green"]["failed"] == "0"
        assert _summary()["decision"]["red_team_failure_count"] == 0
    else:
        assert red["tests_green"]["failed"] == "1"

    forbidden = _rows("forbidden_claim_scan.csv")
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    assert {r["passed"] for r in forbidden} == {"1"}
    large = _rows("large_artifact_scan.csv")
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}
    manifest = _rows("artifact_manifest.csv")
    assert manifest
    for row in manifest:
        assert os.path.exists(row["path"])
        assert _sha256(row["path"]) == row["sha256"]

    # No raw checkpoint hash may be copied into any committed C72 artifact.
    checkpoint_ids = {r["checkpoint_id"] for r in c72._read_csv("oaci/reports/c65_tables/frozen_universe_checkpoint_map.csv")}
    artifact_text = "\n".join(open(row["path"], errors="ignore").read() for row in manifest)
    assert not any(checkpoint_id in artifact_text for checkpoint_id in checkpoint_ids)
