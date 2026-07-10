from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c74_cache as cache
from oaci.conditioned_ceiling_coverage import c74_analysis as analysis
from oaci.conditioned_ceiling_coverage import c74_preprocessing_replay as preprocessing_replay
from oaci.conditioned_ceiling_coverage import c74_t2_source_wz_instrumentation as c74


def test_c74_locked_protocol_and_roles_are_intact():
    protocol = cache.load_locked_protocol()
    assert protocol["new_variable_roles"]["T2"]["units"] == 216
    assert protocol["new_variable_roles"]["T3_HO"]["units"] == 1052
    t2, t3 = cache.locked_unit_sets()
    assert len(t2) == 216
    assert len(t3) == 1052
    assert t2.isdisjoint(t3)


def test_c74_stage_manifests_are_balanced_and_t2_only():
    pilot = []
    expansion = []
    for target in range(1, 10):
        p0 = cache.stage_rows("P0_pilot", target)
        p1 = cache.stage_rows("P1_expansion", target)
        assert len(p0) == 6
        assert len(p1) == 18
        assert {int(row["target"]) for row in p0 + p1} == {target}
        pilot.extend(p0)
        expansion.extend(p1)
    assert len({row["checkpoint_id"] for row in pilot}) == 54
    assert len({row["checkpoint_id"] for row in expansion}) == 162
    assert {row["checkpoint_id"] for row in pilot}.isdisjoint(
        {row["checkpoint_id"] for row in expansion}
    )


def test_c74_authorization_is_exact_and_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("C74_AUTHORIZATION_TOKEN", cache.AUTH_TOKEN)
    assert cache.authorization_ok(cache.AUTH_TOKEN)
    assert not cache.authorization_ok("")
    assert not cache.authorization_ok(f" {cache.AUTH_TOKEN}")
    assert not cache.authorization_ok(cache.AUTH_TOKEN + "\n")


def test_c74_invalid_authorization_stops_before_eeg_load(tmp_path):
    with pytest.raises(PermissionError, match="exact CLI authorization"):
        c74.instrument_stage_target(
            stage="P0_pilot", target_id=1, authorization_token="not-authorized",
            datalake_root=str(tmp_path / "missing"), num_threads=1,
        )


def test_c74_deterministic_npz_is_content_addressed(tmp_path):
    arrays = {
        "ids": np.asarray(["a", "b"], dtype="<U1"),
        "values": np.arange(8, dtype=np.float32).reshape(2, 4),
    }
    first = cache.write_content_addressed_npz(tmp_path, "example", arrays)
    second = cache.write_content_addressed_npz(tmp_path, "example", arrays)
    assert first == second
    cache.verify_shard(first, required_fields=set(arrays))
    assert len(list(tmp_path.glob("example_sha256_*.npz"))) == 1


def test_c74_unit_manifest_self_hash_detects_drift(tmp_path):
    payload = cache.self_hashed_manifest({"unit_id": "u1", "shards": []})
    path = tmp_path / "unit_manifest.json"
    cache.atomic_json(path, payload)
    assert cache.verify_unit_manifest(path, rehash_payloads=True)["unit_id"] == "u1"
    drifted = json.loads(path.read_text())
    drifted["unit_id"] = "u2"
    path.write_text(json.dumps(drifted))
    with pytest.raises(RuntimeError, match="self-hash"):
        cache.verify_unit_manifest(path, rehash_payloads=False)


def test_c74_physical_schemas_keep_target_labels_out_of_unlabeled_view():
    assert "target_class_label" in c74.CONSTRUCTION_FIELDS
    assert "target_class_label" in c74.EVALUATION_FIELDS
    assert "target_class_label" in c74.ORACLE_FIELDS
    assert not (c74.TARGET_UNLABELED_FIELDS & c74.FORBIDDEN_TARGET_UNLABELED)
    assert "source_class_label" in c74.SOURCE_FIELDS
    assert "target_class_label" not in c74.SOURCE_FIELDS


def test_c74_runner_source_has_no_environment_authorization_path():
    source = Path(c74.__file__).read_text()
    cache_source = Path(cache.__file__).read_text()
    assert "os.getenv" not in source
    assert "os.environ" not in source
    assert "os.getenv" not in cache_source
    assert "os.environ" not in cache_source


def test_c74_balanced_accuracy_and_ridge_are_deterministic():
    y = np.asarray([0, 0, 1, 1])
    prediction = np.asarray([0, 1, 1, 1])
    assert analysis._balanced_accuracy(y, prediction) == pytest.approx(0.75)

    groups = np.repeat(np.arange(4), 4)
    x = np.column_stack((np.arange(16, dtype=float), np.arange(16, dtype=float) ** 2))
    outcome = 0.2 + 0.01 * x[:, 0]
    first = analysis._ridge_loto_predictions(x, outcome, groups)
    second = analysis._ridge_loto_predictions(x, outcome, groups)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_c74_analysis_never_loads_same_label_oracle_in_primary_feature_path():
    source = Path(analysis.__file__).read_text()
    feature_body = source.split("def _extract_unit_features", 1)[1].split("def _ridge_loto_predictions", 1)[0]
    assert "same_label_oracle" not in feature_body
    counterfactual_body = source.split("def _counterfactuals", 1)[1].split("def _content_and_abi_tables", 1)[0]
    assert '_descriptor(manifest, "same_label_oracle")' not in counterfactual_body
    assert '_load(_descriptor(manifest, "same_label_oracle")' not in counterfactual_body

    allowed = {
        "checkpoint_Wb", "strict_source_trial", "target_unlabeled_representation",
        "target_construction_labels", "target_evaluation_labels",
    }
    fake = {
        "unit_id": "u1", "same_label_oracle_rows": 576,
        "view_isolation": {"oracle_available_to_primary_smoke": False, "passed": True},
        "shards": [{"kind": kind, "path": f"/{kind}"} for kind in sorted(allowed | {"same_label_oracle"})],
    }
    restricted = analysis._restricted_unit_manifest(fake, allowed)
    body = json.dumps(restricted, sort_keys=True)
    assert "same_label_oracle" not in body
    assert "oracle_available" not in body
    assert {item["kind"] for item in restricted["shards"]} == allowed


def test_c74_cross_node_replay_tolerances_are_locked_and_strict():
    assert preprocessing_replay.REPLICATES == ("nodecpu01", "nodecpu02")
    assert preprocessing_replay.INPUT_MAX_ABS_TOLERANCE == 1e-5
    assert preprocessing_replay.INPUT_MEAN_ABS_TOLERANCE == 1e-7
    assert preprocessing_replay.Z_MAX_ABS_TOLERANCE == 1e-4
    assert preprocessing_replay.LOGIT_MAX_ABS_TOLERANCE == 1e-4
    assert preprocessing_replay.PROBABILITY_MAX_ABS_TOLERANCE == 1e-5


def test_c74_incremental_null_permutes_only_the_new_block():
    rng = np.random.default_rng(74)
    records = []
    for target in range(1, 10):
        for value in (-1.0, -0.3, 0.3, 1.0):
            records.append({
                "manifest": {"target_id": target},
                "source_features": rng.normal(size=4),
                "construction_features": np.asarray([value, value * value]),
                "shared_features": rng.normal(size=3),
                "source_representation_features": rng.normal(size=10),
                "target_representation_features": rng.normal(size=10),
                "evaluation_bAcc": 0.5 + 0.1 * value,
            })
    rows = analysis._incremental_prediction(records)
    construction = rows[1]
    assert construction["incremental_R2"] > 0.5
    assert construction["incremental_exceeds_null_p95"] == 1
    assert construction["null_scheme"] == "permute_new_block_within_target_keep_prior_blocks_and_outcome_fixed"
    assert "target_blocked_null_R2_p95" not in construction
