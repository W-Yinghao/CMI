"""Shadow-only int64 RNG, interval, and replicate persistence tests."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory import c85t_monte_carlo as monte
from oaci.theory import c85t_result_manifest as manifest
from oaci.theory import c85t_rng as rng


def test_int64_rademacher_bytes_are_distinct_from_historical_uint8() -> None:
    legacy_low, legacy_high = rng.draw_s9_rademacher_prefixes(
        "SHADOW_RADEMACHER_A", 0
    )
    low, high = rng.draw_s9_rademacher_int64("SHADOW_RADEMACHER_A", 0)
    assert low.dtype == high.dtype == np.dtype("<i8")
    assert low.shape == legacy_low.shape == (51,)
    assert high.shape == legacy_high.shape == (46,)
    assert not np.array_equal(low, legacy_low) or not np.array_equal(high, legacy_high)
    assert low.tobytes() != legacy_low.tobytes()
    assert high.tobytes() != legacy_high.tobytes()


def test_operative_s9_draw_call_requests_numpy_int64_exactly() -> None:
    source = inspect.getsource(rng.draw_s9_rademacher_int64)
    assert "dtype=np.int64" in source
    assert "dtype=np.uint8" not in source


def test_probability_intervals_persist_raw_and_clipped_values() -> None:
    interval = monte.probability_interval_v2(0.0, 0.1)
    assert interval["raw_95pct_mc_interval"] == [-0.196, 0.196]
    assert interval["reported_95pct_mc_interval"] == [0.0, 0.196]
    assert interval["interval_clipped"] is True
    interior = monte.probability_interval_v2(0.5, 0.01)
    assert interior["raw_95pct_mc_interval"] == interior["reported_95pct_mc_interval"]
    assert interior["interval_clipped"] is False


@pytest.fixture(scope="module")
def near_shadow() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    return monte.simulate_near_optimal_selection_v2(
        scenario_id="SHADOW_NORMAL_A",
        utilities=(1.0, 0.98, 0.7),
        epsilon=0.03,
        tau=0.1,
        pairwise_sigma=0.05,
    )


def test_saved_near_replicates_reproduce_every_aggregate(
    tmp_path: Path,
    near_shadow: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    summary, arrays = near_shadow
    path = tmp_path / "S6_replicates.npz"
    manifest.write_deterministic_npz(path, arrays)
    loaded = manifest.read_deterministic_npz(path)
    replay = monte.summarize_near_replicates_v2(
        "SHADOW_NORMAL_A", loaded, summary["geometry"]
    )
    assert replay == summary
    assert loaded["replicate_id"].dtype == np.dtype("<u2")
    assert loaded["selected_action"].dtype == np.dtype("<u2")
    assert loaded["top1"].dtype == np.dtype("uint8")
    assert loaded["selection_regret"].dtype == np.dtype("<f8")


def test_missing_or_duplicate_replicate_fails(
    near_shadow: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    summary, arrays = near_shadow
    duplicate = {key: value.copy() for key, value in arrays.items()}
    duplicate["replicate_id"][1] = 0
    with pytest.raises(DecisionContractError, match="missing, duplicated, or reordered"):
        monte.summarize_near_replicates_v2(
            "SHADOW_NORMAL_A", duplicate, summary["geometry"]
        )
    missing = {key: value[:-1] for key, value in arrays.items()}
    with pytest.raises(DecisionContractError, match="field drifted"):
        monte.summarize_near_replicates_v2(
            "SHADOW_NORMAL_A", missing, summary["geometry"]
        )


@pytest.fixture(scope="module")
def s9_shadow() -> tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, object]]]:
    return monte.simulate_full_information_designs_v2(
        scenario_id="SHADOW_RADEMACHER_A",
        stratum_masses=(0.7, 0.3),
        sigmas=(0.03, 0.15),
        passive_allocation=(51, 13),
        neyman_allocation=(18, 46),
        population_mean_losses=(0.2, 0.24, 0.6, 0.8),
        action1_offset=0.04,
    )


def test_saved_s9_replicates_and_raw_digests_replay(
    tmp_path: Path,
    s9_shadow: tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, object]]],
) -> None:
    summary, arrays, digest_rows = s9_shadow
    path = tmp_path / "S9_replicates.npz"
    manifest.write_deterministic_npz(path, arrays)
    loaded = manifest.read_deterministic_npz(path)
    replay = monte._summarize_s9_arrays_v2(
        loaded, np.asarray((0.2, 0.24, 0.6, 0.8), dtype="<f8")
    )
    for key in ("analytic_variance", "universal_active_superiority_claim"):
        replay[key] = summary[key]
    assert replay == summary
    assert len(digest_rows) == 4096
    assert [row["replicate_id"] for row in digest_rows] == list(range(4096))
    assert all(row["dtype"] == "<i8" for row in digest_rows)
    first_low, first_high = rng.draw_s9_rademacher_int64(
        "SHADOW_RADEMACHER_A", 0
    )
    combined = hashlib.sha256(first_low.tobytes() + first_high.tobytes()).hexdigest()
    assert digest_rows[0]["combined_sha256"] == combined
    assert loaded["passive_selected_action"].dtype == np.dtype("uint8")
    assert loaded["neyman_D_hat"].dtype == np.dtype("<f8")
    assert loaded["paired_passive_minus_neyman_D_hat"].shape == (4096,)


def test_object_or_nonfinite_array_cannot_be_persisted(tmp_path: Path) -> None:
    with pytest.raises(DecisionContractError, match="non-object and finite"):
        manifest.write_deterministic_npz(
            tmp_path / "object.npz", {"bad": np.asarray([object()], dtype=object)}
        )
    with pytest.raises(DecisionContractError, match="non-object and finite"):
        manifest.write_deterministic_npz(
            tmp_path / "nan.npz", {"bad": np.asarray([np.nan])}
        )


def _shadow_v2_result() -> dict[str, object]:
    return {
        "schema_version": manifest.RESULT_SCHEMA_V2,
        "final_gate": manifest.SUCCESS_GATE_V2,
        "scenario_count": 11,
        "formal_theorem_statuses": {f"T{i}": "OPEN" for i in range(1, 8)},
        "real_project_data_access": 0,
        "active_acquisition": 0,
    }


def _populate_shadow_writer(
    writer: manifest.AtomicResultWriterV2,
    near_shadow: tuple[dict[str, object], dict[str, np.ndarray]],
    s9_shadow: tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, object]]],
) -> None:
    near_summary, near_arrays = near_shadow
    s9_summary, s9_arrays, digest_rows = s9_shadow
    writer.write_json(
        "exact_scenario_results.json", {f"S{i}": {"shadow": True} for i in range(11)}
    )
    writer.write_npz("S6_replicates.npz", near_arrays)
    writer.write_npz("S7_replicates.npz", near_arrays)
    writer.write_npz("S9_replicates.npz", s9_arrays)
    writer.write_json(
        "monte_carlo_summary.json",
        {
            "S6": {**near_summary, "scenario_id": "S6"},
            "S7": {**near_summary, "scenario_id": "S7"},
            "S9": s9_summary,
            "S9_population_mean_losses": [0.2, 0.24, 0.6, 0.8],
        },
    )
    digest_path = writer.path("S9_raw_draw_digest_registry.csv")
    with digest_path.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=tuple(digest_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(digest_rows)
    rows = []
    for theorem_id, filename in zip(
        (f"T{i}" for i in range(1, 8)), manifest.PROOF_FILENAMES_V2
    ):
        proof_path = writer.write_text(
            f"c85t_proof_candidates/{filename}",
            f"# {theorem_id} shadow candidate\n\nFormal status: OPEN\n",
        )
        rows.append(
            {
                "theorem_id": theorem_id,
                "historical_status": "OPEN",
                "candidate_disposition": "INCOMPLETE_OPEN",
                "formal_status": "OPEN",
                "check_class": "PROOF_CANDIDATE_SCHEMA_AND_INTERNAL_CONSISTENCY",
                "proof_candidate_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            }
        )
    disposition_path = writer.path("proof_candidate_dispositions.csv")
    with disposition_path.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    writer.write_json("authorization_consumed.json", {"shadow": True})


def test_atomic_v2_manifest_replays_all_required_shadow_artifacts(
    tmp_path: Path,
    near_shadow: tuple[dict[str, object], dict[str, np.ndarray]],
    s9_shadow: tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, object]]],
) -> None:
    root = tmp_path / "complete"
    with manifest.AtomicResultWriterV2(root) as writer:
        _populate_shadow_writer(writer, near_shadow, s9_shadow)
        writer.publish(_shadow_v2_result())
    observed = manifest.replay_manifest_v2(root)
    assert observed["schema_version"] == manifest.MANIFEST_SCHEMA_V2
    assert observed["counts"] == {
        "scenario_results": 11,
        "S6_S7_logical_replicate_rows": 8192,
        "S9_logical_replicate_design_rows": 8192,
        "S9_raw_draw_digest_rows": 4096,
        "proof_candidates": 7,
        "formal_theorem_status_OPEN": 7,
    }


@pytest.mark.parametrize("point", ["before_result", "before_manifest", "before_publish"])
def test_atomic_v2_failure_leaves_no_final_root(
    tmp_path: Path,
    near_shadow: tuple[dict[str, object], dict[str, np.ndarray]],
    s9_shadow: tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, object]]],
    point: str,
) -> None:
    root = tmp_path / point
    writer: manifest.AtomicResultWriterV2
    with pytest.raises(RuntimeError, match="C85T_V2_SHADOW_FAILURE"):
        with manifest.AtomicResultWriterV2(root, failure_injection=point) as writer:
            _populate_shadow_writer(writer, near_shadow, s9_shadow)
            writer.publish(_shadow_v2_result())
    assert not root.exists()
    assert writer.failed_root is not None and writer.failed_root.is_dir()
