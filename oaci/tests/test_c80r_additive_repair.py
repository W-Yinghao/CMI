"""Synthetic/schema-only tests for the C80R additive repair."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c80_label_budget_frontier as frontier
from oaci.conditioned_ceiling_coverage import c80r_existing_field_adapter as adapter


def _fixture(seed: int = 19) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), 40)
    logits = rng.normal(size=(81, len(labels), 4))
    logits[:, np.arange(len(labels)), labels] += np.linspace(-0.2, 1.0, 81)[:, None]
    return logits, labels


def test_repair_protocol_hash_and_inheritance_replay() -> None:
    protocol, observed = adapter.load_repair_protocol()
    assert observed == adapter.REPAIR_PROTOCOL_SHA_PATH.read_text().strip()
    assert protocol["scientific_inheritance"]["estimands_changed"] is False
    assert protocol["scientific_inheritance"]["registry_bound_cells"] == 80
    assert protocol["authorization_guard_repair"]["canonical_protocol_hash_field"] == "lock.protocol.sha256"
    assert protocol["protected_state_at_protocol_lock"]["real_budget_statistics"] == 0


@pytest.mark.parametrize(
    ("blocker", "left", "right", "stable", "expected"),
    [
        (True, 1, 1, True, "C80-E_protocol_dependence_view_or_provenance_blocker"),
        (False, None, 1, True, "C80-D_no_registered_budget_achieves_stable_material_actionability"),
        (False, 1, None, True, "C80-D_no_registered_budget_achieves_stable_material_actionability"),
        (False, 4, 8, False, "C80-B_actionability_frontier_exists_but_required_budget_is_seed_heterogeneous"),
        (False, 32, "FULL", True, "C80-C_material_actionability_requires_near_full_construction_labels"),
        (False, 16, 32, True, "C80-A_stable_low_regret_label_budget_frontier_across_training_seeds"),
    ],
)
def test_taxonomy_precedence_is_mutually_exclusive(
    blocker: bool, left: int | str | None, right: int | str | None, stable: bool, expected: str,
) -> None:
    assert adapter.classify_taxonomy(
        blocker=blocker,
        bstar_seed3=left,
        bstar_seed4=right,
        cross_seed_stability_pass=stable,
    ) == expected


def test_near_full_is_ordinal_and_full_is_not_numeric_61() -> None:
    assert adapter.NEAR_FULL == (32, "FULL")
    assert adapter.classify_taxonomy(
        blocker=False, bstar_seed3="FULL", bstar_seed4="FULL", cross_seed_stability_pass=True,
    ).startswith("C80-C_")
    protocol, _ = adapter.load_repair_protocol()
    assert protocol["taxonomy"]["numeric_interpolation_between_32_and_FULL"] is False
    assert "not_numeric_61" in protocol["taxonomy"]["FULL_semantics"]


def test_vectorized_endpoint_matches_locked_scalar_formula() -> None:
    logits, labels = _fixture()
    indices = np.concatenate([np.where(labels == class_id)[0][:5] for class_id in range(4)])
    metrics = adapter.endpoint_metrics_all_candidates(logits, labels, indices)
    vector_score = frontier.score_from_endpoint_metrics(metrics)
    scalar_score = frontier.score_candidates_from_logits(logits, labels, indices)
    np.testing.assert_allclose(vector_score, scalar_score, atol=1e-14, rtol=0)


def test_nested_selection_fixture_is_deterministic_and_full_is_chain_invariant() -> None:
    logits, labels = _fixture()
    first = adapter.selection_fixture(logits, labels, seed=3, target=1, level=0, chains=4)
    second = adapter.selection_fixture(logits, labels, seed=3, target=1, level=0, chains=4)
    np.testing.assert_array_equal(first["scores"], second["scores"])
    np.testing.assert_array_equal(first["orders"], second["orders"])
    assert first["scores"].shape == (4, 7, 81)
    assert first["orders"].shape == (4, 7, 10)
    np.testing.assert_array_equal(first["scores"][:, -1], np.broadcast_to(first["scores"][0, -1], (4, 81)))
    np.testing.assert_array_equal(first["orders"][:, -1], np.broadcast_to(first["orders"][0, -1], (4, 10)))


def test_evaluation_fixture_reports_regret_topk_coverage_and_reliability() -> None:
    logits, labels = _fixture()
    selection = adapter.selection_fixture(logits, labels, seed=4, target=2, level=1, chains=3)
    utility = np.linspace(0.0, 1.0, 81)
    good = utility >= 0.75
    result = adapter.evaluate_selection_fixture(selection, utility, good)
    assert result["regret"].shape == (3, 7)
    assert result["reliability"].shape == (3, 7)
    for k in (1, 5, 10):
        assert result[f"top{k}"].shape == (3, 7)
        assert result[f"coverage_top{k}"].shape == (3, 7)
    assert np.all((result["regret"] >= 0) & (result["regret"] <= 1))


def test_simultaneous_band_and_paired_stability_are_reproducible() -> None:
    effects = np.full((8, 7), 0.12)
    effects[:, 0] = np.linspace(0.07, 0.14, 8)
    first = adapter.simultaneous_target_band(effects, seed=8020, replicates=256)
    second = adapter.simultaneous_target_band(effects, seed=8020, replicates=256)
    np.testing.assert_array_equal(first["lower"], second["lower"])
    stable = adapter.paired_cross_seed_stability(effects, effects, 4, 8, replicates=256)
    assert stable["pass"] is True
    assert stable["Bstar_grid_distance"] == 1
    absent = adapter.paired_cross_seed_stability(effects, effects, None, 4, replicates=32)
    assert absent["pass"] is False
    assert absent["reason"] == "one_or_both_Bstar_absent"


def test_generic_qualification_preserves_closure() -> None:
    effects = np.full((7, 7), 0.20)
    qualified = adapter.qualification_for_effects(effects)
    assert qualified["Bstar"] == 1
    effects[:, -1] = -0.20
    blocked = adapter.qualification_for_effects(effects)
    assert blocked["Bstar"] is None


def test_repaired_run_real_fails_before_any_array_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    absent = tmp_path / "no_authorization.json"
    monkeypatch.setattr(adapter, "REPAIRED_AUTHORIZATION_PATH", absent)
    monkeypatch.setattr(adapter, "load_repaired_lock", lambda: ({}, "f" * 64))
    calls = {"np_load": 0, "unlabeled": 0, "label": 0}

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["np_load"] += 1
        raise AssertionError("array loader reached before authorization")

    monkeypatch.setattr(adapter.np, "load", forbidden)
    monkeypatch.setattr(adapter, "_load_unlabeled", lambda *args, **kwargs: calls.__setitem__("unlabeled", 1))
    monkeypatch.setattr(adapter, "_load_label_view", lambda *args, **kwargs: calls.__setitem__("label", 1))
    with pytest.raises(RuntimeError, match="authorization record is absent"):
        adapter.run_real()
    assert calls == {"np_load": 0, "unlabeled": 0, "label": 0}


def test_historical_guard_uses_nested_protocol_sha_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    protocol_sha = "a" * 64
    lock = tmp_path / "lock.json"
    authorization = tmp_path / "authorization.json"
    sha = tmp_path / "protocol.sha256"
    lock.write_text(json.dumps({"protocol": {"commit": "c", "path": "p", "sha256": protocol_sha}}))
    authorization.write_text(json.dumps({"protocol_sha256": protocol_sha}))
    sha.write_text(protocol_sha + "\n")
    monkeypatch.setattr(frontier, "LOCK_PATH", lock)
    monkeypatch.setattr(frontier, "AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(frontier, "PROTOCOL_SHA_PATH", sha)
    assert frontier.assert_c80e_authorized()["protocol_sha256"] == protocol_sha
    lock.write_text(json.dumps({"protocol_sha256": protocol_sha}))
    with pytest.raises(RuntimeError, match="protocol schema mismatch"):
        frontier.assert_c80e_authorized()


def test_selection_stage_source_contains_no_evaluation_view_access() -> None:
    source = Path(adapter.__file__).read_text()
    selection_block = source.split("def _selection_stage", 1)[1].split("def _load_selection", 1)[0]
    assert "target_evaluation_view" not in selection_block
    assert "target_construction_view" in selection_block
    assert "same_label_oracle" in selection_block  # frozen-manifest audit field only
    assert "_evaluation_stage" in source


def test_schema_dry_run_reports_zero_outcomes() -> None:
    audit = adapter.schema_dry_run()
    assert audit["real_budget_statistics"] == 0
    assert audit["evaluation_label_reads"] == 0
    assert audit["run_real_fail_closed"] is True


def test_replacement_lock_replays_adapter_manifests_and_new_authorization_absence() -> None:
    lock, observed = adapter.load_repaired_lock()
    assert observed == adapter.REPAIRED_LOCK_SHA_PATH.read_text().strip()
    assert observed == "e18f2b5f1d79b6fcd96207339c5842e30b7aecb5bc22b8939a475487068b1b82"
    assert lock["protocol"]["sha256"] == adapter.REPAIR_PROTOCOL_SHA_PATH.read_text().strip()
    assert lock["implementation"]["commit"].startswith("e5cb41a")
    adapter_binding = next(
        row for row in lock["implementation"]["files"]
        if row["path"].endswith("c80r_existing_field_adapter.py")
    )
    assert adapter_binding["sha256"] == adapter._sha256_file(Path(adapter.__file__))
    assert len(lock["field_and_view_manifests"]) == 11
    assert lock["registry"]["bound_cells"] == 80
    assert lock["report_schema"]["selection_outputs_frozen_before_evaluation_open"] is True
    assert lock["authorization"]["received"] is False
    assert not adapter.REPAIRED_AUTHORIZATION_PATH.exists()


def test_synthetic_result_synthesis_runs_all_paths_and_exact_taxonomy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    table_dir = tmp_path / "tables"
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(adapter, "RESULT_TABLE_DIR", table_dir)
    monkeypatch.setattr(adapter, "RESULT_PATH", result_path)
    selection = {
        "cell_seed": np.repeat([3, 4], 16),
        "cell_target": np.tile(np.repeat(frontier.PRIMARY_TARGETS, 2), 2),
        "cell_level": np.tile([0, 1], 16),
        "full_class_counts": np.full((32, 4), 61, dtype=int),
    }
    monkeypatch.setattr(adapter, "_load_selection", lambda manifest: selection)
    cell_rows = []
    regime_rows = []
    geometry_rows = []
    for seed in (3, 4):
        for target in frontier.PRIMARY_TARGETS:
            for level in (0, 1):
                for budget in frontier.BUDGETS:
                    cell_rows.append({
                        "seed": seed, "target": target, "level": level, "budget": budget,
                        "expected_standardized_regret": 0.1,
                        "regret_reduction_vs_source": 0.2,
                        "reliability": 0.5,
                        "top1": 0.25, "top5": 0.75, "top10": 0.9,
                        "source_top1": 0.0, "source_top5": 0.5, "source_top10": 0.6,
                        "coverage_top1": 0.3, "coverage_top5": 0.8, "coverage_top10": 0.95,
                        "material_actionability": 1,
                    })
                    regime_rows.append({
                        "seed": seed, "target": target, "level": level, "budget": budget,
                        "ERM_fraction": 0.1, "OACI_fraction": 0.5, "SRC_fraction": 0.4,
                    })
                    geometry_rows.append({
                        "seed": seed, "target": target, "level": level, "budget": budget,
                        "raw_M": 81,
                        "effective_M_epsilon_0.05": 2 + target + level,
                        "top_two_gap": 0.01 * target + 0.001 * level,
                        "regret_reduction_vs_source": 0.2 + 0.001 * target,
                    })
    result = adapter._summarize_results(
        cell_rows, regime_rows, geometry_rows,
        {"lock": {"protocol": {"sha256": "a" * 64}}, "lock_sha256": "b" * 64},
        {"manifest_sha256": "c" * 64},
    )
    assert result["primary_taxonomy"] == "C80-A_stable_low_regret_label_budget_frontier_across_training_seeds"
    assert result["all_five_paths_unconditional"] is True
    registry = list(csv.DictReader((table_dir / "registry_execution_ledger.csv").open()))
    assert [row["path"] for row in registry] == ["P1", "P2", "S1", "S2", "S3"]
    cross_seed = list(csv.DictReader((table_dir / "cross_seed_frontier_stability.csv").open()))
    assert len(cross_seed) == 7
    assert all(row["registered_stability_pass"] == "1" for row in cross_seed)
    assert len(list(csv.DictReader((table_dir / "reliability_actionability_summary.csv").open()))) == 14
    assert len(list(csv.DictReader((table_dir / "topgap_multiplicity_moderation.csv").open()))) == 378
