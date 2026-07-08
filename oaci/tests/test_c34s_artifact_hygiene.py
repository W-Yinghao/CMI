"""C34S artifact slimming / report payload hygiene tests."""
from __future__ import annotations

import csv
import json
import os

from oaci.continuous_regret import artifact_hygiene, schema


REPORT_JSON = "oaci/reports/C34_CONTINUOUS_LOCAL_REGRET_AUDIT.json"
TABLE_DIR = "oaci/reports/c34_tables"


def _load_compact():
    with open(REPORT_JSON) as f:
        return json.load(f)


def _table_rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def test_compact_json_loads_and_has_no_row_payloads():
    d = _load_compact()
    assert d["schema_version"] == "C34S-compact-v1"
    assert d["payload_policy"] == "row_level_payload_in_c34_tables_csv"
    assert d["config"]["config_hash"] == schema.LOCKED_C19_CONFIG_HASH
    assert d["verdict"]["primary_cases"] == [
        schema.M2,
        schema.M7,
        schema.M8,
    ]
    for heavy_key in ("endpoint_registry", "selected_pairs", "source_direction",
                      "source_objective_components", "gauge_locality"):
        assert heavy_key not in d
    assert os.path.getsize(REPORT_JSON) < 200_000


def test_table_manifest_resolves_and_hashes_match():
    d = _load_compact()
    manifest = d["table_manifest"]
    assert {r["table"] for r in manifest} == set(artifact_hygiene.TABLES)
    for row in manifest:
        path = os.path.join("oaci/reports", row["path"])
        assert os.path.exists(path)
        assert row["bytes"] == os.path.getsize(path)
        assert row["rows"] == artifact_hygiene.csv_row_count(path)
        assert row["sha256"] == artifact_hygiene.sha256_file(path)


def test_key_c34_numbers_match_committed_csvs():
    d = _load_compact()
    selected = _table_rows("selected_vs_continuous_better_pairs.csv")
    cont = [r for r in selected if r["comparison"] == "nearest_continuous_better"]
    binary_miss = [r for r in selected if r["comparison"] == "nearest_binary_joint_good" and
                   r["selected_joint_good"] == "0"]
    agg = d["key_aggregates"]["selected_pair_regret"]
    assert agg["n_selected_continuous_better_pairs"] == len(cont) == 153
    assert agg["continuous_raw_pareto_nonworse_count"] == sum(
        float(r["target_bacc_delta"]) >= 0 and float(r["target_nll_delta"]) >= 0 and
        float(r["target_ece_delta"]) >= 0 for r in cont) == 72
    assert agg["continuous_raw_endpoint_backward_count"] == sum(
        float(r["target_bacc_delta"]) < 0 or float(r["target_nll_delta"]) < 0 or
        float(r["target_ece_delta"]) < 0 for r in cont) == 81
    assert agg["continuous_joint_min_negative_count"] == sum(
        float(r["joint_min_margin_delta"]) < 0 for r in cont) == 33
    assert agg["binary_miss_count"] == len(binary_miss) == 81
    assert agg["binary_threshold_tiny_count"] == sum(int(r["threshold_artifact"]) for r in binary_miss) == 0
    assert agg["binary_endpoint_tradeoff_count"] == sum(int(r["endpoint_tradeoff"]) for r in binary_miss) == 15


def test_no_scientific_value_changes_in_compact_summary():
    d = _load_compact()
    assert d["key_aggregates"]["selected_pair_regret"]["real_endpoint_regret_fraction"] == 0.9411764705882353
    assert d["key_aggregates"]["selected_pair_regret"]["threshold_only_fraction_among_binary_misses"] == 0.0
    assert d["key_aggregates"]["gauge_locality"]["meaningful_regret_gauge_unseen_fraction"] == 0.05251141552511415
    assert d["key_aggregates"]["gauge_locality"]["target_unlabeled_pm1_regret_delta_vs_source"] == 0.03750105883097604
    assert all(d["gates"].values())
