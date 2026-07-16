from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory.c85t_execution_context_v3 import (
    canonical_json_bytes,
    create_validated_c85t_execution_context,
)
from oaci.theory.c85t_result_manifest import (
    read_deterministic_npz,
    write_deterministic_npz,
)
from oaci.theory.c85t_semantic_replay_v3 import validate_result_semantics_v3
from oaci.theory.c85t_transaction_v3 import AtomicExecutionBundleV3
from oaci.tests.c85tr2_test_support import (
    create_shadow_authorized_repository,
    populate_shadow_bundle,
)


@pytest.fixture
def staged_shadow(tmp_path: Path):
    fixture = create_shadow_authorized_repository(tmp_path)
    context = create_validated_c85t_execution_context(
        fixture["lock_path"], fixture["authorization_path"], fixture["output_root"]
    )
    bundle = AtomicExecutionBundleV3(context)
    shadow = populate_shadow_bundle(bundle, context, fixture["statements"])
    bundle.write_json("C85T_RESULT.json", shadow["result"])
    yield fixture, context, shadow
    context.close()


def _validate(fixture, context, shadow):
    return validate_result_semantics_v3(
        context.staging_bundle_root,
        context=context,
        contract=shadow["contract"],
        statements=fixture["statements"],
        shadow_expected_exact=shadow["exact"],
        shadow_expected_s9_arrays=shadow["s9_arrays"],
        shadow_expected_s9_digest_rows=shadow["digest_rows"],
    )


def test_semantic_replay_derives_all_counts_from_shadow_artifacts(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    observed = _validate(fixture, context, shadow)
    assert observed["scenario_results"] == 11
    assert observed["scenario_keys"] == [f"S{i}" for i in range(11)]
    assert observed["S6_S7_logical_replicate_rows"] == 8192
    assert observed["S9_logical_replicate_design_rows"] == 8192
    assert observed["S9_raw_draw_digest_rows"] == 4096
    assert observed["proof_candidates"] == 7
    assert observed["formal_theorem_status_OPEN"] == 7
    assert observed["protected_counters_zero"] is True


def test_exact_scenario_key_omission_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "exact_scenario_results.json"
    exact = json.loads(path.read_text())
    exact.pop("S4")
    path.write_bytes(canonical_json_bytes(exact))
    with pytest.raises(DecisionContractError, match="exact scenario keys"):
        _validate(fixture, context, shadow)


def test_invalid_s9_action_range_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "S9_replicates.npz"
    arrays = read_deterministic_npz(path)
    arrays["passive_selected_action"][0] = 4
    path.unlink()
    write_deterministic_npz(path, arrays)
    with pytest.raises(DecisionContractError, match="out of range"):
        _validate(fixture, context, shadow)


def test_duplicate_replicate_id_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "S6_replicates.npz"
    arrays = read_deterministic_npz(path)
    arrays["replicate_id"][1] = 0
    path.unlink()
    write_deterministic_npz(path, arrays)
    with pytest.raises(DecisionContractError, match="replicate IDs"):
        _validate(fixture, context, shadow)


@pytest.mark.parametrize("field,value", [("dtype", "uint8"), ("L_count", "50")])
def test_invalid_s9_digest_dtype_or_count_fails(
    staged_shadow, field: str, value: str
) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "S9_raw_draw_digest_registry.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DecisionContractError, match="dtype/count"):
        _validate(fixture, context, shadow)


def test_invalid_s9_combined_digest_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "S9_raw_draw_digest_registry.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["combined_sha256"] = "0" * 64
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DecisionContractError, match="digest replay drifted"):
        _validate(fixture, context, shadow)


def test_proof_file_csv_hash_mismatch_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    proof = context.staging_bundle_root / "c85t_proof_candidates/T1_blackwell_monotonicity.md"
    proof.write_text(proof.read_text() + "\nTAMPERED\n")
    with pytest.raises(DecisionContractError, match="proof hash drifted"):
        _validate(fixture, context, shadow)


def test_proof_statement_sha_mismatch_fails_even_when_file_hash_is_updated(
    staged_shadow,
) -> None:
    fixture, context, shadow = staged_shadow
    proof = context.staging_bundle_root / "c85t_proof_candidates/T1_blackwell_monotonicity.md"
    text = proof.read_text().replace(
        "Statement SHA-256: `", "Statement SHA-256: `" + "0" * 64 + "`\nIgnored: `", 1
    )
    proof.write_text(text)
    dispositions = context.staging_bundle_root / "proof_candidate_dispositions.csv"
    with dispositions.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["proof_candidate_sha256"] = hashlib.sha256(proof.read_bytes()).hexdigest()
    with dispositions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DecisionContractError, match="statement SHA drifted"):
        _validate(fixture, context, shadow)


def test_saved_array_aggregate_mismatch_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "monte_carlo_summary.json"
    summary = json.loads(path.read_text())
    summary["S7"]["top_1_probability"] = 0.5
    path.write_bytes(canonical_json_bytes(summary))
    with pytest.raises(DecisionContractError, match="aggregate replay drifted"):
        _validate(fixture, context, shadow)


def test_non_open_theorem_status_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "proof_candidate_dispositions.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["formal_status"] = "PROVED"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DecisionContractError, match="proof disposition drifted"):
        _validate(fixture, context, shadow)


def test_result_attempt_identity_mismatch_fails(staged_shadow) -> None:
    fixture, context, shadow = staged_shadow
    path = context.staging_bundle_root / "C85T_RESULT.json"
    result = json.loads(path.read_text())
    result["attempt_id"] = "wrong-attempt"
    path.write_bytes(canonical_json_bytes(result))
    with pytest.raises(DecisionContractError, match="result identity drifted: attempt_id"):
        _validate(fixture, context, shadow)
