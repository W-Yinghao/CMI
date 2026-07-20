from __future__ import annotations

import copy
from pathlib import Path

import pytest

from oaci.theory.c85_decision_experiments import DecisionContractError
from oaci.theory.c85t_execution_context_v3 import (
    ValidatedC85TExecutionContext,
    create_validated_c85t_execution_context,
    replay_lifecycle_v3,
    validate_registered_execution_context,
)
from oaci.theory import c85t_registered_v3 as registered
from oaci.tests.c85tr2_test_support import create_shadow_authorized_repository


def _create(tmp_path: Path):
    fixture = create_shadow_authorized_repository(tmp_path)
    context = create_validated_c85t_execution_context(
        fixture["lock_path"], fixture["authorization_path"], fixture["output_root"]
    )
    return fixture, context


def test_committed_paths_and_exclusive_receipt_create_valid_context(tmp_path: Path) -> None:
    fixture, context = _create(tmp_path)
    assert validate_registered_execution_context(context) is context
    assert context.external_consumption_receipt_path == fixture["external_receipt_path"]
    assert context.external_consumption_receipt_path.is_file()
    events = replay_lifecycle_v3(context.lifecycle_path)
    assert [row["stage"] for row in events] == [
        "PREFLIGHT_STARTED",
        "PREFLIGHT_COMPLETED",
        "AUTHORIZATION_CONSUMED",
    ]
    assert events[-1]["authorization_binding_sha256"] == fixture["binding_sha"]
    context.close()


def test_generic_mapping_and_direct_constructor_cannot_create_context(tmp_path: Path) -> None:
    fixture = create_shadow_authorized_repository(tmp_path)
    with pytest.raises(DecisionContractError):
        ValidatedC85TExecutionContext()
    with pytest.raises((TypeError, AttributeError)):
        create_validated_c85t_execution_context(  # type: ignore[arg-type]
            {"execution_lock_sha256": fixture["lock_sha"]},
            fixture["authorization_path"],
            fixture["output_root"],
        )
    fabricated = object.__new__(ValidatedC85TExecutionContext)
    with pytest.raises(DecisionContractError):
        validate_registered_execution_context(fabricated)
    with pytest.raises(DecisionContractError):
        registered._s9_int64_draws(0, context=fabricated)


def test_context_copy_deepcopy_and_cross_root_mutation_fail(tmp_path: Path) -> None:
    _, context = _create(tmp_path)
    with pytest.raises(DecisionContractError):
        copy.copy(context)
    with pytest.raises(DecisionContractError):
        copy.deepcopy(context)
    original = context._output_root
    context._output_root = original.with_name("wrong-root")
    with pytest.raises(DecisionContractError, match="receipt binding drifted"):
        validate_registered_execution_context(context)
    context._output_root = original
    assert validate_registered_execution_context(context) is context
    context.close()


def test_authorization_is_globally_single_use(tmp_path: Path) -> None:
    fixture, context = _create(tmp_path)
    with pytest.raises(DecisionContractError, match="already consumed"):
        create_validated_c85t_execution_context(
            fixture["lock_path"], fixture["authorization_path"], fixture["output_root"]
        )
    wrong_root = fixture["output_root"].with_name("not-the-authorized-root")
    with pytest.raises(DecisionContractError, match="output root binding drifted"):
        create_validated_c85t_execution_context(
            fixture["lock_path"], fixture["authorization_path"], wrong_root
        )
    context.close()


@pytest.mark.parametrize("mode", ["delete", "tamper"])
def test_receipt_deletion_or_tamper_blocks_dispatch(tmp_path: Path, mode: str) -> None:
    _, context = _create(tmp_path)
    receipt = context.external_consumption_receipt_path
    if mode == "delete":
        receipt.unlink()
    else:
        receipt.write_text("{}\n")
    with pytest.raises(DecisionContractError, match="absent or tampered"):
        validate_registered_execution_context(context)
    context.close()


def test_v3_source_has_no_historical_mint_or_generic_consume_api() -> None:
    theory = Path(__file__).resolve().parents[1] / "theory"
    text = "\n".join(path.read_text() for path in theory.glob("*v3.py"))
    for forbidden in (
        "_CAPABILITY_SENTINEL",
        "_ISSUED_CAPABILITIES",
        "_issue_capability",
        "consume_authorization_once",
    ):
        assert forbidden not in text
