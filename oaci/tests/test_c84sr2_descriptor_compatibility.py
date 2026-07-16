from copy import deepcopy

import pytest

from oaci.multidataset.c84s_common import C84SContractError
from oaci.multidataset.c84sr1_context_enumerator import (
    LEVEL_INTERVENTION_IDS, resolve_level_intervention_id,
)
from oaci.multidataset.c84sr2_runtime_guard import field_descriptor_compatibility_rows


def _fixture(*, level=0, panel="A", seed=5, provenance="C84C", sidecar_field=True):
    raw = {
        "level_intervention_id": LEVEL_INTERVENTION_IDS[level],
        "model_reuse_provenance": provenance,
    }
    sidecar = {"level": level, "panel": panel, "seed": seed}
    if sidecar_field:
        sidecar["level_intervention_id"] = LEVEL_INTERVENTION_IDS[level]
    return raw, sidecar


def test_native_sidecar_must_match_authoritative_field_descriptor():
    raw, sidecar = _fixture(level=1, panel="B", seed=6, provenance="C84F")
    assert resolve_level_intervention_id(raw, sidecar) == LEVEL_INTERVENTION_IDS[1]
    sidecar["level_intervention_id"] = LEVEL_INTERVENTION_IDS[0]
    with pytest.raises(C84SContractError, match="sidecar/complete-field"):
        resolve_level_intervention_id(raw, sidecar)


def test_historical_omission_allowed_only_for_exact_c84c_scope():
    raw, sidecar = _fixture(sidecar_field=False)
    assert resolve_level_intervention_id(raw, sidecar) == LEVEL_INTERVENTION_IDS[0]
    for key, value in (("model_reuse_provenance", "C84F"),):
        changed = deepcopy(raw)
        changed[key] = value
        with pytest.raises(C84SContractError, match="exact historical C84C"):
            resolve_level_intervention_id(changed, sidecar)
    for field, value in (("panel", "B"), ("seed", 6), ("level", 1)):
        changed = deepcopy(sidecar)
        changed[field] = value
        if field == "level":
            raw_changed = deepcopy(raw)
            raw_changed["level_intervention_id"] = LEVEL_INTERVENTION_IDS[1]
        else:
            raw_changed = raw
        with pytest.raises(C84SContractError, match="exact historical C84C"):
            resolve_level_intervention_id(raw_changed, changed)


def test_raw_descriptor_is_required_and_locked_by_level():
    raw, sidecar = _fixture()
    del raw["level_intervention_id"]
    with pytest.raises(C84SContractError, match="complete-field descriptor lacks"):
        resolve_level_intervention_id(raw, sidecar)
    raw, sidecar = _fixture()
    raw["level_intervention_id"] = LEVEL_INTERVENTION_IDS[1]
    with pytest.raises(C84SContractError, match="locked level definition"):
        resolve_level_intervention_id(raw, sidecar)


def test_real_frozen_descriptor_audit_has_exact_1701_plus_243_partition():
    rows = field_descriptor_compatibility_rows()
    assert len(rows) == 1944
    assert sum(row["sidecar_intervention_present"] == 1 for row in rows) == 1701
    missing = [row for row in rows if row["sidecar_intervention_present"] == 0]
    assert len(missing) == 243
    assert {(row["panel"], row["training_seed"], row["level"], row["model_reuse_provenance"]) for row in missing} == {
        ("A", 5, 0, "C84C")
    }
    assert {row["dataset"] for row in missing} == {"Lee2019_MI", "Cho2017", "PhysionetMI"}
