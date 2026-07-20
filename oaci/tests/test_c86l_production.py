"""C86L production entrypoint — prepared but inert. Guards, no real data."""
from __future__ import annotations

import pytest

from oaci.active_testing.c86l_production import (
    AUTHORIZATION_PHRASE,
    C86LContractError,
    C86LInputBinding,
    C86LNotAuthorized,
    execute,
    validate_readiness,
)


def _ready_binding(**over):
    kw = dict(
        acquisition_unlabeled_root="/roots/c86l/acq_unlabeled",
        label_oracle_root="/roots/c86l/label_oracle",
        contribution_store_root="/roots/c86l/contrib",
        construction_evaluation_disjoint_declared=True,
    )
    kw.update(over)
    return C86LInputBinding(**kw)


def test_readiness_ok_on_wired_binding():
    out = validate_readiness(_ready_binding())
    assert out["ready"] is True and out["authorized"] is False
    assert out["one_query_one_physical_label"] is True
    assert out["contexts_per_trial"] == 8
    assert out["trigger"] == AUTHORIZATION_PHRASE


def test_roots_must_be_distinct():
    with pytest.raises(C86LContractError):
        validate_readiness(_ready_binding(
            label_oracle_root="/roots/c86l/acq_unlabeled"))   # collides with acquisition root


def test_all_roots_required():
    with pytest.raises(C86LContractError):
        validate_readiness(_ready_binding(contribution_store_root=""))


def test_held_outcome_not_a_c86l_root():
    b = _ready_binding()
    with pytest.raises(C86LContractError):
        validate_readiness(_ready_binding(acquisition_unlabeled_root=b.held_outcome_identity))


def test_construction_evaluation_must_be_disjoint_and_distinct():
    with pytest.raises(C86LContractError):
        validate_readiness(_ready_binding(construction_evaluation_disjoint_declared=False))
    with pytest.raises(C86LContractError):
        validate_readiness(_ready_binding(
            evaluation_registry_id="c84f_construction_trial_ids"))   # same as construction


# --- the boundary: execution refuses without a separate direct authorization ---
def test_execute_refuses_without_authorization():
    with pytest.raises(C86LNotAuthorized):
        execute(_ready_binding())
    with pytest.raises(C86LNotAuthorized):
        execute(_ready_binding(), authorization="please")


def test_execute_authorized_requires_output_root():
    # with the phrase the guard passes; the real build still needs an explicit output_root
    from oaci.active_testing.c86l_production import C86LContractError
    with pytest.raises(C86LContractError):
        execute(_ready_binding(), authorization=AUTHORIZATION_PHRASE)
