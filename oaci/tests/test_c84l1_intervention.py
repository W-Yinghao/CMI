from __future__ import annotations

import pytest

from oaci.multidataset import c84l1_intervention as intervention
from oaci.multidataset import c84l1_protocols as protocol


def _fixture(dataset: str = "Lee2019_MI", panel: str = "A", rows: int = 8):
    return intervention.synthetic_source_panel(dataset, panel, rows_per_cell=rows)


def _apply(fixture, **overrides):
    arguments = {
        "dataset": "Lee2019_MI",
        "panel": "A",
        "level": 1,
        "source_subjects": fixture["subjects"],
        "source_labels": fixture["labels"],
        "source_trial_ids": fixture["trial_ids"],
    }
    arguments.update(overrides)
    return intervention.apply_level_intervention(**arguments)


def _drop_indices(fixture, indices):
    removed = set(indices)
    return {
        key: tuple(value for index, value in enumerate(values) if index not in removed)
        for key, values in fixture.items()
    }


def test_registered_first_subject_left_hand_cell_is_the_only_deleted_cell():
    fixture = _fixture()
    result = _apply(fixture)
    assert result.level_intervention_id == protocol.LEVEL1_ID
    assert result.deleted_source_subject == 31
    assert result.deleted_class == "left_hand"
    assert len(result.deleted_indices) == 8
    assert len(result.pre_cell_counts) == 24
    assert len(result.post_cell_counts) == 23
    assert (31, 0, 8) in result.pre_cell_counts
    assert not any(subject == 31 and label == 0 for subject, label, _ in result.post_cell_counts)
    assert (31, 1, 8) in result.post_cell_counts


def test_all_six_registered_cells_match_the_locked_first_subject_order():
    for (dataset, panel), expected_subject in protocol.DELETED_SUBJECTS.items():
        fixture = intervention.synthetic_source_panel(dataset, panel)
        result = intervention.apply_level_intervention(
            dataset=dataset,
            panel=panel,
            level=1,
            source_subjects=fixture["subjects"],
            source_labels=fixture["labels"],
            source_trial_ids=fixture["trial_ids"],
        )
        assert result.deleted_source_subject == expected_subject
        assert result.deleted_class == protocol.DELETED_CLASS


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requested_deleted_subject": 4}, "substitute the registered deleted subject"),
        ({"requested_deleted_class": "right_hand"}, "substitute the registered deleted class"),
        ({"target_dependent_choice": True}, "target- or outcome-dependent"),
        ({"outcome_dependent_choice": True}, "target- or outcome-dependent"),
    ],
)
def test_numeric_min_class_target_or_outcome_substitutions_fail(overrides, message):
    with pytest.raises(intervention.C84L1InterventionError, match=message):
        _apply(_fixture(), **overrides)


def test_registered_deleted_cell_absent_before_deletion_fails():
    fixture = _fixture()
    indices = [
        index for index, (subject, label) in enumerate(zip(fixture["subjects"], fixture["labels"]))
        if subject == 31 and label == 0
    ]
    with pytest.raises(intervention.C84L1InterventionError, match="all 24"):
        _apply(_drop_indices(fixture, indices))


def test_registered_deleted_cell_below_minimum_fails():
    fixture = _fixture()
    indices = [
        index for index, (subject, label) in enumerate(zip(fixture["subjects"], fixture["labels"]))
        if subject == 31 and label == 0
    ][:1]
    with pytest.raises(intervention.C84L1InterventionError, match="below the locked minimum"):
        _apply(_drop_indices(fixture, indices))


def test_second_cell_absent_before_deletion_fails():
    fixture = _fixture()
    retained_subject = next(subject for subject in fixture["subjects"] if subject != 31)
    indices = [
        index for index, (subject, label) in enumerate(zip(fixture["subjects"], fixture["labels"]))
        if subject == retained_subject and label == 1
    ]
    with pytest.raises(intervention.C84L1InterventionError, match="all 24"):
        _apply(_drop_indices(fixture, indices))


def test_retained_cell_below_minimum_fails():
    fixture = _fixture()
    retained_subject = next(subject for subject in fixture["subjects"] if subject != 31)
    indices = [
        index for index, (subject, label) in enumerate(zip(fixture["subjects"], fixture["labels"]))
        if subject == retained_subject and label == 1
    ][:1]
    with pytest.raises(intervention.C84L1InterventionError, match="below the locked minimum"):
        _apply(_drop_indices(fixture, indices))


@pytest.mark.parametrize(
    "protected",
    [
        {
            "source_audit_trial_ids_before": ("a", "b"),
            "source_audit_trial_ids_after": ("a",),
        },
        {
            "target_trial_ids_before": ("a", "b"),
            "target_trial_ids_after": ("b", "a"),
        },
    ],
)
def test_source_audit_and_target_rows_cannot_be_changed(protected):
    with pytest.raises(intervention.C84L1InterventionError, match="was changed"):
        _apply(_fixture(), **protected)


def test_level0_is_an_identity_operation_and_rejects_deletion_requests():
    fixture = _fixture()
    result = _apply(fixture, level=0)
    assert result.level_intervention_id == protocol.LEVEL0_ID
    assert result.keep_indices == tuple(range(len(fixture["trial_ids"])))
    assert result.deleted_indices == ()
    assert result.pre_trial_count == result.post_trial_count
    with pytest.raises(intervention.C84L1InterventionError, match="level 0 cannot"):
        _apply(fixture, level=0, requested_deleted_subject=31)


def test_source_trial_ids_must_be_unique_and_otherwise_unchanged():
    fixture = _fixture()
    duplicate = dict(fixture)
    duplicate["trial_ids"] = (fixture["trial_ids"][0],) + fixture["trial_ids"][:-1]
    with pytest.raises(intervention.C84L1InterventionError, match="not unique"):
        _apply(duplicate)
