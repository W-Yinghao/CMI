from __future__ import annotations

import numpy as np

from cedar_eeg.config import P0Thresholds
from cedar_eeg.eval.noninferiority import crossfit_task_bacc
from cedar_eeg.probes.crossfit_grouped import crossfit_conditional_domain_probe, make_folds
from cedar_eeg.red_team import RedTeamFailure, validate_p0_result
from cedar_eeg.surgery.latent_mask import apply_diagonal_mask, mask_from_drop_dims, rank_latent_dimensions
from cedar_eeg.surgery.selection import SurgeryCandidate, SurgeryDecision, decide_p0, target_eval_warnings


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 180
    y = np.repeat([0, 1], n // 2)
    d = np.tile(np.repeat([0, 1, 2], n // 6), 2)
    groups = np.repeat(np.arange(18), 10)
    z = rng.normal(0.0, 0.35, size=(n, 6))
    z[:, 0] += (d - 1) * 2.5           # domain-rich / task-light
    z[:, 1] += (2 * y - 1) * 2.5       # task-rich / domain-light
    z[:, 2] += (d - 1) * 0.4 + y * 0.2
    return z, y, d, groups


def test_grouped_folds_keep_groups_disjoint():
    _, _, _, groups = _synthetic()
    for tr, ev in make_folds(len(groups), groups=groups, n_splits=3, seed=1):
        assert set(groups[tr]).isdisjoint(set(groups[ev]))


def test_latent_rank_prefers_domain_rich_task_light_dim():
    z, y, d, _ = _synthetic()
    ranked = rank_latent_dimensions(z, y, d)
    assert ranked[0] == 0
    assert ranked.index(0) < ranked.index(1)


def test_domain_mask_reduces_conditional_leakage_without_task_collapse():
    z, y, d, groups = _synthetic()
    base_leak = crossfit_conditional_domain_probe(
        z, y, d, n_classes=2, n_domains=3, groups=groups, n_splits=3, seed=2
    )
    base_task = crossfit_task_bacc(z, y, groups=groups, n_classes=2, n_splits=3, seed=2)
    keep = mask_from_drop_dims(z.shape[1], [0])
    z_masked = apply_diagonal_mask(z, keep)
    masked_leak = crossfit_conditional_domain_probe(
        z_masked, y, d, n_classes=2, n_domains=3, groups=groups, n_splits=3, seed=3
    )
    masked_task = crossfit_task_bacc(z_masked, y, groups=groups, n_classes=2, n_splits=3, seed=3)
    assert masked_leak["advantage_mean"] < base_leak["advantage_mean"] * 0.70
    assert base_task - masked_task <= 0.01


def test_p0_abstains_when_task_drop_is_too_large():
    cand = SurgeryCandidate(
        name="drop_task_dim",
        dropped_units=(1,),
        leakage_before=0.30,
        leakage_after=0.10,
        source_bacc_before=0.90,
        source_bacc_after=0.70,
    )
    decision, reasons = decide_p0(cand, P0Thresholds())
    assert decision == SurgeryDecision.ABSTAIN
    assert any("source_bacc_drop" in r for r in reasons)


def test_p0_accepts_clean_leakage_surgery_candidate():
    cand = SurgeryCandidate(
        name="drop_domain_dim",
        dropped_units=(0,),
        leakage_before=0.40,
        leakage_after=0.20,
        source_bacc_before=0.90,
        source_bacc_after=0.895,
        r3_before=0.0,
        r3_after=0.0,
        stability=0.95,
        random_control_drop_frac=0.01,
    )
    decision, reasons = decide_p0(cand, P0Thresholds())
    assert decision == SurgeryDecision.ACCEPT
    assert reasons == []


def test_target_metrics_are_diagnostic_only_for_source_decision():
    cand = SurgeryCandidate(
        name="drop_domain_dim",
        dropped_units=(0,),
        leakage_before=0.40,
        leakage_after=0.20,
        source_bacc_before=0.90,
        source_bacc_after=0.895,
        target_bacc_before=0.90,
        target_bacc_after=0.50,
        stability=0.95,
        random_control_drop_frac=0.01,
    )
    decision, reasons = decide_p0(cand, P0Thresholds())
    assert decision == SurgeryDecision.ACCEPT
    assert not reasons
    assert target_eval_warnings(cand, P0Thresholds())


def test_red_team_rejects_target_metric_in_decision_reasons():
    payload = {
        "project": "CEDAR-EEG",
        "phase": "P0_frozen_latent",
        "groups_present": True,
        "claim_boundary": "target metrics are evaluation-only; leakage reduction is not a target-generalization guarantee.",
        "baseline": {"permutation_null": {"advantage_mean": 0.0}},
        "candidates": [
            {
                "decision": "ABSTAIN",
                "reasons": ["target_bacc_drop 0.2 > 0.01"],
                "utility": 1.0,
                "candidate": {"name": "x", "random_control_drop_frac": 0.0, "target_bacc_drop": 0.2},
            }
        ],
        "selected": None,
    }
    try:
        validate_p0_result(payload)
    except RedTeamFailure:
        return
    raise AssertionError("red team must reject target-dependent decision reasons")


def test_red_team_accepts_clean_p0_payload():
    payload = {
        "project": "CEDAR-EEG",
        "phase": "P0_frozen_latent",
        "groups_present": True,
        "claim_boundary": "target metrics are evaluation-only; leakage reduction is not a target-generalization guarantee.",
        "baseline": {"permutation_null": {"advantage_mean": 0.0}},
        "candidates": [
            {
                "decision": "ACCEPT",
                "reasons": [],
                "utility": 2.0,
                "candidate": {"name": "x", "random_control_drop_frac": 0.0, "target_bacc_drop": None},
            }
        ],
        "selected": {
            "decision": "ACCEPT",
            "reasons": [],
            "utility": 2.0,
            "candidate": {"name": "x", "random_control_drop_frac": 0.0, "target_bacc_drop": None},
        },
    }
    res = validate_p0_result(payload)
    assert res.passed
    assert "target_labels_quarantined_from_decisions" in res.checks
