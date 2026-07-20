from __future__ import annotations

import pytest

from oaci.multidataset import c84s_taxonomy as taxonomy


METHODS = ("U5", "U7", "U11", "U13", "U14", "U15")


def decisions(method=None, q2=False, stable=True):
    return {
        item: {
            "Q1_pass": item == method,
            "Q2_pass": item == method and q2,
            "panel_seed_Q1_all_directional": stable if item == method else True,
            "panel_seed_Q2_all_within_margin": stable if item == method else True,
        }
        for item in METHODS
    }


def classify(per_dataset, *, hetero=None, loto=None, blocker=False, hidden=False):
    return taxonomy.classify_c84(
        dataset_decisions=per_dataset,
        level_heterogeneity=hetero or {method: False for method in METHODS},
        loto_preserved_methods=loto or {dataset: ["U13"] for dataset in taxonomy.DATASETS},
        blocker=blocker, hidden_level_q1_pass=hidden,
    )["gate"]


def test_blocker_has_highest_precedence() -> None:
    per = {dataset: decisions("U13", q2=True) for dataset in taxonomy.DATASETS}
    assert classify(per, blocker=True) == taxonomy.GATE_E


def test_same_method_q1_q2_across_datasets_is_A() -> None:
    per = {dataset: decisions("U13", q2=True) for dataset in taxonomy.DATASETS}
    assert classify(per) == taxonomy.GATE_A


def test_same_method_q1_only_across_datasets_is_B() -> None:
    per = {dataset: decisions("U13") for dataset in taxonomy.DATASETS}
    assert classify(per) == taxonomy.GATE_B


def test_different_methods_cannot_support_cross_dataset_claim() -> None:
    chosen = dict(zip(taxonomy.DATASETS, ("U5", "U13", "U14")))
    per = {dataset: decisions(chosen[dataset]) for dataset in taxonomy.DATASETS}
    loto = {dataset: [chosen[dataset]] for dataset in taxonomy.DATASETS}
    assert classify(per, loto=loto) == taxonomy.GATE_D


def test_level_or_panel_instability_is_D() -> None:
    per = {dataset: decisions("U13") for dataset in taxonomy.DATASETS}
    hetero = {method: method == "U13" for method in METHODS}
    assert classify(per, hetero=hetero) == taxonomy.GATE_D
    unstable = {dataset: decisions("U13", stable=False) for dataset in taxonomy.DATASETS}
    assert classify(unstable) == taxonomy.GATE_D


def test_stable_null_is_C_but_hidden_level_pass_is_D() -> None:
    per = {dataset: decisions() for dataset in taxonomy.DATASETS}
    loto = {dataset: ["NO_Q1"] for dataset in taxonomy.DATASETS}
    assert classify(per, loto=loto) == taxonomy.GATE_C
    assert classify(per, loto=loto, hidden=True) == taxonomy.GATE_D


@pytest.mark.parametrize(
    "bstars,hetero,expected",
    [
        ({"Lee2019_MI": 1, "Cho2017": 2, "PhysionetMI": 2}, False, "C84-L1"),
        ({"Lee2019_MI": 8, "Cho2017": "FULL", "PhysionetMI": 8}, False, "C84-L2"),
        ({"Lee2019_MI": 1, "Cho2017": 8, "PhysionetMI": 2}, False, "C84-L3"),
        ({"Lee2019_MI": 1, "Cho2017": 2, "PhysionetMI": 2}, True, "C84-L3"),
        ({"Lee2019_MI": 1, "Cho2017": None, "PhysionetMI": 2}, False, "C84-L4"),
    ],
)
def test_label_frontier_taxonomy(bstars, hetero, expected) -> None:
    assert taxonomy.classify_label_frontier(bstars, registered_heterogeneity=hetero) == expected

