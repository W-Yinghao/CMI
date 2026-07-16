"""Same-method C84 scientific and label-frontier taxonomy."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .c84s_common import require


GATE_A = "C84-A_same_zero_label_selector_matches_B1_across_all_external_datasets"
GATE_B = "C84-B_same_zero_label_selector_improves_source_across_all_external_datasets_but_not_B1"
GATE_C = "C84-C_no_registered_zero_label_selector_materially_improves_source_in_any_external_dataset"
GATE_D = "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous"
GATE_E = "C84-E_multidataset_protocol_field_view_analysis_or_provenance_blocker"
DATASETS = ("Lee2019_MI", "Cho2017", "PhysionetMI")


def dataset_method_sets(decisions: Mapping[str, Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    a_set = {
        method for method, row in decisions.items()
        if bool(row["Q1_pass"]) and bool(row["Q2_pass"])
    }
    b_set = {method for method, row in decisions.items() if bool(row["Q1_pass"])}
    return a_set, b_set


def classify_c84(
    *,
    dataset_decisions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    level_heterogeneity: Mapping[str, bool],
    loto_preserved_methods: Mapping[str, Sequence[str]],
    blocker: bool = False,
    hidden_level_q1_pass: bool = False,
) -> dict[str, Any]:
    require(set(dataset_decisions) == set(DATASETS), "C84 dataset decision coverage drift")
    require(set(loto_preserved_methods) == set(DATASETS), "C84 LOTO dataset coverage drift")
    a_sets, b_sets = {}, {}
    categories: dict[str, str] = {}
    for dataset in DATASETS:
        a_sets[dataset], b_sets[dataset] = dataset_method_sets(dataset_decisions[dataset])
        categories[dataset] = "A" if a_sets[dataset] else "B" if b_sets[dataset] else "C"
    common_a = set.intersection(*(a_sets[dataset] for dataset in DATASETS))
    common_b = set.intersection(*(b_sets[dataset] for dataset in DATASETS))
    all_methods = set().union(*(set(rows) for rows in dataset_decisions.values()))
    active_level_heterogeneity = any(level_heterogeneity.get(method, False) for method in all_methods)

    if common_a:
        stable_a = {
            method for method in common_a
            if all(
                method in set(loto_preserved_methods[dataset])
                and dataset_decisions[dataset][method]["panel_seed_Q1_all_directional"]
                and dataset_decisions[dataset][method]["panel_seed_Q2_all_within_margin"]
                for dataset in DATASETS
            )
        }
    else:
        stable_a = set()
    if common_b:
        stable_b = {
            method for method in common_b
            if all(
                method in set(loto_preserved_methods[dataset])
                and dataset_decisions[dataset][method]["panel_seed_Q1_all_directional"]
                for dataset in DATASETS
            )
        }
    else:
        stable_b = set()

    category_mismatch = len(set(categories.values())) > 1
    method_identity_failure = (
        all(category == "A" for category in categories.values()) and not common_a
    ) or (
        all(category == "B" for category in categories.values()) and not common_b
    )
    stability_failure = bool((common_a and not stable_a) or (common_b and not stable_b))
    one_dataset_only_positive = any(b_sets.values()) and not all(b_sets.values())
    heterogeneity = bool(
        category_mismatch or method_identity_failure or stability_failure
        or active_level_heterogeneity or one_dataset_only_positive
    )
    if blocker:
        gate = GATE_E
    elif heterogeneity:
        gate = GATE_D
    elif stable_a:
        gate = GATE_A
    elif not common_a and stable_b:
        gate = GATE_B
    elif not any(b_sets.values()) and not hidden_level_q1_pass:
        stable_c = all(len(loto_preserved_methods[dataset]) > 0 for dataset in DATASETS)
        gate = GATE_C if stable_c else GATE_D
    else:
        gate = GATE_D
    return {
        "gate": gate,
        "dataset_categories": categories,
        "A_sets": {dataset: sorted(a_sets[dataset]) for dataset in DATASETS},
        "B_sets": {dataset: sorted(b_sets[dataset]) for dataset in DATASETS},
        "A_intersection": sorted(common_a),
        "B_intersection": sorted(common_b),
        "stable_A_methods": sorted(stable_a),
        "stable_B_methods": sorted(stable_b),
        "level_heterogeneity": active_level_heterogeneity,
        "method_identity_failure": method_identity_failure,
        "stability_failure": stability_failure,
        "hidden_level_Q1_pass": bool(hidden_level_q1_pass),
    }


def classify_label_frontier(
    bstars: Mapping[str, int | str | None],
    *,
    registered_heterogeneity: bool = False,
) -> str:
    require(set(bstars) == set(DATASETS), "label-frontier dataset coverage drift")
    if any(value is None for value in bstars.values()):
        return "C84-L4"
    order = {1: 0, 2: 1, 4: 2, 8: 3, "FULL": 4}
    indices = [order[value] for value in bstars.values()]
    distance = max(indices) - min(indices)
    if registered_heterogeneity or distance > 1:
        return "C84-L3"
    maximum = max(indices)
    if maximum <= order[4]:
        return "C84-L1"
    return "C84-L2"

