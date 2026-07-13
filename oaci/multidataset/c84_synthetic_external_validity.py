"""Synthetic-only calibration for the C84 fixed-zoo external-validity design.

The public functions operate on generated target summaries.  This module has no
dataset-loader, raw-array, training, forward, or model-inference entrypoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import c84_fixed_zoo_protocol as protocol


GATE_A = "C84-A_same_zero_label_selector_matches_B1_across_all_external_datasets"
GATE_B = "C84-B_same_zero_label_selector_improves_source_across_all_external_datasets_but_not_B1"
GATE_C = "C84-C_no_registered_zero_label_selector_materially_improves_source_in_any_external_dataset"
GATE_D = "C84-D_external_dataset_source_panel_seed_or_target_heterogeneous"
GATE_E = "C84-E_multidataset_protocol_field_view_analysis_or_provenance_blocker"

LABEL_L1 = "C84-L1_common_small_budget_frontier_at_or_below_4"
LABEL_L2 = "C84-L2_common_frontier_exists_at_8_or_FULL"
LABEL_L3 = "C84-L3_dataset_heterogeneous_label_budget"
LABEL_L4 = "C84-L4_no_registered_common_grid_frontier"

METHODS = protocol.PRIMARY_ZERO_METHODS
DATASETS = protocol.DATASET_ORDER
COMMON_BUDGETS = protocol.COMMON_BUDGETS
TARGET_COUNTS = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
RAK_DRAW_COUNT = 65536
MATERIAL_MARGIN = 0.05
NONINFERIORITY_MARGIN = 0.05


def _rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"C84_SYNTHETIC_V1|{label}".encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _signs(label: str, targets: int, draws: int = RAK_DRAW_COUNT) -> np.ndarray:
    return _rng(label).choice(np.array([-1.0, 1.0]), size=(draws, targets), replace=True)


def target_cluster_maxT(
    target_effects: np.ndarray,
    *,
    null_margin: float,
    label: str,
) -> dict[str, np.ndarray]:
    """One-sided shared target-cluster Rademacher max-T calibration.

    Columns form one simultaneous method/budget family.  Rows, and only rows,
    are treated as scientific target clusters.
    """
    effects = np.asarray(target_effects, dtype=float)
    if effects.ndim != 2 or effects.shape[0] < 8 or effects.shape[1] < 1:
        raise ValueError("C84 max-T requires target-by-family effects")
    if not np.isfinite(effects).all():
        raise ValueError("C84 max-T effects must be finite")
    centered = effects - float(null_margin)
    observed = centered.mean(axis=0)
    signs = _signs(label, effects.shape[0])
    null = signs @ centered / effects.shape[0]
    null_max = null.max(axis=1)
    pvalue = (1.0 + (null_max[:, None] >= observed[None, :]).sum(axis=0)) / (len(null_max) + 1.0)
    critical = float(np.quantile(null_max, 0.95, method="higher"))
    simultaneous_lower = effects.mean(axis=0) - critical
    return {
        "mean": effects.mean(axis=0),
        "pvalue": pvalue,
        "simultaneous_lower": simultaneous_lower,
        "draws": np.array([len(null_max)], dtype=int),
        "scientific_clusters": np.array([effects.shape[0]], dtype=int),
    }


def aggregate_target_cells(cell_effects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average level and four panel-by-seed cells within target first."""
    cells = np.asarray(cell_effects, dtype=float)
    if cells.ndim != 3 or cells.shape[1:] != (2, 4):
        raise ValueError("C84 target cells must have shape target x level x panel-seed")
    if not np.isfinite(cells).all():
        raise ValueError("C84 target cells must be finite")
    return cells.mean(axis=(1, 2)), cells.mean(axis=(0, 1))


def q1_decision(cell_effects: np.ndarray, *, family_p: float) -> dict[str, Any]:
    target, panel_seed = aggregate_target_cells(cell_effects)
    return {
        "mean_effect": float(target.mean()),
        "familywise_p": float(family_p),
        "favorable_targets": int((target > 0.0).sum()),
        "target_count": len(target),
        "worst_target": float(target.min()),
        "positive_panel_seed_cells": int((panel_seed > 0.0).sum()),
        "pass": bool(
            target.mean() >= MATERIAL_MARGIN
            and family_p <= 0.05
            and (target > 0.0).mean() >= 0.75
            and target.min() >= -0.10
            and (panel_seed > 0.0).sum() >= 3
        ),
    }


def q2_decision(
    cell_differences: np.ndarray,
    *,
    family_p: float,
    simultaneous_upper: float,
) -> dict[str, Any]:
    target, panel_seed = aggregate_target_cells(cell_differences)
    return {
        "mean_difference": float(target.mean()),
        "familywise_p": float(family_p),
        "simultaneous_upper": float(simultaneous_upper),
        "within_margin_targets": int((target <= NONINFERIORITY_MARGIN).sum()),
        "target_count": len(target),
        "worst_target_excess": float(target.max()),
        "within_margin_panel_seed_cells": int((panel_seed <= NONINFERIORITY_MARGIN).sum()),
        "pass": bool(
            target.mean() <= NONINFERIORITY_MARGIN
            and simultaneous_upper <= NONINFERIORITY_MARGIN
            and family_p <= 0.05
            and (target <= NONINFERIORITY_MARGIN).mean() >= 0.75
            and target.max() <= 0.20
            and (panel_seed <= NONINFERIORITY_MARGIN).sum() >= 3
        ),
    }


def classify_cross_dataset(
    *,
    q1_methods: Mapping[str, Iterable[str]],
    q2_methods: Mapping[str, Iterable[str]],
    stable: Mapping[str, bool] | None = None,
    blocker: bool = False,
) -> dict[str, Any]:
    q1 = {dataset: set(q1_methods[dataset]) for dataset in DATASETS}
    q2 = {dataset: set(q2_methods[dataset]) for dataset in DATASETS}
    stable = dict(stable or {dataset: True for dataset in DATASETS})
    a = {dataset: q1[dataset] & q2[dataset] for dataset in DATASETS}
    common_a = set.intersection(*(a[dataset] for dataset in DATASETS))
    common_b = set.intersection(*(q1[dataset] for dataset in DATASETS))
    all_stable = all(stable.get(dataset, False) for dataset in DATASETS)
    if blocker:
        gate = GATE_E
    elif not all_stable:
        gate = GATE_D
    elif common_a:
        gate = GATE_A
    elif common_b:
        gate = GATE_B
    elif all(not q1[dataset] for dataset in DATASETS):
        gate = GATE_C
    else:
        gate = GATE_D
    return {
        "gate": gate,
        "A_intersection": sorted(common_a),
        "B_intersection": sorted(common_b),
        "dataset_A": {dataset: sorted(a[dataset]) for dataset in DATASETS},
        "dataset_B": {dataset: sorted(q1[dataset]) for dataset in DATASETS},
        "all_stable": all_stable,
    }


def budget_star(qualifies: Sequence[bool]) -> int | str | None:
    if len(qualifies) != len(COMMON_BUDGETS):
        raise ValueError("C84 common-grid qualification vector length drift")
    for index, passed in enumerate(qualifies):
        if passed and all(qualifies[index:]):
            return COMMON_BUDGETS[index]
    return None


def classify_label_frontier(stars: Mapping[str, int | str | None]) -> str:
    values = [stars[dataset] for dataset in DATASETS]
    if any(value is None for value in values):
        return LABEL_L4
    ordinals = [COMMON_BUDGETS.index(value) for value in values]
    if max(ordinals) - min(ordinals) > 1:
        return LABEL_L3
    if max(ordinals) <= COMMON_BUDGETS.index(4):
        return LABEL_L1
    return LABEL_L2


def atomic_manifest_write(
    final_directory: Path,
    artifacts: Mapping[str, bytes],
    *,
    inject_failure: bool = False,
) -> dict[str, Any]:
    """Synthetic all-or-none manifest freeze used to calibrate C84 failure handling."""
    final_directory = Path(final_directory)
    if final_directory.exists():
        raise ValueError("C84 final directory must not pre-exist")
    staging = Path(tempfile.mkdtemp(prefix=".c84-staging-", dir=final_directory.parent))
    try:
        manifest = []
        for index, (name, payload) in enumerate(sorted(artifacts.items())):
            if Path(name).name != name:
                raise ValueError("C84 synthetic artifact names must be basenames")
            path = staging / name
            path.write_bytes(payload)
            manifest.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            if inject_failure and index == 0:
                raise RuntimeError("injected C84 synthetic atomic failure")
        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        (staging / "artifact_manifest.json").write_bytes(manifest_bytes)
        staging.rename(final_directory)
        return {"artifacts": len(manifest), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _method_sets(method: str | None = None) -> dict[str, set[str]]:
    return {dataset: ({method} if method else set()) for dataset in DATASETS}


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_calibration() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def record(scenario: str, object_: str, expected: str, observed: str, detail: str) -> None:
        rows.append({
            "scenario": scenario,
            "calibration_object": object_,
            "expected": expected,
            "observed": observed,
            "passed": int(expected == observed),
            "real_EEG_arrays_loaded": 0,
            "real_labels_read": 0,
            "detail": detail,
        })

    empty = _method_sets()
    observed = classify_cross_dataset(q1_methods=empty, q2_methods=empty)["gate"]
    record("S0", "all_selectors_null", GATE_C, observed, "no registered Q1 method in any dataset")

    q1 = _method_sets()
    q1["Lee2019_MI"] = {"U13"}
    observed = classify_cross_dataset(q1_methods=q1, q2_methods=empty)["gate"]
    record("S1", "one_dataset_only", GATE_D, observed, "COTT supports Lee only")

    q1 = _method_sets("U13")
    observed = classify_cross_dataset(q1_methods=q1, q2_methods=empty)["gate"]
    record("S2", "same_method_improves_S1", GATE_B, observed, "COTT common Q1 intersection")

    q1 = {"Lee2019_MI": {"U13"}, "Cho2017": {"U7"}, "PhysionetMI": {"U14"}}
    observed = classify_cross_dataset(q1_methods=q1, q2_methods=empty)["gate"]
    record("S3", "different_methods", GATE_D, observed, "method identity intersection is empty")

    q1 = _method_sets("U13")
    q2 = _method_sets("U13")
    observed = classify_cross_dataset(q1_methods=q1, q2_methods=q2)["gate"]
    record("S4", "same_method_noninferior_B1", GATE_A, observed, "COTT common Q1+Q2 intersection")

    stable = {dataset: dataset != "Cho2017" for dataset in DATASETS}
    observed = classify_cross_dataset(q1_methods=q1, q2_methods=q2, stable=stable)["gate"]
    record("S5", "source_panel_heterogeneity", GATE_D, observed, "Cho panel stability gate fails")

    stable = {dataset: dataset != "PhysionetMI" for dataset in DATASETS}
    observed = classify_cross_dataset(q1_methods=q1, q2_methods=q2, stable=stable)["gate"]
    record("S6", "training_seed_heterogeneity", GATE_D, observed, "Physionet seed stability gate fails")

    stable = {dataset: dataset != "Lee2019_MI" for dataset in DATASETS}
    observed = classify_cross_dataset(q1_methods=q1, q2_methods=q2, stable=stable)["gate"]
    record("S7", "target_composition_heterogeneity", GATE_D, observed, "Lee LOTO preservation gate fails")

    stars = {dataset: budget_star([True] * len(COMMON_BUDGETS)) for dataset in DATASETS}
    observed = classify_label_frontier(stars)
    record("S8", "stable_B1_frontier", LABEL_L1, observed, json.dumps(stars, sort_keys=True))

    stars = {"Lee2019_MI": 1, "Cho2017": 4, "PhysionetMI": "FULL"}
    observed = classify_label_frontier(stars)
    record("S9", "dataset_heterogeneous_frontier", LABEL_L3, observed, json.dumps(stars, sort_keys=True))

    # Association and top-k are deliberately not inputs to Q1 taxonomy.
    observed = classify_cross_dataset(q1_methods=empty, q2_methods=empty)["gate"]
    record("S10", "ranking_association_without_regret", GATE_C, observed, "Spearman=0.9;Q1 sets remain empty")
    record("S11", "topk_without_Q1", GATE_C, observed, "top5 improves;Q1 sets remain empty")

    target_effects = np.zeros((22, 2), dtype=float)
    max_t = target_cluster_maxT(target_effects, null_margin=MATERIAL_MARGIN, label="S12")
    observed = "TARGET_CLUSTER_N_22_NULL_NOT_REJECTED" if np.all(max_t["pvalue"] > 0.05) else "FALSE_REJECTION"
    record("S12", "pooled_pseudoreplication_trap", "TARGET_CLUSTER_N_22_NULL_NOT_REJECTED", observed, "trial/model rows never enter N")

    physionet_construction_min = 23 // 2
    finite_common = [budget for budget in COMMON_BUDGETS if isinstance(budget, int)]
    observed = "COMMON_ONLY" if max(finite_common) <= physionet_construction_min < min(protocol.EXTENDED_BUDGETS) else "AVAILABILITY_DRIFT"
    record("S13", "Physionet_low_label_availability", "COMMON_ONLY", observed, f"metadata construction minimum={physionet_construction_min}")

    with tempfile.TemporaryDirectory(prefix="c84p-atomic-") as temporary:
        final = Path(temporary) / "result"
        try:
            atomic_manifest_write(final, {"a.csv": b"x\n", "b.json": b"{}\n"}, inject_failure=True)
            observed = "UNEXPECTED_SUCCESS"
        except RuntimeError:
            observed = "NO_FINAL_DIRECTORY" if not final.exists() else "PARTIAL_VISIBLE"
    record("S14", "atomic_field_result_failure", "NO_FINAL_DIRECTORY", observed, "injected failure after first staging artifact")

    # Exercise target counts 20, 22, and 76 with the exact fixed draw count.
    for dataset, count in TARGET_COUNTS.items():
        effects = np.full((count, 1), 0.20)
        evidence = target_cluster_maxT(effects, null_margin=MATERIAL_MARGIN, label=f"power|{dataset}")
        expected = "PASS"
        observed = "PASS" if evidence["pvalue"][0] <= 0.05 and evidence["draws"][0] == RAK_DRAW_COUNT else "FAIL"
        record(f"MAXT_{dataset}", "target_cluster_calibration", expected, observed, f"targets={count};draws={RAK_DRAW_COUNT}")

    # Verify Q1/Q2 panel-seed aggregation against all locked components.
    q1_cells = np.full((22, 2, 4), 0.20)
    q2_cells = np.zeros((22, 2, 4))
    q1_evidence = target_cluster_maxT(q1_cells.mean(axis=1), null_margin=MATERIAL_MARGIN, label="q1-gate")
    q2_evidence = target_cluster_maxT(NONINFERIORITY_MARGIN - q2_cells.mean(axis=1), null_margin=0.0, label="q2-gate")
    q1_gate = q1_decision(q1_cells, family_p=float(q1_evidence["pvalue"][0]))
    q2_gate = q2_decision(q2_cells, family_p=float(q2_evidence["pvalue"][0]), simultaneous_upper=0.01)
    record("GATE_Q1", "panel_seed_aggregation", "PASS", "PASS" if q1_gate["pass"] else "FAIL", "22 targets x 2 levels x 4 panel-seed cells")
    record("GATE_Q2", "panel_seed_aggregation", "PASS", "PASS" if q2_gate["pass"] else "FAIL", "22 targets x 2 levels x 4 panel-seed cells")

    output = protocol.TABLE_DIR / "synthetic_calibration.csv"
    _write_csv(output, rows)
    passed = sum(row["passed"] for row in rows)
    if passed != len(rows):
        raise RuntimeError(f"C84 synthetic calibration failed: {passed}/{len(rows)}")
    return {
        "checks": len(rows),
        "passed": passed,
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "dataset_loader_imported": False,
        "gate": protocol.FAIL_GATE,
    }


def main() -> int:
    print(json.dumps(run_calibration(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
