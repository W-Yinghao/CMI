"""Production-path C84SR1 synthetic orchestration and taxonomy calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .c84s_common import require, sha256_file, write_json
from .c84sr1_common import (
    DATASET_TARGET_COUNTS, SCORE_METHODS, context_id, context_identity,
)
from .c84sr1_context_enumerator import CandidateDescriptor, ContextDescriptor
from .c84sr1_q0_store import synthetic_payload
from .c84sr1_stage_a_labels import run_stage_a_from_rows
from .c84sr1_stage_b_selection import run_stage_b
from .c84sr1_stage_c_evaluation import run_stage_c


SELECTED_BY_METHOD = {
    "S1": 1, "U5": 5, "U7": 7, "U11": 11,
    "U13": 13, "U14": 14, "U15": 15,
}
Q0_SELECTED = {1: 21, 2: 22, 4: 24, 8: 28, 16: 36, 32: 52}


def synthetic_trial_counts() -> dict[str, list[int]]:
    return {
        "Lee2019_MI": [100] * 22,
        "Cho2017": [200] * 20,
        "PhysionetMI": [36, 43, 57] + [45] * 73,
    }


def synthetic_label_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], list[str]]]:
    registry: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    target_trials: dict[tuple[str, str], list[str]] = {}
    for dataset, counts in synthetic_trial_counts().items():
        require(len(counts) == DATASET_TARGET_COUNTS[dataset], "synthetic target-count contract drift")
        for target_index, count in enumerate(counts, 1):
            target = str(target_index)
            class0 = count // 2
            ids = []
            for trial_index in range(count):
                trial_id = f"{dataset}|subject={target}|session=0|run=0|trial={trial_index:05d}"
                row = {
                    "dataset": dataset, "target_subject_id": target,
                    "target_trial_id": trial_id, "session": "0", "run": "0",
                }
                registry.append(dict(row))
                labels.append({**row, "canonical_class_label": int(trial_index >= class0)})
                ids.append(trial_id)
            target_trials[(dataset, target)] = ids
    require(len(registry) == len(labels) == 9621, "synthetic 9,621-row label arithmetic drift")
    return registry, labels, target_trials


def synthetic_contexts() -> list[ContextDescriptor]:
    zoos: dict[tuple[str, str, int, int], tuple[CandidateDescriptor, ...]] = {}
    for dataset in DATASET_TARGET_COUNTS:
        for panel in ("A", "B"):
            for seed in (5, 6):
                for level in (0, 1):
                    candidates = []
                    for index in range(81):
                        regime = "ERM" if index == 0 else "OACI" if index <= 40 else "SRC"
                        order = 0 if index == 0 else index if index <= 40 else index - 40
                        unit_id = f"syn_{dataset[:4]}_{panel}{seed}{level}_{index:02d}"
                        candidates.append(CandidateDescriptor(
                            dataset=dataset, panel=panel, training_seed=seed, level=level,
                            regime=regime, epoch=order, trajectory_order=order,
                            unit_id=unit_id, level_intervention_id=f"SYN_LEVEL_{level}",
                            source_audit_path="SYNTHETIC", source_audit_sha256="0" * 64,
                            target_artifact_path="SYNTHETIC", target_artifact_sha256="0" * 64,
                            training_sidecar_path="SYNTHETIC", training_sidecar_sha256="0" * 64,
                            target_sidecar_path="SYNTHETIC", target_sidecar_sha256="0" * 64,
                        ))
                    zoos[(dataset, panel, seed, level)] = tuple(candidates)
    contexts = []
    for dataset, target_count in DATASET_TARGET_COUNTS.items():
        for target_index in range(1, target_count + 1):
            for panel in ("A", "B"):
                for seed in (5, 6):
                    for level in (0, 1):
                        identity = context_identity(dataset, str(target_index), panel, seed, level)
                        contexts.append(ContextDescriptor(
                            context_id=context_id(identity), dataset=dataset,
                            target_subject_id=str(target_index), panel=panel,
                            training_seed=seed, level=level,
                            candidates=zoos[(dataset, panel, seed, level)],
                        ))
    require(len(contexts) == 944 and len({row.context_id for row in contexts}) == 944,
            "synthetic context arithmetic drift")
    return contexts


class SyntheticLoader:
    def __init__(self, target_trials: Mapping[tuple[str, str], Sequence[str]]):
        self.target_trials = target_trials

    def __call__(self, context: ContextDescriptor) -> dict[str, Any]:
        trial_ids = np.asarray(self.target_trials[(context.dataset, context.target_subject_id)], dtype=str)
        candidates = context.candidates
        return {
            "candidate_ids": [row.unit_id for row in candidates],
            "regimes": [row.regime for row in candidates],
            "trajectory_orders": [row.trajectory_order for row in candidates],
            "source_probabilities": np.full((81, 4, 2), 0.5, dtype=float),
            "source_labels": np.asarray([0, 0, 1, 1], dtype=int),
            "source_domains": np.asarray([1, 1, 2, 2], dtype=int),
            "source_trial_ids": np.asarray(["s0", "s1", "s2", "s3"], dtype=str),
            "target_logits": np.zeros((81, len(trial_ids), 2), dtype=float),
            "target_trial_ids": trial_ids,
        }


def synthetic_score_provider(
    context: ContextDescriptor, data: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    del context, data
    output = {}
    for method in SCORE_METHODS:
        score = -np.arange(81, dtype=float)
        score[SELECTED_BY_METHOD[method]] = 100.0
        output[method] = score
    return output


def synthetic_q0_builder(**kwargs: Any) -> dict[str, np.ndarray]:
    payload = synthetic_payload(
        kwargs["identity"], kwargs["candidate_ids"], chains=int(kwargs["chains"]),
    )
    codes = np.asarray(payload["finite_budget_code"], dtype=np.uint8)
    for budget, selected in Q0_SELECTED.items():
        mask = codes == budget
        if not np.any(mask):
            continue
        order = np.asarray([selected] + [index for index in range(81) if index != selected], dtype=np.uint8)
        payload["finite_candidate_order"][mask] = order
        payload["finite_selected_index"][mask] = selected
    full_order = np.asarray([80] + list(range(80)), dtype=np.uint8)
    payload["FULL_candidate_order"][0] = full_order
    payload["FULL_selected_index"][0] = 80
    return payload


def _utility_values(context: ContextDescriptor, scenario: str) -> np.ndarray:
    utility = np.linspace(0.01, 0.15, 81, dtype=float)
    utility[0], utility[80], utility[1] = 0.0, 1.0, 0.2
    for index in SELECTED_BY_METHOD.values():
        if index != 1:
            utility[index] = 0.1
    q_values = {1: 0.90, 2: 0.92, 4: 0.94, 8: 0.96, 16: 0.97, 32: 0.98}
    if scenario == "A_L1":
        utility[13] = 0.98
        q_values[1] = 0.96
    elif scenario == "B_L1":
        utility[13] = 0.85
        q_values[1] = 0.98
    elif scenario == "D_METHOD_L3":
        selected = {"Lee2019_MI": 13, "Cho2017": 14, "PhysionetMI": 15}[context.dataset]
        utility[selected] = 0.85
        if context.dataset == "Lee2019_MI":
            q_values.update({1: 0.95})
        elif context.dataset == "Cho2017":
            q_values.update({1: 0.2, 2: 0.2, 4: 0.2, 8: 0.95})
        else:
            q_values.update({1: 0.2, 2: 0.2, 4: 0.2, 8: 0.2})
    elif scenario == "D_LEVEL_L1":
        utility[13] = 0.95 if context.level == 0 else 0.1
        q_values[1] = 0.98
    elif scenario == "C_L2":
        q_values.update({1: 0.2, 2: 0.2, 4: 0.2, 8: 0.9, 16: 0.95, 32: 0.97})
    elif scenario == "C_L4":
        utility[1] = 1.0
        q_values = {budget: 1.0 for budget in q_values}
    elif scenario not in {"C_L1", "E_L1"}:
        raise ValueError(f"unknown C84SR1 synthetic scenario: {scenario}")
    for budget, index in Q0_SELECTED.items():
        utility[index] = q_values[budget]
    return utility


def utility_provider(scenario: str):
    def provider(
        context: ContextDescriptor, data: Mapping[str, Any],
        evaluation_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        del data, evaluation_rows
        utility = _utility_values(context, scenario)
        metrics = np.column_stack((utility, 1.0 - utility, 1.0 - utility))
        return utility, metrics
    return provider


def run_production_path_calibration(
    *, root: str | Path, full_chains: int = 2048, branch_chains: int = 8,
) -> dict[str, Any]:
    root = Path(root)
    require(not root.exists(), "C84SR1 synthetic calibration root exists")
    root.mkdir(parents=True)
    calibration_started = time.monotonic()
    registry, labels, target_trials = synthetic_label_rows()
    contexts = synthetic_contexts()
    loader = SyntheticLoader(target_trials)
    stage_a_root = root / "stage_a"
    stage_a = run_stage_a_from_rows(
        guard_receipt={"C84S_authorized": True}, frozen_registry_rows=registry,
        label_rows=labels, output_root=stage_a_root,
    )
    handoff = stage_a_root / "C84S_STAGE_A_HANDOFF.json"
    seal = stage_a_root / "C84S_STAGE_A_EVALUATION_SEAL.json"

    full_selection = root / "full_scale_selection"
    full_selection_started = time.monotonic()
    full_b = run_stage_b(
        stage_a_handoff_path=handoff, final_root=full_selection,
        contexts=contexts, context_loader=loader,
        score_provider=synthetic_score_provider, q0_builder=synthetic_q0_builder,
        chains=full_chains, synthetic=True,
    )
    require(full_b["Q0_records"] == 9110448, "full-scale synthetic Q0 arithmetic drift")
    full_selection_seconds = time.monotonic() - full_selection_started
    full_result_root = root / "full_scale_result"
    full_result_started = time.monotonic()
    full_result = run_stage_c(
        selection_root=full_selection, evaluation_seal_path=seal,
        final_root=full_result_root, contexts=contexts, context_loader=loader,
        utility_provider=utility_provider("C_L1"), q0_chains=full_chains,
        maxT_draws=65536, synthetic=True,
    )
    full_result_seconds = time.monotonic() - full_result_started

    branch_selection = root / "branch_selection"
    run_stage_b(
        stage_a_handoff_path=handoff, final_root=branch_selection,
        contexts=contexts, context_loader=loader,
        score_provider=synthetic_score_provider, q0_builder=synthetic_q0_builder,
        chains=branch_chains, synthetic=True,
    )
    scenarios = ("A_L1", "B_L1", "C_L1", "D_METHOD_L3", "D_LEVEL_L1", "C_L2", "C_L4", "E_L1")
    branch_results = {}
    for scenario in scenarios:
        result = run_stage_c(
            selection_root=branch_selection, evaluation_seal_path=seal,
            final_root=root / f"result_{scenario}", contexts=contexts,
            context_loader=loader, utility_provider=utility_provider(scenario),
            q0_chains=branch_chains, maxT_draws=64,
            synthetic=True, blocker=scenario == "E_L1",
        )
        branch_results[scenario] = {
            "primary_gate": result["primary_gate"],
            "label_frontier_tag": result["label_frontier_tag"],
            "result_sha256": sha256_file(root / f"result_{scenario}/C84S_RESULT.json"),
        }
    expected = {
        "A_L1": ("C84-A_same_zero_label_selector_matches_B1_across_all_external_datasets", "C84-L1"),
        "B_L1": ("C84-B_same_zero_label_selector_improves_source_across_all_external_datasets_but_not_B1", "C84-L1"),
        "C_L1": ("C84-C_no_registered_zero_label_selector_materially_improves_source_in_any_external_dataset", "C84-L1"),
        "D_METHOD_L3": ("C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous", "C84-L3"),
        "D_LEVEL_L1": ("C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous", "C84-L1"),
        "C_L2": ("C84-C_no_registered_zero_label_selector_materially_improves_source_in_any_external_dataset", "C84-L2"),
        "C_L4": ("C84-C_no_registered_zero_label_selector_materially_improves_source_in_any_external_dataset", "C84-L4"),
        "E_L1": ("C84-E_multidataset_protocol_field_view_analysis_or_provenance_blocker", "C84-L1"),
    }
    for scenario, wanted in expected.items():
        observed = branch_results[scenario]
        require((observed["primary_gate"], observed["label_frontier_tag"]) == wanted,
                f"synthetic taxonomy calibration drift: {scenario}: {observed}")
    summary = {
        "schema_version": "c84sr1_production_path_synthetic_calibration_v1",
        "status": "PASS", "contexts": 944, "candidates_per_context": 81,
        "full_scale_Q0_chains": full_chains,
        "full_scale_Q0_records": full_b["Q0_records"],
        "full_scale_method_context_rows": full_result["method_context_rows"],
        "full_scale_primary_gate": full_result["primary_gate"],
        "full_scale_label_frontier_tag": full_result["label_frontier_tag"],
        "stage_A_sha256": stage_a["sha256"],
        "full_selection_manifest_sha256": full_b["sha256"],
        "full_result_sha256": sha256_file(full_result_root / "C84S_RESULT.json"),
        "branch_chains": branch_chains,
        "branch_results": branch_results,
        "timing_seconds": {
            "full_selection": full_selection_seconds,
            "full_stage_C": full_result_seconds,
            "total": time.monotonic() - calibration_started,
        },
        "external_root_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "precomputed_method_context_rows_injected": False,
        "real_field_array_access": 0, "real_target_label_access": 0,
        "real_selector_scores": 0, "real_scientific_statistics": 0,
    }
    summary_sha = write_json(root / "C84SR1_SYNTHETIC_CALIBRATION.json", summary)
    return {**summary, "summary_sha256": summary_sha, "root": str(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_production_path_calibration(root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
