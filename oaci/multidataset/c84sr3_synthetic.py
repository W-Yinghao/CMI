"""Full production-path synthetic calibration for the C84SR3 V5 repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .c84s_common import require, sha256_file, write_json
from .c84sr1_stage_a_labels import run_stage_a_from_rows
from .c84sr1_synthetic import (
    Q0_SELECTED, SyntheticLoader, synthetic_contexts, synthetic_label_rows,
    synthetic_score_provider, utility_provider,
)
from .c84sr3_common import METHOD_CONTEXT_ROWS, Q0_RECORDS
from .c84sr3_q0_store import synthetic_payload
from .c84sr3_stage_b_selection import run_stage_b
from .c84sr3_stage_c_evaluation import run_stage_c


def synthetic_q0_builder(**kwargs: Any) -> dict[str, np.ndarray]:
    payload = synthetic_payload(
        kwargs["identity"], kwargs["candidate_ids"], chains=int(kwargs["chains"]),
    )
    codes = np.asarray(payload["finite_budget_code"], dtype=np.uint8)
    for budget, selected in Q0_SELECTED.items():
        mask = codes == budget
        if not np.any(mask):
            continue
        order = np.asarray(
            [selected] + [index for index in range(81) if index != selected],
            dtype=np.uint8,
        )
        payload["finite_candidate_order"][mask] = order
        payload["finite_selected_index"][mask] = selected
    full_order = np.asarray([80] + list(range(80)), dtype=np.uint8)
    payload["FULL_candidate_order"][0] = full_order
    payload["FULL_selected_index"][0] = 80
    return payload


def run_calibration(
    *, root: str | Path, full_chains: int = 2048, branch_chains: int = 8,
) -> dict[str, Any]:
    root = Path(root)
    require(not root.exists(), "C84SR3 synthetic calibration root exists")
    root.mkdir(parents=True)
    started = time.monotonic()
    registry, labels, target_trials = synthetic_label_rows()
    contexts = synthetic_contexts()
    loader = SyntheticLoader(target_trials)
    stage_a_root = root / "stage_a"
    stage_a = run_stage_a_from_rows(
        guard_receipt={"C84S_authorized": True},
        frozen_registry_rows=registry, label_rows=labels,
        output_root=stage_a_root,
    )
    handoff = stage_a_root / "C84S_STAGE_A_HANDOFF.json"
    seal = stage_a_root / "C84S_STAGE_A_EVALUATION_SEAL.json"

    full_selection = root / "full_scale_selection"
    selection_started = time.monotonic()
    full_b = run_stage_b(
        stage_a_handoff_path=handoff, final_root=full_selection,
        contexts=contexts, context_loader=loader,
        score_provider=synthetic_score_provider,
        q0_builder=synthetic_q0_builder, chains=full_chains,
        synthetic=True,
    )
    require(full_b["Q0_records"] == Q0_RECORDS,
            "C84SR3 full-scale Q0 arithmetic drift")
    selection_seconds = time.monotonic() - selection_started
    full_result_root = root / "full_scale_result"
    result_started = time.monotonic()
    full_result = run_stage_c(
        selection_root=full_selection, evaluation_seal_path=seal,
        final_root=full_result_root, contexts=contexts,
        context_loader=loader, utility_provider=utility_provider("C_L1"),
        q0_chains=full_chains, maxT_draws=65536, synthetic=True,
    )
    require(full_result["method_context_rows"] == METHOD_CONTEXT_ROWS,
            "C84SR3 full-scale method-context arithmetic drift")
    result_seconds = time.monotonic() - result_started

    branch_selection = root / "branch_selection"
    run_stage_b(
        stage_a_handoff_path=handoff, final_root=branch_selection,
        contexts=contexts, context_loader=loader,
        score_provider=synthetic_score_provider,
        q0_builder=synthetic_q0_builder, chains=branch_chains,
        synthetic=True,
    )
    scenarios = (
        "A_L1", "B_L1", "C_L1", "D_METHOD_L3", "D_LEVEL_L1",
        "C_L2", "C_L4", "E_L1",
    )
    branch_results = {}
    for scenario in scenarios:
        result_root = root / f"result_{scenario}"
        result = run_stage_c(
            selection_root=branch_selection, evaluation_seal_path=seal,
            final_root=result_root, contexts=contexts,
            context_loader=loader, utility_provider=utility_provider(scenario),
            q0_chains=branch_chains, maxT_draws=64, synthetic=True,
            blocker=scenario == "E_L1",
        )
        branch_results[scenario] = {
            "primary_gate": result["primary_gate"],
            "label_frontier_tag": result["label_frontier_tag"],
            "result_sha256": sha256_file(result_root / "C84S_RESULT.json"),
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
                f"C84SR3 taxonomy calibration drift: {scenario}: {observed}")

    availability_path = full_selection / "q0_budget_availability.csv"
    availability_text = availability_path.read_text(encoding="utf-8")
    require("Lee2019_MI,32,SECONDARY,22,0,22,25,25,0,INPUT_UNAVAILABLE_ALL_TARGETS" in availability_text,
            "C84SR3 Lee B32 synthetic availability witness absent")
    require("Cho2017,32,SECONDARY,20,20,0,50,50,1,FEASIBLE" in availability_text,
            "C84SR3 Cho B32 synthetic availability witness absent")

    summary = {
        "schema_version": "c84sr3_production_path_synthetic_calibration_v1",
        "status": "PASS", "contexts": 944, "candidates_per_context": 81,
        "full_scale_Q0_chains": full_chains,
        "full_scale_Q0_records": full_b["Q0_records"],
        "full_scale_method_context_rows": full_result["method_context_rows"],
        "full_scale_primary_gate": full_result["primary_gate"],
        "full_scale_label_frontier_tag": full_result["label_frontier_tag"],
        "stage_A_sha256": stage_a["sha256"],
        "selection_freeze_sha256": full_b["sha256"],
        "scientific_result_sha256": sha256_file(full_result_root / "C84S_RESULT.json"),
        "availability_sha256": sha256_file(availability_path),
        "Lee_B32_status": "INPUT_UNAVAILABLE_NO_SELECTION_OR_RESULT_ROW",
        "Cho_B32_status": "OPERATIVE_SECONDARY",
        "branch_chains": branch_chains, "branch_results": branch_results,
        "timing_seconds": {
            "full_selection": selection_seconds,
            "full_stage_C": result_seconds,
            "total": time.monotonic() - started,
        },
        "external_root_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "precomputed_method_context_rows_injected": False,
        "real_field_array_access": 0, "real_target_label_access": 0,
        "real_selector_scores": 0, "real_scientific_statistics": 0,
    }
    digest = write_json(root / "C84SR3_SYNTHETIC_CALIBRATION.json", summary)
    return {**summary, "summary_sha256": digest, "root": str(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_calibration(root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
