"""Receipt-bound C85T V3 coordinator; C85TR2 does not invoke run-locked."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85t_exact_scenarios import as_fraction
from .c85t_execution_context_v3 import (
    create_validated_c85t_execution_context,
    sha256_file,
)
from .c85t_monte_carlo import _summarize_s9_arrays_v2, summarize_near_replicates_v2
from .c85t_registered_v3 import (
    execute_proof_candidate_pipeline_v3,
    execute_registered_exact_scenarios_v3,
    execute_registered_monte_carlo_v3,
)
from .c85t_result_manifest import read_deterministic_npz
from .c85t_semantic_replay_v3 import RESULT_SCHEMA_V3, SUCCESS_GATE_V3
from .c85t_transaction_v3 import AtomicExecutionBundleV3, preserve_primary_exception_v3


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise DecisionContractError("C85T V3 cannot write an empty CSV")
    if path.exists():
        raise DecisionContractError("C85T V3 CSV path must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(values[0]))
        writer.writeheader()
        writer.writerows(values)


def run_real(
    *, lock_path: Path, authorization_record: Path, output_root: Path
) -> dict[str, Any]:
    context = create_validated_c85t_execution_context(
        lock_path,
        authorization_record,
        output_root,
    )
    bundle = AtomicExecutionBundleV3(context)
    lock = context.lock
    repo_root = lock_path.resolve().parents[2]
    contract = json.loads((repo_root / lock["v2_generator_path"]).read_text())
    operationalization = json.loads(
        (repo_root / lock["c85tl_operationalization_path"]).read_text()
    )
    statements = operationalization["proof_statements"]
    try:
        context._lifecycle.append("EXACT_SCENARIOS_STARTED")
        exact = execute_registered_exact_scenarios_v3(contract, context=context)
        exact_path = bundle.write_json("exact_scenario_results.json", exact)
        context._lifecycle.append(
            "EXACT_SCENARIOS_COMPLETED",
            artifact_or_receipt_sha256=sha256_file(exact_path),
        )

        context._lifecycle.append("MONTE_CARLO_STARTED")
        monte = execute_registered_monte_carlo_v3(contract, context=context)
        summaries: dict[str, Any] = {}
        for scenario_id in ("S6", "S7"):
            path = bundle.write_npz(
                f"{scenario_id}_replicates.npz", monte[scenario_id]["arrays"]
            )
            loaded = read_deterministic_npz(path)
            summaries[scenario_id] = summarize_near_replicates_v2(
                scenario_id,
                loaded,
                monte[scenario_id]["summary"]["geometry"],
            )
        s9_path = bundle.write_npz("S9_replicates.npz", monte["S9"]["arrays"])
        s9_loaded = read_deterministic_npz(s9_path)
        scenarios = {row["id"]: row for row in contract["scenarios"]}
        population = np.asarray(
            [
                float(as_fraction(value))
                for value in scenarios["S9"]["population_mean_losses"]
            ],
            dtype="<f8",
        )
        summaries["S9"] = _summarize_s9_arrays_v2(s9_loaded, population)
        summaries["S9"]["analytic_variance"] = monte["S9"]["summary"][
            "analytic_variance"
        ]
        summaries["S9"]["universal_active_superiority_claim"] = False
        summaries["S9_population_mean_losses"] = population.tolist()
        _write_csv(
            bundle.path("S9_raw_draw_digest_registry.csv"),
            monte["S9"]["raw_draw_digest_rows"],
        )
        summary_path = bundle.write_json("monte_carlo_summary.json", summaries)
        context._lifecycle.append(
            "MONTE_CARLO_COMPLETED",
            artifact_or_receipt_sha256=sha256_file(summary_path),
        )

        context._lifecycle.append("PROOF_CANDIDATES_STARTED")
        dispositions = execute_proof_candidate_pipeline_v3(
            statements=statements,
            exact_results=exact,
            output_dir=bundle.path("c85t_proof_candidates"),
            dispositions_path=bundle.path("proof_candidate_dispositions.csv"),
            context=context,
        )
        disposition_path = bundle.path("proof_candidate_dispositions.csv")
        context._lifecycle.append(
            "PROOF_CANDIDATES_COMPLETED",
            artifact_or_receipt_sha256=sha256_file(disposition_path),
        )

        result = {
            "schema_version": RESULT_SCHEMA_V3,
            "final_gate": SUCCESS_GATE_V3,
            "execution_lock_sha256": context.execution_lock_sha256,
            "execution_lock_commit": context.execution_lock_commit,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "authorization_file_sha256": context.authorization_file_sha256,
            "authorization_id": context.authorization_id,
            "attempt_id": context.attempt_id,
            "output_root": str(context.output_root),
            "HEAD": context.head,
            "scenario_count": len(exact),
            "S6_S7_logical_replicate_rows": 8192,
            "S9_logical_replicate_design_rows": 8192,
            "S9_raw_draw_digest_rows": len(monte["S9"]["raw_draw_digest_rows"]),
            "proof_candidate_count": len(dispositions),
            "proof_candidate_dispositions": dispositions,
            "formal_theorem_statuses": {f"T{index}": "OPEN" for index in range(1, 8)},
            "real_project_data_access": 0,
            "active_acquisition": 0,
            "C85V_authorized": False,
            "C85E_authorized": False,
            "manuscript_modified": False,
        }
        return bundle.publish(result, contract=contract, statements=statements)
    except BaseException as primary:
        try:
            recovered = preserve_primary_exception_v3(context, primary)
        except BaseException:
            recovered = None
        if recovered is not None:
            return recovered
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-locked")
    run.add_argument("--execution-lock", type=Path, required=True)
    run.add_argument("--authorization-record", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command != "run-locked":
        raise DecisionContractError("unknown C85T V3 command")
    result = run_real(
        lock_path=args.execution_lock.resolve(),
        authorization_record=args.authorization_record.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
