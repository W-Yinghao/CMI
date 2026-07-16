"""C84SR3 Stage-C evaluation over the immutable V3 selection freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import c84s_evaluation as evaluation
from .c84s_common import read_json, require, sha256_file, write_json
from .c84sr1_common import MAXT_DRAWS, Q0_CHAINS
from .c84sr1_context_enumerator import ContextDescriptor, enumerate_contexts
from .c84sr1_field_reader import FrozenZooReader, zoo_then_target_key
from .c84sr1_stage_c_evaluation import (
    _evaluation_labels_for_context, _load_evaluation_seal, _load_selection_tables,
)
from .c84sr3_analysis import run_analysis_and_freeze_v3
from .c84sr3_common import (
    METHOD_CONTEXT_ROWS, Q0_RECORDS, Q0_STAGE_C_MC_ROWS,
    Q0_STAGE_C_REGIME_ROWS, finite_budgets,
)
from .c84sr3_method_context_materialization import materialize_context
from .c84sr3_q0_store import read_context_shard


def verify_selection_freeze(
    root: Path, *, expected_contexts: int = 944,
    expected_q0_records: int = Q0_RECORDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "C84S_SELECTION_FREEZE_MANIFEST_V3.json"
    handoff_path = root / "C84S_STAGE_B_HANDOFF_V2.json"
    require(manifest_path.is_file() and handoff_path.is_file(),
            "C84SR3 Stage-B freeze handoff absent")
    manifest = read_json(manifest_path)
    handoff = read_json(handoff_path)
    manifest_sha = sha256_file(manifest_path)
    require(manifest["schema_version"] == "c84sr3_selection_freeze_manifest_v3",
            "C84SR3 selection-freeze schema drift")
    require(manifest["status"] == "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
            "C84SR3 Stage-B selection is not frozen")
    require(handoff["selection_freeze_manifest_sha256"] == manifest_sha,
            "C84SR3 Stage-B handoff manifest identity drift")
    require(handoff["evaluation_descriptor_received"] is False and
            manifest["evaluation_label_descriptor_received"] is False,
            "evaluation descriptor reached C84SR3 Stage B")
    require(manifest["same_label_oracle_accessed"] is False,
            "oracle access recorded in C84SR3 selection freeze")
    require(manifest["contexts"] == expected_contexts and
            manifest["Q0_records"] == expected_q0_records,
            "C84SR3 selection-freeze arithmetic drift")
    require(manifest["Lee_Q0_B32_status"] ==
            "INPUT_UNAVAILABLE_NO_SELECTION_OR_RESULT_ROW",
            "Lee B32 availability status drift")
    for identity in manifest["artifacts"].values():
        path = root / identity["path"]
        require(path.is_file() and sha256_file(path) == identity["sha256"],
                f"C84SR3 Stage-B artifact identity drift: {identity['path']}")
    return manifest, {
        "status": manifest["status"], "sha256": manifest_sha, "root": str(root),
    }


def run_stage_c(
    *, selection_root: str | Path, evaluation_seal_path: str | Path,
    final_root: str | Path,
    contexts: Sequence[ContextDescriptor] | None = None,
    context_loader: Callable[[ContextDescriptor], Mapping[str, Any]] | None = None,
    utility_provider: Callable[[ContextDescriptor, Mapping[str, Any], Sequence[Mapping[str, Any]]], tuple[np.ndarray, np.ndarray]] | None = None,
    q0_chains: int = Q0_CHAINS, maxT_draws: int = MAXT_DRAWS,
    synthetic: bool = False, blocker: bool = False,
    failure_injection_context: int | None = None,
) -> dict[str, Any]:
    selection_root = Path(selection_root)
    contexts = sorted(
        list(enumerate_contexts() if contexts is None else contexts),
        key=zoo_then_target_key,
    )
    expected_q0_records = sum(
        len(finite_budgets(context.dataset)) * int(q0_chains) + 1
        for context in contexts
    )
    manifest, selection_identity = verify_selection_freeze(
        selection_root, expected_contexts=len(contexts),
        expected_q0_records=expected_q0_records,
    )
    evaluation_descriptor, evaluation_rows = _load_evaluation_seal(evaluation_seal_path)
    require(int(manifest["Q0_chains"]) == int(q0_chains),
            "C84SR3 Stage-B/Stage-C Q0 chain count drift")
    if not synthetic:
        require(len(contexts) == 944 and q0_chains == Q0_CHAINS and
                maxT_draws == MAXT_DRAWS, "real C84SR3 Stage-C scope reduction")
    scores, fixed, shards = _load_selection_tables(selection_root, contexts)
    loader = FrozenZooReader(include_source=False) if context_loader is None else context_loader
    method_rows: list[dict[str, Any]] = []
    q0_regime_rows: list[dict[str, Any]] = []
    q0_mc_rows: list[dict[str, Any]] = []
    for position, context in enumerate(contexts):
        if failure_injection_context is not None and position == failure_injection_context:
            raise RuntimeError("injected C84SR3 Stage-C materialization failure")
        data = dict(loader(context))
        candidate_ids = [row.unit_id for row in context.candidates]
        require(list(map(str, data["candidate_ids"])) == candidate_ids,
                "C84SR3 Stage-C candidate identity drift")
        if utility_provider is None:
            evaluation_index, labels = _evaluation_labels_for_context(
                context, np.asarray(data["target_trial_ids"], dtype=str),
                evaluation_rows,
            )
            utility, evaluation_metrics = evaluation.context_candidate_utility(
                np.asarray(data["target_logits"], dtype=float)[:, evaluation_index], labels,
            )
        else:
            utility, evaluation_metrics = utility_provider(context, data, evaluation_rows)
        shard_identity = shards[context.context_id]
        q0_payload, replay = read_context_shard(
            selection_root / shard_identity["path"],
            expected_sha256=shard_identity["sha256"], chains=q0_chains,
        )
        require(replay["context_id"] == context.context_id,
                "C84SR3 Q0 shard/context mismatch")
        rows, regimes, diagnostics = materialize_context(
            identity=context.identity(), candidate_ids=candidate_ids,
            regimes=[row.regime for row in context.candidates],
            utility=np.asarray(utility, dtype=float),
            evaluation_metrics=np.asarray(evaluation_metrics, dtype=float),
            score_vectors=scores[context.context_id],
            fixed_selected_indices=fixed[context.context_id],
            q0_payload=q0_payload, q0_chains=q0_chains,
        )
        method_rows.extend(rows)
        q0_regime_rows.extend(regimes)
        q0_mc_rows.extend(diagnostics)
    if not synthetic:
        require(len(method_rows) == METHOD_CONTEXT_ROWS,
                "C84SR3 Stage-C method-context row-count drift")
        require(len(q0_regime_rows) == Q0_STAGE_C_REGIME_ROWS,
                "C84SR3 Stage-C Q0 regime row-count drift")
        require(len(q0_mc_rows) == Q0_STAGE_C_MC_ROWS,
                "C84SR3 Stage-C Q0 MC row-count drift")
    return run_analysis_and_freeze_v3(
        method_rows, q0_regime_rows=q0_regime_rows, q0_mc_rows=q0_mc_rows,
        selection_freeze_identity=selection_identity,
        evaluation_view_identity=evaluation_descriptor,
        final_root=final_root, draws=maxT_draws, blocker=blocker,
        synthetic=synthetic,
    )


def run_stage_c_real(
    *, selection_root: str | Path, evaluation_seal_path: str | Path,
    final_root: str | Path,
) -> dict[str, Any]:
    attempt_path = Path(final_root).parent / f"{Path(final_root).name}.stage_c_attempt.json"
    require(not attempt_path.exists(), "C84SR3 Stage-C attempt exists")
    attempt = {
        "schema_version": "c84sr3_stage_attempt_v2", "stage": "Stage_C",
        "status": "STARTED", "started_at_unix_ns": time.time_ns(),
        "selection_mutation": 0, "construction_descriptor_received": 0,
        "evaluation_descriptor_received": 1,
    }
    write_json(attempt_path, attempt)
    try:
        result = run_stage_c(
            selection_root=selection_root,
            evaluation_seal_path=evaluation_seal_path,
            final_root=final_root,
        )
        attempt.update({"status": "COMPLETE", "finished_at_unix_ns": time.time_ns(), "result": result})
        write_json(attempt_path, attempt)
        return result
    except BaseException as error:
        attempt.update({
            "status": "FAILED", "finished_at_unix_ns": time.time_ns(),
            "error_type": type(error).__name__, "error": str(error),
            "error_notes": list(getattr(error, "__notes__", ())),
            "partial_final_root": Path(final_root).exists(),
        })
        write_json(attempt_path, attempt)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--evaluation-seal", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = run_stage_c_real(
        selection_root=args.selection_root,
        evaluation_seal_path=args.evaluation_seal,
        final_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
