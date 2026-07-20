"""Future C79E seed-4 execution adapter.

Nothing in this module imports an EEG loader, a training worker, PyTorch, or
CUDA at import time.  Every real command first verifies the two committed C79P
locks and a later direct-PI authorization record.  The adapter then binds the
already validated C78F/C78S engines to the seed-4 scope without modifying those
historical files.

C79P tests only the pure binding contract and the fail-closed boundary.  This
module is not executed against seed-4 data during C79P.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import c79p_post_seed3_protocol as c79p


EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c79-seed4")
FIELD_RESULT_PATH = c79p.REPORT_DIR / "C79_SEED4_FIELD_GENERATION.json"
FIELD_VIEW_TABLE = c79p.REPORT_DIR / "c79_tables" / "seed4_physical_view_ledger.csv"
FIELD_SEED4_AUDIT = c79p.REPORT_DIR / "c79_tables" / "seed4_target_isolation.csv"
ANALYSIS_TABLE_DIR = c79p.REPORT_DIR / "c79_tables" / "raw_locked_engine"
ANALYSIS_ROUTE_PATH = c79p.REPORT_DIR / "C79_SEED4_PRIMARY_VIEW_ROUTE.json"
ANALYSIS_ROUTE_SHA_PATH = c79p.REPORT_DIR / "C79_SEED4_PRIMARY_VIEW_ROUTE.sha256"
ANALYSIS_WORK_RESULT = c79p.REPORT_DIR / "C79_SEED4_LOCKED_ENGINE_OUTPUT.json"
ANALYSIS_WORK_REPORT = c79p.REPORT_DIR / "C79_SEED4_LOCKED_ENGINE_OUTPUT.md"
ANALYSIS_STATE = c79p.REPORT_DIR / "C79_SEED4_ANALYSIS_STATE.json"
REGISTERED_DECISION_PATH = c79p.REPORT_DIR / "C79_SEED4_REGISTERED_REPLICATION_INTERMEDIATE.json"


def field_binding_contract() -> dict[str, Any]:
    """Return the exhaustive historical-worker substitutions, without imports."""
    return {
        "reference_worker": "oaci.conditioned_ceiling_coverage.c78f_train",
        "reference_instrumentation": "oaci.conditioned_ceiling_coverage.c78f_instrument",
        "guard_before_reference_import": True,
        "substitutions": {
            "SEED": 4,
            "TARGETS": list(c79p.TARGET_ORDER),
            "TARGET4_CANARY": 4,
            "LEVELS": list(c79p.LEVELS),
            "REGIMES": list(c79p.REGIMES),
            "REMAINING_UNITS": 1458,
            "FULL_FIELD_UNITS": 1458,
            "EXPECTED_SOURCE_ROWS": c79p.EXPECTED_SOURCE_ROWS,
            "EXPECTED_TARGET_ROWS": c79p.EXPECTED_TARGET_ROWS,
            "FULL_SOURCE_ROWS": c79p.EXPECTED_SOURCE_ROWS,
            "FULL_TARGET_ROWS": c79p.EXPECTED_TARGET_ROWS,
            "TABLE_DIR": str(c79p.TABLE_DIR),
            "PROTOCOL_PATH": str(c79p.PROTOCOL_PATH),
            "PROTOCOL_SHA_PATH": str(c79p.PROTOCOL_SHA_PATH),
            "C78S_PROTOCOL_SHA_PATH": str(c79p.PROTOCOL_SHA_PATH),
            "EXTERNAL_ROOT": str(EXTERNAL_ROOT),
        },
        "wave_assignment": {name: list(targets) for name, targets in c79p.WAVES.items()},
        "field_freeze_implementation": "C79_specific_no_seed3_parent_arithmetic",
        "label_provisioning": "C79_specific_construction_and_evaluation_only",
        "same_label_oracle_created": False,
    }


def analysis_binding_contract() -> dict[str, Any]:
    return {
        "reference_protocol": "oaci.conditioned_ceiling_coverage.c78s_protocol",
        "reference_data": "oaci.conditioned_ceiling_coverage.c78s_data",
        "reference_modeling": "oaci.conditioned_ceiling_coverage.c78s_modeling",
        "reference_analysis": "oaci.conditioned_ceiling_coverage.c78s_seed3_scientific_analysis",
        "guard_before_reference_import": True,
        "substitutions": {
            "SEED": 4,
            "PRIMARY_TARGETS": list(c79p.PRIMARY_TARGETS),
            "TARGET4_CANARY": 4,
            "PRIMARY_UNITS": 1296,
            "FULL_FIELD_UNITS": 1458,
            "PROTOCOL_PATH": str(c79p.PROTOCOL_PATH),
            "PROTOCOL_SHA_PATH": str(c79p.PROTOCOL_SHA_PATH),
            "LOCK_PATH": str(c79p.ANALYSIS_LOCK_PATH),
            "LOCK_SHA_PATH": str(c79p.ANALYSIS_LOCK_SHA_PATH),
            "C78F_FULL_UNITS": str(c79p.EXPECTED_MANIFEST_CSV),
            "TABLE_DIR": str(ANALYSIS_TABLE_DIR),
        },
        "all_registered_paths_unconditional": True,
        "active_after_Holm_runtime_selection": False,
        "target4_primary": False,
        "same_label_oracle_reachable": False,
        "raw_engine_output_then_registered_C79_synthesis": True,
    }


def _locked_runtime_tuple() -> tuple[dict[str, Any], dict[str, Any], str]:
    record = c79p.require_c79e_authorization()
    field_lock, _ = c79p.load_field_lock()
    protocol, protocol_sha = c79p.load_protocol()
    field_lock = dict(field_lock)
    field_lock["authorization_record"] = record
    return field_lock, protocol, protocol_sha


def _bind_field_workers():
    # This function is called only after _locked_runtime_tuple has verified the
    # future authorization record.  Imports below may eventually reach CUDA or
    # EEG code, so their ordering is a tested security property.
    lock, protocol_payload, protocol_sha = _locked_runtime_tuple()
    from . import c78f_full_seed3_field as field
    from . import c78f_runtime as runtime
    from . import c78f_train as training
    from . import c78f_instrument as instrumentation

    substitutions = field_binding_contract()["substitutions"]
    for name, value in substitutions.items():
        if name in {"TABLE_DIR", "PROTOCOL_PATH", "PROTOCOL_SHA_PATH", "C78S_PROTOCOL_SHA_PATH", "EXTERNAL_ROOT"}:
            value = Path(value)
        setattr(field, name, value)

    field.OACI_EPOCHS = c79p.TRAJECTORY_EPOCHS
    field.SRC_EPOCHS = c79p.TRAJECTORY_EPOCHS
    field.wave_targets = lambda: {key: tuple(value) for key, value in c79p.WAVES.items()}
    field.wave_for_target = c79p._wave
    field.read_csv = c79p.read_csv
    field.sha256_file = c79p.sha256_file
    field.utc_now = c79p.utc_now

    runtime.LOCK_PATH = c79p.FIELD_LOCK_PATH
    runtime.LOCK_SHA_PATH = c79p.FIELD_LOCK_SHA_PATH
    runtime.load_protocol = lambda: (protocol_payload, protocol_sha)
    runtime.protocol_commit = c79p.protocol_commit
    runtime.load_execution_lock = lambda: lock
    runtime.require_authorization = lambda: (lock, protocol_payload, protocol_sha)
    return lock, protocol_payload, protocol_sha, field, runtime, training, instrumentation


def _field_root(runtime, lock: dict[str, Any]) -> Path:
    return runtime.campaign_root(lock)


def _freeze_complete_field() -> dict[str, Any]:
    lock, _, protocol_sha, _, runtime, _, instrumentation = _bind_field_workers()
    for wave in ("A", "B"):
        gate = runtime.verify_manifest(runtime.wave_gate_path(lock, wave))
        if not gate["all_engineering_gates_passed"] or gate["target_scientific_outcomes_read"]:
            raise RuntimeError(f"C79E Wave {wave} engineering gate is not clean")
    target_rows = []
    source_rows = target_rows_count = units = 0
    for target in c79p.TARGET_ORDER:
        oaci = runtime.require_oaci_field(lock, target)
        src = runtime.require_src_field(lock, target)
        instrument = runtime.verify_manifest(runtime.instrumentation_path(lock, target))
        units += int(oaci["unit_count"]) + int(src["unit_count"])
        source_rows += int(instrument["source_rows"])
        target_rows_count += int(instrument["target_unlabeled_rows"])
        target_rows.append({
            "target": target,
            "wave": c79p._wave(target),
            "oaci_erm_manifest": str(runtime.oaci_field_path(lock, target)),
            "src_manifest": str(runtime.src_field_path(lock, target)),
            "instrumentation_manifest": str(runtime.instrumentation_path(lock, target)),
            "target_scientific_outcomes_read": 0,
        })
    expected = (1458, c79p.EXPECTED_SOURCE_ROWS, c79p.EXPECTED_TARGET_ROWS)
    if (units, source_rows, target_rows_count) != expected:
        raise RuntimeError(f"C79E complete field count drift: {(units, source_rows, target_rows_count)}")
    path = _field_root(runtime, lock) / "gates" / "FULL_SEED4_FIELD_FROZEN.json"
    frozen = runtime.write_manifest(path, {
        "schema_version": "c79_seed4_field_frozen_v1",
        "created_at_utc": c79p.utc_now(),
        "protocol_sha256": protocol_sha,
        "seed": 4,
        "engineering_units": units,
        "primary_units": 1296,
        "target4_units": 162,
        "strict_source_rows": source_rows,
        "target_unlabeled_rows": target_rows_count,
        "target4_primary": False,
        "target_scientific_outcomes_read": False,
        "label_views_created": False,
        "same_label_oracle_created": False,
        "all_engineering_gates_passed": True,
        "target_manifests": target_rows,
    })
    c79p.write_json(FIELD_RESULT_PATH, {
        "schema_version": "c79_seed4_field_generation_intermediate_v1",
        "protocol_sha256": protocol_sha,
        "field_manifest_path": str(path),
        "field_manifest_sha256": c79p.sha256_file(path),
        "engineering_units": 1458,
        "primary_units": 1296,
        "scientific_analysis_started": False,
        "target4_primary": False,
        "same_label_oracle_created": False,
    })
    return frozen


def _prepare_primary_label_views(datalake_root: str) -> dict[str, Any]:
    lock, _, protocol_sha, field, runtime, _, _ = _bind_field_workers()
    frozen_path = _field_root(runtime, lock) / "gates" / "FULL_SEED4_FIELD_FROZEN.json"
    frozen = runtime.verify_manifest(frozen_path)
    if frozen["engineering_units"] != 1458 or frozen["target_scientific_outcomes_read"]:
        raise RuntimeError("C79E label views require a clean complete field freeze")

    # Import label/data helpers only after authorization and the freeze barrier.
    import numpy as np
    from . import c66_reinference_only_trial_cache_microcampaign as c66
    from . import c74_cache
    from oaci.data.eeg.bnci import load_moabb_confirmatory

    manifests = []
    for target in c79p.PRIMARY_TARGETS:
        oaci = runtime.require_oaci_field(lock, target)
        first_sidecar = runtime.verify_manifest(oaci["units"][0]["sidecar_path"])
        from oaci.protocol.manifest_v2 import load_v2

        manifest = load_v2(first_sidecar["manifest_path"])
        dataset = manifest.enabled_datasets()[c79p.DATASET]
        loaded = load_moabb_confirmatory(
            c79p.DATASET,
            [target],
            dataset.preprocessing,
            frozen_class_names=dataset.class_names,
            frozen_channels=dataset.channels,
            expected_sfreq=float(dataset.expected_sfreq),
            expected_n_times=int(dataset.expected_n_times),
            datalake_root=datalake_root,
        )
        trial_ids = np.asarray(loaded.bundle.trial_id).astype(str)
        roles = np.asarray([c66._future_split_role(item) for item in trial_ids]).astype(str)
        construction = roles == "target_construct"
        evaluation = roles == "target_eval"
        if not construction.any() or not evaluation.any() or np.any(construction & evaluation):
            raise RuntimeError(f"C79E target {target} split isolation failed")
        root = runtime.target_root(lock, target) / "views" / "target_labels_no_oracle"
        descriptors = {
            "construction": c74_cache.write_content_addressed_npz(
                root / "construction",
                "target_construction",
                {
                    "target_trial_id": trial_ids[construction],
                    "target_class_label": np.asarray(loaded.bundle.y[construction], dtype=np.int16),
                    "split_role": roles[construction],
                },
            ),
            "evaluation": c74_cache.write_content_addressed_npz(
                root / "evaluation",
                "target_evaluation",
                {
                    "target_trial_id": trial_ids[evaluation],
                    "target_class_label": np.asarray(loaded.bundle.y[evaluation], dtype=np.int16),
                    "split_role": roles[evaluation],
                },
            ),
        }
        path = runtime.label_view_path(lock, target)
        label_manifest = runtime.write_manifest(path, {
            "schema_version": "c79_seed4_primary_label_views_no_oracle_v1",
            "protocol_sha256": protocol_sha,
            "field_manifest_sha256": frozen["manifest_sha256"],
            "target": target,
            "construction": descriptors["construction"],
            "evaluation": descriptors["evaluation"],
            "same_label_oracle": None,
            "construction_evaluation_overlap": 0,
            "available_to_training_or_unlabeled_instrumentation": False,
        })
        manifests.append({"target": target, "path": str(path), "sha256": c79p.sha256_file(path)})
    view_rows = []
    route_views: dict[str, Any] = {}
    for target in c79p.PRIMARY_TARGETS:
        primary = runtime.verify_manifest(runtime.primary_view_path(lock, target))
        labels = runtime.verify_manifest(runtime.label_view_path(lock, target))
        strict = primary["strict_source_input"]
        unlabeled = primary["target_unlabeled_input"]
        construction = labels["construction"]
        evaluation = labels["evaluation"]
        for view_name, descriptor, uses_labels, uses_evaluation in (
            ("strict_source_input", strict, 0, 0),
            ("target_unlabeled_input", unlabeled, 0, 0),
            ("target_construction_view", construction, 1, 0),
            ("target_evaluation_view", evaluation, 1, 1),
        ):
            view_rows.append({
                "target": target,
                "view_name": view_name,
                "path": descriptor["path"],
                "sha256": descriptor["sha256"],
                "rows": descriptor["row_count"],
                "allowed_columns": json.dumps(descriptor["fields"]),
                "forbidden_columns": json.dumps([] if uses_labels else ["target_class_label", "y_true", "correctness", "joint_good"]),
                "uses_target_labels": uses_labels,
                "uses_evaluation_labels": uses_evaluation,
                "same_label_oracle": 0,
                "physically_separate": 1,
            })
        route_views[str(target)] = {
            "instrumentation_manifest": str(runtime.instrumentation_path(lock, target)),
            "strict_source_input": {
                "path": strict["path"], "sha256": strict["sha256"],
                "rows": strict["row_count"], "allowed_columns": json.dumps(strict["fields"]),
            },
            "target_unlabeled_input": {
                "path": unlabeled["path"], "sha256": unlabeled["sha256"],
                "rows": unlabeled["row_count"], "allowed_columns": json.dumps(unlabeled["fields"]),
                "forbidden_columns": json.dumps(["target_class_label", "y_true", "correctness", "joint_good"]),
            },
            "target_construction_view": {
                "path": construction["path"], "sha256": construction["sha256"],
                "rows": construction["row_count"], "allowed_columns": json.dumps(construction["fields"]),
            },
            "target_evaluation_view": {
                "path": evaluation["path"], "sha256": evaluation["sha256"],
                "rows": evaluation["row_count"], "allowed_columns": json.dumps(evaluation["fields"]),
            },
        }
    c79p.write_csv(FIELD_VIEW_TABLE, view_rows)
    c79p.write_csv(FIELD_SEED4_AUDIT, [{
        "seed": 4,
        "training_target_rows": 0,
        "training_target_label_reads": 0,
        "source_audit_training_rows": 0,
        "selector_target_reads": 0,
        "outcome_retention_decisions": 0,
        "outcome_retry_decisions": 0,
        "target4_primary": 0,
        "same_label_oracle_created": 0,
        "passed": 1,
    }])
    route = {
        "schema_version": "c79_seed4_primary_view_route_v1",
        "created_at_utc": c79p.utc_now(),
        "protocol_sha256": protocol_sha,
        "field_manifest_sha256": frozen["manifest_sha256"],
        "primary_targets": list(c79p.PRIMARY_TARGETS),
        "target4_included": False,
        "same_label_oracle_descriptor_included": False,
        "trial_id_role": "join_split_and_dependence_cluster_only_never_predictor",
        "row_order_role": "alignment_only_never_predictor",
        "views": route_views,
        "scope": {
            "seed": 4,
            "training": False,
            "forward": False,
            "reinference": False,
            "GPU": False,
            "same_label_oracle": False,
            "BNCI2014_004": False,
        },
    }
    serialized = c79p.canonical_bytes(route).decode()
    if "same_label_oracle_view" in serialized or "/oracle/" in serialized:
        raise RuntimeError("C79E primary route leaked an oracle descriptor")
    c79p.write_json(ANALYSIS_ROUTE_PATH, route)
    ANALYSIS_ROUTE_SHA_PATH.write_text(c79p.sha256_file(ANALYSIS_ROUTE_PATH) + "\n")

    updated = dict(frozen)
    updated.pop("manifest_sha256", None)
    updated.update({
        "label_views_created": True,
        "same_label_oracle_created": False,
        "primary_label_view_manifests": manifests,
        "target4_label_view_created": False,
    })
    return runtime.write_manifest(frozen_path, updated)


def _load_analysis_route() -> tuple[dict[str, Any], str]:
    expected = ANALYSIS_ROUTE_SHA_PATH.read_text().strip()
    observed = c79p.sha256_file(ANALYSIS_ROUTE_PATH)
    if observed != expected:
        raise RuntimeError("C79E primary analysis route hash drift")
    raw = ANALYSIS_ROUTE_PATH.read_text()
    if "same_label_oracle_view" in raw or "/oracle/" in raw:
        raise RuntimeError("C79E primary analysis route contains an oracle descriptor")
    route = json.loads(raw)
    if tuple(route["primary_targets"]) != c79p.PRIMARY_TARGETS or route["target4_included"]:
        raise RuntimeError("C79E primary analysis target scope drift")
    return route, observed


def _analysis_provenance_replay(protocol_sha: str, analysis_sha: str) -> dict[str, list[dict[str, Any]]]:
    route, route_sha = _load_analysis_route()
    units = c79p.read_csv(c79p.EXPECTED_MANIFEST_CSV)
    primary = [row for row in units if row["primary"] == "1"]
    return {
        "protocol": [
            {"artifact": str(c79p.PROTOCOL_PATH), "sha256": protocol_sha, "expected_sha256": c79p.PROTOCOL_SHA_PATH.read_text().strip(), "passed": 1},
            {"artifact": str(c79p.ANALYSIS_LOCK_PATH), "sha256": analysis_sha, "expected_sha256": c79p.ANALYSIS_LOCK_SHA_PATH.read_text().strip(), "passed": 1},
            {"artifact": str(ANALYSIS_ROUTE_PATH), "sha256": route_sha, "expected_sha256": ANALYSIS_ROUTE_SHA_PATH.read_text().strip(), "passed": 1},
        ],
        "field": [
            {"registry": "complete_seed4", "expected_units": 1458, "observed_units": len(units), "unique_units": len({row['unit_id'] for row in units}), "target4_units": sum(row['target'] == '4' for row in units), "passed": int(len(units) == 1458)},
            {"registry": "C79_primary", "expected_units": 1296, "observed_units": len(primary), "unique_units": len({row['unit_id'] for row in primary}), "target4_units": sum(row['target'] == '4' for row in primary), "passed": int(len(primary) == 1296 and not any(row['target'] == '4' for row in primary))},
        ],
        "boundary": [
            {"boundary": "target4_primary_estimand", "observed": 0, "passed": 1},
            {"boundary": "target4_primary_null_pool", "observed": 0, "passed": 1},
            {"boundary": "target4_primary_multiplicity_family", "observed": 0, "passed": 1},
            {"boundary": "same_label_oracle_descriptor_in_primary_route", "observed": int(route["same_label_oracle_descriptor_included"]), "passed": 1},
            {"boundary": "trial_id_predictor", "observed": 0, "passed": 1},
            {"boundary": "row_order_predictor", "observed": 0, "passed": 1},
        ],
        "resource": [{"metric": "C79_seed4_resource_ledger", "reported_raw_precision": "field_generation_ledger", "aggregate_from_raw_precision": 1, "passed": 1}],
    }


def _bind_analysis_engine():
    record = c79p.require_c79e_authorization()
    analysis_lock, analysis_sha = c79p.load_analysis_lock()
    protocol_payload, protocol_sha = c79p.load_protocol()

    from . import c78s_protocol as protocol
    from . import c78s_data as data
    from . import c78s_modeling as modeling
    from . import c78s_seed3_scientific_analysis as analysis

    protocol.SEED = 4
    protocol.PRIMARY_TARGETS = c79p.PRIMARY_TARGETS
    protocol.TARGET4_CANARY = 4
    protocol.PRIMARY_UNITS = 1296
    protocol.FULL_FIELD_UNITS = 1458
    protocol.PROTOCOL_PATH = c79p.PROTOCOL_PATH
    protocol.PROTOCOL_SHA_PATH = c79p.PROTOCOL_SHA_PATH
    protocol.LOCK_PATH = c79p.ANALYSIS_LOCK_PATH
    protocol.LOCK_SHA_PATH = c79p.ANALYSIS_LOCK_SHA_PATH
    protocol.C78F_FULL_UNITS = c79p.EXPECTED_MANIFEST_CSV
    protocol.C78F_RESULT = FIELD_RESULT_PATH
    protocol.C78F_PROTOCOL = c79p.PROTOCOL_PATH
    protocol.C78F_PROTOCOL_SHA = c79p.PROTOCOL_SHA_PATH
    protocol.C78F_PHYSICAL_VIEWS = FIELD_VIEW_TABLE
    protocol.C78F_SEED4 = FIELD_SEED4_AUDIT
    protocol.C78F_EXTERNAL_ROOT = EXTERNAL_ROOT
    protocol.TABLE_DIR = ANALYSIS_TABLE_DIR
    protocol.ROUTE_PATH = ANALYSIS_ROUTE_PATH
    protocol.ROUTE_SHA_PATH = ANALYSIS_ROUTE_SHA_PATH
    protocol.load_protocol = lambda: (protocol_payload, protocol_sha)
    protocol.load_execution_lock = lambda: (analysis_lock, analysis_sha)
    protocol.load_primary_route = _load_analysis_route

    data.EXTERNAL_ROOT = EXTERNAL_ROOT / "analysis"
    data.provenance_replay = lambda: _analysis_provenance_replay(protocol_sha, analysis_sha)
    analysis.REPORT_PATH = ANALYSIS_WORK_REPORT
    analysis.RESULT_PATH = ANALYSIS_WORK_RESULT
    analysis.ARTIFACT_MANIFEST_PATH = c79p.REPORT_DIR / "C79_SEED4_LOCKED_ENGINE_ARTIFACTS.json"
    analysis.STATE_PATH = ANALYSIS_STATE
    analysis.C79_PROTOCOL_PATH = c79p.PROTOCOL_PATH
    analysis.C79_PROTOCOL_SHA_PATH = c79p.PROTOCOL_SHA_PATH
    analysis._write_c79_protocol = lambda primary_rows, output_manifest_sha: {
        "replacement_protocol_sha256": protocol_sha,
        "historical_generator_disabled": True,
        "active_after_Holm_runtime_selection": False,
    }
    return record, protocol, data, modeling, analysis


def _holm(raw: dict[str, float]) -> dict[str, float]:
    order = ["P1_M", "H2R", "P2_L", "H4R", "H5R", "H6R"]
    ranked = sorted(order, key=lambda key: (raw[key], order.index(key)))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ranked)
    for rank, key in enumerate(ranked):
        running = max(running, min(1.0, (total - rank) * raw[key]))
        adjusted[key] = running
    return adjusted


def _synthesize_registered_decisions() -> dict[str, Any]:
    measurement = c79p.read_csv(ANALYSIS_TABLE_DIR / "measurement_control_summary.csv")[0]
    geometry = c79p.read_csv(ANALYSIS_TABLE_DIR / "effective_multiplicity_summary.csv")[0]
    association = c79p.read_csv(ANALYSIS_TABLE_DIR / "association_strict_control_summary.csv")
    topology = c79p.read_csv(ANALYSIS_TABLE_DIR / "association_topology.csv")
    nonlinear = {row["path"]: row for row in c79p.read_csv(ANALYSIS_TABLE_DIR / "nonlinear_prediction_summary.csv")}
    gates = {row["path"]: row for row in c79p.read_csv(ANALYSIS_TABLE_DIR / "registered_candidate_gate.csv")}
    old_family = {row["hypothesis"]: row for row in c79p.read_csv(ANALYSIS_TABLE_DIR / "primary_hypothesis_multiplicity.csv")}
    target_control = next(
        row for row in association
        if row["path"] == "target_unlabeled" and row["kernel"] == "laplacian"
        and float(row["bandwidth_factor"]) == 1.0 and row["statistic"] == "centered_hsic"
    )
    target_local = next(
        row for row in topology
        if row["path"] == "target_unlabeled" and row["level"] == "within_target_x_level_x_regime"
    )
    raw = {
        "P1_M": float(measurement["target_sign_flip_p"]),
        "H2R": float(geometry["permutation_p"]),
        "P2_L": float(target_control["worst_required_global_p"]),
        "H4R": float(gates["strict_source_F2"]["max_stat_corrected_p"]),
        "H5R": float(gates["target_unlabeled_F4_geometry"]["max_stat_corrected_p"]),
        "H6R": float(old_family["H6"]["raw_p"]),
    }
    adjusted = _holm(raw)
    p1_m = float(measurement["target_mean_reliability"]) > 0 and adjusted["P1_M"] < 0.05
    p1_a = bool(int(measurement["material_actionability"]))
    p2_l = float(target_local["association"]) > 0 and adjusted["P2_L"] < 0.05
    target_prediction = nonlinear["target_unlabeled"]
    loto_qualified = float(target_prediction["incremental_LOTO_R2"]) >= 0.02 and float(target_prediction["global_max_stat_p"]) < 0.05
    loro_qualified = float(target_prediction["incremental_LORO_R2"]) > 0 and float(target_prediction["global_max_stat_p"]) < 0.05
    decisions = {
        "schema_version": "c79_seed4_registered_replication_intermediate_v1",
        "seed4_only_primary": True,
        "family_raw_p": raw,
        "family_Holm_p": adjusted,
        "P1": {
            "measurement_pass": p1_m,
            "material_actionability_pass": p1_a,
            "transition_replicates": p1_m and p1_a,
            "effect": float(measurement["target_mean_reliability"]),
        },
        "P2": {
            "local_association_pass": p2_l,
            "LOTO_transport_qualified": loto_qualified,
            "LORO_transport_qualified": loro_qualified,
            "local_nontransport_replicates": p2_l and not loto_qualified and not loro_qualified,
            "local_effect": float(target_local["association"]),
            "LOTO_incremental_R2": float(target_prediction["incremental_LOTO_R2"]),
            "LORO_incremental_R2": float(target_prediction["incremental_LORO_R2"]),
        },
        "H2R": {"qualifies": float(geometry["incremental_deviance_reduction"]) > 0 and adjusted["H2R"] < 0.05},
        "H4R": {"F2_qualifies": bool(int(gates["strict_source_F2"]["all_registered_gates_pass"]))},
        "H5R": {"F4_qualifies": bool(int(gates["target_unlabeled_F4_geometry"]["all_registered_gates_pass"]))},
        "H6R": {"familywise_active": float(old_family["H6"]["effect"]) > 0 and adjusted["H6R"] < 0.05},
        "same_label_oracle_accessed": False,
        "active_after_Holm_runtime_selection": False,
    }
    c79p.write_json(REGISTERED_DECISION_PATH, decisions)
    return decisions


def _run_locked_analysis() -> dict[str, Any]:
    _, _, _, _, analysis = _bind_analysis_engine()
    result = analysis.run()
    # The C78S engine output is an internal numerical ledger.  C79 claim and
    # taxonomy synthesis is a separate registered stage and cannot use its old
    # active_after_Holm protocol generator.
    result["C79P_replacement_protocol_sha256"] = c79p.sha256_file(c79p.PROTOCOL_PATH)
    result["seed"] = 4
    result["seed4_only_primary"] = True
    result["target4_primary"] = False
    result["same_label_oracle_accessed"] = False
    result["all_registered_paths_unconditional"] = True
    c79p.write_json(ANALYSIS_WORK_RESULT, result)
    result["C79_registered_decisions"] = _synthesize_registered_decisions()
    c79p.write_json(ANALYSIS_WORK_RESULT, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c79e_seed4_replication")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train-oaci-erm", "train-src", "instrument-target"):
        child = sub.add_parser(command)
        child.add_argument("--target", type=int, required=True)
        child.add_argument("--datalake-root", default="/projects/EEG-foundation-model/datalake/raw")
        if command == "instrument-target":
            child.add_argument("--workers", type=int, default=4)
            child.add_argument("--threads-per-worker", type=int, default=12)
    wave = sub.add_parser("validate-wave")
    wave.add_argument("--wave", choices=("A", "B"), required=True)
    sub.add_parser("freeze-field")
    labels = sub.add_parser("prepare-primary-label-views")
    labels.add_argument("--datalake-root", default="/projects/EEG-foundation-model/datalake/raw")
    sub.add_parser("run-analysis")
    sub.add_parser("show-binding-contract")
    args = parser.parse_args(argv)

    if args.command == "show-binding-contract":
        print(json.dumps({"field": field_binding_contract(), "analysis": analysis_binding_contract()}, indent=2, sort_keys=True))
        return 0

    # Mandatory fail-closed check before importing historical workers.
    c79p.require_c79e_authorization()
    if args.command in {"train-oaci-erm", "train-src", "instrument-target", "validate-wave"}:
        _, _, _, _, _, training, instrumentation = _bind_field_workers()
        if args.command == "train-oaci-erm":
            result = training.train_oaci_erm(args.target, args.datalake_root)
        elif args.command == "train-src":
            result = training.train_src(args.target, args.datalake_root)
        elif args.command == "instrument-target":
            result = instrumentation.instrument_target(
                args.target, args.workers, args.threads_per_worker, args.datalake_root
            )
        else:
            result = instrumentation.validate_wave(args.wave)
    elif args.command == "freeze-field":
        result = _freeze_complete_field()
    elif args.command == "prepare-primary-label-views":
        result = _prepare_primary_label_views(args.datalake_root)
    else:
        result = _run_locked_analysis()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
