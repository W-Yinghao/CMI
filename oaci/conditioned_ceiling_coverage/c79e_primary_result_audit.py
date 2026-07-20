"""Freeze C79E seed-4-only registered outputs before cross-seed synthesis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import c78s_modeling as modeling


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c79_tables"
RAW = TABLE_DIR / "raw_locked_engine"
INTERMEDIATE = REPORT_DIR / "C79_SEED4_REGISTERED_REPLICATION_INTERMEDIATE.json"
STATE = REPORT_DIR / "C79_SEED4_ANALYSIS_STATE.json"
REGISTRY = REPORT_DIR / "c79p_tables" / "c79_post_seed3_scientific_registry.csv"
ANALYSIS_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c79-seed4/analysis/"
    "protocol_e350b7f0c4ee3dfc/implementation_dd4043ad7dd67552"
)
PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)
FAMILY_ORDER = ("P1_M", "H2R", "P2_L", "H4R", "H5R", "H6R")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty C79E table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def i(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def one(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one row: {path}")
    return rows[0]


def holm(raw: dict[str, float]) -> dict[str, float]:
    ranked = sorted(FAMILY_ORDER, key=lambda key: (raw[key], FAMILY_ORDER.index(key)))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ranked):
        running = max(running, min(1.0, (len(ranked) - rank) * raw[key]))
        adjusted[key] = running
    return adjusted


def regime_composition() -> list[dict[str, Any]]:
    unlabeled_manifest = json.loads((ANALYSIS_ROOT / "unlabeled_feature_cache_manifest.json").read_text())
    labeled_manifest = json.loads((ANALYSIS_ROOT / "split_label_analysis_cache_manifest.json").read_text())
    with np.load(unlabeled_manifest["descriptor"]["path"], allow_pickle=False) as shard:
        target = shard["target_id"].astype(int)
        level = shard["level"].astype(int)
        regime = shard["regime"].astype(str)
    with np.load(labeled_manifest["descriptor"]["path"], allow_pickle=False) as shard:
        construction_score = shard["F5"][:, 14].astype(float)
    selected = []
    for target_id in PRIMARY_TARGETS:
        for level_id in (0, 1):
            indices = np.where((target == target_id) & (level == level_id))[0]
            if len(indices) != 81:
                raise RuntimeError("C79E P1 candidate field is not exactly 81")
            selected.append(str(regime[indices[int(np.argmax(construction_score[indices]))]]))
    return [
        {
            "regime": name,
            "selected_cells": selected.count(name),
            "total_cells": len(selected),
            "selected_fraction": selected.count(name) / len(selected),
            "ERM_is_anchor_not_trajectory": int(name == "ERM"),
            "diagnostic_only_not_checkpoint_recommendation": 1,
        }
        for name in ("ERM", "OACI", "SRC")
    ]


def audit() -> dict[str, Any]:
    state_events = [json.loads(line) for line in STATE.read_text().splitlines() if line.strip()]
    required_events = {
        "C78S_started", "unlabeled_features_frozen", "H1_complete",
        "H3_H4_H5_complete", "H2_complete", "C78S_primary_outputs_complete",
    }
    if not required_events.issubset({row["event"] for row in state_events}):
        raise RuntimeError("C79E locked analysis state is incomplete")
    if sum(row["event"] == "prediction_path_complete" for row in state_events) != 4:
        raise RuntimeError("C79E did not run all four registered linear blocks")
    if sum(row["event"] == "krr_path_complete" for row in state_events) != 2:
        raise RuntimeError("C79E did not run both registered nonlinear paths")
    if sum(row["event"] == "association_null_scheme_complete" for row in state_events) != 6:
        raise RuntimeError("C79E did not run all six association null schemes")

    decision = json.loads(INTERMEDIATE.read_text())
    measurement = one(RAW / "measurement_control_summary.csv")
    geometry = one(RAW / "effective_multiplicity_summary.csv")
    association_summary = read_csv(RAW / "association_strict_control_summary.csv")
    topology = read_csv(RAW / "association_topology.csv")
    nonlinear = {row["path"]: row for row in read_csv(RAW / "nonlinear_prediction_summary.csv")}
    gates = {row["path"]: row for row in read_csv(RAW / "registered_candidate_gate.csv")}
    old_family = {row["hypothesis"]: row for row in read_csv(RAW / "primary_hypothesis_multiplicity.csv")}
    target_control = next(
        row for row in association_summary
        if row["path"] == "target_unlabeled" and row["kernel"] == "laplacian"
        and f(row, "bandwidth_factor") == 1.0 and row["statistic"] == "centered_hsic"
    )
    target_local = next(
        row for row in topology
        if row["path"] == "target_unlabeled" and row["level"] == "within_target_x_level_x_regime"
    )
    raw_p = {
        "P1_M": f(measurement, "target_sign_flip_p"),
        "H2R": f(geometry, "permutation_p"),
        "P2_L": f(target_control, "worst_required_global_p"),
        "H4R": f(gates["strict_source_F2"], "max_stat_corrected_p"),
        "H5R": f(gates["target_unlabeled_F4_geometry"], "max_stat_corrected_p"),
        "H6R": f(old_family["H6"], "raw_p"),
    }
    adjusted = holm(raw_p)
    if any(abs(raw_p[key] - float(decision["family_raw_p"][key])) > 1e-15 for key in FAMILY_ORDER):
        raise RuntimeError("C79E registered raw-p replay mismatch")
    if any(abs(adjusted[key] - float(decision["family_Holm_p"][key])) > 1e-15 for key in FAMILY_ORDER):
        raise RuntimeError("C79E registered Holm replay mismatch")

    p1_measurement_pass = f(measurement, "target_mean_reliability") > 0 and adjusted["P1_M"] < 0.05
    p1_action_pass = bool(i(measurement, "material_actionability"))
    p2_local_pass = f(target_local, "association") > 0 and adjusted["P2_L"] < 0.05
    p2_prediction = nonlinear["target_unlabeled"]
    p2_loto_qualified = f(p2_prediction, "incremental_LOTO_R2") >= 0.02 and f(p2_prediction, "global_max_stat_p") < 0.05
    p2_loro_qualified = f(p2_prediction, "incremental_LORO_R2") > 0 and f(p2_prediction, "global_max_stat_p") < 0.05
    p1_replicates = p1_measurement_pass and p1_action_pass
    p2_replicates = p2_local_pass and not p2_loto_qualified and not p2_loro_qualified
    if p1_replicates and p2_replicates:
        primary = "C79-A_seed4_replicates_P1_information_conditioned_transition_and_P2_local_nontransport"
    elif p1_replicates:
        primary = "C79-B_seed4_replicates_P1_only"
    elif p2_replicates:
        primary = "C79-C_seed4_replicates_P2_only"
    else:
        primary = "C79-E_seed4_does_not_replicate_either_core_pattern"

    write_csv(TABLE_DIR / "p1_measurement_reliability.csv", [{
        "effect": f(measurement, "target_mean_reliability"),
        "target_bootstrap_ci_low": f(measurement, "target_bootstrap_ci_low"),
        "target_bootstrap_ci_high": f(measurement, "target_bootstrap_ci_high"),
        "raw_p": raw_p["P1_M"], "Holm_p": adjusted["P1_M"],
        "Holm_reject_0.05": int(adjusted["P1_M"] < 0.05),
        "measurement_pass": int(p1_measurement_pass), "targets": 8, "cells": 16,
    }])
    topk_rows = []
    for k in (1, 5, 10):
        prior = f(measurement, f"prior_oracle_best_in_predicted_top{k}")
        full = f(measurement, f"full_oracle_best_in_predicted_top{k}")
        topk_rows.append({
            "k": k, "candidate_count": 81, "random_baseline": k / 81,
            "source_bAcc_hit": prior, "construction_hit": full,
            "construction_minus_source": full - prior,
            "construction_minus_random": full - k / 81,
            "materiality_component": int(k in (5, 10) and full - prior >= 0.05),
        })
    write_csv(TABLE_DIR / "p1_topk_actionability.csv", topk_rows)
    write_csv(TABLE_DIR / "p1_regret_materiality.csv", [{
        "source_standardized_regret": f(measurement, "prior_standardized_regret"),
        "construction_standardized_regret": f(measurement, "full_standardized_regret"),
        "regret_reduction": f(measurement, "standardized_regret_reduction"),
        "positive_targets": i(measurement, "positive_regret_reduction_targets"),
        "materiality_threshold": 0.05,
        "material_regret": i(measurement, "material_regret"),
        "material_topk": i(measurement, "material_topk"),
        "P1_A_pass": int(p1_action_pass),
    }])
    p1_cells = read_csv(RAW / "measurement_control_actionability_cells.csv")
    write_csv(TABLE_DIR / "p1_target_level_effects.csv", p1_cells)
    write_csv(TABLE_DIR / "p1_regime_selection_composition.csv", regime_composition())
    write_csv(TABLE_DIR / "p1_failure_reason_ledger.csv", [{
        "component": "P1_M", "passed": int(p1_measurement_pass),
        "reason": "PASS" if p1_measurement_pass else "fixed_six_member_Holm_not_rejected",
        "P1_A_pass": int(p1_action_pass), "P1_transition_replicates": int(p1_replicates),
    }])

    registry_rows = read_csv(REGISTRY)
    h2_registry = [row for row in registry_rows if row["path_id_and_hypothesis_role"].startswith("H2R")]
    write_csv(TABLE_DIR / "h2r_model_registry_replay.csv", h2_registry)
    geometry_rows = read_csv(RAW / "effective_multiplicity_prefix_ledger.csv")
    raw_X = np.asarray([[math.log(f(row, "raw_M"))] for row in geometry_rows])
    full_X = np.asarray([
        [math.log(f(row, "raw_M")), math.log(max(f(row, "effective_M_epsilon_0.05"), 1)), -math.log(f(row, "top_two_gap") + 1e-6)]
        for row in geometry_rows
    ])
    y = np.asarray([f(row, "top1_miss") for row in geometry_rows])
    targets = np.asarray([i(row, "target_id") for row in geometry_rows])
    h2_replay = modeling.crossfit_logistic_deviance(raw_X, full_X, y, targets)
    if abs(float(h2_replay["incremental_deviance_reduction"]) - f(geometry, "incremental_deviance_reduction")) > 1e-12:
        raise RuntimeError("C79E H2 deviance replay mismatch")
    write_csv(TABLE_DIR / "h2r_held_target_deviance.csv", [{
        "raw_deviance": f(geometry, "raw_deviance"), "full_deviance": f(geometry, "full_deviance"),
        "incremental_deviance_reduction": f(geometry, "incremental_deviance_reduction"),
        "positive_means_improvement": 1, "qualifies": int(f(geometry, "incremental_deviance_reduction") > 0 and adjusted["H2R"] < 0.05),
    }])
    write_csv(TABLE_DIR / "h2r_permutation_summary.csv", [{
        "observed": f(geometry, "incremental_deviance_reduction"),
        "null_mean": f(geometry, "null_mean"), "null_p95": f(geometry, "null_p95"),
        "replicates": i(geometry, "null_replicates"), "raw_p": raw_p["H2R"],
        "Holm_p": adjusted["H2R"], "passed": int(f(geometry, "incremental_deviance_reduction") > 0 and adjusted["H2R"] < 0.05),
    }])
    target_deviance = []
    for target_id in PRIMARY_TARGETS:
        mask = targets == target_id
        raw_pred = np.clip(np.asarray(h2_replay["raw_prediction"])[mask], 1e-9, 1 - 1e-9)
        full_pred = np.clip(np.asarray(h2_replay["full_prediction"])[mask], 1e-9, 1 - 1e-9)
        yy = y[mask]
        raw_dev = -2 * float(np.sum(yy * np.log(raw_pred) + (1 - yy) * np.log(1 - raw_pred)))
        full_dev = -2 * float(np.sum(yy * np.log(full_pred) + (1 - yy) * np.log(1 - full_pred)))
        target_deviance.append({"target": target_id, "rows": int(mask.sum()), "raw_deviance": raw_dev, "full_deviance": full_dev, "deviance_reduction": raw_dev - full_dev})
    write_csv(TABLE_DIR / "h2r_target_heterogeneity.csv", target_deviance)

    write_csv(TABLE_DIR / "p2_local_association.csv", [{
        "fixed_control_association": f(target_control, "association"),
        "local_within_target_level_regime_association": f(target_local, "association"),
        "positive_trajectory_cells": round(f(target_local, "positive_group_fraction") * i(target_local, "group_count")),
        "trajectory_cells": i(target_local, "group_count"),
        "worst_six_control_raw_p": raw_p["P2_L"], "Holm_p": adjusted["P2_L"],
        "local_association_pass": int(p2_local_pass),
    }])
    topology_folds = read_csv(RAW / "association_topology_folds.csv")
    p2_cells = [row for row in topology_folds if row["path"] == "target_unlabeled" and row["level"] == "within_target_x_level_x_regime"]
    if len(p2_cells) != 32:
        raise RuntimeError("C79E P2 trajectory-cell count drift")
    write_csv(TABLE_DIR / "p2_trajectory_cell_effects.csv", p2_cells)
    null_rows = read_csv(RAW / "association_null_summary.csv")
    p2_nulls = [row for row in null_rows if row["path"] == "target_unlabeled" and row["kernel"] == "laplacian" and f(row, "bandwidth_factor") == 1.0 and row["statistic"] == "centered_hsic"]
    if len(p2_nulls) != 6:
        raise RuntimeError("C79E P2 blocked-control family count drift")
    write_csv(TABLE_DIR / "p2_blocked_control_summary.csv", p2_nulls)
    p2_loto_rows = [row for row in read_csv(RAW / "nonlinear_leave_target.csv") if row["path"] == "target_unlabeled"]
    p2_loto_rows.append({
        "path": "AGGREGATE", "target_id": "ALL", "prior_rho": "", "full_rho": "", "delta_rho": "",
        "increment_residual_rho": f(p2_prediction, "LOTO_median_increment_residual_rho"),
        "incremental_R2": f(p2_prediction, "incremental_LOTO_R2"), "positive_increment": i(p2_prediction, "positive_targets"),
    })
    write_csv(TABLE_DIR / "p2_loto_transport.csv", p2_loto_rows)
    p2_loro_rows = [row for row in read_csv(RAW / "nonlinear_leave_regime.csv") if row["path"] == "target_unlabeled"]
    p2_loro_rows.append({"path": "AGGREGATE", "held_regime": "ALL", "row_count": 1296, "increment_residual_rho": f(p2_prediction, "incremental_LORO_R2")})
    write_csv(TABLE_DIR / "p2_loro_transport.csv", p2_loro_rows)
    heterogeneity = [row for row in topology_folds if row["path"] == "target_unlabeled" and row["level"] in {"within_target", "within_regime"}]
    write_csv(TABLE_DIR / "p2_target_regime_heterogeneity.csv", heterogeneity)
    write_csv(TABLE_DIR / "p2_failure_reason_ledger.csv", [
        {"component": "P2_L", "passed": int(p2_local_pass), "reason": "PASS" if p2_local_pass else "worst_control_and_Holm_gate_failed"},
        {"component": "P2_T_LOTO", "passed": int(not p2_loto_qualified), "reason": "transport_did_not_qualify"},
        {"component": "P2_T_LORO", "passed": int(not p2_loro_qualified), "reason": "transport_did_not_qualify"},
    ])

    h4 = gates["strict_source_F2"]
    h5 = gates["target_unlabeled_F4_geometry"]
    write_csv(TABLE_DIR / "h4r_f2_qualification.csv", [{**h4, "qualified": i(h4, "all_registered_gates_pass"), "allowed_interpretation": "registered_F2_candidate_status_only"}])
    write_csv(TABLE_DIR / "h5r_f4_qualification.csv", [{**h5, "qualified": i(h5, "all_registered_gates_pass"), "allowed_interpretation": "registered_F4_candidate_status_only"}])
    write_csv(TABLE_DIR / "h4r_h5r_failure_reason_ledger.csv", [
        {"path": "H4R_F2", "qualified": i(h4, "all_registered_gates_pass"), "reason": "registered_candidate_did_not_qualify", "universal_impossibility_claim": 0},
        {"path": "H5R_F4", "qualified": i(h5, "all_registered_gates_pass"), "reason": "registered_candidate_did_not_qualify", "universal_impossibility_claim": 0},
    ])

    leave_target = read_csv(RAW / "leave_target_out_prediction.csv")
    per_target: dict[int, dict[str, float]] = defaultdict(dict)
    for row in leave_target:
        if row["outcome"] == "continuous_joint_utility":
            per_target[i(row, "target_id")][row["path"]] = f(row, "incremental_R2")
    h6_rows = []
    for target_id in PRIMARY_TARGETS:
        values = per_target[target_id]
        effect = values["construction_F5_positive"] - max(values["strict_source_F2"], values["target_unlabeled_F4_geometry"])
        h6_rows.append({"target": target_id, "construction_increment": values["construction_F5_positive"], "F2_increment": values["strict_source_F2"], "F4_increment": values["target_unlabeled_F4_geometry"], "H6_effect": effect})
    if abs(float(np.mean([row["H6_effect"] for row in h6_rows])) - f(old_family["H6"], "effect")) > 1e-12:
        raise RuntimeError("C79E H6 target-effect replay mismatch")
    write_csv(TABLE_DIR / "h6r_positive_control_effect.csv", [{
        "effect": f(old_family["H6"], "effect"), "raw_p": raw_p["H6R"], "Holm_p": adjusted["H6R"],
        "familywise_active": int(f(old_family["H6"], "effect") > 0 and adjusted["H6R"] < 0.05),
        "allowed_interpretation": "registered_diagnostic_positive_control_only",
    }])
    family_rows = [{"family_order": rank + 1, "path": key, "raw_p": raw_p[key], "Holm_p": adjusted[key], "Holm_reject_0.05": int(adjusted[key] < 0.05)} for rank, key in enumerate(FAMILY_ORDER)]
    write_csv(TABLE_DIR / "h6r_Holm_family_replay.csv", family_rows)
    write_csv(TABLE_DIR / "h6r_target_level_effects.csv", h6_rows)

    generated = [
        "p1_measurement_reliability.csv", "p1_topk_actionability.csv", "p1_regret_materiality.csv",
        "p1_target_level_effects.csv", "p1_regime_selection_composition.csv", "p1_failure_reason_ledger.csv",
        "h2r_model_registry_replay.csv", "h2r_held_target_deviance.csv", "h2r_permutation_summary.csv", "h2r_target_heterogeneity.csv",
        "p2_local_association.csv", "p2_trajectory_cell_effects.csv", "p2_blocked_control_summary.csv",
        "p2_loto_transport.csv", "p2_loro_transport.csv", "p2_target_regime_heterogeneity.csv", "p2_failure_reason_ledger.csv",
        "h4r_f2_qualification.csv", "h5r_f4_qualification.csv", "h4r_h5r_failure_reason_ledger.csv",
        "h6r_positive_control_effect.csv", "h6r_Holm_family_replay.csv", "h6r_target_level_effects.csv",
    ]
    manifest = {
        "schema_version": "c79_seed4_primary_output_freeze_v1",
        "seed4_only_primary": True,
        "primary_taxonomy": primary,
        "P1_transition_replicates": p1_replicates,
        "P2_local_nontransport_replicates": p2_replicates,
        "H2R_qualifies": bool(decision["H2R"]["qualifies"]),
        "H4R_F2_qualifies": bool(decision["H4R"]["F2_qualifies"]),
        "H5R_F4_qualifies": bool(decision["H5R"]["F4_qualifies"]),
        "H6R_familywise_active": bool(decision["H6R"]["familywise_active"]),
        "target4_primary": False,
        "same_label_oracle_accessed": False,
        "active_after_Holm_runtime_selection": False,
        "registered_paths_completed": 10,
        "tables": [{"path": str(TABLE_DIR / name), "sha256": sha256_file(TABLE_DIR / name)} for name in generated],
    }
    freeze_path = REPORT_DIR / "C79_SEED4_PRIMARY_OUTPUT_FREEZE.json"
    freeze_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    audit()
