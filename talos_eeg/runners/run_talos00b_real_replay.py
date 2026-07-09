"""TALOS_00B real frozen-feature adapter replay.

This is a P0 frozen-feature replay over the CEDAR_01F handoff artifacts. It is
not P1 training and makes no source-free deployment claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from cedar_eeg.data.feature_handoff import validate_handoff_manifest
from cedar_eeg.data.feature_schema import sha256_file, stable_json_hash
from cedar_eeg.data.load_frozen_features import FrozenFeatureBundle, load_frozen_feature_npz
from talos_eeg.adapters.diagonal_affine import fit_diagonal_affine, fit_diagonal_logit_affine
from talos_eeg.adapters.logit_bias import fit_logit_bias_temperature, fit_temperature_only
from talos_eeg.adapters.trust_region import (
    TrustRegionBounds,
    array_hash,
    identity_adapter,
    predict_proba,
    stable_payload_hash,
    trust_region_report,
)
from talos_eeg.data.source_state import SourceState, build_source_state, validate_source_state_schema
from talos_eeg.eval.collapse_guards import evaluate_collapse_guards
from talos_eeg.eval.metrics import final_metrics, metrics_hash, unlabeled_diagnostics
from talos_eeg.red_team.adapter_determinism import validate_adapter_determinism
from talos_eeg.red_team.target_label_quarantine import validate_target_label_quarantine


TALOS00B_VARIANTS: tuple[str, ...] = (
    "ERM_NO_ADAPT",
    "TTA_CONTROL_REPLAY",
    "TALOS_L",
    "TALOS_D",
    "TALOS_LD",
)
TALOS00B_TALOS_VARIANTS: tuple[str, ...] = ("TALOS_L", "TALOS_D", "TALOS_LD")


def _rows(bundle: FrozenFeatureBundle, role_name: str) -> dict[str, np.ndarray]:
    role = np.asarray(bundle.role).astype(str)
    keep = role == role_name
    if not np.any(keep):
        raise ValueError(f"artifact {bundle.metadata.get('path')} has no {role_name} rows")
    return {
        "z": bundle.z[keep],
        "y": bundle.y[keep],
        "domain": bundle.domain[keep],
        "groups": bundle.groups[keep],
        "role": role[keep],
    }


def _fit_adapter(variant: str, source_state: SourceState, z_target: np.ndarray, bounds: TrustRegionBounds):
    if variant == "ERM_NO_ADAPT":
        return identity_adapter(variant, source_state.n_features, source_state.n_classes)
    if variant == "TTA_CONTROL_REPLAY":
        return fit_temperature_only(source_state, z_target, bounds=bounds)
    if variant == "TALOS_L":
        return fit_logit_bias_temperature(source_state, z_target, bounds=bounds)
    if variant == "TALOS_D":
        return fit_diagonal_affine(source_state, z_target, bounds=bounds)
    if variant == "TALOS_LD":
        return fit_diagonal_logit_affine(source_state, z_target, bounds=bounds)
    raise ValueError(f"unexpected TALOS_00B variant: {variant}")


def _scenario_target_y(target_y: np.ndarray, scenario: str, seed: int, fold_key: str) -> np.ndarray | None:
    if scenario == "true_y_final_only":
        return target_y
    if scenario == "target_y_removed":
        return None
    if scenario == "target_y_permuted":
        fold_seed = int(stable_payload_hash({"seed": seed, "fold_key": fold_key})[:8], 16)
        rng = np.random.default_rng(fold_seed)
        return rng.permutation(target_y)
    raise ValueError(f"unknown scenario: {scenario}")


def _artifact_key(record: dict[str, Any]) -> str:
    return f"{record.get('dataset')}|{record.get('backbone')}|seed{record.get('seed')}|fold{record.get('fold_id')}"


def _variant_key(record: dict[str, Any], variant: str) -> str:
    return f"{_artifact_key(record)}|{variant}"


def _source_audit_metrics(source_state: SourceState, source_audit: dict[str, np.ndarray] | None) -> dict[str, Any] | None:
    if source_audit is None:
        return None
    adapter = identity_adapter("SOURCE_AUDIT_IDENTITY", source_state.n_features, source_state.n_classes)
    proba = predict_proba(source_state, source_audit["z"], adapter)
    return final_metrics(source_audit["y"], proba)


def _run_artifact_scenario(
    *,
    record: dict[str, Any],
    bundle: FrozenFeatureBundle,
    scenario: str,
    seed: int,
    bounds: TrustRegionBounds,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_train = _rows(bundle, "source_train")
    try:
        source_audit = _rows(bundle, "source_audit")
    except ValueError:
        source_audit = None
    target = _rows(bundle, "target_audit")
    source_state = build_source_state(source_train["z"], source_train["y"])
    source_schema = validate_source_state_schema(source_state).to_dict()
    source_audit = source_audit if source_audit is not None else None
    y_final = _scenario_target_y(target["y"], scenario, seed, _artifact_key(record))

    scenario_variants = []
    fold_metrics = []
    adapter_norms = []
    collapse_guards = []
    calibration_rows = []
    for variant in TALOS00B_VARIANTS:
        adapter = _fit_adapter(variant, source_state, target["z"], bounds)
        proba = predict_proba(source_state, target["z"], adapter)
        pred = proba.argmax(axis=1).astype(np.int64)
        diagnostics = unlabeled_diagnostics(proba)
        collapse = evaluate_collapse_guards(proba).to_dict()
        trust = trust_region_report(adapter, bounds)
        variant_key = _variant_key(record, variant)
        pre_final_payload = {
            "variant": variant_key,
            "adapter_state_hash": adapter.hash(),
            "predictions_hash": array_hash(pred),
            "proba_hash": array_hash(proba),
            "unlabeled_diagnostics": diagnostics,
            "collapse": collapse,
            "trust_region": trust,
        }
        final_payload = final_metrics(y_final, proba) if y_final is not None else None
        rec = {
            "variant": variant_key,
            "adapter_state_hash": adapter.hash(),
            "predictions_hash": array_hash(pred),
            "proba_hash": array_hash(proba),
            "metrics_before_final_y_hash": metrics_hash(pre_final_payload),
            "final_metrics_hash": stable_payload_hash(final_payload) if final_payload is not None else None,
        }
        scenario_variants.append(rec)
        if scenario == "true_y_final_only":
            if final_payload is None:
                raise ValueError("true_y_final_only scenario missing final metrics")
            base_fields = {
                "dataset": record.get("dataset"),
                "backbone": record.get("backbone"),
                "seed": record.get("seed"),
                "fold_id": record.get("fold_id"),
                "artifact_path": record.get("path"),
                "variant": variant,
                "variant_key": variant_key,
                "target_bacc": float(final_payload["bacc"]),
                "target_nll": float(final_payload["nll"]),
                "target_ece": float(final_payload["ece"]),
                "target_n_eval": int(final_payload["n_eval"]),
                "entropy_mean": float(diagnostics["entropy_mean"]),
                "confidence_mean": float(diagnostics["confidence_mean"]),
                "mean_prediction": json.dumps(diagnostics["mean_prediction"], sort_keys=True),
                "predicted_label_counts": json.dumps(diagnostics["predicted_label_counts"], sort_keys=True),
                "collapse_passed": bool(collapse["passed"]),
                "collapse_warnings": json.dumps(collapse["warnings"], sort_keys=True),
                "boundary_hit": bool(trust["boundary_hits"]),
                "boundary_hits": json.dumps(trust["boundary_hits"], sort_keys=True),
                "log_t_abs": float(trust["log_t_abs"]),
                "beta_norm": float(trust["beta_norm"]),
                "diag_max_abs_delta": float(trust["diag_max_abs_delta"]),
                "shift_norm": float(trust["shift_norm"]),
                "adapter_state_hash": adapter.hash(),
                "predictions_hash": array_hash(pred),
                "proba_hash": array_hash(proba),
            }
            fold_metrics.append(base_fields)
            adapter_norms.append(
                {
                    key: base_fields[key]
                    for key in (
                        "dataset",
                        "backbone",
                        "seed",
                        "fold_id",
                        "variant",
                        "boundary_hit",
                        "boundary_hits",
                        "log_t_abs",
                        "beta_norm",
                        "diag_max_abs_delta",
                        "shift_norm",
                    )
                }
            )
            collapse_guards.append(
                {
                    "dataset": record.get("dataset"),
                    "backbone": record.get("backbone"),
                    "seed": record.get("seed"),
                    "fold_id": record.get("fold_id"),
                    "variant": variant,
                    **collapse,
                }
            )
            calibration_rows.append(
                {
                    "dataset": record.get("dataset"),
                    "backbone": record.get("backbone"),
                    "seed": record.get("seed"),
                    "fold_id": record.get("fold_id"),
                    "variant": variant,
                    "target_nll": float(final_payload["nll"]),
                    "target_ece": float(final_payload["ece"]),
                }
            )

    source_summary = {
        "dataset": record.get("dataset"),
        "backbone": record.get("backbone"),
        "seed": record.get("seed"),
        "fold_id": record.get("fold_id"),
        "artifact_path": record.get("path"),
        "source_train_rows": int(len(source_train["y"])),
        "source_audit_rows": int(0 if source_audit is None else len(source_audit["y"])),
        "target_audit_rows": int(len(target["y"])),
        "n_features": int(source_state.n_features),
        "n_classes": int(source_state.n_classes),
        "source_state_hash": source_state.hash(),
        "source_state_schema": source_schema,
        "source_audit_metrics": _source_audit_metrics(source_state, source_audit),
    }
    scenario_payload = {
        "scenario": scenario,
        "variant_ranking": [_variant_key(record, variant) for variant in TALOS00B_VARIANTS],
        "reported_variant": "REPORT_ALL_FROZEN_ORDER_NO_DEPLOYMENT_SELECTION",
        "target_labels_used_for": "final_metrics_only" if y_final is not None else "not_available",
        "variants": scenario_variants,
    }
    return scenario_payload, fold_metrics, adapter_norms, collapse_guards, calibration_rows, source_summary


def _feature_handoff_validation(manifest_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(manifest_path)
    handoff = validate_handoff_manifest(manifest_path)
    records = []
    for rec in handoff["per_artifact_hashes"]:
        observed = sha256_file(rec["path"])
        records.append(
            {
                "path": rec["path"],
                "dataset": rec.get("dataset"),
                "backbone": rec.get("backbone"),
                "fold_id": rec.get("fold_id"),
                "expected_sha256": rec.get("file_sha256"),
                "observed_sha256": observed,
                "status": "PASS" if observed == rec.get("file_sha256") else "FAIL",
            }
        )
    validation = {
        "cedar01f_handoff_manifest": str(manifest_path),
        "cedar01f_handoff_hash": sha256_file(manifest_path),
        "cedar01f_canonical_payload_hash": handoff.get("canonical_payload_hash"),
        "feature_artifacts_expected": 18,
        "feature_artifacts_loaded": len(records),
        "per_artifact_hash_check": "PASS" if len(records) == 18 and all(r["status"] == "PASS" for r in records) else "FAIL",
        "records": records,
    }
    if validation["per_artifact_hash_check"] != "PASS":
        raise ValueError("CEDAR_01F handoff hash validation failed")
    return handoff, validation


def _build_variant_freeze(seed: int, bounds: TrustRegionBounds) -> dict[str, Any]:
    payload = {
        "phase": "TALOS_00B_real_frozen_feature_adapter_replay",
        "allowed_variants": list(TALOS00B_VARIANTS),
        "forbidden_variants": [
            "TALOS_LR",
            "TALOS_FULL",
            "rank_r_affine",
            "geometry_loss_full_variant",
            "CMI",
            "CEDAR_mask",
            "pruning",
            "surgery",
            "safety_gate",
            "harm_router",
        ],
        "runtime_variant_addition_allowed": False,
        "target_labels_allowed_for": ["final_metrics_only"],
        "trust_region_bounds": bounds.to_dict(),
        "seed": int(seed),
    }
    payload["variant_universe_hash"] = stable_payload_hash(payload)
    return payload


def _validate_variant_freeze(payload: dict[str, Any]) -> dict[str, Any]:
    checks = []
    warnings: list[str] = []
    if tuple(payload.get("allowed_variants", ())) != TALOS00B_VARIANTS:
        raise ValueError("TALOS_00B allowed variant universe changed")
    checks.append("allowed_variants_exact")
    if payload.get("runtime_variant_addition_allowed") is not False:
        raise ValueError("runtime variant addition must be disabled")
    checks.append("runtime_addition_disabled")
    if payload.get("target_labels_allowed_for") != ["final_metrics_only"]:
        raise ValueError("target labels must be final_metrics_only")
    checks.append("target_labels_final_metrics_only")
    expected = dict(payload)
    observed_hash = expected.pop("variant_universe_hash", None)
    if observed_hash != stable_payload_hash(expected):
        raise ValueError("variant_universe_hash mismatch")
    checks.append("variant_universe_hash_recomputable")
    return {
        "passed": True,
        "checks": checks,
        "warnings": warnings,
        "allowed_variants": list(TALOS00B_VARIANTS),
        "variant_universe_hash": observed_hash,
    }


def _aggregate_variant_table(per_fold_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_fold_metrics:
        by_key.setdefault((str(row["backbone"]), str(row["variant"])), []).append(row)
    aggregates = []
    lookup = {}
    for (backbone, variant), rows in sorted(by_key.items()):
        bacc = np.asarray([float(r["target_bacc"]) for r in rows], dtype=np.float64)
        nll = np.asarray([float(r["target_nll"]) for r in rows], dtype=np.float64)
        ece = np.asarray([float(r["target_ece"]) for r in rows], dtype=np.float64)
        rec = {
            "backbone": backbone,
            "variant": variant,
            "n_folds": int(len(rows)),
            "mean_target_bacc": float(bacc.mean()),
            "worst_fold_target_bacc": float(bacc.min()),
            "mean_target_nll": float(nll.mean()),
            "mean_target_ece": float(ece.mean()),
            "boundary_hit_count": int(sum(bool(r["boundary_hit"]) for r in rows)),
            "collapse_warning_count": int(
                sum(len(json.loads(str(r["collapse_warnings"]))) for r in rows)
            ),
            "mean_entropy": float(np.mean([float(r["entropy_mean"]) for r in rows])),
        }
        aggregates.append(rec)
        lookup[(backbone, variant)] = rec
    for rec in aggregates:
        erm = lookup.get((rec["backbone"], "ERM_NO_ADAPT"))
        tta = lookup.get((rec["backbone"], "TTA_CONTROL_REPLAY"))
        rec["delta_bacc_vs_erm"] = (
            float(rec["mean_target_bacc"] - erm["mean_target_bacc"]) if erm else float("nan")
        )
        rec["delta_bacc_vs_tta_control"] = (
            float(rec["mean_target_bacc"] - tta["mean_target_bacc"]) if tta else float("nan")
        )
    return aggregates


def _scientific_gate(variant_table: list[dict[str, Any]], red_team: dict[str, Any]) -> dict[str, Any]:
    failures = []
    warnings = []
    for name, payload in red_team.items():
        if not payload.get("passed", False):
            failures.append(f"{name}:failed")
        for warning in payload.get("warnings", []):
            warnings.append(f"{name}:{warning}")

    by_backbone: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in variant_table:
        by_backbone.setdefault(str(rec["backbone"]), {})[str(rec["variant"])] = rec
    backbone_results = {}
    for backbone, records in sorted(by_backbone.items()):
        erm = records["ERM_NO_ADAPT"]
        tta = records["TTA_CONTROL_REPLAY"]
        eligible = []
        for variant in TALOS00B_TALOS_VARIANTS:
            rec = records[variant]
            rec = dict(rec)
            rec["eligible_no_boundary_no_collapse"] = (
                int(rec["boundary_hit_count"]) == 0 and int(rec["collapse_warning_count"]) == 0
            )
            eligible.append(rec)
        eligible_clean = [r for r in eligible if r["eligible_no_boundary_no_collapse"]]
        if eligible_clean:
            best = max(eligible_clean, key=lambda r: (float(r["mean_target_bacc"]), -float(r["mean_target_nll"]), r["variant"]))
        else:
            best = max(eligible, key=lambda r: (float(r["mean_target_bacc"]), -float(r["mean_target_nll"]), r["variant"]))
        delta_erm = float(best["mean_target_bacc"] - erm["mean_target_bacc"])
        delta_tta = float(best["mean_target_bacc"] - tta["mean_target_bacc"])
        pass_backbone = bool(
            best["eligible_no_boundary_no_collapse"]
            and delta_erm >= 0.020
            and delta_tta >= -0.005
        )
        neutral = bool(delta_erm >= -0.005 and delta_tta >= -0.005)
        backbone_results[backbone] = {
            "best_post_hoc_variant": best["variant"],
            "post_hoc_readout_only_no_deployment_selection": True,
            "mean_bacc_best": best["mean_target_bacc"],
            "mean_bacc_erm": erm["mean_target_bacc"],
            "mean_bacc_tta_control": tta["mean_target_bacc"],
            "delta_bacc_vs_erm": delta_erm,
            "delta_bacc_vs_tta_control": delta_tta,
            "eligible_no_boundary_no_collapse": best["eligible_no_boundary_no_collapse"],
            "pass": pass_backbone,
            "neutral_or_inconclusive": neutral,
        }

    pass_count = sum(1 for rec in backbone_results.values() if rec["pass"])
    neutral_count = sum(1 for rec in backbone_results.values() if rec["neutral_or_inconclusive"])
    if failures or warnings:
        outcome = "FAIL"
        reasons = failures + warnings
    elif pass_count == len(backbone_results) and backbone_results:
        outcome = "PASS"
        reasons = []
    elif pass_count == 1 and neutral_count == len(backbone_results):
        outcome = "CONDITIONAL_PASS"
        reasons = ["one_backbone_pass_other_neutral"]
    else:
        outcome = "FAIL"
        reasons = ["talos_variants_do_not_satisfy_both_backbone_gate"]
    return {
        "outcome": outcome,
        "reasons": reasons,
        "backbone_results": backbone_results,
        "p1_status": "BLOCKED_UNTIL_PM_APPROVAL",
        "source_free_deployment_claim": False,
    }


def build_talos00b_payload(
    *,
    handoff_manifest: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    bounds = TrustRegionBounds()
    handoff, handoff_validation = _feature_handoff_validation(handoff_manifest)
    variant_config = _build_variant_freeze(seed, bounds)
    variant_freeze = _validate_variant_freeze(variant_config)

    scenario_payloads = {
        "true_y_final_only": {
            "scenario": "true_y_final_only",
            "variant_ranking": [],
            "reported_variant": "REPORT_ALL_FROZEN_ORDER_NO_DEPLOYMENT_SELECTION",
            "variants": [],
        },
        "target_y_removed": {
            "scenario": "target_y_removed",
            "variant_ranking": [],
            "reported_variant": "REPORT_ALL_FROZEN_ORDER_NO_DEPLOYMENT_SELECTION",
            "variants": [],
        },
        "target_y_permuted": {
            "scenario": "target_y_permuted",
            "variant_ranking": [],
            "reported_variant": "REPORT_ALL_FROZEN_ORDER_NO_DEPLOYMENT_SELECTION",
            "variants": [],
        },
    }
    repeat_payload = {
        "scenario": "true_y_final_only",
        "variant_ranking": [],
        "reported_variant": "REPORT_ALL_FROZEN_ORDER_NO_DEPLOYMENT_SELECTION",
        "variants": [],
    }
    per_fold_metrics: list[dict[str, Any]] = []
    adapter_norms: list[dict[str, Any]] = []
    collapse_guards: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    source_state_summary: list[dict[str, Any]] = []

    records = sorted(
        handoff["per_artifact_hashes"],
        key=lambda r: (str(r["backbone"]), int(r["fold_id"]), str(r["path"])),
    )
    for record in records:
        bundle = load_frozen_feature_npz(record["path"])
        for scenario in scenario_payloads:
            scenario_rec, fold_rows, adapter_rows, collapse_rows, calibration, source_summary = _run_artifact_scenario(
                record=record,
                bundle=bundle,
                scenario=scenario,
                seed=seed,
                bounds=bounds,
            )
            scenario_payloads[scenario]["variant_ranking"].extend(scenario_rec["variant_ranking"])
            scenario_payloads[scenario]["variants"].extend(scenario_rec["variants"])
            if scenario == "true_y_final_only":
                per_fold_metrics.extend(fold_rows)
                adapter_norms.extend(adapter_rows)
                collapse_guards.extend(collapse_rows)
                calibration_rows.extend(calibration)
                source_state_summary.append(source_summary)
        repeat_rec, _, _, _, _, _ = _run_artifact_scenario(
            record=record,
            bundle=bundle,
            scenario="true_y_final_only",
            seed=seed,
            bounds=bounds,
        )
        repeat_payload["variant_ranking"].extend(repeat_rec["variant_ranking"])
        repeat_payload["variants"].extend(repeat_rec["variants"])

    target_quarantine = validate_target_label_quarantine(scenario_payloads).to_dict()
    adapter_determinism = validate_adapter_determinism(scenario_payloads["true_y_final_only"], repeat_payload).to_dict()
    handoff_red_team = {
        "passed": handoff_validation["per_artifact_hash_check"] == "PASS",
        "checks": [
            "cedar01f_handoff_manifest_validated",
            "feature_artifacts_loaded_18",
            "per_artifact_hash_check_pass",
        ],
        "warnings": [],
        "cedar01f_handoff_hash": handoff_validation["cedar01f_handoff_hash"],
    }
    red_team = {
        "handoff_immutability": handoff_red_team,
        "target_label_quarantine": target_quarantine,
        "adapter_determinism": adapter_determinism,
        "variant_freeze": variant_freeze,
    }
    variant_table = _aggregate_variant_table(per_fold_metrics)
    scientific_gate = _scientific_gate(variant_table, red_team)
    payload = {
        "project": "TALOS-EEG",
        "phase": "TALOS_00B_real_frozen_feature_adapter_replay",
        "status": scientific_gate["outcome"],
        "preflight_baseline_commit": "b8dbb70",
        "real_frozen_feature_replay": True,
        "p1_training": False,
        "target_labels_final_metric_only": True,
        "source_free_deployment_claim": False,
        "cmi_pruning_mask_surgery_safety_active": False,
        "handoff_validation": handoff_validation,
        "variant_config": variant_config,
        "red_team": red_team,
        "scientific_gate": scientific_gate,
        "variant_table": variant_table,
        "per_fold_metrics": per_fold_metrics,
        "adapter_norms": adapter_norms,
        "collapse_guards": collapse_guards,
        "calibration_metrics": calibration_rows,
        "source_state_summary": source_state_summary,
        "target_label_quarantine_scenarios": scenario_payloads,
    }
    payload["talos00b_payload_hash"] = stable_payload_hash(payload)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(payload: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "run_manifest.json": {
            key: payload[key]
            for key in (
                "project",
                "phase",
                "status",
                "preflight_baseline_commit",
                "real_frozen_feature_replay",
                "p1_training",
                "target_labels_final_metric_only",
                "source_free_deployment_claim",
                "cmi_pruning_mask_surgery_safety_active",
                "talos00b_payload_hash",
            )
        },
        "feature_handoff_validation.json": payload["handoff_validation"],
        "red_team.json": payload["red_team"],
        "target_label_quarantine.json": payload["red_team"]["target_label_quarantine"],
        "adapter_norms.json": payload["adapter_norms"],
        "collapse_guards.json": payload["collapse_guards"],
        "calibration_metrics.json": payload["calibration_metrics"],
        "source_state_summary.json": payload["source_state_summary"],
        "scientific_gate.json": payload["scientific_gate"],
        "talos00b_summary.json": payload,
    }
    for name, data in json_outputs.items():
        with (out / name).open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    _write_csv(out / "per_fold_metrics.csv", payload["per_fold_metrics"])
    _write_csv(out / "variant_table.csv", payload["variant_table"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--handoff-manifest",
        default="results/cedar/feature_supply/cedar01f_bnci2014_001_seed0/CEDAR_01F_HANDOFF_MANIFEST.json",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/talos/talos00b_bnci2014_001_seed0")
    args = ap.parse_args()
    payload = build_talos00b_payload(handoff_manifest=args.handoff_manifest, seed=args.seed)
    write_outputs(payload, args.out_dir)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "phase": payload["phase"],
                "out_dir": args.out_dir,
                "feature_artifacts_loaded": payload["handoff_validation"]["feature_artifacts_loaded"],
                "red_team_failures": [
                    name for name, item in payload["red_team"].items() if not item.get("passed", False)
                ],
                "talos00b_payload_hash": payload["talos00b_payload_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
