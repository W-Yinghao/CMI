"""TALOS_00A implementation and red-team preflight.

This runner uses synthetic data only. It must not run the real CEDAR_01F
BNCI2014 frozen-feature readout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from talos_eeg.adapters.diagonal_affine import fit_diagonal_affine, fit_diagonal_logit_affine
from talos_eeg.adapters.logit_bias import fit_logit_bias_temperature
from talos_eeg.adapters.trust_region import (
    TrustRegionBounds,
    array_hash,
    identity_adapter,
    predict_proba,
    stable_payload_hash,
    trust_region_report,
)
from talos_eeg.data.feature_handoff import assert_no_real_feature_readout, talos00a_handoff_boundary
from talos_eeg.data.source_state import build_source_state, validate_source_state_schema
from talos_eeg.eval.collapse_guards import evaluate_collapse_guards
from talos_eeg.eval.metrics import final_metrics, metrics_hash, unlabeled_diagnostics
from talos_eeg.red_team.adapter_determinism import validate_adapter_determinism
from talos_eeg.red_team.target_label_quarantine import validate_target_label_quarantine
from talos_eeg.red_team.variant_freeze import (
    ALLOWED_VARIANTS,
    build_variant_freeze_config,
    validate_variant_freeze,
)


def make_synthetic_preflight(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_source = 180
    n_target = 90
    n_features = 6
    y_source = np.tile(np.array([0, 1], dtype=np.int64), n_source // 2)
    y_target = np.tile(np.array([0, 1], dtype=np.int64), n_target // 2)
    rng.shuffle(y_source)
    rng.shuffle(y_target)
    centers = np.vstack(
        [
            np.array([-1.0, 0.8, 0.0, 0.3, -0.2, 0.1]),
            np.array([1.0, -0.6, 0.2, -0.3, 0.2, -0.1]),
        ]
    )
    z_source = centers[y_source] + rng.normal(0.0, 0.45, size=(n_source, n_features))
    target_shift = np.array([0.35, -0.25, 0.15, 0.20, 0.05, -0.10])
    target_scale = np.array([1.08, 0.92, 1.05, 0.97, 1.02, 0.95])
    z_target = centers[y_target] * target_scale + target_shift + rng.normal(0.0, 0.50, size=(n_target, n_features))
    return {
        "z_source": z_source.astype(np.float64),
        "y_source": y_source.astype(np.int64),
        "z_target": z_target.astype(np.float64),
        "y_target": y_target.astype(np.int64),
    }


def _adapter_for_variant(variant: str, source_state, z_target: np.ndarray, bounds: TrustRegionBounds):
    if variant in {"ERM", "TTA_CONTROL_REPLAY"}:
        return identity_adapter(variant, source_state.n_features, source_state.n_classes)
    if variant == "TALOS_L":
        return fit_logit_bias_temperature(source_state, z_target, bounds=bounds)
    if variant == "TALOS_D":
        return fit_diagonal_affine(source_state, z_target, bounds=bounds)
    if variant == "TALOS_LD":
        return fit_diagonal_logit_affine(source_state, z_target, bounds=bounds)
    raise ValueError(f"unexpected variant: {variant}")


def _scenario_target_y(data: dict[str, np.ndarray], scenario: str, seed: int) -> np.ndarray | None:
    if scenario == "true_y_final_only":
        return data["y_target"]
    if scenario == "target_y_removed":
        return None
    if scenario == "target_y_permuted":
        rng = np.random.default_rng(seed + 991)
        return rng.permutation(data["y_target"])
    raise ValueError(f"unknown scenario: {scenario}")


def run_preflight_scenario(
    *,
    data: dict[str, np.ndarray],
    scenario: str,
    seed: int,
    bounds: TrustRegionBounds,
) -> dict[str, Any]:
    source_state = build_source_state(data["z_source"], data["y_source"])
    y_final = _scenario_target_y(data, scenario, seed)
    variants = []
    collapse_by_variant = {}
    for variant in ALLOWED_VARIANTS:
        adapter = _adapter_for_variant(variant, source_state, data["z_target"], bounds)
        proba = predict_proba(source_state, data["z_target"], adapter)
        pred = proba.argmax(axis=1).astype(np.int64)
        diagnostics = unlabeled_diagnostics(proba)
        collapse = evaluate_collapse_guards(proba).to_dict()
        trust = trust_region_report(adapter, bounds)
        pre_final_payload = {
            "variant": variant,
            "adapter_state_hash": adapter.hash(),
            "predictions_hash": array_hash(pred),
            "unlabeled_diagnostics": diagnostics,
            "collapse": collapse,
            "trust_region": trust,
        }
        final_payload = final_metrics(y_final, proba) if y_final is not None else None
        rec = {
            "variant": variant,
            "adapter_state": adapter.to_dict(),
            "adapter_state_hash": adapter.hash(),
            "predictions_hash": array_hash(pred),
            "proba_hash": array_hash(proba),
            "metrics_before_final_y": diagnostics,
            "metrics_before_final_y_hash": metrics_hash(pre_final_payload),
            "collapse": collapse,
            "trust_region": trust,
            "final_metrics_FINAL_ONLY": final_payload,
            "final_metrics_hash": stable_payload_hash(final_payload) if final_payload is not None else None,
        }
        variants.append(rec)
        collapse_by_variant[variant] = collapse

    return {
        "scenario": scenario,
        "seed": int(seed),
        "target_labels_used_for": "final_metrics_only" if y_final is not None else "not_available",
        "variant_ranking": list(ALLOWED_VARIANTS),
        "reported_variant": "REPORT_ALL_FROZEN_ORDER",
        "source_state_hash": source_state.hash(),
        "variants": variants,
        "collapse_by_variant": collapse_by_variant,
    }


def build_preflight_payload(seed: int = 0) -> dict[str, Any]:
    assert_no_real_feature_readout(())
    data = make_synthetic_preflight(seed)
    bounds = TrustRegionBounds()
    source_state = build_source_state(data["z_source"], data["y_source"])
    source_schema = validate_source_state_schema(source_state)
    variant_config = build_variant_freeze_config(seed=seed, bounds=bounds, steps=1)
    variant_freeze = validate_variant_freeze(variant_config)

    scenario_names = ("true_y_final_only", "target_y_removed", "target_y_permuted")
    scenarios = {
        name: run_preflight_scenario(data=data, scenario=name, seed=seed, bounds=bounds)
        for name in scenario_names
    }
    repeat = run_preflight_scenario(data=data, scenario="true_y_final_only", seed=seed, bounds=bounds)
    target_label_quarantine = validate_target_label_quarantine(scenarios)
    adapter_determinism = validate_adapter_determinism(scenarios["true_y_final_only"], repeat)

    red_team = {
        "target_label_quarantine": target_label_quarantine.to_dict(),
        "adapter_determinism": adapter_determinism.to_dict(),
        "variant_freeze": variant_freeze.to_dict(),
        "source_state_schema": source_schema.to_dict(),
    }
    all_passed = all(
        bool(item.get("passed"))
        for item in red_team.values()
    )
    trust_reports = [
        rec["trust_region"]
        for rec in scenarios["true_y_final_only"]["variants"]
    ]
    collapse_reports = [
        rec["collapse"]
        for rec in scenarios["true_y_final_only"]["variants"]
    ]
    if not all(rep["within_bounds"] for rep in trust_reports):
        all_passed = False
    if not all(rep["passed"] for rep in collapse_reports):
        all_passed = False

    payload = {
        "project": "TALOS-EEG",
        "phase": "TALOS_00A_adapter_implementation_red_team_preflight",
        "status": "PASS" if all_passed else "FAIL",
        "design_baseline_commit": "96894a7",
        "implementation_scope": "synthetic_smoke_only_no_real_eeg_readout",
        "real_eeg_readout_run": False,
        "scientific_readout": False,
        "talos00b_real_replay_approved": False,
        "source_free_deployment_claim": False,
        "handoff_boundary": talos00a_handoff_boundary().to_dict(),
        "variant_config": variant_config,
        "synthetic_data_hash": stable_payload_hash(
            {
                "z_source": array_hash(data["z_source"]),
                "y_source": array_hash(data["y_source"]),
                "z_target": array_hash(data["z_target"]),
                "y_target": array_hash(data["y_target"]),
            }
        ),
        "source_state_schema": source_schema.to_dict(),
        "red_team": red_team,
        "scenarios": scenarios,
        "repeat_true_y_final_only": repeat,
    }
    payload["preflight_payload_hash"] = stable_payload_hash(payload)
    return payload


def write_preflight_outputs(payload: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "run_manifest.json": {
            key: payload[key]
            for key in (
                "project",
                "phase",
                "status",
                "design_baseline_commit",
                "implementation_scope",
                "real_eeg_readout_run",
                "scientific_readout",
                "talos00b_real_replay_approved",
                "source_free_deployment_claim",
                "handoff_boundary",
                "variant_config",
                "synthetic_data_hash",
                "preflight_payload_hash",
            )
        },
        "preflight_summary.json": payload,
        "target_label_quarantine.json": payload["red_team"]["target_label_quarantine"],
        "adapter_determinism.json": payload["red_team"]["adapter_determinism"],
        "variant_universe_freeze.json": payload["red_team"]["variant_freeze"],
        "source_state_schema.json": payload["red_team"]["source_state_schema"],
        "collapse_guards.json": {
            variant: rec["collapse"]
            for variant, rec in zip(
                ALLOWED_VARIANTS,
                payload["scenarios"]["true_y_final_only"]["variants"],
            )
        },
        "trust_region_bounds.json": {
            variant: rec["trust_region"]
            for variant, rec in zip(
                ALLOWED_VARIANTS,
                payload["scenarios"]["true_y_final_only"]["variants"],
            )
        },
    }
    for name, data in outputs.items():
        with (out / name).open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/talos/talos00a_preflight")
    args = ap.parse_args()
    payload = build_preflight_payload(seed=args.seed)
    write_preflight_outputs(payload, args.out_dir)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "phase": payload["phase"],
                "out_dir": args.out_dir,
                "real_eeg_readout_run": payload["real_eeg_readout_run"],
                "preflight_payload_hash": payload["preflight_payload_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
