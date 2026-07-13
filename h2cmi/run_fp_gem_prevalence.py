"""P13 fixed-reservoir prevalence stress on frozen P12 TSMNet checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from h2cmi import run_fp_gem as p12
from h2cmi.config import TTAConfig
from h2cmi.data.real_eeg import load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.grid_io import hash_state, stable_hash_int
from h2cmi.run_spdim_probe import TensorDomainDataset, _loader, _metrics, _prediction_fields
from h2cmi.tta.class_conditional import B1A_VARIANTS_BY_NAME, ClassConditionalTTA
from h2cmi.w1_repaired_split import indices_from_trial_ids


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "h2cmi/results/fp_gem_prevalence"
CONFIG_PATH = OUT / "fp_gem_prevalence_config.json"
MANIFEST_PATH = OUT / "fp_gem_prevalence_manifest.json"
DATASET = "Lee2019_MI"
SUBJECTS = tuple(range(1, 55))
SEEDS = (0, 1, 2)
Q_VALUES = ("0.1", "0.5", "0.9")
METHODS = (
    "source_only_tsmnet",
    "rct",
    "spdim_geodesic",
    "spdim_bias",
    "Joint-GEM",
    "FP-GEM",
)
ADAPTIVE_METHODS = METHODS[1:]
P12_COMMIT = "3bba1d0bb4f803948e855b6fa707e64b13f2a99a"
P12_RESULT_SHA256 = "f3e4ca699b81e4fa2cab404109aa2dfe7aa1fbe58f25e2779d3d11651e40d48d"
P12_RUNNER_SHA256 = "720b91b1b43cdf6a983be1cb8413430a06b98d6f4923166fa14614041ec46abd"
P12_CONFIG_SHA256 = "d44fd98aa5913eb45908b7fd398b04e5a268dd4aaa75f15bcc96819f424bf165"
P13_CONFIG_SHA256 = "12acd01fbad33cdc5feadf2fe54da0c7423960ab6f1bfa7c8a7005ff76b87e2f"
P13_MANIFEST_SHA256 = "8c5b160fcec5ffeaded7faaf196f9753d7e0f7f15e583f8a18a5651ddf1c5802"
P13_MANIFEST_SEMANTIC_SHA256 = "29febb846ab5935dfed398953b28cbc2da86862842edf4c851a21515df71263f"


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )


def launch_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_config() -> dict[str, Any]:
    if p12.sha256_file(CONFIG_PATH) != P13_CONFIG_SHA256:
        raise RuntimeError("P13 config checksum mismatch")
    config = json.loads(CONFIG_PATH.read_text())
    if (
        config["dataset"] != DATASET
        or tuple(config["subjects"]) != SUBJECTS
        or tuple(config["source_seeds"]) != SEEDS
        or tuple(f"{value:.1f}" for value in config["q_values"]) != Q_VALUES
        or tuple(config["methods"]) != METHODS
    ):
        raise RuntimeError("P13 frozen scientific scope mismatch")
    if config["execution"]["fresh_source_training_permitted"]:
        raise RuntimeError("P13 may not fresh-train source models")
    return config


def load_manifest(config: dict[str, Any]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    if p12.sha256_file(MANIFEST_PATH) != P13_MANIFEST_SHA256:
        raise RuntimeError("P13 manifest file checksum mismatch")
    payload = json.loads(MANIFEST_PATH.read_text())
    if payload["semantic_sha256"] != P13_MANIFEST_SEMANTIC_SHA256:
        raise RuntimeError("P13 manifest semantic checksum mismatch")
    units = {
        (int(unit["target_subject"]), int(unit["source_seed"])): unit
        for unit in payload["units"]
    }
    if len(units) != 162:
        raise RuntimeError(f"P13 unit coverage mismatch: {len(units)}")
    for key, unit in units.items():
        if tuple(unit["batches"]) != Q_VALUES:
            raise RuntimeError(f"P13 q coverage mismatch for {key}")
        if unit["batches"]["0.5"]["trial_ids"] != unit["adapt_reservoir_trial_ids"]:
            raise RuntimeError(f"q=0.5 differs from P12 reservoir for {key}")
        if any(len(unit["batches"][q]["trial_ids"]) != 50 for q in Q_VALUES):
            raise RuntimeError(f"P13 adaptation batch-size mismatch for {key}")
        if set(unit["eval_trial_ids"]) & set(unit["adapt_reservoir_trial_ids"]):
            raise RuntimeError(f"P13 adaptation/evaluation overlap for {key}")
    return units, payload


def validate_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    expected = {
        "p12_results": P12_RESULT_SHA256,
        "p12_runner": P12_RUNNER_SHA256,
        "p12_config": P12_CONFIG_SHA256,
    }
    actual = {
        "p12_results": p12.sha256_file(ROOT / config["p12"]["results_path"]),
        "p12_runner": p12.sha256_file(ROOT / config["p12"]["runner_path"]),
        "p12_config": p12.sha256_file(ROOT / config["p12"]["config_path"]),
    }
    if actual != expected:
        raise RuntimeError(f"P12 frozen input changed: {actual}")
    p12_config = p12.load_config()
    p12_frozen = p12.validate_frozen_inputs(p12_config)
    units, manifest = load_manifest(config)
    del units, manifest
    return actual | {
        "p12_commit": P12_COMMIT,
        "p12_split_semantic": p12_frozen["manifest_semantic"],
        "external_spdim_commit": p12_frozen["external_spdim_commit"],
        "p13_config": P13_CONFIG_SHA256,
        "p13_manifest": P13_MANIFEST_SHA256,
        "p13_manifest_semantic": P13_MANIFEST_SEMANTIC_SHA256,
    }


def load_p12_hash_references(config: dict[str, Any], target: int, seed: int) -> dict[str, dict[str, str]]:
    references = {}
    with (ROOT / config["p12"]["results_path"]).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"] == DATASET
                and int(row["target_subject"]) == target
                and int(row["source_seed"]) == seed
            ):
                references[row["method"]] = {
                    "prediction_hash": row["prediction_hash"],
                    "logits_hash": row["logits_hash"],
                    "checkpoint_hash": row["source_checkpoint_file_sha256"],
                    "source_model_sha256": row["source_model_sha256"],
                }
    if set(references) != set(METHODS):
        raise RuntimeError(f"P12 reference coverage mismatch for {(target, seed)}")
    return references


def load_p12_metric_rows(config: dict[str, Any], target: int, seed: int) -> dict[str, dict[str, Any]]:
    rows = {}
    with (ROOT / config["p12"]["results_path"]).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["dataset"] == DATASET
                and int(row["target_subject"]) == target
                and int(row["source_seed"]) == seed
            ):
                rows[row["method"]] = row
    if set(rows) != set(METHODS):
        raise RuntimeError("P12 metric-row coverage mismatch")
    return rows


def break_spd_running_buffer_aliases(model: torch.nn.Module) -> int:
    """Separate official SPD BN train/test buffers before state_dict copy-in."""
    cloned = 0
    for module in model.modules():
        for name in ("running_mean", "running_var", "running_mean_test", "running_var_test"):
            if name in module._buffers and module._buffers[name] is not None:
                module._buffers[name] = module._buffers[name].clone()
                cloned += 1
    return cloned


def load_p12_checkpoint(
    unit: dict[str, Any],
    ep,
    p12_config: dict[str, Any],
    bn,
    TSMNet,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = Path(unit["checkpoint_path"])
    if not checkpoint.exists():
        raise RuntimeError(f"P12 checkpoint missing: {checkpoint}")
    observed_file_hash = p12.sha256_file(checkpoint)
    if observed_file_hash != unit["checkpoint_sha256"]:
        raise RuntimeError("P12 checkpoint file checksum mismatch")
    source_subjects = [subject for subject in SUBJECTS if subject != int(unit["target_subject"])]
    domain_values = sorted(source_subjects + [int(unit["target_subject"])])
    cfg = p12_config["p9_pipeline"]["source_training"]
    model = p12._build_model(
        TSMNet,
        bn,
        ep=ep,
        domain_values=domain_values,
        args=argparse.Namespace(
            temporal_filters=cfg["temporal_filters"],
            spatial_filters=cfg["spatial_filters"],
            subspace_dims=cfg["subspace_dims"],
            spd_device=cfg["spd_device"],
        ),
        device=device,
        dtype=dtype,
    )
    cloned_buffer_count = break_spd_running_buffer_aliases(model)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    actual_state_hash = hash_state(model)
    if actual_state_hash != unit["source_model_sha256"]:
        raise RuntimeError(
            f"P12 checkpoint state hash mismatch after alias-safe load: {actual_state_hash}"
        )
    return model, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": observed_file_hash,
        "source_model_sha256": actual_state_hash,
        "alias_safe_buffer_clone_count": cloned_buffer_count,
        "fresh_source_training_performed": False,
    }


def trainer_without_source_fit(p12_config: dict[str, Any], Trainer, device, dtype):
    cfg = p12_config["p9_pipeline"]["source_training"]
    return Trainer(
        max_epochs=cfg["epochs"],
        callbacks=[],
        loss=torch.nn.CrossEntropyLoss(),
        device=device,
        dtype=dtype,
    )


def adapt_methods(
    *,
    source_state: dict[str, torch.Tensor],
    source_only_logits: torch.Tensor,
    density,
    source_prior: np.ndarray,
    ep,
    adapt_idx: np.ndarray,
    eval_idx: np.ndarray,
    target: int,
    seed: int,
    split_hash: str,
    domain_values: list[int],
    p12_config: dict[str, Any],
    bn,
    TSMNet,
    trainer,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run frozen methods from an ordered EEG batch; q and target labels are not inputs."""
    parameter_t = torch.tensor(
        p12_config["p9_pipeline"]["official_adaptation"]["parameter_t"],
        dtype=torch.float64,
        device="cpu",
    )
    adapt_ds = TensorDomainDataset(
        ep.X[adapt_idx],
        np.zeros(len(adapt_idx), dtype=np.int64),
        np.full(len(adapt_idx), target, dtype=np.int64),
    )
    adapt_loader = _loader(adapt_ds, len(adapt_ds), False, seed)
    target_domains_adapt = np.full(len(adapt_idx), target, dtype=np.int64)
    target_domains_eval = np.full(len(eval_idx), target, dtype=np.int64)

    rct_model, rct_info = p12.refit_rct(
        source_state,
        ep=ep,
        domain_values=domain_values,
        adapt_idx=adapt_idx,
        target=target,
        config=p12_config,
        bn=bn,
        TSMNet=TSMNet,
        device=device,
        dtype=dtype,
    )
    adapt_features, _, adapt_hook_error = p12.capture_preclassifier(
        rct_model, ep.X[adapt_idx], target_domains_adapt, device=device, dtype=dtype
    )
    eval_features, rct_logits, eval_hook_error = p12.capture_preclassifier(
        rct_model, ep.X[eval_idx], target_domains_eval, device=device, dtype=dtype
    )
    if max(adapt_hook_error, eval_hook_error) > 1e-7:
        raise RuntimeError("P13 target feature-hook semantic mismatch")

    official_cfg = p12_config["p9_pipeline"]["official_adaptation"]
    geodesic_model, _ = p12.refit_rct(
        source_state,
        ep=ep,
        domain_values=domain_values,
        adapt_idx=adapt_idx,
        target=target,
        config=p12_config,
        bn=bn,
        TSMNet=TSMNet,
        device=device,
        dtype=dtype,
    )
    geodesic_parameters_before = p12.hash_parameters(geodesic_model)
    best_t = trainer.get_information_maximization_geodesic(
        geodesic_model,
        test_dataloader=adapt_loader,
        parameter_t=parameter_t.clone(),
        epochs=official_cfg["epochs"],
        lr=official_cfg["learning_rate"],
    )
    if p12.hash_parameters(geodesic_model) != geodesic_parameters_before:
        raise RuntimeError("P13 SPDIM geodesic changed frozen model parameters")
    geodesic_logits = p12.forward_logits(
        geodesic_model,
        ep.X[eval_idx],
        target_domains_eval,
        device=device,
        dtype=dtype,
        parameter_t=best_t.detach(),
    )

    bias_model, _ = p12.refit_rct(
        source_state,
        ep=ep,
        domain_values=domain_values,
        adapt_idx=adapt_idx,
        target=target,
        config=p12_config,
        bn=bn,
        TSMNet=TSMNet,
        device=device,
        dtype=dtype,
    )
    bias_parameters_before = p12.hash_parameters(bias_model)
    trainer.predict(bias_model, adapt_loader, parameter_t=parameter_t.clone())
    best_mean = trainer.get_information_maximization_bias(
        bias_model,
        test_dataloader=adapt_loader,
        parameter_t=parameter_t.clone(),
        epochs=official_cfg["epochs"],
        lr=official_cfg["learning_rate"],
    )
    if p12.hash_parameters(bias_model) != bias_parameters_before:
        raise RuntimeError("P13 SPDIM bias changed frozen model parameters")
    bias_logits = p12.forward_logits(
        bias_model,
        ep.X[eval_idx],
        target_domains_eval,
        device=device,
        dtype=dtype,
        parameter_t=parameter_t,
        fm_mean=best_mean.detach(),
    )

    tta_cfg = TTAConfig(
        transform="diag_affine",
        lowrank=p12_config["geometry_em"].get("lowrank", 4),
        trust_region=p12_config["geometry_em"]["trust_region_a"],
        trust_region_b=p12_config["geometry_em"]["trust_region_b"],
        logdet_weight=p12_config["geometry_em"]["logdet_weight"],
        prior_anchor_strength=p12_config["geometry_em"]["prior_anchor_strength"],
        dirichlet=p12_config["geometry_em"]["dirichlet"],
        em_iters=p12_config["geometry_em"]["outer_iterations"],
        em_lr=p12_config["geometry_em"]["learning_rate"],
    )
    tta = ClassConditionalTTA(density, source_prior, tta_cfg, 2, str(device))
    tta_seed = stable_hash_int("P12", DATASET, target, seed, split_hash)
    parameters_before = p12.hash_parameters(rct_model)
    classifier_before = hash_state(rct_model.classifier)
    density_before = hash_state(density)
    joint = tta.fit_variant(
        adapt_features,
        B1A_VARIANTS_BY_NAME["joint_iterative_diag"],
        tta_seed=tta_seed,
    )
    fixed = tta.fit_variant(
        adapt_features,
        B1A_VARIANTS_BY_NAME["gen_iterative_diag"],
        tta_seed=tta_seed,
    )
    if not np.array_equal(fixed.pi_T.detach().cpu().numpy(), source_prior.astype(np.float32)):
        raise RuntimeError("P13 FP-GEM changed the fixed source empirical prior")
    if (
        p12.hash_parameters(rct_model) != parameters_before
        or hash_state(rct_model.classifier) != classifier_before
        or hash_state(density) != density_before
    ):
        raise RuntimeError("P13 GEM changed a frozen model/density component")
    with torch.no_grad():
        joint_logits = rct_model.classifier(joint.transform.apply(eval_features))
        fixed_logits = rct_model.classifier(fixed.transform.apply(eval_features))

    method_logits = {
        "source_only_tsmnet": source_only_logits,
        "rct": rct_logits,
        "spdim_geodesic": geodesic_logits,
        "spdim_bias": bias_logits,
        "Joint-GEM": joint_logits,
        "FP-GEM": fixed_logits,
    }
    if not all(torch.isfinite(logits).all() for logits in method_logits.values()):
        raise RuntimeError("P13 produced non-finite logits")
    geometry = {}
    for method, fit in (("Joint-GEM", joint), ("FP-GEM", fixed)):
        a = fit.transform.a.detach().cpu().numpy().astype(np.float32)
        b = fit.transform.b.detach().cpu().numpy().astype(np.float32)
        geometry[method] = {
            "log_scale_vector": a.tolist(),
            "translation_vector": b.tolist(),
            "log_scale_vector_hash": p12.hash_array(a),
            "translation_vector_hash": p12.hash_array(b),
            "log_scale_norm": float(np.linalg.norm(a)),
            "translation_norm": float(np.linalg.norm(b)),
            "fitted_prior": fit.pi_T.detach().cpu().numpy().tolist(),
            "source_prior": source_prior.tolist(),
            "tta_seed": tta_seed,
        }
    return method_logits, {
        "geometry": geometry,
        "feature_hook": {
            "adapt_semantic_max_abs_error": adapt_hook_error,
            "eval_semantic_max_abs_error": eval_hook_error,
            "dimension": int(adapt_features.shape[1]),
        },
        "rct": rct_info,
        "geodesic_parameter_t": float(best_t.detach().cpu().item()),
        "bias_mean_sha256": p12.hash_array(best_mean.detach().cpu().numpy()),
    }


def prediction_payload(method_logits: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    payload = {}
    for method, logits in method_logits.items():
        values = logits.detach().cpu().numpy().astype(np.float32)
        predictions = values.argmax(axis=1).astype(np.int64)
        payload[method] = {
            "prediction_hash": p12.hash_array(predictions),
            "logits_hash": p12.hash_array(values),
            "prediction_vector": predictions.tolist(),
        }
    return payload


def verify_p12_center(
    predictions: dict[str, dict[str, Any]],
    geometry: dict[str, Any],
    references: dict[str, dict[str, str]],
    p12_raw: dict[str, Any],
) -> dict[str, Any]:
    method_checks = {}
    for method in METHODS:
        method_checks[method] = {
            "prediction_hash_match": predictions[method]["prediction_hash"]
            == references[method]["prediction_hash"],
            "logits_hash_match": predictions[method]["logits_hash"]
            == references[method]["logits_hash"],
        }
    geometry_checks = {}
    for method, prefix in (("Joint-GEM", "joint"), ("FP-GEM", "fp")):
        center_a = np.asarray(p12_raw["geometry"][f"{prefix}_a"], dtype=np.float32)
        center_b = np.asarray(p12_raw["geometry"][f"{prefix}_b"], dtype=np.float32)
        geometry_checks[method] = {
            "log_scale_hash_match": geometry[method]["log_scale_vector_hash"] == p12.hash_array(center_a),
            "translation_hash_match": geometry[method]["translation_vector_hash"] == p12.hash_array(center_b),
        }
    all_method_hashes_match = all(
        all(checks.values()) for checks in method_checks.values()
    )
    all_geometry_hashes_match = all(
        all(checks.values()) for checks in geometry_checks.values()
    )
    return {
        "method_hash_checks": method_checks,
        "geometry_hash_checks": geometry_checks,
        "source_only_prediction_hash_reproduced": method_checks["source_only_tsmnet"][
            "prediction_hash_match"
        ],
        "all_six_prediction_and_logits_hashes_match": all_method_hashes_match,
        "both_gem_geometry_hashes_match": all_geometry_hashes_match,
        "p12_center_reuse_approved": all_method_hashes_match and all_geometry_hashes_match,
    }


def geometry_with_displacement(
    geometry: dict[str, Any], p12_raw: dict[str, Any]
) -> dict[str, Any]:
    output = deepcopy(geometry)
    for method, prefix in (("Joint-GEM", "joint"), ("FP-GEM", "fp")):
        center_a = np.asarray(p12_raw["geometry"][f"{prefix}_a"], dtype=np.float32)
        center_b = np.asarray(p12_raw["geometry"][f"{prefix}_b"], dtype=np.float32)
        current_a = np.asarray(output[method]["log_scale_vector"], dtype=np.float32)
        current_b = np.asarray(output[method]["translation_vector"], dtype=np.float32)
        da = float(np.linalg.norm(current_a - center_a))
        db = float(np.linalg.norm(current_b - center_b))
        output[method]["log_scale_displacement_from_q05"] = da
        output[method]["translation_displacement_from_q05"] = db
        output[method]["geometry_displacement_from_q05"] = math.sqrt(da * da + db * db)
    return output


def prepare_unit(
    target: int,
    seed: int,
    hardware_group: str,
    config: dict[str, Any],
    units: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (target, seed)
    if key not in units:
        raise RuntimeError(f"P13 unit absent from manifest: {key}")
    unit = units[key]
    if unit["hardware_group"] != hardware_group:
        raise RuntimeError(f"P13 hardware-group mismatch for {key}")
    if git_status():
        raise RuntimeError("P13 GPU execution requires a clean worktree")
    frozen = validate_frozen_inputs(config)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("P13 requires an allocated CUDA device")
    dtype = torch.float32
    runtime = p12.runtime_environment(device)
    p12.check_hardware_group(hardware_group, runtime)
    if runtime["sys_executable"] != config["execution"]["runtime_python"]:
        raise RuntimeError("P13 runtime Python mismatch")
    p12_config = p12.load_config()
    external = p12_config["p9_pipeline"]["external_spdim_path"]
    bn, TSMNet, Trainer = p12._import_official(external)
    ep = load_dataset(DATASET, MOABB_CLASS[DATASET]().subject_list)
    source_model, checkpoint_info = load_p12_checkpoint(
        unit, ep, p12_config, bn, TSMNet, device, dtype
    )
    references = load_p12_hash_references(config, target, seed)
    if {item["checkpoint_hash"] for item in references.values()} != {unit["checkpoint_sha256"]}:
        raise RuntimeError("P13/P12 checkpoint reference mismatch")
    source_subjects = [subject for subject in SUBJECTS if subject != target]
    source_idx = np.where(np.isin(ep.subject, np.asarray(source_subjects, dtype=np.int64)))[0]
    train_idx, val_idx = p12._source_train_val_split(
        source_idx,
        ep.y,
        ep.subject,
        p12_config["p9_pipeline"]["source_training"]["validation_fraction"],
    )
    source_state = deepcopy(source_model.state_dict())
    eval_idx = indices_from_trial_ids(unit["eval_trial_ids"])
    target_domains_eval = np.full(len(eval_idx), target, dtype=np.int64)
    parameter_t = torch.tensor(
        p12_config["p9_pipeline"]["official_adaptation"]["parameter_t"],
        dtype=torch.float64,
        device="cpu",
    )
    source_only_logits = p12.forward_logits(
        source_model,
        ep.X[eval_idx],
        target_domains_eval,
        device=device,
        dtype=dtype,
        parameter_t=parameter_t,
    )
    source_only_pred = prediction_payload({"source_only_tsmnet": source_only_logits})[
        "source_only_tsmnet"
    ]
    if source_only_pred["prediction_hash"] != references["source_only_tsmnet"]["prediction_hash"]:
        raise RuntimeError("P13 checkpoint failed P12 source-only prediction reproduction")
    if source_only_pred["logits_hash"] != references["source_only_tsmnet"]["logits_hash"]:
        raise RuntimeError("P13 checkpoint failed P12 source-only logits reproduction")

    source_features, _, source_hook_error = p12.capture_preclassifier(
        source_model,
        ep.X[train_idx],
        ep.subject[train_idx],
        device=device,
        dtype=dtype,
    )
    if source_hook_error > 1e-7:
        raise RuntimeError("P13 source feature-hook semantic mismatch")
    density, source_prior, density_info = p12.fit_source_density(
        source_features,
        ep.y[train_idx],
        dataset=DATASET,
        target=target,
        seed=seed,
        config=p12_config,
        device=device,
    )
    p12_raw_path = Path(unit["p12_raw_path"])
    if p12.sha256_file(p12_raw_path) != unit["p12_raw_sha256"]:
        raise RuntimeError("P13 P12 raw-unit checksum mismatch")
    p12_raw = json.loads(p12_raw_path.read_text())
    if density_info["density_state_sha256"] != p12_raw["source_density"]["density_state_sha256"]:
        raise RuntimeError("P13 source density does not reproduce P12")
    trainer = trainer_without_source_fit(p12_config, Trainer, device, dtype)
    return {
        "unit": unit,
        "ep": ep,
        "p12_config": p12_config,
        "bn": bn,
        "TSMNet": TSMNet,
        "trainer": trainer,
        "source_model": source_model,
        "source_state": source_state,
        "source_only_logits": source_only_logits,
        "density": density,
        "source_prior": source_prior,
        "eval_idx": eval_idx,
        "domain_values": sorted(source_subjects + [target]),
        "references": references,
        "p12_raw": p12_raw,
        "checkpoint_info": checkpoint_info,
        "density_info": density_info,
        "source_hook_error": source_hook_error,
        "runtime": runtime,
        "frozen": frozen,
        "train_idx_sha256": p12._sha_indices(train_idx),
        "val_idx_sha256": p12._sha_indices(val_idx),
    }


def run_checkpoint_gate(
    target: int,
    seed: int,
    hardware_group: str,
    config: dict[str, Any],
    units: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    if (target, seed) != (1, 0):
        raise RuntimeError("P13 checkpoint gate is frozen to Lee2019_MI target 1 seed 0")
    start = time.time()
    context = prepare_unit(target, seed, hardware_group, config, units)
    unit = context["unit"]
    center_idx = indices_from_trial_ids(unit["batches"]["0.5"]["trial_ids"])
    method_logits, diagnostics = adapt_methods(
        source_state=context["source_state"],
        source_only_logits=context["source_only_logits"],
        density=context["density"],
        source_prior=context["source_prior"],
        ep=context["ep"],
        adapt_idx=center_idx,
        eval_idx=context["eval_idx"],
        target=target,
        seed=seed,
        split_hash=unit["p12_split_hash"],
        domain_values=context["domain_values"],
        p12_config=context["p12_config"],
        bn=context["bn"],
        TSMNet=context["TSMNet"],
        trainer=context["trainer"],
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    predictions = prediction_payload(method_logits)
    center = verify_p12_center(
        predictions, diagnostics["geometry"], context["references"], context["p12_raw"]
    )
    if not center["source_only_prediction_hash_reproduced"] or not center["p12_center_reuse_approved"]:
        raise RuntimeError(f"P13 checkpoint/q=0.5 reproduction gate failed: {center}")
    return {
        "status": "pass",
        "mode": "checkpoint_gate_no_performance",
        "dataset": DATASET,
        "target_subject": target,
        "source_seed": seed,
        "performance_metrics_computed": False,
        "evaluation_labels_accessed": False,
        "target_labels_passed_to_adaptation": False,
        "target_performance_selection": False,
        "checkpoint": context["checkpoint_info"],
        "source_density_sha256": context["density_info"]["density_state_sha256"],
        "source_hook_error": context["source_hook_error"],
        "center_reproduction": center,
        "method_hashes": {
            method: {
                "prediction_hash": predictions[method]["prediction_hash"],
                "logits_hash": predictions[method]["logits_hash"],
            }
            for method in METHODS
        },
        "runtime": context["runtime"],
        "provenance": {
            "launch_commit": launch_commit(),
            "runner_sha256": p12.sha256_file(__file__),
            "config_sha256": P13_CONFIG_SHA256,
            "manifest_sha256": P13_MANIFEST_SHA256,
            "frozen_inputs": context["frozen"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "clean_worktree": True,
            "fresh_source_training_performed": False,
        },
        "elapsed_seconds": time.time() - start,
    }


def run_unit(
    target: int,
    seed: int,
    hardware_group: str,
    config: dict[str, Any],
    units: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    start = time.time()
    context = prepare_unit(target, seed, hardware_group, config, units)
    unit = context["unit"]
    q_outputs = {}

    # Exact deterministic reconstruction of the P12 center is required only because
    # P12 retained prediction hashes, not vectors needed for disagreement. It creates
    # no new accepted q=0.5 result row.
    center_idx = indices_from_trial_ids(unit["batches"]["0.5"]["trial_ids"])
    center_logits, center_diagnostics = adapt_methods(
        source_state=context["source_state"],
        source_only_logits=context["source_only_logits"],
        density=context["density"],
        source_prior=context["source_prior"],
        ep=context["ep"],
        adapt_idx=center_idx,
        eval_idx=context["eval_idx"],
        target=target,
        seed=seed,
        split_hash=unit["p12_split_hash"],
        domain_values=context["domain_values"],
        p12_config=context["p12_config"],
        bn=context["bn"],
        TSMNet=context["TSMNet"],
        trainer=context["trainer"],
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    center_predictions = prediction_payload(center_logits)
    center_check = verify_p12_center(
        center_predictions,
        center_diagnostics["geometry"],
        context["references"],
        context["p12_raw"],
    )
    if not center_check["p12_center_reuse_approved"]:
        raise RuntimeError(f"P13 q=0.5 exact-center gate failed: {center_check}")
    q_outputs["0.5"] = {
        "predictions": center_predictions,
        "geometry": geometry_with_displacement(
            center_diagnostics["geometry"], context["p12_raw"]
        ),
        "diagnostics": center_diagnostics,
    }

    for q in ("0.1", "0.9"):
        adapt_idx = indices_from_trial_ids(unit["batches"][q]["trial_ids"])
        logits, diagnostics = adapt_methods(
            source_state=context["source_state"],
            source_only_logits=context["source_only_logits"],
            density=context["density"],
            source_prior=context["source_prior"],
            ep=context["ep"],
            adapt_idx=adapt_idx,
            eval_idx=context["eval_idx"],
            target=target,
            seed=seed,
            split_hash=unit["p12_split_hash"],
            domain_values=context["domain_values"],
            p12_config=context["p12_config"],
            bn=context["bn"],
            TSMNet=context["TSMNet"],
            trainer=context["trainer"],
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
        q_outputs[q] = {
            "predictions": prediction_payload(logits),
            "geometry": geometry_with_displacement(diagnostics["geometry"], context["p12_raw"]),
            "diagnostics": diagnostics,
            "logits": logits,
        }

    # Evaluation labels are first read after every q and every adaptation method is fit.
    y_eval = np.asarray(context["ep"].y[context["eval_idx"]], dtype=np.int64)
    if np.bincount(y_eval, minlength=2).astype(int).tolist() != [25, 25]:
        raise RuntimeError("P13 evaluation block is not the frozen balanced P12 block")
    p12_metrics = load_p12_metric_rows(config, target, seed)
    results = []
    geometry_rows = []
    for q in Q_VALUES:
        batch = unit["batches"][q]
        for method in METHODS:
            prediction = q_outputs[q]["predictions"][method]
            if q == "0.5" or method == "source_only_tsmnet":
                reference = p12_metrics[method]
                metrics = {
                    "acc": float(reference["acc"]),
                    "bacc": float(reference["bacc"]),
                    "macro_f1": float(reference["macro_f1"]),
                }
                origin = "P12_q05_reused" if q == "0.5" else "P12_source_only_reused"
            else:
                predictions = np.asarray(prediction["prediction_vector"], dtype=np.int64)
                metrics = _metrics(y_eval, predictions)
                origin = "P13_new_prevalence_adaptation"
            results.append({
                "dataset": DATASET,
                "target_subject": target,
                "source_seed": seed,
                "q": float(q),
                "method": method,
                "checkpoint_hash": unit["checkpoint_sha256"],
                "adaptation_manifest_hash": batch["adaptation_manifest_hash"],
                "n_adapt": batch["n_adapt"],
                "class_counts_adapt": batch["class_counts_adapt"],
                "n_eval": len(y_eval),
                "class_counts_eval": [25, 25],
                **metrics,
                "prediction_hash": prediction["prediction_hash"],
                "logits_hash": prediction["logits_hash"],
                "prediction_vector": prediction["prediction_vector"],
                "status": "ok",
                "failure_reason": "",
                "result_origin": origin,
                "target_labels_passed_to_adaptation": False,
                "q_passed_to_method": False,
                "target_performance_selection": False,
            })
        for method in ("Joint-GEM", "FP-GEM"):
            geometry_rows.append({
                "dataset": DATASET,
                "target_subject": target,
                "source_seed": seed,
                "q": float(q),
                "method": method,
                "checkpoint_hash": unit["checkpoint_sha256"],
                "adaptation_manifest_hash": batch["adaptation_manifest_hash"],
                **q_outputs[q]["geometry"][method],
            })

    return {
        "status": "ok",
        "dataset": DATASET,
        "target_subject": target,
        "source_seed": seed,
        "results": results,
        "geometry": geometry_rows,
        "checkpoint": context["checkpoint_info"],
        "source_density": context["density_info"],
        "q05_center_reproduction": center_check,
        "split": {
            "p12_split_hash": unit["p12_split_hash"],
            "adapt_reservoir_trial_ids": unit["adapt_reservoir_trial_ids"],
            "eval_trial_ids": unit["eval_trial_ids"],
            "adapt_eval_disjoint": True,
            "class_counts_eval": [25, 25],
        },
        "provenance": {
            "launch_commit": launch_commit(),
            "runner_sha256": p12.sha256_file(__file__),
            "config_sha256": P13_CONFIG_SHA256,
            "manifest_sha256": P13_MANIFEST_SHA256,
            "runtime": context["runtime"],
            "hardware_group": hardware_group,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "clean_worktree": True,
            "fresh_source_training_performed": False,
            "official_pretrained_weight_used": False,
            "third_party_vendored": False,
            "target_labels_passed_to_adaptation": False,
            "q_passed_to_method": False,
            "target_performance_selection": False,
            "frozen_inputs": context["frozen"],
        },
        "elapsed_seconds": time.time() - start,
    }


def grouped_units(units: dict[tuple[int, int], dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return sorted(
        [unit for unit in units.values() if unit["hardware_group"] == group],
        key=lambda unit: (int(unit["target_subject"]), int(unit["source_seed"])),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "checkpoint-gate", "run"), required=True)
    parser.add_argument("--target-subject", type=int)
    parser.add_argument("--source-seed", type=int)
    parser.add_argument("--hardware-group", choices=("V100", "A100"))
    parser.add_argument("--group-index", type=int)
    args = parser.parse_args()
    config = load_config()
    units, _ = load_manifest(config)
    frozen = validate_frozen_inputs(config)
    if args.mode == "dry-run":
        groups = {name: len(grouped_units(units, name)) for name in ("V100", "A100")}
        print(json.dumps({
            "status": "pass",
            "units": len(units),
            "groups": groups,
            "frozen_inputs": frozen,
            "fresh_source_training_permitted": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.hardware_group is None:
        raise SystemExit("--hardware-group is required for GPU modes")
    if args.mode == "checkpoint-gate":
        target = 1 if args.target_subject is None else args.target_subject
        seed = 0 if args.source_seed is None else args.source_seed
        output = Path(config["execution"]["raw_root"]) / "checkpoint_gate.json"
        try:
            payload = run_checkpoint_gate(target, seed, args.hardware_group, config, units)
            atomic_json(output, payload)
            print(json.dumps({"status": "pass", "out": str(output)}, sort_keys=True))
            return 0
        except Exception as exc:
            failure = {
                "status": "blocked",
                "mode": "checkpoint_gate_no_performance",
                "dataset": DATASET,
                "target_subject": target,
                "source_seed": seed,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "fresh_source_training_performed": False,
            }
            atomic_json(output, failure)
            print(json.dumps({"status": "blocked", "out": str(output), "failure": failure["failure_reason"]}))
            return 1
    if args.group_index is None:
        raise SystemExit("--group-index is required for run mode")
    group = grouped_units(units, args.hardware_group)
    if args.group_index < 0 or args.group_index >= len(group):
        raise SystemExit(f"group index {args.group_index} outside {args.hardware_group} coverage")
    unit = group[args.group_index]
    target = int(unit["target_subject"])
    seed = int(unit["source_seed"])
    output = Path(config["execution"]["raw_root"]) / f"units/{DATASET}_target{target}_seed{seed}.json"
    if output.exists():
        existing = json.loads(output.read_text())
        if existing.get("status") == "ok" and len(existing.get("results", [])) == 18:
            print(json.dumps({"status": "skip_complete", "out": str(output)}, sort_keys=True))
            return 0
    try:
        payload = run_unit(target, seed, args.hardware_group, config, units)
        atomic_json(output, payload)
        print(json.dumps({"status": "ok", "out": str(output), "rows": len(payload["results"])}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "dataset": DATASET,
            "target_subject": target,
            "source_seed": seed,
            "hardware_group": args.hardware_group,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "fresh_source_training_performed": False,
            "provenance": {
                "launch_commit": launch_commit(),
                "runner_sha256": p12.sha256_file(__file__),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            },
        }
        atomic_json(output, failure)
        print(json.dumps({"status": "failed", "out": str(output), "failure": failure["failure_reason"]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
