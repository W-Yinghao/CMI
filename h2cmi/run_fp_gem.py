"""P12 Fixed-Prior Geometry EM on the frozen official P9 TSMNet pipeline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import mne
import moabb
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.data.real_eeg import load_dataset
from h2cmi.data.real_metadata import MOABB_CLASS
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.grid_io import hash_state, stable_hash_int
from h2cmi.run_spdim_probe import (
    OFFICIAL_SHA,
    TensorDomainDataset,
    _build_model,
    _git_sha,
    _import_official,
    _loader,
    _metrics,
    _prediction_fields,
    _set_seed,
    _sha_indices,
    _source_train_val_split,
)
from h2cmi.tta.class_conditional import B1A_VARIANTS_BY_NAME, ClassConditionalTTA
from h2cmi.w1_repaired_split import (
    SPLIT_FAMILY,
    indices_from_trial_ids,
    load_manifest_csv,
    manifest_hash,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "h2cmi/results/fp_gem_main/fp_gem_config.json"
UNITS_PATH = ROOT / "h2cmi/results/fp_gem_main/fp_gem_units.csv"
SELECTED_DATASETS = ("BNCI2014_001", "Lee2019_MI")
SELECTED_SEEDS = (0, 1, 2)
NEW_METHODS = ("Joint-GEM", "FP-GEM")
P9_METHODS = ("source_only_tsmnet", "rct", "spdim_geodesic", "spdim_bias")
P9_RESULTS_SHA256 = "95b8f69556a140dc020415753c9694cf9ebdeed1abb0766dd24f523c491289c3"
P9_RUNNER_SHA256 = "946b28b93f0ddbce395ade7c6a13d30b20f368fe7a1ae22fbefa01f291e82be8"
P9_CONFIG_SHA256 = "6f27455570996064b8e8ea360b1e0324a9b8ea2e5995d35297a66697a76e6a6b"
MANIFEST_HASH = "231246def0ac1dd8cef02920b77502767467738a839ca0a99673117df31b6d8e"
MANIFEST_FILE_SHA256 = "e9ebe6e9421bdcf10f8a952623285cec0842f5cb6b868e8147f13dde23e8a712"
FROZEN_CONFIG_SHA256 = "15543cb1b912f06872c9b4146f2f9da903e5bef5ab7056974ee20adf5b24c0d6"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_array(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(tuple(x.shape)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def hash_parameters(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )


def launch_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_config() -> dict[str, Any]:
    actual = sha256_file(CONFIG_PATH)
    if FROZEN_CONFIG_SHA256.startswith("__"):
        raise RuntimeError("FP-GEM runner has an unresolved frozen config checksum")
    if actual != FROZEN_CONFIG_SHA256:
        raise RuntimeError(f"FP-GEM config checksum mismatch: {actual} != {FROZEN_CONFIG_SHA256}")
    config = json.loads(CONFIG_PATH.read_text())
    if tuple(config["datasets"]) != SELECTED_DATASETS or tuple(config["source_seeds"]) != SELECTED_SEEDS:
        raise RuntimeError("FP-GEM dataset/seed scope differs from the frozen P12 config")
    return config


def validate_frozen_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "p9_results": ROOT / config["p9_pipeline"]["results_path"],
        "p9_runner": ROOT / config["p9_pipeline"]["runner_path"],
        "p9_config": ROOT / config["p9_pipeline"]["config_path"],
        "manifest": ROOT / config["split"]["manifest_path"],
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    expected = {
        "p9_results": P9_RESULTS_SHA256,
        "p9_runner": P9_RUNNER_SHA256,
        "p9_config": P9_CONFIG_SHA256,
        "manifest": MANIFEST_FILE_SHA256,
    }
    mismatches = {key: (actual[key], expected[key]) for key in actual if actual[key] != expected[key]}
    if mismatches:
        raise RuntimeError(f"frozen P9 input checksum mismatch: {mismatches}")
    rows = load_manifest_csv(paths["manifest"])
    semantic = manifest_hash(rows)
    if semantic != MANIFEST_HASH:
        raise RuntimeError(f"repaired manifest semantic hash mismatch: {semantic}")
    external = config["p9_pipeline"]["external_spdim_path"]
    external_sha = _git_sha(external)
    if external_sha != OFFICIAL_SHA == config["p9_pipeline"]["external_spdim_commit"]:
        raise RuntimeError(f"external SPDIM commit mismatch: {external_sha}")
    if Path(external).resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("third-party SPDIM checkout is vendored inside the repository")
    return actual | {"manifest_semantic": semantic, "external_spdim_commit": external_sha}


def p9_source_hashes(config: dict[str, Any]) -> dict[tuple[str, int, int], str]:
    path = ROOT / config["p9_pipeline"]["results_path"]
    grouped: dict[tuple[str, int, int], dict[str, set[str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in SELECTED_DATASETS or int(row["source_seed"]) not in SELECTED_SEEDS:
                continue
            key = (row["dataset"], int(row["target_subject"]), int(row["source_seed"]))
            group = grouped.setdefault(key, {"methods": set(), "hashes": set()})
            group["methods"].add(row["method"])
            group["hashes"].add(row["source_model_sha256"])
    out = {}
    for key, group in grouped.items():
        if group["methods"] != set(P9_METHODS) or len(group["hashes"]) != 1:
            raise RuntimeError(f"P9 source key is incomplete or inconsistent: {key} {group}")
        out[key] = next(iter(group["hashes"]))
    if len(out) != 189:
        raise RuntimeError(f"expected 189 selected P9 source keys, found {len(out)}")
    return out


def manifest_map(config: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = load_manifest_csv(ROOT / config["split"]["manifest_path"])
    out = {}
    for row in rows:
        if row["dataset"] not in SELECTED_DATASETS or int(row["source_seed"]) not in SELECTED_SEEDS:
            continue
        key = (row["dataset"], int(row["target_subject"]), int(row["source_seed"]))
        if key in out:
            raise RuntimeError(f"duplicate manifest key {key}")
        if row["split_family"] != SPLIT_FAMILY:
            raise RuntimeError(f"unexpected split family for {key}")
        if not row["adapt_eval_disjoint"] or not row["both_classes_adapt"] or not row["both_classes_eval"]:
            raise RuntimeError(f"invalid repaired split gate for {key}")
        if not row["target_labels_hidden_from_adaptation"]:
            raise RuntimeError(f"target labels not hidden for {key}")
        out[key] = row
    if len(out) != 189:
        raise RuntimeError(f"expected 189 selected manifest units, found {len(out)}")
    return out


def load_units() -> list[dict[str, str]]:
    with UNITS_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 189:
        raise RuntimeError(f"expected 189 frozen execution units, found {len(rows)}")
    if [int(row["unit_index"]) for row in rows] != list(range(189)):
        raise RuntimeError("FP-GEM unit_index sequence is not exact")
    return rows


def resolve_unit(args: argparse.Namespace) -> tuple[str, int, int, str]:
    if args.hardware_group is not None or args.group_index is not None:
        if args.hardware_group is None or args.group_index is None:
            raise RuntimeError("--hardware-group and --group-index must be provided together")
        matches = [row for row in load_units() if row["hardware_group"] == args.hardware_group]
        matches.sort(key=lambda row: int(row["hardware_group_index"]))
        if args.group_index < 0 or args.group_index >= len(matches):
            raise RuntimeError(f"group index out of range for {args.hardware_group}: {args.group_index}")
        row = matches[args.group_index]
        return row["dataset"], int(row["target_subject"]), int(row["source_seed"]), row["hardware_group"]
    if args.unit_index is not None:
        row = load_units()[args.unit_index]
        return row["dataset"], int(row["target_subject"]), int(row["source_seed"]), row["hardware_group"]
    if args.dataset is None or args.target_subject is None or args.seed is None:
        raise RuntimeError("provide --unit-index or dataset/target-subject/seed")
    hardware = "A100" if args.dataset == "Lee2019_MI" and args.seed == 2 and args.target_subject >= 27 else "V100"
    return args.dataset, args.target_subject, args.seed, hardware


def runtime_environment(device: torch.device) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    capability = list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
    return {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": gpu_name,
        "compute_capability": capability,
        "moabb_version": moabb.__version__,
        "mne_version": mne.__version__,
        "conda_default_env_inherited_label": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "hostname": platform.node(),
    }


def check_hardware_group(expected: str, runtime: dict[str, Any]) -> None:
    name = runtime["device_name"]
    if expected == "V100" and "V100" not in name:
        raise RuntimeError(f"unit requires V100-family source reproduction, allocated {name}")
    if expected == "A100" and "A100" not in name:
        raise RuntimeError(f"unit requires A100 source reproduction, allocated {name}")


def capture_preclassifier(
    model: torch.nn.Module,
    X: np.ndarray,
    domains: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        if len(inputs) != 1:
            raise RuntimeError("TSMNet classifier pre-hook received unexpected inputs")
        captured.append(inputs[0].detach())

    handle = model.classifier.register_forward_pre_hook(hook)
    features, logits = [], []
    max_error = 0.0
    parameter_t = torch.tensor(1.0, dtype=torch.float64, device="cpu")
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[start:start + batch_size], dtype=dtype, device=device)
                db = torch.as_tensor(domains[start:start + batch_size], dtype=torch.long)
                captured.clear()
                out = model(inputs=xb, domains=db, parameter_t=parameter_t, fm_mean=None)
                if len(captured) != 1:
                    raise RuntimeError(f"classifier feature hook fired {len(captured)} times")
                feat = captured[0]
                direct = model.classifier(feat)
                max_error = max(max_error, float((out - direct).abs().max().detach().cpu()))
                features.append(feat)
                logits.append(out.detach())
    finally:
        handle.remove()
    if not features:
        raise RuntimeError("no pre-classifier features captured")
    return torch.cat(features), torch.cat(logits), max_error


def train_source_model(
    *,
    ep,
    dataset: str,
    target: int,
    seed: int,
    source_idx: np.ndarray,
    domain_values: list[int],
    expected_state_hash: str,
    config: dict[str, Any],
    bn,
    TSMNet,
    Trainer,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = config["p9_pipeline"]["source_training"]
    train_idx, val_idx = _source_train_val_split(source_idx, ep.y, ep.subject, cfg["validation_fraction"])
    raw_root = Path(config["execution"]["raw_root"])
    cache = raw_root / "source_checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / f"{expected_state_hash}.pt"
    sidecar = cache / f"{expected_state_hash}.json"
    # P9 seeded immediately before TSMNet construction. Keep that exact order:
    # model initialization consumes the seeded stream, and training must continue
    # from the resulting RNG state rather than resetting it after construction.
    _set_seed(seed)
    model = _build_model(
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
    reused = False
    train_seconds = 0.0
    if checkpoint.exists() and sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if (meta.get("dataset"), int(meta.get("target_subject", -1)), int(meta.get("source_seed", -1))) != (
            dataset, target, seed
        ):
            raise RuntimeError("cached source checkpoint unit key mismatch")
        if meta.get("source_model_sha256") != expected_state_hash:
            raise RuntimeError("cached source checkpoint sidecar hash mismatch")
        if meta.get("checkpoint_file_sha256") != sha256_file(checkpoint):
            raise RuntimeError("cached source checkpoint file checksum mismatch")
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        reused = True
    else:
        train_ds = TensorDomainDataset(ep.X[train_idx], ep.y[train_idx], ep.subject[train_idx])
        val_ds = TensorDomainDataset(ep.X[val_idx], ep.y[val_idx], ep.subject[val_idx])
        train_loader = _loader(train_ds, cfg["batch_size"], True, seed + target)
        val_loader = _loader(val_ds, len(val_ds), False, seed)
        trainer = Trainer(
            max_epochs=cfg["epochs"],
            callbacks=[],
            loss=torch.nn.CrossEntropyLoss(),
            device=device,
            dtype=dtype,
        )
        start = time.time()
        trainer.fit(
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            parameter_t=torch.tensor(cfg["parameter_t"], dtype=torch.float64, device="cpu"),
        )
        train_seconds = time.time() - start
    actual_state_hash = hash_state(model)
    if actual_state_hash != expected_state_hash:
        raise RuntimeError(
            f"reproduced P9 source state mismatch for {(dataset, target, seed)}: "
            f"{actual_state_hash} != {expected_state_hash}"
        )
    if not reused:
        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        tmp = checkpoint.with_suffix(".pt.tmp")
        torch.save(cpu_state, tmp)
        os.replace(tmp, checkpoint)
        atomic_json(sidecar, {
            "dataset": dataset,
            "target_subject": target,
            "source_seed": seed,
            "source_model_sha256": expected_state_hash,
            "checkpoint_file_sha256": sha256_file(checkpoint),
            "p9_runner_sha256": P9_RUNNER_SHA256,
            "p9_config_sha256": P9_CONFIG_SHA256,
            "source_idx_sha256": _sha_indices(source_idx),
            "train_idx_sha256": _sha_indices(train_idx),
            "val_idx_sha256": _sha_indices(val_idx),
        })
    return model, train_idx, val_idx, {
        "source_model_sha256_expected": expected_state_hash,
        "source_model_sha256_actual": actual_state_hash,
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_file_sha256": sha256_file(checkpoint),
        "source_checkpoint_reused": reused,
        "source_train_seconds": train_seconds,
        "source_idx_sha256": _sha_indices(source_idx),
        "train_idx_sha256": _sha_indices(train_idx),
        "val_idx_sha256": _sha_indices(val_idx),
    }


def fit_source_density(
    features: torch.Tensor,
    labels: np.ndarray,
    *,
    dataset: str,
    target: int,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ClassConditionalDensity, np.ndarray, dict[str, Any]]:
    cfg = config["source_density"]
    density_cfg = DensityConfig(
        n_components=cfg["n_components"],
        cov_rank=cfg["cov_rank"],
        df=cfg["degrees_of_freedom"],
        eig_floor=cfg["eigenvalue_floor"],
        init_scale=cfg["init_scale"],
    )
    density_seed = stable_hash_int("P12", dataset, target, seed, "source_density")
    fork_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(density_seed)
        density = ClassConditionalDensity(features.shape[1], 2, density_cfg).to(device)
        ds = TensorDataset(features.detach(), torch.as_tensor(labels, dtype=torch.long, device=device))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(density_seed)
        loader = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            density.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        start = time.time()
        first_loss = last_loss = float("nan")
        for epoch in range(cfg["epochs"]):
            density.train()
            losses = []
            for xb, yb in loader:
                loss = -density.log_prob(xb, yb).mean() / density.dim
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite source density loss at epoch {epoch}")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(density.parameters(), cfg["gradient_clip"])
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(losses))
            if epoch == 0:
                first_loss = epoch_loss
            last_loss = epoch_loss
            scheduler.step()
        density_seconds = time.time() - start
    density.eval()
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float64)
    source_prior = counts / counts.sum()
    with torch.no_grad():
        logp = density.log_prob_all(features)
    if not torch.isfinite(logp).all():
        raise RuntimeError("non-finite fitted source density log probabilities")
    return density, source_prior, {
        "density_seed": density_seed,
        "density_state_sha256": hash_state(density),
        "density_fit_seconds": density_seconds,
        "density_first_epoch_nll_per_dim": first_loss,
        "density_final_epoch_nll_per_dim": last_loss,
        "density_log_prob_min": float(logp.min().detach().cpu()),
        "density_log_prob_max": float(logp.max().detach().cpu()),
        "source_train_class_counts": counts.astype(int).tolist(),
        "source_empirical_prior": source_prior.tolist(),
    }


def refit_rct(
    source_state: dict[str, torch.Tensor],
    *,
    ep,
    domain_values: list[int],
    adapt_idx: np.ndarray,
    target: int,
    config: dict[str, Any],
    bn,
    TSMNet,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = config["p9_pipeline"]["source_training"]
    model = _build_model(
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
    model.load_state_dict(source_state)
    before_parameters = hash_parameters(model)
    classifier_before = hash_state(model.classifier)
    x_adapt = torch.as_tensor(ep.X[adapt_idx], dtype=dtype, device=device)
    dummy_y = torch.zeros(len(adapt_idx), dtype=torch.long, device=device)
    d_adapt = torch.full((len(adapt_idx),), target, dtype=torch.long)
    start = time.time()
    model.eval()
    model.domainadapt_finetune(x_adapt, dummy_y, d_adapt, "refit")
    seconds = time.time() - start
    after_parameters = hash_parameters(model)
    classifier_after = hash_state(model.classifier)
    if before_parameters != after_parameters or classifier_before != classifier_after:
        raise RuntimeError("RCT changed frozen TSMNet parameters or classifier")
    return model, {
        "rct_seconds": seconds,
        "dummy_labels_unique": [0],
        "parameters_sha256_before_rct": before_parameters,
        "parameters_sha256_after_rct": after_parameters,
        "classifier_sha256_before_rct": classifier_before,
        "classifier_sha256_after_rct": classifier_after,
        "rct_model_state_sha256": hash_state(model),
    }


def run_unit(
    dataset: str,
    target: int,
    seed: int,
    hardware_group: str,
    *,
    config: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if dataset not in SELECTED_DATASETS or seed not in SELECTED_SEEDS:
        raise RuntimeError("unit is outside frozen P12 dataset/seed scope")
    if smoke and (dataset, target, seed) != ("BNCI2014_001", 1, 0):
        raise RuntimeError("P12 smoke unit is frozen to BNCI2014_001 target 1 seed 0")
    status = git_status()
    if status:
        raise RuntimeError(f"P12 GPU launch requires a clean worktree:\n{status}")
    frozen = validate_frozen_inputs(config)
    p9_hashes = p9_source_hashes(config)
    manifests = manifest_map(config)
    key = (dataset, target, seed)
    if key not in p9_hashes or key not in manifests:
        raise RuntimeError(f"unit key absent from frozen P9/manifest evidence: {key}")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("P12 smoke/fleet requires an allocated CUDA device")
    dtype = torch.float32
    runtime = runtime_environment(device)
    if runtime["sys_executable"] != config["execution"]["runtime_python"]:
        raise RuntimeError(f"unexpected Python runtime: {runtime['sys_executable']}")
    check_hardware_group(hardware_group, runtime)

    external = config["p9_pipeline"]["external_spdim_path"]
    bn, TSMNet, Trainer = _import_official(external)
    ep = load_dataset(dataset, MOABB_CLASS[dataset]().subject_list)
    split = manifests[key]
    source_subjects = [int(value) for value in split["source_subject_ids"]]
    source_idx = np.where(np.isin(ep.subject, np.asarray(source_subjects, dtype=np.int64)))[0]
    adapt_idx = indices_from_trial_ids(split["adapt_trial_ids"])
    eval_idx = indices_from_trial_ids(split["eval_trial_ids"])
    if set(adapt_idx.tolist()) & set(eval_idx.tolist()):
        raise RuntimeError("adaptation/evaluation overlap")
    if min(split["class_counts_adapt"]) <= 0 or min(split["class_counts_eval"]) <= 0:
        raise RuntimeError("manifest class-count gate failed")
    domain_values = sorted(source_subjects + [target])

    source_model, train_idx, val_idx, source_info = train_source_model(
        ep=ep,
        dataset=dataset,
        target=target,
        seed=seed,
        source_idx=source_idx,
        domain_values=domain_values,
        expected_state_hash=p9_hashes[key],
        config=config,
        bn=bn,
        TSMNet=TSMNet,
        Trainer=Trainer,
        device=device,
        dtype=dtype,
    )
    source_state = deepcopy(source_model.state_dict())
    source_features, source_logits, source_hook_error = capture_preclassifier(
        source_model,
        ep.X[train_idx],
        ep.subject[train_idx],
        device=device,
        dtype=dtype,
    )
    if source_features.shape[1] != config["feature_space"]["dimension"]:
        raise RuntimeError(f"unexpected TSMNet feature dimension {source_features.shape}")
    if hash_state(source_model) != source_info["source_model_sha256_actual"]:
        raise RuntimeError("source feature extraction changed the frozen P9 source state")
    if source_hook_error > 1e-7:
        raise RuntimeError(f"source feature-hook semantic mismatch: {source_hook_error}")
    density, source_prior, density_info = fit_source_density(
        source_features,
        ep.y[train_idx],
        dataset=dataset,
        target=target,
        seed=seed,
        config=config,
        device=device,
    )

    rct_model, rct_info = refit_rct(
        source_state,
        ep=ep,
        domain_values=domain_values,
        adapt_idx=adapt_idx,
        target=target,
        config=config,
        bn=bn,
        TSMNet=TSMNet,
        device=device,
        dtype=dtype,
    )
    target_domains_adapt = np.full(len(adapt_idx), target, dtype=np.int64)
    target_domains_eval = np.full(len(eval_idx), target, dtype=np.int64)
    adapt_features, adapt_logits, adapt_hook_error = capture_preclassifier(
        rct_model,
        ep.X[adapt_idx],
        target_domains_adapt,
        device=device,
        dtype=dtype,
    )
    eval_features, eval_logits, eval_hook_error = capture_preclassifier(
        rct_model,
        ep.X[eval_idx],
        target_domains_eval,
        device=device,
        dtype=dtype,
    )
    if max(adapt_hook_error, eval_hook_error) > 1e-7:
        raise RuntimeError("target feature-hook semantic mismatch")
    if not all(torch.isfinite(tensor).all() for tensor in (
        source_features, source_logits, adapt_features, adapt_logits, eval_features, eval_logits
    )):
        raise RuntimeError("non-finite feature or logit tensor")

    tta_cfg = TTAConfig(
        transform="diag_affine",
        lowrank=config["geometry_em"].get("lowrank", 4),
        trust_region=config["geometry_em"]["trust_region_a"],
        trust_region_b=config["geometry_em"]["trust_region_b"],
        logdet_weight=config["geometry_em"]["logdet_weight"],
        prior_anchor_strength=config["geometry_em"]["prior_anchor_strength"],
        dirichlet=config["geometry_em"]["dirichlet"],
        em_iters=config["geometry_em"]["outer_iterations"],
        em_lr=config["geometry_em"]["learning_rate"],
    )
    tta = ClassConditionalTTA(density, source_prior, tta_cfg, 2, str(device))
    tta_seed = stable_hash_int("P12", dataset, target, seed, split["split_hash"])
    model_params_before_gem = hash_parameters(rct_model)
    classifier_before_gem = hash_state(rct_model.classifier)
    density_before_gem = hash_state(density)
    start = time.time()
    joint = tta.fit_variant(
        adapt_features,
        B1A_VARIANTS_BY_NAME["joint_iterative_diag"],
        tta_seed=tta_seed,
    )
    joint_seconds = time.time() - start
    start = time.time()
    fixed = tta.fit_variant(
        adapt_features,
        B1A_VARIANTS_BY_NAME["gen_iterative_diag"],
        tta_seed=tta_seed,
    )
    fixed_seconds = time.time() - start
    if not np.allclose(fixed.pi_T.detach().cpu().numpy(), source_prior, atol=0.0, rtol=0.0):
        raise RuntimeError("FP-GEM changed the fixed source empirical prior")
    if hash_parameters(rct_model) != model_params_before_gem or hash_state(rct_model.classifier) != classifier_before_gem:
        raise RuntimeError("GEM changed frozen TSMNet parameters or classifier")
    if hash_state(density) != density_before_gem:
        raise RuntimeError("GEM changed the frozen source density")

    common = {
        "dataset": dataset,
        "target_subject": target,
        "source_seed": seed,
        "split_family": SPLIT_FAMILY,
        "split_hash": split["split_hash"],
        "manifest_hash": MANIFEST_HASH,
        "source_subject_ids": source_subjects,
        "n_source": int(len(source_idx)),
        "n_adapt": int(len(adapt_idx)),
        "n_eval": int(len(eval_idx)),
        "class_counts_adapt": split["class_counts_adapt"],
        "class_counts_eval": split["class_counts_eval"],
        "adapt_eval_disjoint": True,
        "both_classes_adapt": min(split["class_counts_adapt"]) > 0,
        "both_classes_eval": min(split["class_counts_eval"]) > 0,
        "source_idx_sha256": _sha_indices(source_idx),
        "adapt_idx_sha256": _sha_indices(adapt_idx),
        "eval_idx_sha256": _sha_indices(eval_idx),
        "target_labels_passed_to_adaptation": False,
        "target_performance_selection": False,
        "source_checkpoint": source_info,
        "source_density": density_info,
        "rct": rct_info,
        "feature_hook": {
            "module": "TSMNet.classifier",
            "type": "register_forward_pre_hook",
            "dimension": int(source_features.shape[1]),
            "source_semantic_max_abs_error": source_hook_error,
            "adapt_semantic_max_abs_error": adapt_hook_error,
            "eval_semantic_max_abs_error": eval_hook_error,
            "source_features_sha256": hash_array(source_features.detach().cpu().numpy()),
            "adapt_features_sha256": hash_array(adapt_features.detach().cpu().numpy()),
            "eval_features_sha256": hash_array(eval_features.detach().cpu().numpy()),
        },
        "geometry": {
            "tta_seed": tta_seed,
            "joint_pi_fit": joint.pi_T.detach().cpu().numpy().tolist(),
            "fp_pi_fit": fixed.pi_T.detach().cpu().numpy().tolist(),
            "source_empirical_prior": source_prior.tolist(),
            "joint_objective": joint.objective,
            "fp_objective": fixed.objective,
            "joint_a": joint.transform.a.detach().cpu().numpy().tolist(),
            "joint_b": joint.transform.b.detach().cpu().numpy().tolist(),
            "fp_a": fixed.transform.a.detach().cpu().numpy().tolist(),
            "fp_b": fixed.transform.b.detach().cpu().numpy().tolist(),
            "joint_responsibility_hash": hash_array(joint.r_last_used.detach().cpu().numpy()),
            "fp_responsibility_hash": hash_array(fixed.r_last_used.detach().cpu().numpy()),
            "joint_seconds": joint_seconds,
            "fp_seconds": fixed_seconds,
        },
        "provenance": {
            "launch_commit": launch_commit(),
            "git_status_porcelain": status,
            "clean_worktree": True,
            "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "runner_sha256": sha256_file(__file__),
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": sha256_file(CONFIG_PATH),
            "frozen_inputs": frozen,
            "runtime": runtime,
            "hardware_group": hardware_group,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "official_pretrained_weight_used": False,
            "third_party_vendored": False,
        },
    }

    with torch.no_grad():
        joint_logits = rct_model.classifier(joint.transform.apply(eval_features))
        fixed_logits = rct_model.classifier(fixed.transform.apply(eval_features))
    if not torch.isfinite(joint_logits).all() or not torch.isfinite(fixed_logits).all():
        raise RuntimeError("non-finite GEM evaluation logits")
    if smoke:
        return common | {
            "status": "pass",
            "mode": "smoke",
            "performance_metrics_computed": False,
            "evaluation_labels_accessed": False,
            "joint_prediction_shape": list(joint_logits.shape),
            "fp_prediction_shape": list(fixed_logits.shape),
            "joint_prediction_hash": hash_array(joint_logits.argmax(1).detach().cpu().numpy()),
            "fp_prediction_hash": hash_array(fixed_logits.argmax(1).detach().cpu().numpy()),
            "joint_logits_hash": hash_array(joint_logits.detach().cpu().numpy()),
            "fp_logits_hash": hash_array(fixed_logits.detach().cpu().numpy()),
        }

    # Runtime target labels are first read after both adaptation fits are complete.
    y_eval = np.asarray(ep.y[eval_idx], dtype=np.int64)
    results = []
    for method, logits, fit, seconds in (
        ("Joint-GEM", joint_logits, joint, joint_seconds),
        ("FP-GEM", fixed_logits, fixed, fixed_seconds),
    ):
        logits_np = logits.detach().cpu().numpy()
        predictions = logits_np.argmax(1).astype(np.int64)
        results.append({
            "dataset": dataset,
            "target_subject": target,
            "source_seed": seed,
            "split_family": SPLIT_FAMILY,
            "method": method,
            "n_adapt": int(len(adapt_idx)),
            "n_eval": int(len(eval_idx)),
            "class_counts_adapt": split["class_counts_adapt"],
            "class_counts_eval": split["class_counts_eval"],
            **_metrics(y_eval, predictions),
            **_prediction_fields(y_eval, predictions, logits_np),
            "status": "ok",
            "failure_reason": "",
            "source_model_sha256": source_info["source_model_sha256_actual"],
            "source_checkpoint_file_sha256": source_info["source_checkpoint_file_sha256"],
            "density_state_sha256": density_info["density_state_sha256"],
            "transform_a": fit.transform.a.detach().cpu().numpy().tolist(),
            "transform_b": fit.transform.b.detach().cpu().numpy().tolist(),
            "pi_fit": fit.pi_T.detach().cpu().numpy().tolist(),
            "fit_objective": fit.objective,
            "adapt_seconds": seconds,
            "target_label_leakage_detected": False,
            "target_performance_selection_detected": False,
            "classifier_frozen": True,
            "backbone_frozen": True,
        })
    return common | {
        "status": "ok",
        "mode": "run",
        "performance_metrics_computed": True,
        "evaluation_labels_accessed_only_after_adaptation": True,
        "results": results,
    }


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    frozen = validate_frozen_inputs(config)
    p9 = p9_source_hashes(config)
    manifest = manifest_map(config)
    units = load_units()
    unit_keys = {(row["dataset"], int(row["target_subject"]), int(row["source_seed"])) for row in units}
    manifest_keys = set(manifest)
    p9_keys = set(p9)
    hardware_counts = {group: sum(row["hardware_group"] == group for row in units) for group in ("V100", "A100")}
    external = config["p9_pipeline"]["external_spdim_path"]
    bn, TSMNet, _Trainer = _import_official(external)
    ep = load_dataset("BNCI2014_001", MOABB_CLASS["BNCI2014_001"]().subject_list)
    source_cfg = config["p9_pipeline"]["source_training"]
    model = _build_model(
        TSMNet,
        bn,
        ep=ep,
        domain_values=sorted(int(value) for value in np.unique(ep.subject)),
        args=argparse.Namespace(
            temporal_filters=source_cfg["temporal_filters"],
            spatial_filters=source_cfg["spatial_filters"],
            subspace_dims=source_cfg["subspace_dims"],
            spd_device=source_cfg["spd_device"],
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    probe_features, probe_logits, probe_error = capture_preclassifier(
        model,
        ep.X[:1],
        ep.subject[:1],
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    feature_probe = {
        "dataset": "BNCI2014_001",
        "feature_shape": list(probe_features.shape),
        "logit_shape": list(probe_logits.shape),
        "semantic_max_abs_error": probe_error,
        "finite": bool(torch.isfinite(probe_features).all() and torch.isfinite(probe_logits).all()),
        "target_labels_accessed": False,
    }
    expected = config["expected_counts"]
    return {
        "dryrun_pass": (
            unit_keys == manifest_keys == p9_keys
            and len(units) == expected["target_seed_units"]
            and hardware_counts == {"V100": 161, "A100": 28}
            and feature_probe["feature_shape"] == [1, config["feature_space"]["dimension"]]
            and feature_probe["logit_shape"] == [1, 2]
            and feature_probe["semantic_max_abs_error"] <= 1e-7
            and feature_probe["finite"]
        ),
        "datasets": list(SELECTED_DATASETS),
        "source_seeds": list(SELECTED_SEEDS),
        "unit_count": len(units),
        "hardware_group_counts": hardware_counts,
        "feature_hook_cpu_probe": feature_probe,
        "new_rows_expected": len(units) * len(NEW_METHODS),
        "reused_p9_rows_expected": len(units) * len(P9_METHODS),
        "final_rows_expected": len(units) * (len(NEW_METHODS) + len(P9_METHODS)),
        "unit_keys_match_manifest": unit_keys == manifest_keys,
        "unit_keys_match_p9": unit_keys == p9_keys,
        "all_adapt_eval_disjoint": all(row["adapt_eval_disjoint"] for row in manifest.values()),
        "all_eval_both_classes": all(row["both_classes_eval"] for row in manifest.values()),
        "all_adapt_both_classes": all(row["both_classes_adapt"] for row in manifest.values()),
        "target_label_leakage_detected": False,
        "target_performance_selection_detected": False,
        "frozen_inputs": frozen,
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(__file__),
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dry-run", "smoke", "run"), required=True)
    ap.add_argument("--unit-index", type=int)
    ap.add_argument("--hardware-group", choices=("V100", "A100"))
    ap.add_argument("--group-index", type=int)
    ap.add_argument("--dataset", choices=SELECTED_DATASETS)
    ap.add_argument("--target-subject", type=int)
    ap.add_argument("--seed", type=int, choices=SELECTED_SEEDS)
    ap.add_argument("--out", default="")
    return ap


def main() -> int:
    args = parser().parse_args()
    config = load_config()
    if args.mode == "dry-run":
        report = dry_run(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["dryrun_pass"] else 2
    dataset, target, seed, hardware_group = resolve_unit(args)
    out = Path(args.out) if args.out else Path(config["execution"]["raw_root"]) / (
        "smoke.json" if args.mode == "smoke" else f"units/{dataset}_target{target}_seed{seed}.json"
    )
    if out.exists():
        existing = json.loads(out.read_text())
        if existing.get("status") in {"ok", "pass"}:
            print(json.dumps({"status": "skip_complete", "out": str(out)}, sort_keys=True))
            return 0
    started = time.time()
    try:
        payload = run_unit(
            dataset,
            target,
            seed,
            hardware_group,
            config=config,
            smoke=args.mode == "smoke",
        )
        payload["elapsed_seconds"] = time.time() - started
        atomic_json(out, payload)
        print(json.dumps({
            "status": payload["status"],
            "mode": payload["mode"],
            "dataset": dataset,
            "target_subject": target,
            "source_seed": seed,
            "out": str(out),
            "sha256": sha256_file(out),
        }, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "mode": args.mode,
            "dataset": dataset,
            "target_subject": target,
            "source_seed": seed,
            "hardware_group": hardware_group,
            "failure_reason": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "launch_commit": launch_commit(),
            "runner_sha256": sha256_file(__file__),
            "config_sha256": sha256_file(CONFIG_PATH),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        }
        atomic_json(out, failure)
        print(json.dumps({"status": "failed", "out": str(out), "failure_reason": failure["failure_reason"]}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
