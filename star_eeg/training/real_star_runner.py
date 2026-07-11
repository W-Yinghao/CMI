"""Actual source-only mixed-stream CBraMod runner for STAR smoke/launch.

The module has no evaluation or non-source FACED path. It accepts only the
SHA-named immutable H200 payload from the closure manifest, the H200 Route-B
TUEG pool, and the full FACED source_train anchor manifests.
"""

import hashlib
import json
import math
import os
import random
import socket
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from star_eeg.config import STAR01
from star_eeg.data.anchor_batch_stream import build_anchor_batches, load_shuffled_manifest
from star_eeg.data.checkpoint_registry import CBRAMOD_ROOT, sha256_file
from star_eeg.data.faced_source_train_loader import (
    FACEDSourceTrainAnchorLoader,
    load_anchor_manifest,
)
from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.data.tueg_ssl_batch_stream import (
    TUEGRouteBWindowLoader,
    build_ssl_batches,
)
from star_eeg.objectives.alternating_schedule import (
    SSL_CONT,
    STAR_SHUFFLED,
    STAR_TRUE,
    build_schedule,
)
from star_eeg.training.approval_lock import (
    cell_name,
    validate_approval_lock,
    validate_six_cell_array_environment,
)
from star_eeg.training.persistence import (
    append_telemetry_row,
    atomic_torch_save_no_overwrite,
    atomic_write_json_no_overwrite,
    claim_attempt_directory,
    freeze_completed_attempt,
    no_temporary_files,
    open_telemetry,
    sha256_file as persistence_sha256_file,
    validate_telemetry_file,
)


TRAINED_VARIANTS = (SSL_CONT, STAR_TRUE, STAR_SHUFFLED)


@dataclass(frozen=True)
class RealStarConfig:
    variant: str
    model_seed: int
    optimizer_steps: int
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 5e-2
    scheduler_eta_min: float = 1e-5
    mask_ratio: float = 0.5
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "disabled_fp32"
    ssl_loss: str = "masked_mse_mean"
    zero_grad_semantics: str = "set_to_none_true"
    model_mode: str = "train"

    def validate(self) -> None:
        if self.variant not in TRAINED_VARIANTS:
            raise ValueError(f"variant outside frozen training universe: {self.variant}")
        if self.model_seed not in STAR01.model_seeds:
            raise ValueError("model seed outside frozen universe")
        if self.optimizer_steps <= 0 or self.optimizer_steps > STAR01.continuation_optimizer_steps:
            raise ValueError("optimizer_steps outside 1..3750")
        if self.optimizer_steps % 5:
            raise ValueError("optimizer_steps must end on a complete 4+1 cycle")
        expected = {
            "batch_size": STAR01.batch_size,
            "learning_rate": STAR01.base_learning_rate,
            "weight_decay": STAR01.weight_decay,
            "scheduler_eta_min": STAR01.scheduler_eta_min,
            "mask_ratio": 0.5,
            "gradient_clip_norm": 1.0,
            "mixed_precision": "disabled_fp32",
            "ssl_loss": "masked_mse_mean",
            "zero_grad_semantics": "set_to_none_true",
            "model_mode": "train",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"{field} differs from frozen real-path contract")


def _stable_seed(*parts: object) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def _setup_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _imports():
    if str(CBRAMOD_ROOT) not in sys.path:
        sys.path.insert(0, str(CBRAMOD_ROOT))
    from models.cbramod import CBraMod
    from utils.util import generate_mask

    return CBraMod, generate_mask


def _torch_load(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _state_dict(payload: object) -> Mapping[str, object]:
    if isinstance(payload, dict):
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint has no model state dictionary")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in payload.items()}


def tensor_state_hash(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_hash(array) -> str:
    contiguous = array.astype("float32", copy=False)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _grad_norm(parameters: Sequence[object]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = float(parameter.grad.detach().float().norm(2).cpu())
            total += value * value
    return math.sqrt(total)


def _parameter_delta(before: Sequence[object], parameters: Sequence[object]) -> float:
    total = 0.0
    for old, parameter in zip(before, parameters):
        value = float((parameter.detach() - old).float().norm(2).cpu())
        total += value * value
    return math.sqrt(total)


def _immutable_start(
    manifest_path: Path,
    go_nogo_path: Path,
    model_seed: int,
) -> Mapping[str, object]:
    manifest = json.loads(manifest_path.read_text())
    go_nogo = json.loads(go_nogo_path.read_text())
    if go_nogo.get("status") != "PASS" or go_nogo.get("launcher_restricted_to_sha_named_payload") is not True:
        raise RuntimeError("H200 immutable closure is not launchable")
    row = next(
        (item for item in manifest.get("checkpoints", []) if item.get("tag") == f"H200_s{model_seed}"),
        None,
    )
    if row is None:
        raise RuntimeError(f"immutable manifest missing H200_s{model_seed}")
    path = Path(row["launcher_accepted_path"])
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("launcher path must be a regular SHA-named payload")
    expected_name = f"best.{row['sha256']}.pth"
    if path.name != expected_name:
        raise RuntimeError("launcher path filename does not embed the declared SHA")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError("launcher path is writable")
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError("launcher path SHA mismatch")
    return row


def _new_model_and_head(start_path: Path, model_seed: int, device):
    import torch

    CBraMod, _ = _imports()
    _setup_seed(model_seed)
    model = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
    ).to(device)
    checkpoint = _torch_load(start_path, device)
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    torch.manual_seed(STAR01.head_seed_offset + model_seed)
    torch.cuda.manual_seed_all(STAR01.head_seed_offset + model_seed)
    task_head = torch.nn.Linear(6400, 9).to(device)
    return model, task_head


def strict_reload_training_checkpoint(path: Path, device) -> bool:
    payload = _torch_load(path, device)
    model_seed = int(payload["config"]["model_seed"])
    start_path = Path(payload["source_checkpoint"])
    model, task_head = _new_model_and_head(start_path, model_seed, device)
    model.load_state_dict(payload["model_state"], strict=True)
    task_head.load_state_dict(payload["task_head_state"], strict=True)
    return True


def require_launch_approval(
    config: RealStarConfig,
    launch_approval_path: Optional[Path],
    repo_root: Path,
) -> Mapping[str, object]:
    if config.optimizer_steps <= 20:
        return {
            "status": "NOT_REQUIRED_BOUNDED_SMOKE",
            "approval_manifest_hash": None,
            "approved_cell": cell_name(config.variant, config.model_seed),
            "target_scoring": "BLOCKED",
        }
    if config.optimizer_steps != STAR01.continuation_optimizer_steps:
        raise PermissionError("STAR_00B permits only bounded 5-20 step smoke updates")
    if launch_approval_path is None:
        raise PermissionError("3750-step STAR_01 requires a PM launch approval manifest")
    approval = validate_approval_lock(
        launch_approval_path,
        repo_root=repo_root,
        variant=config.variant,
        model_seed=config.model_seed,
    )
    return {
        **approval,
        "array_environment": validate_six_cell_array_environment(
            config.variant, config.model_seed
        ),
    }


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(repo_root), text=True
    ).strip()


def _hashed_payload(core: Mapping[str, object], hash_field: str) -> Dict[str, object]:
    return {**core, hash_field: canonical_hash(core)}


def _execution_source_hashes(repo_root: Path) -> Dict[str, str]:
    relative_paths = (
        "star_eeg/training/real_star_runner.py",
        "star_eeg/training/persistence.py",
        "star_eeg/training/approval_lock.py",
        "star_eeg/runners/run_star01_train.py",
        "star_eeg/slurm/star01_train_array.sbatch",
    )
    return {
        relative: persistence_sha256_file(repo_root / relative)
        for relative in relative_paths
    }


def run_real_star(
    config: RealStarConfig,
    repo_root: Path,
    immutable_manifest_path: Path,
    immutable_go_nogo_path: Path,
    anchor_manifest_path: Path,
    shuffled_manifest_path: Path,
    faced_lmdb_path: Path,
    contract_dir: Path,
    runtime_output_dir: Path,
    device_name: str,
    attempt_id: str,
    launch_approval_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Run one persistent bounded smoke or approval-locked 3,750-step cell."""
    repo_root = Path(repo_root).resolve()
    config.validate()
    approval_audit = require_launch_approval(
        config,
        launch_approval_path,
        repo_root=repo_root,
    )
    import torch

    if not torch.cuda.is_available() or not str(device_name).startswith("cuda"):
        raise RuntimeError("real STAR mixed-stream smoke requires CUDA")
    device = torch.device(device_name)
    start = _immutable_start(
        immutable_manifest_path,
        immutable_go_nogo_path,
        config.model_seed,
    )
    start_path = Path(start["launcher_accepted_path"])
    source_sha_before = sha256_file(start_path)
    cell = cell_name(config.variant, config.model_seed)
    if config.optimizer_steps == STAR01.continuation_optimizer_steps:
        array_attempt = approval_audit["array_environment"]["attempt_id"]
        approved_attempt = approval_audit["approved_attempt_id"]
        if array_attempt != attempt_id or approved_attempt != attempt_id:
            raise PermissionError(
                "attempt_id differs from the approval-locked Slurm environment"
            )
    attempt_dir = claim_attempt_directory(runtime_output_dir, attempt_id)
    checkpoints_dir = attempt_dir / "checkpoints"
    checkpoints_dir.mkdir()
    attempt_wall_start = time.time()

    artifact_input_paths = {
        "h200_immutable_manifest": Path(immutable_manifest_path),
        "h200_immutable_go_nogo": Path(immutable_go_nogo_path),
        "anchor_manifest": Path(anchor_manifest_path),
        "shuffled_manifest": Path(shuffled_manifest_path),
        "route_b_channel_contract": Path(contract_dir) / "route_b_canonical_channel_order.json",
    }
    artifact_input_hashes = {
        label: persistence_sha256_file(path)
        for label, path in artifact_input_paths.items()
    }
    execution_core = {
        "schema_version": 1,
        "phase": (
            "STAR_01A_BLIND_TRAINING"
            if config.optimizer_steps == STAR01.continuation_optimizer_steps
            else "STAR_00C_BOUNDED_REALPATH_SMOKE"
        ),
        "cell": cell,
        "attempt_id": attempt_id,
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "tracked_worktree_clean": not bool(
            _git(repo_root, "status", "--porcelain", "--untracked-files=no")
        ),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_array_geometry": approval_audit.get("array_environment"),
        "device": str(device),
        "source_checkpoint": str(start_path.resolve()),
        "source_checkpoint_sha": source_sha_before,
        "approval_status": approval_audit["status"],
        "approval_manifest_hash": approval_audit.get("approval_manifest_hash"),
        "approval_file_sha256": approval_audit.get("approval_file_sha256"),
        "target_scoring": "BLOCKED",
        "artifact_input_hashes": artifact_input_hashes,
        "execution_source_hashes": _execution_source_hashes(repo_root),
    }
    execution_manifest = _hashed_payload(
        execution_core, "execution_manifest_hash"
    )
    atomic_write_json_no_overwrite(
        attempt_dir / "execution_manifest.json", execution_manifest
    )

    model, task_head = _new_model_and_head(start_path, config.model_seed, device)
    model.train()
    task_head.train()
    initial_model_hash = tensor_state_hash(model.state_dict())
    initial_head_hash = tensor_state_hash(task_head.state_dict())
    parameter_names = [name for name, _ in model.named_parameters()]
    update_scope_hash = canonical_hash(parameter_names)

    optimizer_parameters = list(model.parameters()) + list(task_head.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=STAR01.continuation_optimizer_steps,
        eta_min=config.scheduler_eta_min,
    )
    criterion_ssl = torch.nn.MSELoss(reduction="mean")
    criterion_anchor = torch.nn.CrossEntropyLoss(reduction="mean")
    _, generate_mask = _imports()

    anchor_manifest = load_anchor_manifest(anchor_manifest_path)
    shuffled_manifest = load_shuffled_manifest(shuffled_manifest_path)
    anchor_batches, _ = build_anchor_batches(anchor_manifest, shuffled_manifest, config.model_seed)
    common_batches, replacement_batches, tueg_mapping = build_ssl_batches(
        repo_root, config.model_seed, contract_dir
    )
    tueg_loader = TUEGRouteBWindowLoader(tueg_mapping, contract_dir)
    faced_loader = FACEDSourceTrainAnchorLoader(faced_lmdb_path, anchor_manifest)
    schedule = build_schedule(config.variant, total_steps=config.optimizer_steps)
    schedule_rows = [asdict(row) for row in schedule]
    common_batch_hashes = [canonical_hash(batch) for batch in common_batches]
    replacement_batch_hashes = [canonical_hash(batch) for batch in replacement_batches]
    anchor_x_hashes = [str(row["x_id_hash"]) for row in anchor_batches]
    stream_core = {
        "common_ssl_stream_hash": canonical_hash(common_batch_hashes),
        "replacement_ssl_stream_hash": canonical_hash(replacement_batch_hashes),
        "anchor_x_stream_hash": canonical_hash(anchor_x_hashes),
        "anchor_true_label_stream_hash": canonical_hash([
            row["true_label_hash"] for row in anchor_batches
        ]),
        "anchor_shuffled_label_stream_hash": canonical_hash([
            row["shuffled_label_hash"] for row in anchor_batches
        ]),
    }
    config_hash = canonical_hash(asdict(config))
    schedule_hash = canonical_hash(schedule_rows)
    stream_hash = canonical_hash(stream_core)
    run_manifest_core = {
        "schema_version": 1,
        "cell": cell,
        "variant": config.variant,
        "model_seed": config.model_seed,
        "attempt_id": attempt_id,
        "config": asdict(config),
        "config_hash": config_hash,
        "schedule_hash": schedule_hash,
        "stream_hash": stream_hash,
        "streams": stream_core,
        "optimizer_steps_expected": config.optimizer_steps,
        "checkpoint_save_steps": [
            step for step in STAR01.checkpoint_save_steps
            if step <= config.optimizer_steps
        ] + ([config.optimizer_steps] if config.optimizer_steps not in STAR01.checkpoint_save_steps else []),
        "primary_checkpoint_step": STAR01.primary_checkpoint_step,
        "primary_checkpoint_fixed_final_step": (
            config.optimizer_steps == STAR01.primary_checkpoint_step
        ),
        "bounded_smoke_checkpoint_is_not_scientific_primary": (
            config.optimizer_steps < STAR01.primary_checkpoint_step
        ),
        "optimizer_update_and_batch_count_matched": True,
        "strict_flop_matched": False,
        "source_checkpoint_sha": source_sha_before,
        "approval_manifest_hash": approval_audit.get("approval_manifest_hash"),
        "target_data_allowed": False,
    }
    run_manifest = _hashed_payload(run_manifest_core, "run_manifest_hash")
    atomic_write_json_no_overwrite(
        attempt_dir / "run_manifest.json", run_manifest
    )

    telemetry: List[Mapping[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)
    optimizer_wall_start = time.time()
    final_checkpoint = None
    telemetry_path = attempt_dir / "telemetry.jsonl"
    telemetry_handle, telemetry_digest = open_telemetry(telemetry_path)
    with telemetry_handle:
        for scheduled in schedule:
            optimizer.zero_grad(set_to_none=True)
            model_parameters = list(model.parameters())
            head_parameters = list(task_head.parameters())
            before_parameters = [
                parameter.detach().clone() for parameter in optimizer_parameters
            ]
            data_stream = None
            id_hash = None
            tensor_hash = None
            label_hash = None
            labels_differ = None
            applied_mask = False

            if scheduled.update_kind == "ssl":
                data_stream = f"tueg_ssl_{scheduled.ssl_stream}"
                source = (
                    common_batches
                    if scheduled.ssl_stream == "common"
                    else replacement_batches
                )
                window_ids = source[int(scheduled.ssl_stream_index) - 1]
                batch_array = tueg_loader.load_batch(window_ids)
                id_hash = canonical_hash(window_ids)
                tensor_hash = _array_hash(batch_array)
                batch = torch.from_numpy(batch_array).to(device)
                objective_seed = _stable_seed(
                    "native-ssl-objective",
                    STAR01.ssl_objective_seed_offset,
                    config.model_seed,
                    scheduled.ssl_stream,
                    scheduled.ssl_stream_index,
                )
                torch.manual_seed(objective_seed)
                torch.cuda.manual_seed_all(objective_seed)
                batch_size, channels, patches, _ = batch.shape
                mask = generate_mask(
                    batch_size,
                    channels,
                    patches,
                    mask_ratio=config.mask_ratio,
                    device=device,
                )
                prediction = model(batch, mask=mask)
                loss = criterion_ssl(prediction[mask == 1], batch[mask == 1])
                applied_mask = True
            else:
                data_stream = "faced_source_train_anchor"
                row = anchor_batches[int(scheduled.anchor_stream_index) - 1]
                batch_array, true_labels_array, _ = faced_loader.load_batch(
                    row["sample_ids"]
                )
                if true_labels_array.tolist() != row["true_labels"]:
                    raise RuntimeError(
                        "FACED loader labels differ from frozen anchor stream"
                    )
                selected_labels = (
                    row["true_labels"]
                    if config.variant == STAR_TRUE
                    else row["shuffled_labels"]
                )
                labels_differ = row["true_labels"] != row["shuffled_labels"]
                id_hash = row["x_id_hash"]
                tensor_hash = _array_hash(batch_array)
                label_hash = canonical_hash(selected_labels)
                batch = torch.from_numpy(batch_array).to(device)
                labels = torch.as_tensor(
                    selected_labels, dtype=torch.long, device=device
                )
                objective_seed = _stable_seed(
                    "source-anchor-objective",
                    STAR01.anchor_stream_seed_offset,
                    config.model_seed,
                    scheduled.anchor_stream_index,
                )
                torch.manual_seed(objective_seed)
                torch.cuda.manual_seed_all(objective_seed)
                embedded = model.patch_embedding(batch, None)
                encoded = model.encoder(embedded)
                features = encoded.mean(2).reshape(encoded.shape[0], -1)
                if features.shape[1] != 6400:
                    raise RuntimeError(
                        f"FACED anchor representation dim {features.shape[1]} != 6400"
                    )
                logits = task_head(features)
                loss = criterion_anchor(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {scheduled.optimizer_step}"
                )
            loss.backward()
            encoder_parameters = list(model.encoder.parameters())
            encoder_grad_before = _grad_norm(encoder_parameters)
            model_grad_before = _grad_norm(model_parameters)
            head_grad_before = _grad_norm(head_parameters)
            model_clip_return = float(
                torch.nn.utils.clip_grad_norm_(
                    model_parameters, config.gradient_clip_norm
                ).detach().cpu()
            )
            head_clip_return = (
                float(
                    torch.nn.utils.clip_grad_norm_(
                        head_parameters, config.gradient_clip_norm
                    ).detach().cpu()
                )
                if head_grad_before > 0
                else 0.0
            )
            encoder_grad_after = _grad_norm(encoder_parameters)
            model_grad_after = _grad_norm(model_parameters)
            head_grad_after = _grad_norm(head_parameters)
            finite_values = [
                float(loss.detach().cpu()),
                encoder_grad_before,
                encoder_grad_after,
                model_grad_before,
                model_grad_after,
                head_grad_before,
                head_grad_after,
                model_clip_return,
                head_clip_return,
            ]
            if not all(math.isfinite(value) for value in finite_values):
                raise RuntimeError(
                    f"non-finite gradient/clip telemetry at step {scheduled.optimizer_step}"
                )
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            learning_rate_after = float(optimizer.param_groups[0]["lr"])
            step_delta = _parameter_delta(
                before_parameters, optimizer_parameters
            )
            current_model_hash = tensor_state_hash(model.state_dict())
            if not math.isfinite(step_delta):
                raise RuntimeError(
                    f"non-finite parameter delta at step {scheduled.optimizer_step}"
                )
            telemetry_row = {
                "step": scheduled.optimizer_step,
                "semantic_slot": scheduled.semantic_slot,
                "data_stream": data_stream,
                "batch_id_hash": id_hash,
                "batch_tensor_hash": tensor_hash,
                "label_hash": label_hash,
                "true_shuffled_labels_differ": labels_differ,
                "loss": float(loss.detach().cpu()),
                "encoder_grad_norm_before_clipping": encoder_grad_before,
                "encoder_grad_norm_after_clipping": encoder_grad_after,
                "model_grad_norm_before_clipping": model_grad_before,
                "model_grad_norm_after_clipping": model_grad_after,
                "temporary_head_grad_norm_before_clipping": head_grad_before,
                "temporary_head_grad_norm_after_clipping": head_grad_after,
                "model_clip_return": model_clip_return,
                "temporary_head_clip_return": head_clip_return,
                "parameter_delta_norm": step_delta,
                "learning_rate": learning_rate_used,
                "learning_rate_after_scheduler_step": learning_rate_after,
                "mask_ratio": config.mask_ratio,
                "mask_applied": applied_mask,
                "nan_inf_status": "PASS",
                "source_checkpoint_sha": source_sha_before,
                "current_model_state_hash": current_model_hash,
            }
            telemetry.append(telemetry_row)
            telemetry_prefix_sha = append_telemetry_row(
                telemetry_handle,
                telemetry_digest,
                telemetry_row,
            )

            if (
                scheduled.optimizer_step in STAR01.checkpoint_save_steps
                or scheduled.optimizer_step == config.optimizer_steps
            ):
                checkpoint_path = checkpoints_dir / (
                    f"step_{scheduled.optimizer_step:04d}.pth"
                )
                payload = {
                    "schema_version": 2,
                    "cell": cell,
                    "attempt_id": attempt_id,
                    "variant": config.variant,
                    "model_seed": config.model_seed,
                    "optimizer_step": scheduled.optimizer_step,
                    "model_state": model.state_dict(),
                    "task_head_state": task_head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "source_checkpoint": str(start_path),
                    "source_checkpoint_sha": source_sha_before,
                    "config": asdict(config),
                    "config_hash": config_hash,
                    "schedule_hash": schedule_hash,
                    "stream_hash": stream_hash,
                    "telemetry_rows": len(telemetry),
                    "telemetry_sha256": telemetry_prefix_sha,
                    "primary_checkpoint": (
                        scheduled.optimizer_step
                        == STAR01.primary_checkpoint_step
                    ),
                    "target_data_used": False,
                    "approval_manifest_hash": approval_audit.get(
                        "approval_manifest_hash"
                    ),
                }
                atomic_torch_save_no_overwrite(checkpoint_path, payload)
                final_checkpoint = checkpoint_path

    if final_checkpoint is None:
        raise RuntimeError("runner did not save a final smoke checkpoint")
    optimizer_loop_wall_seconds = time.time() - optimizer_wall_start
    strict_reload_pass = strict_reload_training_checkpoint(final_checkpoint, device)
    source_sha_after = sha256_file(start_path)
    final_model_hash = tensor_state_hash(model.state_dict())
    final_head_hash = tensor_state_hash(task_head.state_dict())
    common_telemetry = [row for row in telemetry if row["data_stream"] == "tueg_ssl_common"]
    anchor_telemetry = [row for row in telemetry if row["data_stream"] == "faced_source_train_anchor"]
    replacement_telemetry = [row for row in telemetry if row["data_stream"] == "tueg_ssl_replacement"]
    telemetry_validation = validate_telemetry_file(
        telemetry_path, config.optimizer_steps
    )
    final_checkpoint_payload = _torch_load(final_checkpoint, "cpu")
    checkpoint_telemetry_matches = (
        int(final_checkpoint_payload.get("telemetry_rows", -1))
        == telemetry_validation["rows"]
        and final_checkpoint_payload.get("telemetry_sha256")
        == telemetry_validation["sha256"]
    )
    faced_access_audit = faced_loader.access_audit()
    summary_core = {
        "schema_version": 1,
        "cell": cell,
        "attempt_id": attempt_id,
        "variant": config.variant,
        "model_seed": config.model_seed,
        "optimizer_steps": config.optimizer_steps,
        "config": asdict(config),
        "config_hash": config_hash,
        "schedule_hash": schedule_hash,
        "stream_hash": stream_hash,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": float(time.time() - attempt_wall_start),
        "optimizer_loop_wall_seconds": float(optimizer_loop_wall_seconds),
        "encoder_updates": len(telemetry),
        "full_reconstruction_updates": len(common_telemetry) + len(replacement_telemetry),
        "anchor_updates": len(anchor_telemetry),
        "optimizer_update_and_batch_count_matched": True,
        "strict_flop_matched": False,
        "source_checkpoint": str(start_path),
        "source_checkpoint_sha_before": source_sha_before,
        "source_checkpoint_sha_after": source_sha_after,
        "source_checkpoint_unchanged": source_sha_before == source_sha_after,
        "start_model_state_hash": initial_model_hash,
        "final_model_state_hash": final_model_hash,
        "initial_task_head_hash": initial_head_hash,
        "final_task_head_hash": final_head_hash,
        "temporary_head_unchanged": initial_head_hash == final_head_hash,
        "temporary_head_discard_before_evaluation": True,
        "model_update_scope_hash": update_scope_hash,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "task_head_parameter_count": sum(parameter.numel() for parameter in task_head.parameters()),
        "common_ssl_batch_id_hashes": [row["batch_id_hash"] for row in common_telemetry],
        "common_ssl_batch_tensor_hashes": [row["batch_tensor_hash"] for row in common_telemetry],
        "anchor_x_id_hashes": [row["batch_id_hash"] for row in anchor_telemetry],
        "anchor_x_tensor_hashes": [row["batch_tensor_hash"] for row in anchor_telemetry],
        "anchor_label_hashes": [row["label_hash"] for row in anchor_telemetry],
        "replacement_ssl_batch_id_hashes": [row["batch_id_hash"] for row in replacement_telemetry],
        "losses_finite": all(math.isfinite(row["loss"]) for row in telemetry),
        "gradients_finite": all(row["nan_inf_status"] == "PASS" for row in telemetry),
        "parameter_deltas_finite": all(math.isfinite(row["parameter_delta_norm"]) for row in telemetry),
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "checkpoint_strict_reload_pass": strict_reload_pass,
        "checkpoint_telemetry_matches_persisted_payload": checkpoint_telemetry_matches,
        "telemetry_path": str(telemetry_path),
        "telemetry_rows": telemetry_validation["rows"],
        "telemetry_sha256": telemetry_validation["sha256"],
        "faced_loader_access_audit": faced_access_audit,
        "approval_manifest_hash": approval_audit.get("approval_manifest_hash"),
        "real_optimizer_updates_run": True,
        "real_eeg_smoke_run": config.optimizer_steps < STAR01.continuation_optimizer_steps,
        "scientific_training_run": config.optimizer_steps == STAR01.continuation_optimizer_steps,
        "formal_3750_step_training_run": config.optimizer_steps == STAR01.continuation_optimizer_steps,
        "bounded_realpath_smoke": config.optimizer_steps < STAR01.continuation_optimizer_steps,
        "target_metrics_computed": False,
        "target_data_used": False,
    }
    run_summary = _hashed_payload(summary_core, "run_summary_hash")
    atomic_write_json_no_overwrite(
        attempt_dir / "run_summary.json", run_summary
    )
    persisted_summary = json.loads((attempt_dir / "run_summary.json").read_text())
    persisted_summary_core = {
        key: value for key, value in persisted_summary.items()
        if key != "run_summary_hash"
    }
    summary_hash_valid = (
        canonical_hash(persisted_summary_core)
        == persisted_summary.get("run_summary_hash")
    )
    completion_checks = {
        "exact_telemetry_rows": telemetry_validation["rows"] == config.optimizer_steps,
        "final_step_exact": telemetry_validation["final_step"] == config.optimizer_steps,
        "formal_final_step_3750_if_scientific": (
            config.optimizer_steps != STAR01.continuation_optimizer_steps
            or telemetry_validation["final_step"] == STAR01.primary_checkpoint_step
        ),
        "final_checkpoint_strict_reload": strict_reload_pass,
        "source_checkpoint_sha_unchanged": source_sha_before == source_sha_after,
        "no_target_data_access": (
            faced_access_audit["source_val_sample_reads"] == 0
            and faced_access_audit["test_sample_reads"] == 0
            and faced_access_audit["target_tensors_created"] == 0
        ),
        "all_losses_finite": summary_core["losses_finite"],
        "all_gradients_finite": summary_core["gradients_finite"],
        "all_parameter_deltas_finite": summary_core["parameter_deltas_finite"],
        "run_summary_hash_verified": summary_hash_valid,
        "telemetry_hash_verified": (
            telemetry_validation["sha256"]
            == persistence_sha256_file(telemetry_path)
        ),
        "checkpoint_telemetry_hash_verified": checkpoint_telemetry_matches,
        "no_temporary_files": no_temporary_files(attempt_dir),
    }
    if not all(completion_checks.values()):
        raise RuntimeError(f"run completion checks failed: {completion_checks}")
    completion_core = {
        "schema_version": 1,
        "status": "COMPLETE",
        "phase": execution_core["phase"],
        "cell": cell,
        "attempt_id": attempt_id,
        "optimizer_steps": config.optimizer_steps,
        "final_step": telemetry_validation["final_step"],
        "checks": completion_checks,
        "source_checkpoint_sha": source_sha_before,
        "telemetry_sha256": telemetry_validation["sha256"],
        "telemetry_rows": telemetry_validation["rows"],
        "run_manifest_hash": run_manifest["run_manifest_hash"],
        "execution_manifest_hash": execution_manifest["execution_manifest_hash"],
        "run_summary_hash": run_summary["run_summary_hash"],
        "run_summary_file_sha256": persistence_sha256_file(
            attempt_dir / "run_summary.json"
        ),
        "final_checkpoint_sha256": summary_core["checkpoint_sha256"],
        "approval_manifest_hash": approval_audit.get("approval_manifest_hash"),
        "target_metrics_computed": False,
        "target_data_used": False,
        "overwrite_allowed": False,
    }
    completion = _hashed_payload(completion_core, "completion_hash")
    atomic_write_json_no_overwrite(
        attempt_dir / "completion.json", completion
    )
    freeze_completed_attempt(attempt_dir)
    return {
        **run_summary,
        "telemetry": telemetry,
        "completion": completion,
        "attempt_tree_frozen_read_only": True,
    }
