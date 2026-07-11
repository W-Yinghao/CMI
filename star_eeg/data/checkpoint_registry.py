"""Read-only inventory of frozen S2P checkpoints and FACED references."""

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from star_eeg.config import DEPENDENCY_COMMIT
from star_eeg.data.faced_split_contract import canonical_hash, canonical_json_bytes


B1_ROOT = Path("/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1")
H2000_ROOT = Path("/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1_immutable")
RELEASED_PATH = Path("/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth")
CBRAMOD_ROOT = Path("/home/infres/yinwang/eeg2025/CBraMod")

RANDOM_REFERENCE_CONFIG = {
    "tag": "random",
    "model": "CBraMod",
    "init_seed": 0,
    "architecture": {
        "in_dim": 200,
        "out_dim": 200,
        "d_model": 200,
        "dim_feedforward": 800,
        "seq_len": 10,
        "n_layer": 12,
        "nhead": 8,
    },
    "frozen_reference_only": True,
}


@dataclass(frozen=True)
class CheckpointSpec:
    tag: str
    budget_h: Optional[int]
    seed: Optional[int]
    path: str
    kind: str
    usable_as_star_start: bool
    usable_as_reference_only: bool


def registry_specs() -> List[CheckpointSpec]:
    specs = []
    for budget in (200, 500, 1000):
        for seed in (0, 1):
            specs.append(CheckpointSpec(
                tag=f"H{budget}_s{seed}",
                budget_h=budget,
                seed=seed,
                path=str(B1_ROOT / f"H{budget}_s{seed}" / "best.pth"),
                kind="route_b_checkpoint",
                usable_as_star_start=budget == 200,
                usable_as_reference_only=budget != 200,
            ))
    for seed in (0, 1):
        specs.append(CheckpointSpec(
            tag=f"H2000_s{seed}",
            budget_h=2000,
            seed=seed,
            path=str(H2000_ROOT / f"H2000_s{seed}" / "best.pth"),
            kind="route_b_immutable_checkpoint",
            usable_as_star_start=False,
            usable_as_reference_only=True,
        ))
    specs.extend([
        CheckpointSpec(
            tag="released",
            budget_h=None,
            seed=None,
            path=str(RELEASED_PATH),
            kind="released_reference",
            usable_as_star_start=False,
            usable_as_reference_only=True,
        ),
        CheckpointSpec(
            tag="random",
            budget_h=None,
            seed=0,
            path="config://cbramod-random-reference-seed-0",
            kind="random_reference_config",
            usable_as_star_start=False,
            usable_as_reference_only=True,
        ),
    ])
    return specs


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unwrap_state_dict(obj: object) -> Mapping[str, object]:
    state = obj
    if isinstance(state, dict):
        for key in ("model_state", "model", "state_dict", "model_state_dict"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def strict_reload(spec: CheckpointSpec) -> Dict[str, object]:
    """Instantiate CBraMod and strictly load without running inference."""
    if str(CBRAMOD_ROOT) not in sys.path:
        sys.path.insert(0, str(CBRAMOD_ROOT))
    from models.cbramod import CBraMod

    sequence_length = 10 if spec.kind in {"released_reference", "random_reference_config"} else 30
    model = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=sequence_length,
        n_layer=12,
        nhead=8,
    )
    if spec.kind == "random_reference_config":
        return {
            "strict_reload_pass": True,
            "source_git_commit": DEPENDENCY_COMMIT,
            "route_manifest": None,
            "checkpoint_epoch": None,
            "checkpoint_val_loss": None,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    obj = _torch_load(Path(spec.path))
    model.load_state_dict(_unwrap_state_dict(obj), strict=True)
    metadata = obj if isinstance(obj, dict) else {}
    return {
        "strict_reload_pass": True,
        "source_git_commit": metadata.get("git"),
        "route_manifest": metadata.get("route_b_manifest"),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_val_loss": metadata.get("val_loss"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _training_provenance(spec: CheckpointSpec, reload_info: Mapping[str, object]) -> Dict[str, object]:
    if not spec.tag.startswith("H"):
        return {"training_complete": None, "provenance_complete": spec.tag == "random"}
    cell_dir = Path(spec.path).parent
    summary_path = cell_dir / "run_summary.json"
    log_path = cell_dir / "train_log.jsonl"
    if not summary_path.exists() or not log_path.exists():
        return {"training_complete": False, "provenance_complete": False}
    summary = _read_json(summary_path)
    events = _read_jsonl(log_path)
    epochs = [int(row["epoch"]) for row in events if row.get("event") == "epoch"]
    done = [row for row in events if row.get("event") == "done"]
    expected_epochs = int(summary.get("epochs", -1))
    route_manifest = reload_info.get("route_manifest")
    route_manifest_matches = route_manifest == summary.get("cell_manifest")
    checkpoint_matches = (
        int(reload_info.get("checkpoint_epoch") or -1) == int(summary.get("best_epoch", -2))
        and abs(float(reload_info.get("checkpoint_val_loss") or 0.0) - float(summary.get("best_val_loss") or 1.0)) < 1e-12
    )
    complete = all([
        summary.get("route") == "B_33ch_cbramod_only",
        int(summary.get("budget_h", -1)) == int(spec.budget_h),
        int(summary.get("subset_seed", -1)) == int(spec.seed),
        expected_epochs == 50,
        epochs == list(range(1, 51)),
        len(done) == 1,
        summary.get("checkpoint_strict_reload") is True,
        summary.get("target_labels_used") is False,
        summary.get("smoke") is False,
        summary.get("cell_manifest", {}).get("train_val_disjoint") is True,
        route_manifest_matches,
        checkpoint_matches,
    ])
    return {
        "training_complete": bool(complete),
        "provenance_complete": bool(complete),
        "run_summary_path": str(summary_path),
        "train_log_path": str(log_path),
    }


def _contract_hashes(repo_root: Path) -> Dict[str, str]:
    route_path = repo_root / "results/s2p_route_b_33ch_contract/route_b_canonical_channel_order.json"
    faced_path = repo_root / "results/s2p_route_b_33ch_b1_faced/faced_channel_order_manifest.json"
    return {
        "route_b": sha256_file(route_path),
        "faced_reference": sha256_file(faced_path),
    }


def inspect_spec(
    spec: CheckpointSpec,
    repo_root: Path,
    loader: Callable[[CheckpointSpec], Mapping[str, object]] = strict_reload,
) -> Dict[str, object]:
    contract_hashes = _contract_hashes(repo_root)
    if spec.kind == "random_reference_config":
        raw = canonical_json_bytes(RANDOM_REFERENCE_CONFIG)
        reload_info = dict(loader(spec))
        provenance = _training_provenance(spec, reload_info)
        return {
            "tag": spec.tag,
            "budget_h": spec.budget_h,
            "seed": spec.seed,
            "path": spec.path,
            "exists": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "read_only_or_stable": True,
            "strict_reload_pass": bool(reload_info.get("strict_reload_pass")),
            "source_git_commit": reload_info.get("source_git_commit"),
            "route_manifest_hash": None,
            "channel_contract_hash": contract_hashes["faced_reference"],
            "training_complete": provenance["training_complete"],
            "usable_as_star_start": spec.usable_as_star_start,
            "usable_as_reference_only": spec.usable_as_reference_only,
            "kind": spec.kind,
            "provenance_complete": provenance["provenance_complete"],
            "repeated_sha_identical": True,
            "resolved_path_stable": True,
            "parameter_count": reload_info.get("parameter_count"),
        }

    path = Path(spec.path)
    exists = path.exists()
    base = {
        "tag": spec.tag,
        "budget_h": spec.budget_h,
        "seed": spec.seed,
        "path": spec.path,
        "exists": exists,
        "sha256": None,
        "bytes": None,
        "read_only_or_stable": False,
        "strict_reload_pass": False,
        "source_git_commit": None,
        "route_manifest_hash": None,
        "channel_contract_hash": contract_hashes["route_b"] if spec.tag.startswith("H") else contract_hashes["faced_reference"],
        "training_complete": None,
        "usable_as_star_start": spec.usable_as_star_start,
        "usable_as_reference_only": spec.usable_as_reference_only,
        "kind": spec.kind,
        "provenance_complete": False,
        "repeated_sha_identical": False,
        "resolved_path_stable": False,
    }
    if not exists:
        return base
    resolved_before = path.resolve()
    first_sha = sha256_file(resolved_before)
    reload_error = None
    try:
        reload_info = dict(loader(spec))
    except Exception as exc:  # fail closed while preserving the inventory row
        reload_info = {}
        reload_error = f"{type(exc).__name__}: {exc}"
    second_sha = sha256_file(path.resolve())
    resolved_stable = resolved_before == path.resolve()
    repeated_stable = first_sha == second_sha
    provenance = _training_provenance(spec, reload_info) if reload_info else {
        "training_complete": False if spec.tag.startswith("H") else None,
        "provenance_complete": False,
    }
    route_manifest = reload_info.get("route_manifest")
    mode = stat.S_IMODE(path.resolve().stat().st_mode)
    base.update({
        "sha256": first_sha,
        "bytes": path.resolve().stat().st_size,
        "read_only_or_stable": bool((mode & 0o222) == 0 or (repeated_stable and resolved_stable)),
        "strict_reload_pass": bool(reload_info.get("strict_reload_pass", False)),
        "source_git_commit": reload_info.get("source_git_commit"),
        "route_manifest_hash": canonical_hash(route_manifest) if route_manifest is not None else None,
        "training_complete": provenance["training_complete"],
        "provenance_complete": provenance["provenance_complete"],
        "repeated_sha_identical": repeated_stable,
        "resolved_path_stable": resolved_stable,
        "resolved_path": str(resolved_before),
        "mode_octal": oct(mode),
        "parameter_count": reload_info.get("parameter_count"),
        "checkpoint_epoch": reload_info.get("checkpoint_epoch"),
        "strict_reload_error": reload_error,
    })
    return base


def build_checkpoint_inventory(
    repo_root: Path,
    specs: Optional[Iterable[CheckpointSpec]] = None,
    loader: Callable[[CheckpointSpec], Mapping[str, object]] = strict_reload,
) -> Dict[str, object]:
    entries = [inspect_spec(spec, repo_root=repo_root, loader=loader) for spec in (specs or registry_specs())]
    core = {
        "schema_version": 1,
        "dependency_commit": DEPENDENCY_COMMIT,
        "entries": entries,
    }
    h200 = [entry for entry in entries if entry["tag"] in {"H200_s0", "H200_s1"}]
    ready = len(h200) == 2 and all(
        entry["exists"]
        and entry["repeated_sha_identical"]
        and entry["resolved_path_stable"]
        and entry["strict_reload_pass"]
        and entry["training_complete"] is True
        and entry["provenance_complete"] is True
        and bool(entry["source_git_commit"])
        and bool(entry["route_manifest_hash"])
        for entry in h200
    )
    return {
        **core,
        "h200_start_checkpoints_ready": ready,
        "star01_artifact_status": "READY_FOR_PM_REVIEW" if ready else "STAR_01_BLOCKED_ARTIFACT_SUPPLY",
        "checkpoint_inventory_hash": canonical_hash(core),
    }
