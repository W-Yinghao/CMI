#!/usr/bin/env python
"""GPU-only released-tokenizer target gate for S2P-CodeBrain-Bounded.

This is a target-distribution and native-shape canary.  It performs no
optimization, backward pass, checkpoint write, pretraining, or fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import codebrain_bounded_data as CBD


def required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return Path(value).expanduser().resolve()


CODEBRAIN = required_env_path("CODEBRAIN_ROOT")
TOKENIZER = required_env_path("CODEBRAIN_TOKENIZER_PATH")
EXPECTED_TOKENIZER_SHA256 = "e9560b670d64ea4712fd99a48dc2131326b919744c7d3eb504cf57b1ef3af999"
CODEBOOK_SIZE = 4096
EXPECTED_TOKENS_PER_WINDOW = 19 * 30
THRESHOLDS = {
    "minimum_windows_per_stratum": 256,
    "minimum_unique_codes": 16,
    "minimum_effective_perplexity": 8.0,
    "maximum_dominant_code_fraction": 0.95,
    "minimum_unique_sequences": 16,
    "minimum_unique_sequence_fraction": 0.05,
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_tokenizer(device: torch.device):
    sys.path.insert(0, str(CODEBRAIN))
    import Models.modeling_tokenizer  # noqa: F401 - registers tfdual_vq
    from timm.models import create_model

    model = create_model(
        "tfdual_vq", pretrained=True, pretrained_weight=str(TOKENIZER), as_tokenzer=True,
        n_code_t=CODEBOOK_SIZE, n_code_f=CODEBOOK_SIZE, code_dim=32,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_sssm(device: torch.device):
    sys.path.insert(0, str(CODEBRAIN))
    from Models.SSSM import SSSM

    return SSSM(
        in_channels=200, res_channels=200, skip_channels=200, out_channels=200,
        num_res_layers=8, diffusion_step_embed_dim_in=200,
        diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
        s4_lmax=570, s4_d_state=64, s4_dropout=0.1, s4_bidirectional=True,
        s4_layernorm=True, codebook_size_t=CODEBOOK_SIZE,
        codebook_size_f=CODEBOOK_SIZE, if_codebook=True,
    ).eval().to(device)


def stream_metrics(ids: np.ndarray, stream: str, stratum: str) -> dict:
    if ids.ndim != 2 or ids.shape[1] != EXPECTED_TOKENS_PER_WINDOW:
        raise RuntimeError(f"{stratum}/{stream}: target shape {ids.shape} != [N,570]")
    flat = ids.reshape(-1)
    if flat.size == 0 or int(flat.min()) < 0 or int(flat.max()) >= CODEBOOK_SIZE:
        raise RuntimeError(f"{stratum}/{stream}: token id outside [0,{CODEBOOK_SIZE})")
    counts = np.bincount(flat, minlength=CODEBOOK_SIZE).astype(float)
    prob = counts[counts > 0] / counts.sum()
    entropy = float(-(prob * np.log(prob)).sum())
    perplexity = float(np.exp(entropy))
    unique_sequences = int(np.unique(ids, axis=0).shape[0])
    min_sequences = max(
        THRESHOLDS["minimum_unique_sequences"],
        int(math.ceil(ids.shape[0] * THRESHOLDS["minimum_unique_sequence_fraction"])),
    )
    row = {
        "stratum": stratum, "stream": stream, "n_windows": int(ids.shape[0]),
        "tokens_per_window": int(ids.shape[1]), "n_tokens": int(flat.size),
        "unique_codes": int((counts > 0).sum()),
        "codebook_utilization_fraction": float((counts > 0).sum() / CODEBOOK_SIZE),
        "entropy_nats": entropy, "normalized_entropy": float(entropy / math.log(CODEBOOK_SIZE)),
        "effective_perplexity": perplexity, "dominant_code_fraction": float(counts.max() / counts.sum()),
        "unique_sequences": unique_sequences,
        "unique_sequence_fraction": float(unique_sequences / ids.shape[0]),
        "minimum_unique_sequences_required": min_sequences,
    }
    checks = {
        "windows": row["n_windows"] >= THRESHOLDS["minimum_windows_per_stratum"],
        "unique_codes": row["unique_codes"] >= THRESHOLDS["minimum_unique_codes"],
        "perplexity": row["effective_perplexity"] >= THRESHOLDS["minimum_effective_perplexity"],
        "dominant_fraction": row["dominant_code_fraction"] <= THRESHOLDS["maximum_dominant_code_fraction"],
        "unique_sequences": row["unique_sequences"] >= min_sequences,
    }
    row.update({f"check_{name}": bool(value) for name, value in checks.items()})
    row["stream_stratum_gate_pass"] = bool(all(checks.values()))
    return row


def native_shape_canary(tokenizer, device: torch.device, batch: torch.Tensor) -> dict:
    from einops import rearrange
    from Utils.util import generate_mask

    x = batch[:8].to(device) / 100.0
    if list(x.shape) != [8, 19, 30, 200]:
        raise RuntimeError(f"native canary input shape mismatch: {list(x.shape)}")
    with torch.no_grad():
        t1, f1 = tokenizer.get_codebook_indices(x, list(range(20)))
        t2, f2 = tokenizer.get_codebook_indices(x, list(range(20)))
    deterministic = bool(torch.equal(t1, t2) and torch.equal(f1, f2))
    torch.manual_seed(941811)
    mask = generate_mask(8, 19, 30, mask_ratio=0.5, device=device)
    model = build_sssm(device)
    with torch.no_grad():
        logits_t, logits_f = model(x, mask=mask)
        flat_mask = rearrange(mask, "b c p -> b (c p)")
        target_t = t1[flat_mask == 1]
        target_f = f1[flat_mask == 1]
        ce_t = F.cross_entropy(logits_t, target_t)
        ce_f = F.cross_entropy(logits_f, target_f)
    n_masked = int(mask.sum().item())
    checks = {
        "input_shape": list(x.shape) == [8, 19, 30, 200],
        "mask_shape": list(mask.shape) == [8, 19, 30],
        "target_shape": list(t1.shape) == [8, 570] and list(f1.shape) == [8, 570],
        "logit_shape": list(logits_t.shape) == [n_masked, CODEBOOK_SIZE]
                       and list(logits_f.shape) == [n_masked, CODEBOOK_SIZE],
        "masked_target_count": target_t.numel() == n_masked and target_f.numel() == n_masked,
        "finite_loss": bool(torch.isfinite(ce_t) and torch.isfinite(ce_f)),
        "tokenizer_deterministic": deterministic,
    }
    return {
        "input_shape": list(x.shape), "mask_shape": list(mask.shape),
        "target_t_shape": list(t1.shape), "target_f_shape": list(f1.shape),
        "logits_t_shape": list(logits_t.shape), "logits_f_shape": list(logits_f.shape),
        "n_masked": n_masked, "temporal_ce": float(ce_t.cpu()), "frequency_ce": float(ce_f.cpu()),
        "checks": checks, "native_shape_canary_pass": bool(all(checks.values())),
        "optimizer_constructed": False, "backward_called": False, "trainer_called": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--samples-per-stratum", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate_path = out / "codebrain_tokenizer_target_gate.json"
    final_path = out / "codebrain_bounded_go_nogo.json"
    metadata_path = out / "codebrain_metadata_preflight.json"
    result = {
        "tokenizer_target_gate_pass": False, "native_shape_canary_pass": False,
        "exception": None, "thresholds": THRESHOLDS, "training_called": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the native CodeBrain tokenizer/SSSM path")
        actual_sha = file_sha256(TOKENIZER)
        if actual_sha != EXPECTED_TOKENIZER_SHA256:
            raise RuntimeError(f"tokenizer SHA mismatch: {actual_sha}")
        device = torch.device("cuda:0")
        samples = CBD.diagnostic_samples(n_per_stratum=args.samples_per_stratum)
        by_stratum = defaultdict(list)
        for sample in samples:
            by_stratum[sample["stratum"]].append(sample)
        tokenizer = load_tokenizer(device)
        rows = []
        canary_batch = None
        for stratum in ("H200", "H1000_increment", "H2000_increment", "pretrain_val"):
            dataset = CBD.CodeBrainWindowDataset(by_stratum[stratum])
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            temporal, frequency = [], []
            for xb, _ in loader:
                if canary_batch is None:
                    canary_batch = xb.clone()
                x = xb.to(device, non_blocking=True) / 100.0
                with torch.no_grad():
                    ids_t, ids_f = tokenizer.get_codebook_indices(x, list(range(20)))
                temporal.append(ids_t.cpu().numpy())
                frequency.append(ids_f.cpu().numpy())
            rows.append(stream_metrics(np.concatenate(temporal), "temporal", stratum))
            rows.append(stream_metrics(np.concatenate(frequency), "frequency", stratum))
        target_pass = bool(all(row["stream_stratum_gate_pass"] for row in rows))
        canary = native_shape_canary(tokenizer, device, canary_batch)
        result.update({
            "tokenizer_path": "${CODEBRAIN_TOKENIZER_PATH}", "tokenizer_sha256": actual_sha,
            "device": torch.cuda.get_device_name(0), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "samples_per_stratum": args.samples_per_stratum,
            "strata": list(by_stratum), "tokenizer_frozen": not any(p.requires_grad for p in tokenizer.parameters()),
            "tokenizer_target_gate_pass": target_pass,
            "native_shape_canary_pass": canary["native_shape_canary_pass"],
            "native_shape_canary": canary,
            "temporal_target_non_degenerate": bool(all(
                r["stream_stratum_gate_pass"] for r in rows if r["stream"] == "temporal")),
            "frequency_target_non_degenerate": bool(all(
                r["stream_stratum_gate_pass"] for r in rows if r["stream"] == "frequency")),
        })
        write_csv(out / "codebrain_tokenizer_target_metrics.csv", rows)
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    dump_json(gate_path, result)

    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    scientific_preflight_pass = bool(
        metadata.get("budget_metadata_pass")
        and metadata.get("downstream_asset_contract_pass")
        and result.get("tokenizer_target_gate_pass")
        and result.get("native_shape_canary_pass")
    )
    blockers = list(metadata.get("launch_blockers", []))
    if not result.get("tokenizer_target_gate_pass"):
        blockers.append("released tokenizer target utilization gate failed")
    if not result.get("native_shape_canary_pass"):
        blockers.append("native Stage-2 target/logit shape canary failed")
    final = {
        "phase": "S2P_CodeBrain_Bounded_cross_architecture_representation_emergence",
        "budget_metadata_pass": metadata.get("budget_metadata_pass"),
        "downstream_asset_contract_pass": metadata.get("downstream_asset_contract_pass"),
        "tokenizer_target_gate_pass": result.get("tokenizer_target_gate_pass"),
        "temporal_target_non_degenerate": result.get("temporal_target_non_degenerate"),
        "frequency_target_non_degenerate": result.get("frequency_target_non_degenerate"),
        "native_shape_canary_pass": result.get("native_shape_canary_pass"),
        "scientific_preflight_pass": scientific_preflight_pass,
        "launch_blockers": blockers,
        "launch_bounded_stage2_recommended": scientific_preflight_pass,
        "launch_bounded_stage2": False,
        "training_requires_pm_review": True,
        "training_launched": False, "fine_tuning_launched": False,
        "full_1k_9k_replication_launched": False,
        "target_labels_used_for_selection": False,
    }
    dump_json(final_path, final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
