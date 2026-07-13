"""Shared fail-closed helpers for S2P Phase C."""

import csv
import hashlib
import importlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import torch


TAGS = [
    "random",
    "released",
    "H200_s0",
    "H200_s1",
    "H500_s0",
    "H500_s1",
    "H1000_s0",
    "H1000_s1",
    "H2000_s0",
    "H2000_s1",
]
GATE_TAGS = ["random", "released", "H200_s0", "H1000_s0", "H2000_s0"]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            block = fobj.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def truth(value):
    return str(value).lower() == "true"


def validate_manifest(path):
    rows = read_csv(path)
    if len(rows) != 10 or [row["tag"] for row in rows] != TAGS:
        raise RuntimeError("Phase C requires the ordered ten-object immutable manifest")
    for row in rows:
        tag = row["tag"]
        if not all(
            truth(row[field])
            for field in ("strict_reload_pass", "parameter_exact_pass", "feature_equivalence_pass")
        ):
            raise RuntimeError(f"immutable closure did not pass for {tag}")
        if float(row["feature_max_abs_diff"]) != 0.0:
            raise RuntimeError(f"nonzero closure feature difference for {tag}")
        if tag == "random":
            if not row["immutable_path"].startswith("random_init_contract://sha256_"):
                raise RuntimeError("random deterministic contract mismatch")
            continue
        payload = Path(row["immutable_path"])
        if not payload.is_file() or payload.is_symlink():
            raise RuntimeError(f"checkpoint is not a direct payload: {tag}")
        if stat.S_IMODE(payload.stat().st_mode) & 0o222:
            raise RuntimeError(f"checkpoint is writable: {tag}")
        digest = sha256_file(payload)
        if digest != row["immutable_sha256"] or digest not in payload.name:
            raise RuntimeError(f"content-addressed checkpoint mismatch: {tag}")
    return rows


def manifest_row(rows, tag):
    matches = [row for row in rows if row["tag"] == tag]
    if len(matches) != 1:
        raise RuntimeError(f"manifest tag is not unique: {tag}")
    return matches[0]


def configure_determinism(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def import_cbramod(cbramod_root):
    root = str(Path(cbramod_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("models.cbramod")
    return module.CBraMod


def make_model(cbramod_root, seed=0):
    configure_determinism(seed)
    cbramod = import_cbramod(cbramod_root)
    return cbramod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=10,
        n_layer=12,
        nhead=8,
    )


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint is not a dictionary")
    for key in ("model_state", "model", "state_dict", "model_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict) or not obj:
        raise RuntimeError("checkpoint does not contain a state dictionary")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in obj.items()}


def build_model(row, cbramod_root, device):
    model = make_model(cbramod_root, seed=0)
    if row["tag"] != "random":
        state = unwrap_state_dict(
            torch.load(row["immutable_path"], map_location="cpu", weights_only=False)
        )
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"strict state mismatch for {row['tag']}")
    model = model.to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def normalize_patches(data):
    data = np.asarray(data, dtype=np.float32)
    normalized = (data - data.mean(axis=-1, keepdims=True)) / (
        data.std(axis=-1, keepdims=True) + 1e-6
    )
    return np.ascontiguousarray(normalized.astype(np.float32))


@torch.inference_mode()
def extract_features(model, data, device, batch_size):
    chunks = []
    for start in range(0, len(data), batch_size):
        batch = torch.from_numpy(data[start:start + batch_size]).to(device)
        patch = model.patch_embedding(batch, None)
        encoded = model.encoder(patch)
        feature = encoded.mean(dim=2).reshape(encoded.shape[0], -1)
        chunks.append(feature.float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(chunks).astype(np.float32))


def checkpoint_hash(row):
    if row["tag"] == "random":
        return row["immutable_sha256"]
    return sha256_file(row["immutable_path"])


def sanitize_checkpoint_path(row):
    if row["tag"] == "random":
        return row["immutable_path"]
    return "${PHASE_B_ARTIFACT_ROOT}/" + Path(row["immutable_path"]).name
