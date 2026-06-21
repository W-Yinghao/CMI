"""Auditable grid I/O: run manifest, full config signature, strict resume index, persisted
source bundles and compute-resume helpers (review §"audit infrastructure").

The shift/action grids previously keyed resume on result identity alone and hashed only a
few config fields, so a different config (or commit) writing to the same path could silently
skip "done" units, and resume re-trained every source model. This module adds:

  * config_signature(cfg)         -- sha over the FULL nested config.
  * build_manifest / validate_or_create_manifest -- a per-output run manifest (schema,
    commit, config signature, scenarios, normalized CLI, run_signature); a re-run with a
    different signature ABORTS instead of appending into a mixed file.
  * load_done_keys(path, *, item_field) -- strict, keyword-only (no implicit "method"
    default), fails loudly on bad JSON / missing fields / duplicate keys.
  * persisted source bundles with data-hash + config-signature + checkpoint-hash checks,
    and expected_keys() so a runner can SKIP training when a bundle's units are all done.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict

import numpy as np
import torch

SCHEMA_VERSION = 1


# ----------------------------------------------------------------- hashing / git
def full_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(__file__) or ".").decode().strip()
    except Exception:
        return "unknown"


def short_sha() -> str:
    return full_sha()[:12]


def hash_array(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]


def hash_state(model) -> str:
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def config_signature(cfg) -> str:
    """sha over the FULL nested config (every dataclass field), not a hand-picked subset."""
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()[:16]


def variant_id(**parts) -> str:
    """Normalised method id for B1 (responsibility/update/prior/family/alpha/...).
    Keys sorted; None/empty dropped; floats formatted stably. e.g.
    variant_id(resp='gen', update='oneshot', prior='fixed', family='diag') ->
    'family=diag__prior=fixed__resp=gen__update=oneshot'."""
    items = []
    for k in sorted(parts):
        v = parts[k]
        if v is None or v == "":
            continue
        if isinstance(v, float):
            v = f"{v:g}"
        items.append(f"{k}={v}")
    return "__".join(items)


# ----------------------------------------------------------------- manifest
def manifest_path(out_path: str) -> str:
    return out_path + ".manifest.json"


def build_manifest(cfg, scenarios, item_field: str, cli: dict) -> dict:
    core = dict(schema_version=SCHEMA_VERSION, commit_sha=full_sha(),
                config_signature=config_signature(cfg), item_field=item_field,
                scenarios=sorted(scenarios), cli=cli)
    core["run_signature"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]
    core["config"] = asdict(cfg)
    return core


def validate_or_create_manifest(out_path: str, manifest: dict) -> dict:
    """If a manifest already sits beside ``out_path`` it must MATCH (else abort); otherwise
    write the new one. Guards against resuming into a file produced by a different
    config/commit/scenario set."""
    mp = manifest_path(out_path)
    if os.path.exists(mp):
        with open(mp) as f:
            old = json.load(f)
        for k in ("schema_version", "commit_sha", "config_signature", "item_field",
                  "scenarios", "run_signature"):
            if old.get(k) != manifest.get(k):
                raise RuntimeError(
                    f"manifest mismatch on '{k}' for {out_path}: existing={old.get(k)!r} "
                    f"new={manifest.get(k)!r}; use a fresh --out (do not mix runs)")
        return old
    os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


# ----------------------------------------------------------------- strict resume index
def load_done_keys(path: str, *, item_field: str) -> set:
    """Strict, keyword-only resume index keyed by
    (data_seed, target_site, scenario, item_field, cmi). Fails loudly; no silent skips."""
    keys: set = set()
    if not os.path.exists(path):
        return keys
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_no}") from exc
            required = {"data_seed", "target_site", "scenario", item_field, "cmi"}
            missing = required - row.keys()
            if missing:
                raise KeyError(f"{path}:{line_no} missing fields {sorted(missing)}")
            key = (row["data_seed"], row["target_site"], row["scenario"],
                   row[item_field], row["cmi"])
            if key in keys:
                raise ValueError(f"Duplicate result key at {path}:{line_no}: {key}")
            keys.add(key)
    return keys


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
        f.flush(); os.fsync(f.fileno())


def expected_keys(seed: int, site: int, cmi: str, scenarios, items) -> set:
    """Every result key a single source bundle (seed,site,cmi) is responsible for."""
    return {(seed, site, scen, it, cmi) for scen in scenarios for it in items}


# ----------------------------------------------------------------- source bundles
def source_bundle_path(bundle_dir: str, seed: int, site: int, cmi: str) -> str:
    return os.path.join(bundle_dir, f"seed{seed:03d}_site{site}_cmi{cmi}.pt")


def save_source_bundle(path: str, model, cfg, pi_star, source_data_hash: str,
                       commit_sha: str, history) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(schema_version=SCHEMA_VERSION, model_state=model.state_dict(),
                    config=asdict(cfg), config_signature=config_signature(cfg),
                    pi_star=np.asarray(pi_star), source_data_hash=source_data_hash,
                    source_checkpoint_hash=hash_state(model), commit_sha=commit_sha,
                    train_history=history), path)
    return path


def load_source_bundle(path: str, *, build_model, expected_data_hash: str,
                       expected_config_signature: str):
    """Load a bundle and VERIFY data hash, config signature and post-load checkpoint hash.
    Any mismatch aborts (never silently proceeds). Returns (model, bundle_meta)."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    if b.get("source_data_hash") != expected_data_hash:
        raise RuntimeError(f"{path}: source_data_hash mismatch "
                           f"({b.get('source_data_hash')} != {expected_data_hash})")
    if b.get("config_signature") != expected_config_signature:
        raise RuntimeError(f"{path}: config_signature mismatch "
                           f"({b.get('config_signature')} != {expected_config_signature})")
    model = build_model()
    model.load_state_dict(b["model_state"])
    got = hash_state(model)
    if got != b.get("source_checkpoint_hash"):
        raise RuntimeError(f"{path}: checkpoint hash mismatch after load "
                           f"({got} != {b.get('source_checkpoint_hash')})")
    return model, b


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from h2cmi.config import H2Config, core_config
    cfg = core_config(H2Config(n_classes=3))
    print("config_signature:", config_signature(cfg))
    print("variant_id:", variant_id(resp="gen", update="oneshot", prior="fixed",
                                     family="diag", alpha=0.5))
    man = build_manifest(cfg, ["cov", "prior"], "action", {"epochs": 20})
    print("manifest run_signature:", man["run_signature"], "keys:", sorted(man))
    # changing a config field changes the signature
    cfg2 = core_config(H2Config(n_classes=3)); cfg2.tta.em_iters = 999
    assert config_signature(cfg2) != config_signature(cfg), "signature insensitive!"
    print("expected_keys size:", len(expected_keys(0, 1, "off", ["cov", "prior"], ["a", "b"])))
    print("grid_io self-test PASSED")
