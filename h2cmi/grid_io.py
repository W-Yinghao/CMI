"""Auditable grid I/O for sharded, provenance-bound experiments (review §"audit infra").

v2 hardens the v1 single-writer manifest/resume for real one-shard-per-task runs:

  * EXPERIMENT vs SHARD identity. The experiment_signature is over the GLOBAL grid (all
    seeds/sites/scenarios/variants + config + commit); each shard records its own
    shard_spec (its seed/site subset). Two shards `--shard-target-sites 0` and `1` of the
    same experiment therefore SHARE an experiment_signature and can be merged (v1 baked the
    per-invocation seeds/sites into the signature, so real shards never merged).
  * SOURCE-TRAINING provenance. A source bundle is keyed and verified by a
    source_training_signature over ONLY source-training-relevant config (encoder/density/
    cmi/train/aux + seed/site/cmi arm), NOT TTA/gate/responsibility params -- so a frozen
    source model is legitimately reused across adaptation methods, while a stale bundle
    from a different source config/commit is rejected. Metadata is a JSON sidecar verified
    BEFORE the tensors are unpickled (weights_only=True).
  * EXACT-KEY completeness on merge (not just a row count), required manifests, and a
    merged manifest written atomically.
  * Refuse to run on an unknown/dirty git state, or to adopt a non-empty legacy output
    that has no manifest. SHA-256 full digests over sorted state_dict keys incl. dtype/
    shape, and a source-data hash that includes domain levels + DAG metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict

import numpy as np
import torch

SCHEMA_VERSION = 2
_REPO = os.path.dirname(__file__) or "."


# ----------------------------------------------------------------- git
def git_state() -> tuple[str, bool]:
    """(full_commit_sha, dirty). sha='unknown' (and dirty=True) outside a repo."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ("unknown", True)
    try:
        dirty = subprocess.call(["git", "diff", "--quiet", "--ignore-submodules", "HEAD"],
                                cwd=_REPO, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL) != 0
    except Exception:
        dirty = True
    return (sha, dirty)


def full_sha() -> str:
    return git_state()[0]


def short_sha() -> str:
    return full_sha()[:12]


def require_clean_git(*, allow_dirty: bool = False) -> str:
    """Return the commit sha, refusing to run on an unknown or (unless allowed) dirty tree."""
    sha, dirty = git_state()
    if sha == "unknown":
        raise RuntimeError("refusing to run: not a git repo / unknown commit")
    if dirty and not allow_dirty:
        raise RuntimeError(f"refusing to run: dirty working tree at {sha[:12]} "
                           "(commit, or pass --allow-dirty for a dev run)")
    return sha


# ----------------------------------------------------------------- hashing (SHA-256)
def _sha256(*chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c if isinstance(c, bytes) else str(c).encode())
    return h.hexdigest()


def hash_array(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return _sha256(str(x.dtype).encode(), str(x.shape).encode(), x.tobytes())


def hash_state(model) -> str:
    """SHA-256 over sorted state_dict (key + dtype + shape + contiguous bytes)."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        a = sd[k].detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def source_data_hash(X: np.ndarray, y: np.ndarray, domains) -> str:
    """Source identity = X + y + domain levels + DAG factor metadata (not just X,y)."""
    dag_meta = [(f.name, int(f.n_levels), tuple(f.parents), f.handling, float(f.budget))
                for f in domains.dag.factors]
    return _sha256(hash_array(X).encode(), hash_array(y).encode(),
                   hash_array(domains.levels).encode(),
                   json.dumps(dag_meta, sort_keys=True).encode())


def config_signature(cfg) -> str:
    return _sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode())


def source_training_signature(cfg, seed: int, site: int, cmi: str) -> str:
    """Hash of ONLY what affects SOURCE training (so a bundle survives TTA/gate changes but
    not encoder/density/cmi/train/aux/data changes)."""
    src = dict(n_classes=cfg.n_classes, encoder=asdict(cfg.encoder), density=asdict(cfg.density),
               cmi=asdict(cfg.cmi), train=asdict(cfg.train), disentangle=asdict(cfg.disentangle),
               ssl=asdict(cfg.ssl), align=asdict(cfg.align))
    return _sha256(json.dumps(src, sort_keys=True, default=str).encode(),
                   f"seed={seed}".encode(), f"site={site}".encode(), f"cmi={cmi}".encode())[:32]


def experiment_signature(cfg, *, global_seeds, global_sites, scenarios, items, item_field,
                         cmi_arms, commit_sha) -> str:
    payload = dict(schema=SCHEMA_VERSION, commit=commit_sha, config_signature=config_signature(cfg),
                   global_seeds=sorted(int(s) for s in global_seeds),
                   global_sites=sorted(int(s) for s in global_sites),
                   scenarios=sorted(scenarios), items=sorted(items), item_field=item_field,
                   cmi_arms=sorted(cmi_arms))
    return _sha256(json.dumps(payload, sort_keys=True, default=str).encode())[:16]


def variant_id(**parts) -> str:
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


def build_manifest(cfg, *, global_seeds, global_sites, scenarios, items, item_field, cmi_arms,
                   shard_spec: dict, cli: dict) -> dict:
    sha, dirty = git_state()
    return dict(
        schema_version=SCHEMA_VERSION, commit_sha=sha, dirty=dirty,
        config_signature=config_signature(cfg),
        experiment_signature=experiment_signature(
            cfg, global_seeds=global_seeds, global_sites=global_sites, scenarios=scenarios,
            items=items, item_field=item_field, cmi_arms=cmi_arms, commit_sha=sha),
        global_seeds=sorted(int(s) for s in global_seeds),
        global_sites=sorted(int(s) for s in global_sites),
        scenarios=sorted(scenarios), items=sorted(items), item_field=item_field,
        cmi_arms=sorted(cmi_arms), shard_spec=shard_spec, cli=cli, config=asdict(cfg))


def validate_or_create_manifest(out_path: str, manifest: dict) -> dict:
    """Match an existing manifest or create one; refuse to adopt a non-empty legacy output
    that has no manifest, and abort on experiment/shard mismatch."""
    mp = manifest_path(out_path)
    out_nonempty = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    if os.path.exists(mp):
        with open(mp) as f:
            old = json.load(f)
        for k in ("schema_version", "commit_sha", "config_signature", "experiment_signature",
                  "item_field", "shard_spec"):
            if old.get(k) != manifest.get(k):
                raise RuntimeError(f"manifest mismatch on '{k}' for {out_path}: "
                                   f"existing={old.get(k)!r} new={manifest.get(k)!r}; use a fresh --out")
        return old
    if out_nonempty:
        raise RuntimeError(f"{out_path} has results but no manifest {mp}; refusing to adopt "
                           "legacy output (use an explicit migration tool that records the original sha256)")
    os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


# ----------------------------------------------------------------- expected keys
def _keys(seeds, sites, scenarios, items, cmi_arms) -> set:
    return {(int(s), int(t), scen, it, cmi) for s in seeds for t in sites
            for scen in scenarios for it in items for cmi in cmi_arms}


def bundle_expected_keys(seed: int, site: int, cmi: str, scenarios, items) -> set:
    """Result keys a single source bundle (seed,site,cmi) is responsible for."""
    return _keys([seed], [site], scenarios, items, [cmi])


def shard_expected_keys(manifest: dict) -> set:
    s = manifest["shard_spec"]
    return _keys(s["seeds"], s["sites"], manifest["scenarios"], manifest["items"], manifest["cmi_arms"])


def global_expected_keys(manifest: dict) -> set:
    return _keys(manifest["global_seeds"], manifest["global_sites"], manifest["scenarios"],
                 manifest["items"], manifest["cmi_arms"])


# ----------------------------------------------------------------- strict resume index
def load_done_keys(path: str, *, item_field: str) -> set:
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
            key = (row["data_seed"], row["target_site"], row["scenario"], row[item_field], row["cmi"])
            if key in keys:
                raise ValueError(f"Duplicate result key at {path}:{line_no}: {key}")
            keys.add(key)
    return keys


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")
        f.flush(); os.fsync(f.fileno())


# ----------------------------------------------------------------- source bundles
def source_bundle_paths(bundle_root: str, training_signature: str, seed: int, site: int, cmi: str):
    base = os.path.join(bundle_root, training_signature, f"seed{seed:03d}_site{site}_cmi{cmi}")
    return base + ".pt", base + ".json"


def _atomic_torch_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp); os.replace(tmp, path)


def _atomic_json(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def save_source_bundle(pt_path: str, json_path: str, model, *, training_signature: str,
                       source_data_hash: str, pi_star, commit_sha: str, history) -> None:
    os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)
    _atomic_torch_save(model.state_dict(), pt_path)          # tensors only -> weights_only-loadable
    meta = dict(schema_version=SCHEMA_VERSION, source_training_signature=training_signature,
                source_data_hash=source_data_hash, source_checkpoint_hash=hash_state(model),
                pi_star=[float(x) for x in np.asarray(pi_star)],
                source_training_commit_sha=commit_sha,
                train_history_tail=(history[-1] if history else None))
    _atomic_json(meta, json_path)


def load_source_bundle(pt_path: str, json_path: str, *, build_model,
                       expected_training_signature: str, expected_source_data_hash: str):
    """Verify the JSON sidecar BEFORE unpickling tensors; then load weights_only and re-check
    the checkpoint hash. Any mismatch aborts."""
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{json_path}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}")
    if meta.get("source_training_signature") != expected_training_signature:
        raise RuntimeError(f"{json_path}: source_training_signature mismatch")
    if meta.get("source_data_hash") != expected_source_data_hash:
        raise RuntimeError(f"{json_path}: source_data_hash mismatch")
    model = build_model()
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if hash_state(model) != meta.get("source_checkpoint_hash"):
        raise RuntimeError(f"{pt_path}: checkpoint hash mismatch after load")
    return model, meta


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from h2cmi.config import H2Config, core_config
    cfg = core_config(H2Config(n_classes=3))
    common = dict(global_seeds=[0, 1, 2], global_sites=[0, 1, 2, 3, 4], scenarios=["cov", "prior"],
                  items=["identity", "joint"], item_field="action", cmi_arms=["off", "on"])
    m0 = build_manifest(cfg, shard_spec={"seeds": [0], "sites": [1]}, cli={}, **common)
    m1 = build_manifest(cfg, shard_spec={"seeds": [0], "sites": [2]}, cli={}, **common)
    assert m0["experiment_signature"] == m1["experiment_signature"], "shards must share experiment sig"
    assert m0["shard_spec"] != m1["shard_spec"]
    c2 = core_config(H2Config(n_classes=3)); c2.tta.em_iters = 999
    assert source_training_signature(cfg, 0, 0, "off") == source_training_signature(c2, 0, 0, "off"), \
        "TTA change must not invalidate source bundle"
    c3 = core_config(H2Config(n_classes=3)); c3.encoder.z_c_dim = 99
    assert source_training_signature(cfg, 0, 0, "off") != source_training_signature(c3, 0, 0, "off")
    print("git_state:", git_state()[0][:12], "dirty:", git_state()[1])
    print("global expected keys:", len(global_expected_keys(m0)))
    print("grid_io v2 self-test PASSED")
