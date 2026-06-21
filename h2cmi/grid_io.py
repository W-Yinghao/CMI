"""Auditable grid I/O for sharded, provenance-bound experiments (review §"audit infra").

v3 closes three auditability gaps the v2 infra left open:

  * DATA-PROTOCOL identity. The experiment_signature now includes a normalised
    `data_spec` (simulator + n_sites/subjects/sessions/trials + classes/chans/times +
    source_scenario + train-seed policy + difficulty), and `data_spec` is stored in the
    manifest and folded into the source-bundle identity. Two runs that differ only in
    `--subjects/--sessions/--trials/...` therefore get DIFFERENT experiment signatures and
    cannot silently resume / skip-train across each other (v2 hashed only the grid + config).
  * SOURCE-CODE provenance. A `source_code_signature` hashes ONLY the code that can change
    the source model (config.py + train/models/density/cmi/domains/align/disentangle/ssl).
    It is folded into the bundle identity and *verified* on load, so a frozen source model
    is reused across adaptation/TTA edits but a source-code edit invalidates the bundle.
    `pi_star` recorded in the sidecar is also re-checked on load. (v2 stored a commit sha
    but never enforced any code fingerprint -- a bundle saved with a bogus sha still loaded.)
  * RESULT-ROW binding. Every result row carries schema_version/experiment_signature/
    config_signature/runner_commit_sha; `load_done_keys` and `merge` REJECT any row that
    does not belong to this experiment, require each shard's rows to equal its shard_spec
    exactly, and the merged manifest records the SHA-256 of every input shard + manifest and
    of the merged output. (v2 trusted the bare (seed,site,scenario,item,cmi) key, so a row
    from a *different* experiment with the same key could be counted done / merged in.)

Plus: the clean-git guard now also detects UNTRACKED files (porcelain v1, untracked=all) --
an untracked `h2cmi/*.py` can change import behaviour -- while letting a run exclude its own
declared output dir + bundle root so resume over untracked result artifacts still works.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict

import numpy as np
import torch

SCHEMA_VERSION = 3
_REPO = os.path.dirname(__file__) or "."           # the h2cmi package dir

# source-model-affecting code (NOT tta/gate/label/eval): a change here must invalidate a bundle
_SOURCE_CODE_PATHS = ("config.py", "train", "models", "density", "cmi", "domains",
                      "align", "disentangle", "ssl")


# ----------------------------------------------------------------- git
def _repo_root() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _norm_prefixes(prefixes) -> list[str]:
    """Make output-path prefixes repo-root-relative so they can be matched against the
    repo-root-relative paths that `git status --porcelain` prints."""
    root = _repo_root()
    out = []
    for p in prefixes:
        if not p:
            continue
        ap = os.path.abspath(p)
        rel = os.path.relpath(ap, root) if (root and (ap == root or ap.startswith(root + os.sep))) else p
        out.append(rel.rstrip("/"))
    return out


def _porcelain_dirty(out: str, norm_prefixes) -> bool:
    """True if any `git status --porcelain` line denotes a change outside the given (already
    repo-root-relative) output prefixes. Catches untracked files (`?? ...`), not just tracked."""
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()                              # strip "XY " status prefix
        if " -> " in path:                                   # rename: take the destination
            path = path.split(" -> ", 1)[1]
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if any(path == pre or path.startswith(pre + "/") for pre in norm_prefixes):
            continue                                         # under a declared output prefix
        return True
    return False


def git_state(*, ignore_prefixes=()) -> tuple[str, bool]:
    """(full_commit_sha, dirty). Dirty = any tracked change OR any untracked file (porcelain
    v1, --untracked-files=all), EXCEPT entries under an explicitly-declared output prefix.
    sha='unknown' (dirty=True) outside a repo."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ("unknown", True)
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules"],
            cwd=_REPO, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return (sha, True)
    return (sha, _porcelain_dirty(out, _norm_prefixes(ignore_prefixes)))


def full_sha() -> str:
    return git_state()[0]


def short_sha() -> str:
    return full_sha()[:12]


def require_clean_git(*, allow_dirty: bool = False, ignore_prefixes=()) -> str:
    """Return the commit sha, refusing to run on an unknown or (unless allowed) dirty tree.
    `ignore_prefixes` (e.g. the run's output dir + bundle root) are excluded from dirtiness."""
    sha, dirty = git_state(ignore_prefixes=ignore_prefixes)
    if sha == "unknown":
        raise RuntimeError("refusing to run: not a git repo / unknown commit")
    if dirty and not allow_dirty:
        raise RuntimeError(f"refusing to run: dirty working tree at {sha[:12]} "
                           "(commit code, or pass --allow-dirty for a dev run)")
    return sha


# ----------------------------------------------------------------- hashing (SHA-256)
def _sha256(*chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c if isinstance(c, bytes) else str(c).encode())
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
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


def source_code_signature(paths=_SOURCE_CODE_PATHS) -> str:
    """SHA-256 over the *.py under the source-model-affecting modules (sorted by repo-relative
    path, each entry = relpath + file digest). TTA/gate/label/eval/run_* code is excluded, so a
    frozen source model survives adaptation-side edits but not a change to how it is trained."""
    h = hashlib.sha256()
    for rel in paths:
        p = os.path.join(_REPO, rel)
        files = []
        if os.path.isdir(p):
            for root, _dirs, fnames in os.walk(p):
                if "__pycache__" in root:
                    continue
                for fn in fnames:
                    if fn.endswith(".py"):
                        files.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            files.append(p)
        for fpath in sorted(files, key=lambda q: os.path.relpath(q, _REPO)):
            relpath = os.path.relpath(fpath, _REPO)
            h.update(relpath.encode()); h.update(b"\0")
            h.update(sha256_file(fpath).encode()); h.update(b"\0")
    return h.hexdigest()[:32]


def config_signature(cfg) -> str:
    return _sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode())


# ----------------------------------------------------------------- data protocol
def build_data_spec(*, simulator: str, n_sites: int, subjects_per_site: int,
                    sessions_per_subject: int, trials_per_session: int, n_classes: int,
                    n_chans: int, n_times: int, source_scenario: str, train_seed_policy: str,
                    difficulty: str = "standard") -> dict:
    """Normalised description of the data-generating protocol; folded into the experiment
    signature + bundle identity so it cannot silently differ across resumes/shards."""
    return dict(simulator=simulator, n_sites=int(n_sites), subjects_per_site=int(subjects_per_site),
                sessions_per_subject=int(sessions_per_subject),
                trials_per_session=int(trials_per_session), n_classes=int(n_classes),
                n_chans=int(n_chans), n_times=int(n_times), source_scenario=source_scenario,
                train_seed_policy=train_seed_policy, difficulty=difficulty)


def _spec_blob(data_spec) -> bytes:
    return json.dumps(data_spec or {}, sort_keys=True, default=str).encode()


def source_training_signature(cfg, seed: int, site: int, cmi: str, *,
                              source_code_signature: str | None = None,
                              data_spec: dict | None = None) -> str:
    """Bundle identity over what affects the SOURCE model: source config + source-code
    fingerprint + data protocol + seed/site/cmi arm. NOT tta/gate/responsibility -- so a
    frozen source model is reused across adaptation methods, but a source config/code/data
    change yields a new bundle."""
    src = dict(n_classes=cfg.n_classes, encoder=asdict(cfg.encoder), density=asdict(cfg.density),
               cmi=asdict(cfg.cmi), train=asdict(cfg.train), disentangle=asdict(cfg.disentangle),
               ssl=asdict(cfg.ssl), align=asdict(cfg.align))
    return _sha256(json.dumps(src, sort_keys=True, default=str).encode(),
                   f"code={source_code_signature}".encode(), _spec_blob(data_spec),
                   f"seed={seed}".encode(), f"site={site}".encode(), f"cmi={cmi}".encode())[:32]


def experiment_signature(cfg, *, global_seeds, global_sites, scenarios, items, item_field,
                         cmi_arms, commit_sha, data_spec) -> str:
    payload = dict(schema=SCHEMA_VERSION, commit=commit_sha, config_signature=config_signature(cfg),
                   global_seeds=sorted(int(s) for s in global_seeds),
                   global_sites=sorted(int(s) for s in global_sites),
                   scenarios=sorted(scenarios), items=sorted(items), item_field=item_field,
                   cmi_arms=sorted(cmi_arms), data_spec=data_spec)
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
                   shard_spec: dict, cli: dict, data_spec: dict) -> dict:
    sha, dirty = git_state()
    return dict(
        schema_version=SCHEMA_VERSION, commit_sha=sha, dirty=dirty,
        config_signature=config_signature(cfg), data_spec=data_spec,
        experiment_signature=experiment_signature(
            cfg, global_seeds=global_seeds, global_sites=global_sites, scenarios=scenarios,
            items=items, item_field=item_field, cmi_arms=cmi_arms, commit_sha=sha,
            data_spec=data_spec),
        global_seeds=sorted(int(s) for s in global_seeds),
        global_sites=sorted(int(s) for s in global_sites),
        scenarios=sorted(scenarios), items=sorted(items), item_field=item_field,
        cmi_arms=sorted(cmi_arms), shard_spec=shard_spec, cli=cli, config=asdict(cfg))


def validate_or_create_manifest(out_path: str, manifest: dict) -> dict:
    """Match an existing manifest or create one; refuse to adopt a non-empty legacy output
    that has no manifest, and abort on experiment/shard/data-protocol mismatch."""
    mp = manifest_path(out_path)
    out_nonempty = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    if os.path.exists(mp):
        with open(mp) as f:
            old = json.load(f)
        for k in ("schema_version", "commit_sha", "config_signature", "experiment_signature",
                  "item_field", "shard_spec", "data_spec"):
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


# ----------------------------------------------------------------- result-row provenance
_ROW_KEY_FIELDS = ("data_seed", "target_site", "scenario", "cmi")    # + item_field


def row_key(row: dict, item_field: str) -> tuple:
    return (row["data_seed"], row["target_site"], row["scenario"], row[item_field], row["cmi"])


def validate_result_row(row: dict, manifest: dict, *, item_field: str | None = None,
                        line_ref: str = "") -> tuple:
    """Verify a result row was produced by THIS experiment (schema + experiment_signature +
    config_signature + runner commit), and return its key. Raises on any foreign/stale row."""
    item_field = item_field or manifest["item_field"]
    where = f" at {line_ref}" if line_ref else ""
    missing = ({*_ROW_KEY_FIELDS, item_field, "experiment_signature"} - row.keys())
    if missing:
        raise KeyError(f"result row{where} missing fields {sorted(missing)}")
    checks = (("schema_version", manifest.get("schema_version")),
              ("experiment_signature", manifest.get("experiment_signature")),
              ("config_signature", manifest.get("config_signature")),
              ("runner_commit_sha", manifest.get("commit_sha")))
    for field, expected in checks:
        if field in row and row.get(field) != expected:
            raise ValueError(f"foreign result row{where}: {field}={row.get(field)!r} "
                             f"!= experiment {expected!r}")
    if row.get("experiment_signature") != manifest.get("experiment_signature"):
        raise ValueError(f"foreign result row{where}: experiment_signature "
                         f"{row.get('experiment_signature')!r} != {manifest.get('experiment_signature')!r}")
    return row_key(row, item_field)


# ----------------------------------------------------------------- strict resume index
def load_done_keys(path: str, *, item_field: str, manifest: dict | None = None) -> set:
    """Done keys in `path`. With a manifest, every row is provenance-checked (foreign-experiment
    rows abort) and must fall inside the shard_spec; without one, only structural checks run."""
    keys: set = set()
    if not os.path.exists(path):
        return keys
    allowed = shard_expected_keys(manifest) if manifest is not None else None
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_no}") from exc
            if manifest is not None:
                key = validate_result_row(row, manifest, item_field=item_field,
                                          line_ref=f"{path}:{line_no}")
                if key not in allowed:
                    raise ValueError(f"{path}:{line_no}: row {key} is outside this shard_spec")
            else:
                required = {*_ROW_KEY_FIELDS, item_field}
                miss = required - row.keys()
                if miss:
                    raise KeyError(f"{path}:{line_no} missing fields {sorted(miss)}")
                key = row_key(row, item_field)
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
                       source_data_hash: str, source_code_signature: str, pi_star,
                       commit_sha: str, history) -> None:
    os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)
    _atomic_torch_save(model.state_dict(), pt_path)          # tensors only -> weights_only-loadable
    meta = dict(schema_version=SCHEMA_VERSION, source_training_signature=training_signature,
                source_data_hash=source_data_hash, source_code_signature=source_code_signature,
                source_checkpoint_hash=hash_state(model),
                pi_star=[float(x) for x in np.asarray(pi_star)],
                source_training_commit_sha=commit_sha,
                train_history_tail=(history[-1] if history else None))
    _atomic_json(meta, json_path)


def load_source_bundle(pt_path: str, json_path: str, *, build_model,
                       expected_training_signature: str, expected_source_data_hash: str,
                       expected_source_code_signature: str | None = None, expected_pi_star=None):
    """Verify the JSON sidecar BEFORE unpickling tensors (training signature, data hash, and --
    when given -- source-code signature + pi_star), then load weights_only and re-check the
    checkpoint hash. Any mismatch aborts."""
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{json_path}: schema_version {meta.get('schema_version')} != {SCHEMA_VERSION}")
    if meta.get("source_training_signature") != expected_training_signature:
        raise RuntimeError(f"{json_path}: source_training_signature mismatch")
    if meta.get("source_data_hash") != expected_source_data_hash:
        raise RuntimeError(f"{json_path}: source_data_hash mismatch")
    if (expected_source_code_signature is not None
            and meta.get("source_code_signature") != expected_source_code_signature):
        raise RuntimeError(f"{json_path}: source_code_signature mismatch "
                           f"(source-model code changed since this bundle was trained)")
    if expected_pi_star is not None:
        got = np.asarray(meta.get("pi_star", []), dtype=float)
        exp = np.asarray(expected_pi_star, dtype=float)
        if got.shape != exp.shape or not np.allclose(got, exp, atol=1e-8):
            raise RuntimeError(f"{json_path}: pi_star mismatch (stored {got.tolist()} != {exp.tolist()})")
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
    dspec = build_data_spec(simulator="PairedEEGSimulator", n_sites=5, subjects_per_site=3,
                            sessions_per_subject=2, trials_per_session=40, n_classes=3, n_chans=16,
                            n_times=128, source_scenario="population_null",
                            train_seed_policy="train_seed=data_seed")
    common = dict(global_seeds=[0, 1, 2], global_sites=[0, 1, 2, 3, 4], scenarios=["cov", "prior"],
                  items=["identity", "joint"], item_field="action", cmi_arms=["off", "on"],
                  data_spec=dspec)
    m0 = build_manifest(cfg, shard_spec={"seeds": [0], "sites": [1]}, cli={}, **common)
    m1 = build_manifest(cfg, shard_spec={"seeds": [0], "sites": [2]}, cli={}, **common)
    assert m0["experiment_signature"] == m1["experiment_signature"], "shards must share experiment sig"
    assert m0["shard_spec"] != m1["shard_spec"]
    d2 = dict(dspec, trials_per_session=80)
    m2 = build_manifest(cfg, shard_spec={"seeds": [0], "sites": [1]}, cli={},
                        **dict(common, data_spec=d2))
    assert m2["experiment_signature"] != m0["experiment_signature"], "data_spec must bind experiment"
    code = source_code_signature()
    c2 = core_config(H2Config(n_classes=3)); c2.tta.em_iters = 999
    assert source_training_signature(cfg, 0, 0, "off", source_code_signature=code, data_spec=dspec) == \
        source_training_signature(c2, 0, 0, "off", source_code_signature=code, data_spec=dspec), \
        "TTA change must not invalidate source bundle"
    c3 = core_config(H2Config(n_classes=3)); c3.encoder.z_c_dim = 99
    assert source_training_signature(cfg, 0, 0, "off", source_code_signature=code, data_spec=dspec) != \
        source_training_signature(c3, 0, 0, "off", source_code_signature=code, data_spec=dspec)
    print("git_state:", git_state()[0][:12], "dirty:", git_state()[1], "code_sig:", code)
    print("global expected keys:", len(global_expected_keys(m0)))
    print("grid_io v3 self-test PASSED")
