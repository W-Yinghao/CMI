"""ACAR v3 binding DEV runner — the SINGLE frozen command from seven concrete cohort dumps to the binding S2/S4 gate.

    python -m acar.v3.run_dev_binding --input-manifest <seven-cohort.json> --output <new-dir> [--protocol-commit <sha>]

STDLIB-ONLY BOOTSTRAP — the preflight runs before any heavy import or DEV file read, in this exact order:
    parse manifest (stdlib) → output dir absent → validate manifest schema → verify repo: HEAD == protocol commit,
    tag `acar-v3-dev-design-v1^{}` → HEAD, CLEAN worktree (`git status --porcelain` empty) → verify each input file
    exists with declared `full_dump_sha256` (stdlib hashlib) → set single-thread runtime env → import numpy/torch/
    sklearn + v3 modules → apply+verify the environment lock → build a VerifiedBindingContext → open cohort files
    (`build_cohort_input`) and check the four remaining field hashes → `freeze_dev_run`.
Only then does the first real DEV run produce a SELECT (+frozen artifacts) or `DEV_STOP / NO_LOCKBOX_CONSUMED`. No
external Arm-B endpoint or lockbox is touched. There is NO verification-bypass flag.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys

TAG = "acar-v3-dev-design-v1"
_HEXSET = set("0123456789abcdef")
_REQUIRED_HASHES = ("full_dump_sha256", "source_fit_sha256", "deployment_input_sha256", "label_sha256",
                    "subject_list_sha256")


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(root, args):
    return subprocess.check_output(["git", "-C", root] + args, stderr=subprocess.DEVNULL).decode().strip()


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in _HEXSET for c in s)


def verify_protocol(root, protocol_commit, *, require_tag=True):
    """HEAD == the spec's protocol commit AND the immutable tag resolves to that HEAD."""
    if not _is_hex(protocol_commit, 40):
        raise ValueError("protocol_commit must be a full 40-char git SHA-1")
    head = _git(root, ["rev-parse", "HEAD"])
    if head != protocol_commit:
        raise ValueError(f"git HEAD {head} != protocol_commit {protocol_commit}")
    if require_tag:
        try:
            tagged = _git(root, ["rev-list", "-n", "1", TAG])
        except subprocess.CalledProcessError:
            raise ValueError(f"tag {TAG} not found")
        if tagged != head:
            raise ValueError(f"tag {TAG} -> {tagged} != HEAD {head}")
    return head


def verify_clean_worktree(root):
    """The worktree must be byte-clean: no modified tracked files, no untracked files."""
    out = _git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    if out.strip():
        raise ValueError(f"worktree is not clean:\n{out}")


def validate_input_manifest(spec):
    """Exact required schema: protocol_commit (40-hex), exactly the seven config.DISEASE cohorts, each with a unique
    dataset_id + path + disease and all five field-separated SHA-256 (lowercase 64-hex). FAIL-CLOSED (no optional hash)."""
    from acar.config import DISEASE                              # config is stdlib-light (no numpy/torch)
    if not _is_hex(spec.get("protocol_commit", ""), 40):
        raise ValueError("input manifest: protocol_commit must be a 40-hex git SHA-1")
    cohorts = spec.get("cohorts")
    if not isinstance(cohorts, list):
        raise ValueError("input manifest: cohorts must be a list")
    by_d = {}
    seen_ds, seen_path = set(), set()
    for c in cohorts:
        for k in ("dataset_id", "disease", "path", *_REQUIRED_HASHES):
            if k not in c:
                raise ValueError(f"input manifest cohort missing required field {k!r}")
        if c["disease"] not in ("PD", "SCZ"):
            raise ValueError("cohort disease must be PD or SCZ")
        for hk in _REQUIRED_HASHES:
            if not _is_hex(c[hk], 64):
                raise ValueError(f"{c['dataset_id']}: {hk} must be lowercase 64-hex")
        if c["dataset_id"] in seen_ds or c["path"] in seen_path:
            raise ValueError("duplicate cohort dataset_id/path")
        seen_ds.add(c["dataset_id"]); seen_path.add(c["path"])
        by_d.setdefault(c["disease"], []).append(c["dataset_id"])
    for d in ("PD", "SCZ"):
        if sorted(by_d.get(d, [])) != sorted(DISEASE[d]):
            raise ValueError(f"{d} cohorts {sorted(by_d.get(d, []))} != frozen {sorted(DISEASE[d])}")


def input_manifest_sha256(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(input_manifest_path, output, *, protocol_commit=None, repo_root=None):
    """Full stdlib-first preflight, then the binding DEV gate. No bypass flags."""
    root = repo_root or _repo_root()
    with open(input_manifest_path) as f:
        spec = json.load(f)
    if os.path.exists(output):                                  # (1) output absent — before anything else
        raise FileExistsError(f"output dir already exists: {output}")
    validate_input_manifest(spec)                              # (2) exact schema
    commit = protocol_commit or spec["protocol_commit"]
    verify_protocol(root, commit)                             # (3) HEAD == commit + tag -> HEAD
    verify_clean_worktree(root)                               # (4) clean worktree
    for c in spec["cohorts"]:                                 # (5) files exist + declared full-dump hash (stdlib)
        if not os.path.exists(c["path"]):
            raise FileNotFoundError(f"missing cohort dump: {c['path']}")
        if _file_sha256(c["path"]) != c["full_dump_sha256"]:
            raise ValueError(f"{c['dataset_id']}: file SHA-256 != declared full_dump_sha256")
    im_sha = input_manifest_sha256(spec)
    os.environ["OMP_NUM_THREADS"] = "1"                       # (6) set runtime BEFORE importing heavy libs
    from .envlock import apply_runtime, verify_env_lock       # (7) import heavy + apply/verify env lock
    apply_runtime(); env_sha = verify_env_lock()
    from .develop import BindingContext, freeze_dev_run
    from .loader import build_cohort_input
    cmd = "python -m acar.v3.run_dev_binding --input-manifest %s --output %s" % (
        os.path.abspath(input_manifest_path), os.path.abspath(output))
    ctx = BindingContext(commit, TAG, True, env_sha, im_sha, cmd, root)
    inputs = []
    for c in spec["cohorts"]:                                 # (8) open cohort files + check the 4 derived hashes
        ci = build_cohort_input(c["path"], disease=c["disease"], dataset_id=c["dataset_id"],
                                raw_pipeline_sha256=c.get("raw_pipeline_sha256"),
                                dataset_version=c.get("dataset_version"))
        m = ci.manifest
        for hk in ("source_fit_sha256", "deployment_input_sha256", "label_sha256", "subject_list_sha256"):
            if getattr(m, hk) != c[hk]:
                raise ValueError(f"{c['dataset_id']}: declared {hk} != computed")
        inputs.append(ci)
    return freeze_dev_run(ctx, inputs, output)


def main(argv=None):
    p = argparse.ArgumentParser(prog="acar.v3.run_dev_binding")
    p.add_argument("--input-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--protocol-commit", default=None)
    args = p.parse_args(argv)
    res, manifest = run(args.input_manifest, args.output, protocol_commit=args.protocol_commit)
    print(f"verdict={manifest['verdict']} selected={manifest.get('selected')} output={args.output}")
    return 0 if manifest["verdict"] in ("SELECT", "DEV_STOP") else 1


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
