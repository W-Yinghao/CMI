"""ACAR v4 — Option B all-DEV substrate REGENERATION command (B1-gated; FAIL-CLOSED at execution).

This is the ONE frozen command that would (post-B1 sign-off) train a NEW all-DEV V4 external representation substrate for a
disease. It is NOT authorized to train here: after a full STDLIB-FIRST preflight it raises SubstrateTrainingNotAuthorizedError
BEFORE importing torch/cmi or reading any DEV/raw signal. The contract (inputs, output artifacts, runtime lock, atomic
no-overwrite) is frozen so B1 approves an exact, reviewable object. See notes/ACAR_V4_SUBSTRATE_REGEN_COMMAND.md.

Usage (NOT yet runnable to completion — fails closed):
    python -m acar.v4.run_regen_substrate --disease PD \
        --dev-input-manifest /abs/acar_v4_regen_pd_inputs.json --output /abs/new_output_dir
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys

from acar.v4 import regen_substrate as RS
from acar.v4 import regen_envlock as EL


def _git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _verify_commit(root, commit):
    """HEAD == protocol_commit (training happens at the to-be-tagged commit; the tag itself comes AFTER, post-replay)."""
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != commit:
        raise ValueError(f"HEAD != protocol_commit ({head.stdout.strip()!r} vs {commit!r})")


def _verify_clean(root):
    st = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if st.returncode != 0 or st.stdout.strip() != "":
        raise ValueError(f"worktree not clean: [{st.stdout.strip()}]")


def run(input_manifest_path, output, *, disease=None):
    """STDLIB-FIRST, FAIL-CLOSED. Validates the regen request fully, then refuses to train (B1 not signed off). NO torch/cmi
    import, NO DEV/raw read, NO output written. Returns the validated plan report (never trains)."""
    with open(input_manifest_path, "rb") as f:
        raw = f.read()
    input_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    spec = json.loads(raw.decode())
    RS.validate_regen_manifest(spec)                                  # schema + exact DEV scope + provenance hashes
    if disease is not None and spec["disease"] != disease:
        raise ValueError(f"--disease {disease!r} != manifest disease {spec['disease']!r}")
    root = _repo_root()
    _verify_commit(root, spec["protocol_commit"])
    if spec.get("repo_clean_required") is True:
        _verify_clean(root)
    if os.path.exists(output):
        raise FileExistsError(f"output dir already exists (no overwrite): {output}")
    report = RS.validate_substrate_request(spec["disease"], spec["dev_cohorts"], output,
                                           seed=spec.get("seed", 0), env_lock_path=spec["env_lock_path"])
    _verify_env_lock(spec)                                            # env-lock file hash + schema + CAPTURED + pins
    report["input_manifest_sha256"] = input_manifest_sha256
    report["command"] = shlex.join([sys.executable, "-m", "acar.v4.run_regen_substrate", "--disease", spec["disease"],
                                    "--dev-input-manifest", input_manifest_path, "--output", output])
    _require_b1_authorization(spec["disease"])                        # B1 GATE — raises (real training not authorized)
    return _train_and_write(spec, output, report)                    # frozen contract; UNREACHABLE until B1


def _verify_env_lock(spec):
    """Preflight the regen runtime lock (no torch import): the env_lock file's sha must equal the manifest's
    env_lock_sha256; the lock must pass the schema validator; be status CAPTURED_AND_VERIFIED (a SCHEMA-ONLY skeleton is
    rejected — real capture on the training node is required); and pin the SAME protocol_commit + pipeline_config_sha256
    as the manifest."""
    got = _sha256_file(spec["env_lock_path"])
    if got != spec["env_lock_sha256"]:
        raise ValueError(f"env_lock_sha256 mismatch ({got} != {spec['env_lock_sha256']})")
    with open(spec["env_lock_path"]) as f:
        lock = json.load(f)
    EL.validate_regen_env_lock(lock)
    if lock["status"] != "CAPTURED_AND_VERIFIED":
        raise ValueError(f"env lock status must be CAPTURED_AND_VERIFIED, got {lock['status']!r} "
                         "(capture the real runtime on the training node first)")
    if lock["protocol_commit"] != spec["protocol_commit"]:
        raise ValueError("env lock protocol_commit != manifest protocol_commit")
    if lock["pipeline_config_sha256"] != spec["pipeline_config_sha256"]:
        raise ValueError("env lock pipeline_config_sha256 != manifest pipeline_config_sha256")


def _require_b1_authorization(disease):
    raise RS.SubstrateTrainingNotAuthorizedError(
        f"{disease}: all-DEV substrate training is NOT authorized. The request validated (manifest + scope + env lock + "
        f"output-absent + clean worktree + HEAD==protocol_commit all pass), but real GPU/EEGNet training requires explicit "
        f"B1 sign-off (notes/ACAR_V4_SUBSTRATE_REGEN_COMMAND.md). No torch/cmi import, no DEV read, no output written.")


def _train_and_write(spec, output, report):                          # pragma: no cover — gated; never reached pre-B1
    """FROZEN contract for the authorized run (reachable ONLY after _require_b1_authorization is relaxed at B1): atomic
    os.mkdir(output) claim, then lazy torch/braindecode/cmi import, train the all-DEV erm:0 EEGNet (seed 0, regen env lock)
    + fit the source-state, torch.save the encoder, write per-artifact provenance JSONs + artifact_sha256, then the run
    manifest LAST (manifest_sha256; RESULT sentinel). On any abort rmtree(output). NOT implemented here (no training)."""
    raise RS.SubstrateTrainingNotAuthorizedError("unreachable: training body is gated behind B1")


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 all-DEV substrate regeneration (B1-gated; fails closed)")
    ap.add_argument("--disease", choices=sorted(RS.DEV_SCOPE), required=True)
    ap.add_argument("--dev-input-manifest", required=True)
    ap.add_argument("--output", required=True, help="must not exist")
    args = ap.parse_args(argv)
    return run(args.dev_input_manifest, args.output, disease=args.disease)


if __name__ == "__main__":
    main()
