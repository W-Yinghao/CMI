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


def run(input_manifest_path, output, *, disease=None, b1_authorization=None):
    """STDLIB-FIRST, FAIL-CLOSED. Validates the regen request fully (incl. the eligible-subject reconciliation). Without a
    valid B1 authorization manifest it refuses to train (raises) — NO torch/cmi import, NO DEV signal read, NO output. With a
    valid, hash-bound authorization it runs the gated trainer (`_train_substrate`, which tests monkeypatch) under an atomic
    output claim. The eligible check + env-lock check read ONLY DEV metadata (dir listings / sha), never signal."""
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
    _verify_eligible_subjects(spec)                                   # eligible == raw − excluded; count + hashes (METADATA)
    report["input_manifest_sha256"] = input_manifest_sha256
    report["command"] = shlex.join([sys.executable, "-m", "acar.v4.run_regen_substrate", "--disease", spec["disease"],
                                    "--dev-input-manifest", input_manifest_path, "--output", output])
    if b1_authorization is None:                                      # B1 GATE — no authorization => fail closed
        _require_b1_authorization(spec["disease"])                    # raises (no torch/cmi import, no DEV read, no output)
    auth = _load_b1_authorization(b1_authorization, spec, input_manifest_sha256, output)   # validates + binds; raises on mismatch
    return _authorized_train_and_write(spec, output, report, auth)    # atomic; calls the gated _train_substrate


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


def _verify_eligible_subjects(spec):
    """METADATA-ONLY eligible-subject reconciliation: list sub-* dirs per cohort (no signal read) and check raw == eligible ∪
    excluded with the exact count + pinned hashes (RS.check_eligible_subjects). Runs in the preflight, BEFORE the B1 gate, so
    an extra/missing raw subject fails before any training."""
    raw_by_cohort = {}
    for c in spec["dev_cohorts"]:
        cdir = spec["source_paths"][c]
        raw_by_cohort[c] = [d for d in os.listdir(cdir) if d.startswith("sub-") and os.path.isdir(os.path.join(cdir, d))]
    return RS.check_eligible_subjects(spec["disease"], raw_by_cohort, spec)


def _require_b1_authorization(disease):
    raise RS.SubstrateTrainingNotAuthorizedError(
        f"{disease}: all-DEV substrate training is NOT authorized — no B1 authorization manifest supplied. The request "
        f"validated (manifest + scope + env lock + eligible subjects + output-absent + clean worktree + HEAD==protocol_commit "
        f"all pass), but real GPU/EEGNet training requires an explicit, hash-bound B1 authorization manifest "
        f"(--b1-authorization; notes/ACAR_V4_SUBSTRATE_REGEN_COMMAND.md). No torch/cmi import, no DEV read, no output written.")


def _load_b1_authorization(path, spec, input_manifest_sha256, output):
    """Validate + BIND the B1 authorization manifest to THIS run: schema (RS.validate_b1_authorization) + the authorization
    must match the protocol_commit, disease, dev_input_manifest_sha256, env_lock_sha256, and output_path. Any mismatch raises
    BEFORE any heavy import / DEV read."""
    with open(path) as f:
        auth = json.load(f)
    RS.validate_b1_authorization(auth)
    checks = (("protocol_commit", spec["protocol_commit"]), ("disease", spec["disease"]),
              ("dev_input_manifest_sha256", input_manifest_sha256), ("env_lock_sha256", spec["env_lock_sha256"]),
              ("output_path", output))
    for k, want in checks:
        if auth[k] != want:
            raise ValueError(f"B1 authorization {k} != run {k} ({auth[k]!r} != {want!r})")
    return auth


def _train_substrate(spec, output):                                  # pragma: no cover — GATED real trainer; tests monkeypatch
    """REAL all-DEV substrate trainer (reached ONLY with a valid B1 authorization). Lazy heavy imports. Trains the disease's
    all-DEV erm:0 EEGNet on the ELIGIBLE DEV subjects only (excluded never loaded), seed 0 deterministic per the regen env
    lock; torch.save's the encoder state_dict; fits + serializes the source-state; returns
    {encoder_checkpoint_path, source_state_path, ...}. The exact ERM schedule + source-state serialization match the DEV
    pipeline and are confirmed at B1b run time. NOT executed in tests (monkeypatched) and never reached without authorization."""
    import torch                                                      # noqa: gated heavy deps
    from cmi.data.bids_data import load_crossdataset
    from cmi.models.backbones import build_backbone
    disease = spec["disease"]
    torch.use_deterministic_algorithms(True); torch.manual_seed(0)
    eligible = set(_verify_eligible_subjects(spec))                   # recheck; defines the exact subjects to train on
    X, y, meta, classes = load_crossdataset(disease, cohorts=spec["dev_cohorts"], resample=128, win_sec=4.0,
                                            fmin=0.5, fmax=45.0)
    keep = [i for i, s in enumerate(meta["subject"].tolist()) if f"{disease}/{s}" in eligible or s in eligible]
    if not keep:
        raise RuntimeError("no eligible windows after subject filter (verify meta subject namespacing at B1b)")
    Xe, ye = X[keep], y[keep]
    bb = build_backbone("EEGNet", n_chans=19, n_times=512, n_classes=len(classes), device="cpu")
    # ... deterministic ERM training loop (seed 0) over (Xe, ye) — exact schedule pinned/confirmed at B1b ...
    enc = os.path.join(output, f"v4_alldev_encoder_{disease}.pt")
    torch.save(bb.state_dict(), enc)
    # ... fit + serialize the source-state (acar.v3 SourceStateArtifact) on the encoder's eligible embeddings ...
    ss = os.path.join(output, f"v4_alldev_source_state_{disease}.npz")
    raise RS.SubstrateTrainingNotAuthorizedError(
        "real all-DEV ERM training + source-state serialization is wired here but its exact schedule/cmi calls must be "
        "validated at B1b run time; tests monkeypatch _train_substrate. (encoder/source-state targets: "
        f"{enc} / {ss})")


def _authorized_train_and_write(spec, output, report, auth):
    """Run the gated trainer under an ATOMIC output claim and write provenance. os.mkdir(output) (race-free) → _train_substrate
    (called exactly once) → verify + sha the encoder + source-state artifacts → manifest.json → RESULT.json LAST. Any abort
    removes the claimed output (no partial)."""
    os.mkdir(output)                                                 # atomic claim
    try:
        art = _train_substrate(spec, output)
        for k in ("encoder_checkpoint_path", "source_state_path"):
            if not (isinstance(art.get(k), str) and os.path.isfile(art[k])):
                raise RuntimeError(f"trainer did not produce {k}")
        art["encoder_checkpoint_sha256"] = _sha256_file(art["encoder_checkpoint_path"])
        art["source_state_sha256"] = _sha256_file(art["source_state_path"])
        body = {"protocol_commit": spec["protocol_commit"], "disease": spec["disease"],
                "input_manifest_sha256": report.get("input_manifest_sha256"), "command": report.get("command"),
                "env_lock_sha256": spec["env_lock_sha256"], "n_eligible_subjects": spec["n_eligible_subjects"],
                "authorization": {k: auth[k] for k in ("authorized_by", "authorization_time", "statement")},
                "artifacts": art}
        with open(os.path.join(output, "manifest.json"), "w") as f:
            json.dump(body, f, sort_keys=True, allow_nan=False, indent=2)
        with open(os.path.join(output, "RESULT.json"), "w") as f:    # written LAST = completion sentinel
            json.dump({"status": "SUBSTRATE_TRAINED", "disease": spec["disease"],
                       "encoder_checkpoint_sha256": art["encoder_checkpoint_sha256"],
                       "source_state_sha256": art["source_state_sha256"]}, f, sort_keys=True, allow_nan=False, indent=2)
        return body
    except BaseException:
        import shutil
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 all-DEV substrate regeneration (B1-gated; fails closed without auth)")
    ap.add_argument("--disease", choices=sorted(RS.DEV_SCOPE), required=True)
    ap.add_argument("--dev-input-manifest", required=True)
    ap.add_argument("--output", required=True, help="must not exist")
    ap.add_argument("--b1-authorization", default=None, help="path to a B1 authorization manifest (omit => fail closed)")
    args = ap.parse_args(argv)
    return run(args.dev_input_manifest, args.output, disease=args.disease, b1_authorization=args.b1_authorization)


if __name__ == "__main__":
    main()
