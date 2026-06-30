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
    if lock["device_kind"] != "cuda":                                # B1b trains on GPU — reject a CPU lock at the cheap preflight
        raise ValueError(f"env lock device_kind must be 'cuda' for B1b training, got {lock['device_kind']!r} "
                         "(capture the lock on the GPU training node)")
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


def _load_subject_signal(disease, cohort, subject, cohort_dir, pipeline_config):   # pragma: no cover — gated raw read; tests monkeypatch
    """Open ONE eligible subject's raw EEG and return (windows[n,19,512] float, label int) via the SHARED, tested cmi
    pipeline (`cmi.data.bids_data.load_cohort` with a single-subject allowlist — montage→19ch / resample 128 / bandpass
    0.5–45 / 4 s windows / trial z-score). The allowlist=subjects={subject} makes load_cohort skip every other subject at
    file-discovery, so no other subject's signal is opened. Called ONLY for allowlisted (eligible) subjects."""
    from cmi.data.bids_data import load_cohort, COHORTS
    cs = COHORTS[cohort]
    res = load_cohort(cohort_dir, cs["task"], cs["label"],
                      fmin=pipeline_config["bandpass"][0], fmax=pipeline_config["bandpass"][1],
                      resample=pipeline_config["resample_fs"], win_sec=pipeline_config["window_sec"],
                      subjects={subject})                            # discovery-stage filter: excluded never opened
    if res is None:
        raise RuntimeError(f"{cohort}/{subject}: no usable windows (eligible subject expected to have data)")
    Xc, yc, _subs = res
    RS.assert_finite(Xc, f"{cohort}/{subject} windows")             # no NaN/Inf in this subject's signal
    label = RS.single_subject_label(yc, f"{cohort}/{subject}")      # non-empty, all identical, in {0,1}
    return Xc, label


def load_eligible_windows(spec, allowlist, *, signal_loader=None):
    """Load canonical windows for ONLY the eligible subjects. Iterates the ALLOWLIST (cohort-aware 'dsid/sub' ids) and calls
    `signal_loader(disease, cohort, local_subject, cohort_dir, pipeline_config)` per eligible subject — so EXCLUDED subjects
    are NEVER passed to the loader (filtered BEFORE any signal open). Cohort-aware: the SAME local sub-id in two cohorts is
    distinct. Returns (X[N,19,512], y[N], subject_keys[N] cohort-aware). signal_loader defaults to the (gated, real-cmi)
    `_load_subject_signal`; tests inject a fake."""
    import numpy as np
    loader = signal_loader or _load_subject_signal
    cfg = RS.FROZEN_PIPELINE
    cohort_dirs = spec.get("source_paths", {})
    Xs, ys, subj = [], [], []
    for c in spec["dev_cohorts"]:
        for ns in sorted(s for s in allowlist if s.startswith(c + "/")):
            local = ns.split("/", 1)[1]
            w, label = loader(spec["disease"], c, local, cohort_dirs.get(c), cfg)   # excluded never reach here
            w = np.asarray(w, dtype="<f4")
            if w.ndim != 3 or w.shape[1:] != (cfg["canon_channels"], int(cfg["resample_fs"] * cfg["window_sec"])):
                raise ValueError(f"{ns}: windows must be [n,{cfg['canon_channels']},512], got {w.shape}")
            if len(w) == 0:
                raise ValueError(f"{ns}: eligible subject returned 0 windows")
            Xs.append(w); ys += [int(label)] * len(w); subj += [ns] * len(w)
    if not Xs:
        raise RuntimeError("no eligible windows loaded")
    X = np.concatenate(Xs, 0)
    RS.assert_finite(X, "windows")                                  # no NaN/Inf across the whole training set
    return X, np.asarray(ys, dtype=np.int64), subj


def _assert_model_params_finite(bb, what):                          # pragma: no cover — gated torch; tests use RS.assert_finite
    """Every parameter (and grad, if present) of the model must be finite — else abort (no NaN/Inf encoder written)."""
    import torch
    for nm, p in bb.named_parameters():
        if not torch.isfinite(p).all():
            raise ValueError(f"non-finite {what} parameter: {nm}")
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise ValueError(f"non-finite gradient: {nm}")


def _train_encoder_and_save(X, y, enc_path):                         # pragma: no cover — gated torch ERM; tests monkeypatch _train_substrate
    """Deterministic ERM training of the erm:0 EEGNet per RS.TRAINING_SCHEDULE (seed 0, fixed epochs, Adam, CE,
    class-balanced); torch.save the encoder state_dict to enc_path. The backbone returns (logits, z); we train on logits.
    HARD-fails on no CUDA (no silent CPU fallback) and on any non-finite logits/loss/grad/parameter (no NaN encoder written).
    Lazy torch + cmi backbone. Returns (model, device, canonical_state_dict_sha256)."""
    import torch
    import numpy as np
    from cmi.models.backbones import build_backbone
    s = RS.TRAINING_SCHEDULE
    dev = RS.require_cuda(s, torch.cuda.is_available())              # 'cuda' or raise — NEVER CPU fallback
    torch.use_deterministic_algorithms(True); torch.manual_seed(s["seed"]); np.random.seed(s["seed"])
    bb = build_backbone(s["model"], n_chans=s["n_chans"], n_times=s["n_times"], n_classes=s["n_classes"], device=dev)
    Xt = torch.as_tensor(np.asarray(X, dtype="<f4")).to(dev); yt = torch.as_tensor(np.asarray(y)).long().to(dev)
    cw = None
    if s["class_weighting"] == "balanced":
        cnt = torch.bincount(yt, minlength=s["n_classes"]).float().clamp(min=1.0)
        cw = (cnt.sum() / (s["n_classes"] * cnt)).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(bb.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    bb.train()
    n, bs = Xt.shape[0], s["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(s["seed"])
    for _ep in range(s["max_epochs"]):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); logits, _z = bb(Xt[idx]); loss = lossf(logits, yt[idx])
            RS.assert_finite(logits.detach().cpu().numpy(), "logits"); RS.assert_finite(float(loss.detach().cpu()), "loss")
            loss.backward(); _assert_model_params_finite(bb, "post-backward"); opt.step()
            _assert_model_params_finite(bb, "post-step")
    bb.eval()
    sd = bb.state_dict()
    state_sha = RS.canonical_state_dict_sha256({k: v.detach().cpu().numpy() for k, v in sd.items()})
    torch.save(sd, enc_path)
    return bb, dev, state_sha


def _embed(bb, X, dev):                                              # pragma: no cover — gated torch forward; tests monkeypatch _train_substrate
    """Encoder embeddings z (the readout input) for X via the trained backbone (forward -> (logits, z)), batched, no grad."""
    import torch
    import numpy as np
    s = RS.TRAINING_SCHEDULE
    Xt = torch.as_tensor(np.asarray(X, dtype="<f4")).to(dev)
    zs = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], s["batch_size"]):
            _logits, z = bb(Xt[i:i + s["batch_size"]])
            zs.append(z.detach().cpu().numpy())
    Z = np.concatenate(zs, 0)
    RS.assert_finite(Z, "embeddings")                               # never fit the source-state on NaN/Inf embeddings
    return Z


def _fit_and_serialize_source_state(bb, dev, X, y, subj, spec, ss_path):   # pragma: no cover — gated; tests monkeypatch _train_substrate
    """Embed the eligible windows with the trained encoder and fit + serialize the all-DEV source-state f_0 to ss_path (the
    DEV-frozen readout the external adapter consumes). Delegated to the FIXED acar.v3 fitter: fit_source_state_artifact (the
    only DEV f_0 fitter) → freeze_source_state_artifact (no pickle / no z_ev,y_ev) → np.savez. Lazy import."""
    import numpy as np
    from acar.v3.loader import (fit_source_state_artifact, freeze_source_state_artifact, hash_source_fit, env_versions)
    zev = _embed(bb, X, dev)
    yev = np.asarray(y, dtype=np.int64)
    art = fit_source_state_artifact(zev, yev, spec["disease"], hash_source_fit(zev, yev), env_versions())
    blob = freeze_source_state_artifact(art)
    np.savez(ss_path, **blob)
    return art


def _train_substrate(spec, output):                                  # pragma: no cover — gated real trainer; tests monkeypatch
    """REAL all-DEV substrate trainer (reached ONLY with a valid, bound B1 authorization). Orchestration (NOT a placeholder):
    allowlist = eligible subjects (excluded NEVER loaded) → load_eligible_windows (per-eligible-subject raw open) → ERM-train
    the erm:0 EEGNet per RS.TRAINING_SCHEDULE → torch.save encoder → fit + serialize source-state. Returns
    {encoder_checkpoint_path, source_state_path, training_schedule, ...}. The gated inner raw I/O + source-state fitter are
    confirmed at B1b run; tests monkeypatch this whole function (and load_eligible_windows / _load_subject_signal separately)."""
    disease = spec["disease"]
    allowlist = set(_verify_eligible_subjects(spec))                  # eligible cohort-aware ids; excluded NOT included
    X, y, subj = load_eligible_windows(spec, allowlist)              # excluded subjects never opened
    RS.check_training_set(y, subj, allowlist)                        # every eligible subject present; labels {0,1}; both classes
    enc_path = os.path.join(output, f"v4_alldev_encoder_{disease}.pt")
    bb, dev, encoder_state_dict_sha256 = _train_encoder_and_save(X, y, enc_path)
    ss_path = os.path.join(output, f"v4_alldev_source_state_{disease}.npz")
    ss_art = _fit_and_serialize_source_state(bb, dev, X, y, subj, spec, ss_path)
    if int(ss_art.embedding_dim) != RS.FROZEN_PIPELINE["embedding_dim"]:   # backbone width must equal the pinned 16
        raise ValueError(f"trained embedding_dim {ss_art.embedding_dim} != frozen {RS.FROZEN_PIPELINE['embedding_dim']}")
    return {"encoder_checkpoint_path": enc_path, "encoder_state_dict_sha256": encoder_state_dict_sha256,
            "source_state_path": ss_path, "source_state_sha256": ss_art.source_state_sha256,
            "embedding_dim": int(ss_art.embedding_dim), "training_schedule": RS.TRAINING_SCHEDULE,
            "n_train_windows": int(len(X)), "n_eligible_subjects": len(allowlist)}


def _capture_current_runtime():                                     # pragma: no cover — gated (imports torch); tests monkeypatch
    """Snapshot the CURRENT training process's runtime via capture_regen_envlock._probe (SAME version-string methods as the
    captured lock, and it pins threads to 1). Returns the runtime info dict; raises if the training stack fails to import."""
    from acar.v4 import capture_regen_envlock as CAP
    info, stack_ok, note = CAP._probe()
    if not stack_ok:
        raise RuntimeError(f"training stack import failed at runtime: {note}")
    return info


def _verify_runtime_matches_lock(spec):                             # pragma: no cover — gated; tests monkeypatch / RS.check_*
    """The CURRENT runtime must match the captured env lock BEFORE any training. Reads the lock file (already file-hash-verified
    in the stdlib preflight), snapshots this process, and compares (RS.check_runtime_matches_lock): device_kind=cuda is HARD,
    the 3 thread fields ==1, and all library versions + the cuda toolkit/cudnn/driver fields must be non-empty and equal the
    lock (device_name is recorded but NOT required to match). Raises before output is claimed or raw is read."""
    with open(spec["env_lock_path"]) as f:
        lock = json.load(f)
    RS.check_runtime_matches_lock(lock, _capture_current_runtime())


def _authorized_train_and_write(spec, output, report, auth):
    """Run the gated trainer under an ATOMIC output claim and write provenance. FIRST verify the live runtime == env lock
    (no-CUDA / thread / version mismatch → abort with NO output). Then os.mkdir(output) (race-free) → _train_substrate
    (called exactly once) → record BOTH the .pt/.npz file-bytes sha AND the canonical (state_dict / acar.v3) semantic sha →
    manifest.json → RESULT.json LAST. Any abort removes the claimed output (no partial)."""
    _verify_runtime_matches_lock(spec)                              # runtime == lock; fails BEFORE any output / raw read
    os.mkdir(output)                                                 # atomic claim
    try:
        art = _train_substrate(spec, output)
        for k in ("encoder_checkpoint_path", "source_state_path"):
            if not (isinstance(art.get(k), str) and os.path.isfile(art[k])):
                raise RuntimeError(f"trainer did not produce {k}")
        for k in ("encoder_state_dict_sha256", "source_state_sha256"):
            if not (isinstance(art.get(k), str) and len(art[k]) == 64):
                raise RuntimeError(f"trainer did not produce canonical {k}")
        # file-bytes hash (transport/integrity) is DISTINCT from the canonical semantic hash above
        art["encoder_checkpoint_file_sha256"] = _sha256_file(art["encoder_checkpoint_path"])
        art["source_state_file_sha256"] = _sha256_file(art["source_state_path"])
        body = {"protocol_commit": spec["protocol_commit"], "disease": spec["disease"],
                "input_manifest_sha256": report.get("input_manifest_sha256"), "command": report.get("command"),
                "env_lock_sha256": spec["env_lock_sha256"], "n_eligible_subjects": spec["n_eligible_subjects"],
                "authorization": {k: auth[k] for k in ("authorized_by", "authorization_time", "statement")},
                "artifacts": art}
        with open(os.path.join(output, "manifest.json"), "w") as f:
            json.dump(body, f, sort_keys=True, allow_nan=False, indent=2)
        with open(os.path.join(output, "RESULT.json"), "w") as f:    # written LAST = completion sentinel
            json.dump({"status": "SUBSTRATE_TRAINED", "disease": spec["disease"],
                       "encoder_state_dict_sha256": art["encoder_state_dict_sha256"],
                       "encoder_checkpoint_file_sha256": art["encoder_checkpoint_file_sha256"],
                       "source_state_sha256": art["source_state_sha256"],
                       "source_state_file_sha256": art["source_state_file_sha256"]},
                      f, sort_keys=True, allow_nan=False, indent=2)
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
