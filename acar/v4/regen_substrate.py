"""ACAR v4 — Option B all-DEV substrate regeneration SKELETON (code-only; NO training authorized).

NON-BINDING. Implements the COMMAND CONTRACT + ARTIFACT/HASH SCHEMA + FAIL-CLOSED dry-run validator for regenerating a NEW
all-DEV V4 external representation substrate (a disease-specific EEGNet encoder + serialized source-state, pooled over that
disease's DEV cohorts, NO held-out fold). It does NOT train: `train_all_dev_substrate(..., dry_run=False)` raises
`SubstrateTrainingNotAuthorizedError`. Real training requires the separate B1 sign-off (notes/ACAR_V4_SUBSTRATE_REGEN_PLAN.md).

CRITICAL (honesty): the DEV erm_0 dumps are LEAVE-ONE-COHORT-OUT per-fold, so an all-DEV encoder is a NEW substrate (it
cannot bit-reproduce the per-fold feat_hash_te) — never "the recovered original". The fixed-candidate compatibility replay
(compatibility_replay_pass) decides only whether external Arm B may run under the new substrate; it is NOT a new DEV
selection run and reselects nothing.
"""
from __future__ import annotations
import hashlib
import json
import os

from acar.v4.prepare_external_dump import FROZEN_PIPELINE, ENCODER_ARTIFACT_FIELDS

# disease-specific all-DEV training scope (the seven DEV cohorts; NO held-out fold). External sites are NEVER in scope.
DEV_SCOPE = {"PD": ("ds002778", "ds003490", "ds004584"),
             "SCZ": ("ds003944", "ds003947", "ds004000", "ds004367")}
_EXTERNAL_OR_REJECTED = {"zenodo14808296", "ds007526", "ds007020", "14178398", "aszed"}

# the FIXED V4 candidate (DEV exploration #001 selection); the compatibility replay reselects NOTHING.
FIXED_CANDIDATE = {"score_family": "shift_margin", "policy": "benefit_ranked", "loss": "harm_indicator"}
# compatibility-replay numeric pass-line (pre-registered; see ACAR_V4_SUBSTRATE_REGEN_PLAN.md §7)
COVERAGE_MIN = 0.15
BUDGET = 0.10
ALPHA = 0.10
# B1b all-DEV substrate training requires raw_bids (raw → encoder). canonical_features would BYPASS raw→encoder training
# and contradict "new all-DEV external representation substrate"; if ever wanted it is a separate, explicitly-declared protocol.
SOURCE_KINDS = ("raw_bids",)

# PINNED all-DEV training schedule (DECLARED — this is the NEW V4 external substrate schedule, NOT a recovered DEV ERM
# schedule; the original DEV ERM hyperparameters are not reconstructable, so they are fixed here + in
# ACAR_V4_SUBSTRATE_REGEN_COMMAND.md and become part of the substrate's provenance). Pipeline shape matches FROZEN_PIPELINE.
TRAINING_SCHEDULE = {
    "model": "EEGNet", "n_chans": 19, "n_times": 512, "embedding_dim": 16, "n_classes": 2,
    "optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 64,
    "epoch_policy": "fixed", "max_epochs": 100, "loss": "cross_entropy", "class_weighting": "balanced",
    "val_split": 0.0, "device": "cuda", "seed": 0, "deterministic": True,
    "substrate_kind": "NEW all-DEV V4 external representation substrate (declared schedule; NOT a recovered DEV ERM schedule)",
}


class SubstrateTrainingNotAuthorizedError(RuntimeError):
    """Raised by train_all_dev_substrate(dry_run=False) and run_regen_substrate.run() when NO valid B1 authorization manifest
    is supplied. Real all-DEV substrate training is gated behind an explicit, hash-bound B1 authorization manifest; without it
    the CLI fails closed (no torch/cmi import, no DEV read, no output). NEVER trains/regenerates implicitly."""


# EXACT eligible DEV subject universe (subject clusters = calibration/eval unit; v2: 455 = 230 PD + 225 SCZ). The raw BIDS
# dirs may contain MORE sub-* dirs than this (e.g., SCZ ds004000 has 43 raw dirs but only 42 are in the DEV substrate's
# subject_id_te → sub-042 is excluded). Training MUST use exactly these counts; any extra raw subject must be manifest-pinned
# as excluded and NEVER read.
EXACT_ELIGIBLE = {"PD": 230, "SCZ": 225}

# B1 authorization manifest (the explicit, hash-bound human act that unlocks real training)
B1_AUTH_FIELDS = ("protocol_commit", "disease", "dev_input_manifest_sha256", "env_lock_sha256", "output_path",
                  "authorized_by", "authorization_time", "statement")
REQUIRED_AUTH_STATEMENT = ("Authorize all-DEV substrate regeneration for this disease exactly under "
                           "ACAR_V4_SUBSTRATE_REGEN_COMMAND.md")

# C1 compatibility-replay authorization manifest (the explicit, hash-bound human act that unlocks the fixed-candidate DEV
# substrate-compatibility replay). Distinct from B1: it binds the SUBSTRATE-generation commit (b99fa4f, frozen) AND the
# COMPATIBILITY executable-replay commit (the C1 code), so the runner can require HEAD==compatibility_protocol_commit while the
# substrates stay authoritative under their substrate_protocol_commit.
COMPAT_AUTH_FIELDS = ("compatibility_protocol_commit", "substrate_protocol_commit", "substrate_manifest_sha256",
                      "env_lock_sha256", "output_path", "authorized_by", "authorization_time", "statement")
REQUIRED_COMPAT_STATEMENT = ("Authorize fixed-candidate DEV substrate compatibility replay exactly under "
                             "ACAR_V4_SUBSTRATE_REGEN_COMMAND.md")
# the ONLY admissible compatibility-replay result statuses (NO selection / external / binding vocabulary)
SUBSTRATE_COMPAT_STATUSES = ("SUBSTRATE_COMPATIBILITY_PASS", "SUBSTRATE_COMPATIBILITY_FAIL",
                             "OPERATIONALLY_ABORTED_NO_VERDICT")


class SubstrateReplayNotWiredError(RuntimeError):
    """Raised if the gated compatibility-replay body is reached but a required inner step is not validated at the C-run step.
    Distinct from SubstrateCompatibilityNotAuthorizedError (which is the AUTH gate). Tests monkeypatch the replay body."""


def canonical_subject_list_sha256(subjects):
    """Canonical sha-256 over a subject id list (sorted unique, compact JSON). Shared by the manifest builder and the runner
    so the eligible-subject hash is recomputed identically."""
    return hashlib.sha256(json.dumps(sorted(set(subjects)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---- B1b runtime / value / hash safety (PURE; testable without torch / GPU / raw) -------------------------------------
# The exact runtime fields that must MATCH the captured env lock when training actually runs (versions captured with the same
# methods as capture_regen_envlock._probe). device_name is RECORDED but NOT required to match (device_kind=cuda is the hard
# rule); thread fields must all be 1 (the deterministic single-thread runtime the lock pins).
RUNTIME_VERSION_FIELDS = ("python_version", "torch_version", "torchvision_version", "torchaudio_version",
                          "braindecode_version", "moabb_version", "mne_version", "skorch_version",
                          "numpy_version", "scipy_version", "sklearn_version")
# cuda toolkit / cudnn / driver drift from the captured GPU node (compared only when device_kind == cuda, which B1b requires)
RUNTIME_CUDA_FIELDS = ("cuda_version", "cudnn_version", "driver_version")
RUNTIME_THREAD_FIELDS = ("torch_intraop_threads", "torch_interop_threads", "omp_num_threads")


def require_cuda(schedule, cuda_available):
    """Fail closed unless the FROZEN schedule pins cuda AND CUDA is actually available — NEVER fall back to CPU.
    Returns the device string 'cuda' on success."""
    if schedule.get("device") != "cuda":
        raise ValueError(f"TRAINING_SCHEDULE device must be 'cuda', got {schedule.get('device')!r}")
    if not cuda_available:
        raise RuntimeError("CUDA required by the frozen TRAINING_SCHEDULE but torch.cuda.is_available()==False "
                           "(no silent CPU fallback — abort and run on a GPU node)")
    return "cuda"


def check_runtime_matches_lock(lock, runtime):
    """FAIL-CLOSED: the CURRENT training runtime must match the captured env lock. device_kind must be 'cuda' on BOTH (no CPU
    training); the three thread fields must all be 1; every version field (and, since cuda, the cuda toolkit/cudnn/driver
    fields) must be NON-EMPTY and equal to the lock's. Raises on the first mismatch. (device_name is intentionally NOT
    compared — only device_kind=cuda is hard.)"""
    if lock.get("device_kind") != "cuda":
        raise ValueError(f"env lock device_kind must be 'cuda' for B1b training, got {lock.get('device_kind')!r}")
    if runtime.get("device_kind") != "cuda":
        raise RuntimeError(f"current runtime device_kind must be 'cuda' (no CPU training), got "
                           f"{runtime.get('device_kind')!r} — CUDA not available on this node")
    for f in RUNTIME_THREAD_FIELDS:
        if runtime.get(f) != 1:
            raise ValueError(f"runtime {f} must be 1 (deterministic single-thread), got {runtime.get(f)!r}")
        if lock.get(f) != 1:
            raise ValueError(f"env lock {f} must be 1, got {lock.get(f)!r}")
    for f in RUNTIME_VERSION_FIELDS + RUNTIME_CUDA_FIELDS:            # cuda fields compared because device_kind==cuda (B1b)
        rv, lv = runtime.get(f), lock.get(f)
        if not rv or not lv:                                         # empty would vacuously "match" — reject
            raise ValueError(f"{f} must be recorded (non-empty) in BOTH runtime and env lock (got {rv!r} / {lv!r})")
        if rv != lv:
            raise ValueError(f"runtime {f} != env lock ({rv!r} != {lv!r}) — wrong/changed runtime")
    return True


def assert_finite(arr, name):
    """Raise ValueError if `arr` contains any NaN/Inf (or is empty). Pure numpy; works on windows, labels, loss, grads."""
    import numpy as np
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return True


def single_subject_label(y_arr, key):
    """One eligible subject's per-window labels: must be non-empty, all identical, and in {0,1}. Returns the int label.
    Guards against a subject whose windows carry mixed/out-of-range labels (silent training corruption)."""
    import numpy as np
    y = np.asarray(y_arr).ravel()
    if y.size == 0:
        raise ValueError(f"{key}: subject has 0 windows")
    u = {int(v) for v in np.unique(y)}
    if not u <= {0, 1}:
        raise ValueError(f"{key}: labels must be in {{0,1}}, got {sorted(u)}")
    if len(u) != 1:
        raise ValueError(f"{key}: mixed labels within subject ({sorted(u)})")
    return int(y[0])


def check_training_set(y, subj, allowlist):
    """Disease-level fail-closed checks BEFORE training: every eligible subject contributed ≥1 window; no windows from
    non-eligible subjects; all labels in {0,1}; BOTH classes present. y/subj are per-window arrays; allowlist = eligible set."""
    import numpy as np
    yy = np.asarray(y).ravel()
    seen = set(subj)
    allow = set(allowlist)
    missing = sorted(allow - seen)
    if missing:
        raise ValueError(f"eligible subjects with no windows: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    extra = sorted(seen - allow)
    if extra:
        raise ValueError(f"windows from non-eligible subjects present: {extra[:5]}")
    u = {int(v) for v in np.unique(yy)} if yy.size else set()
    if not u <= {0, 1}:
        raise ValueError(f"labels must be in {{0,1}}, got {sorted(u)}")
    if u != {0, 1}:
        raise ValueError(f"training set must contain BOTH classes, got {sorted(u)}")
    return True


def canonical_state_dict_sha256(named_arrays):
    """Serialization-INDEPENDENT canonical hash of an encoder state_dict: sorted by tensor name, each entry hashed as
    name|dtype|shape|C-contiguous little-endian raw bytes. Stable across torch.save format / device changes — the SEMANTIC
    provenance hash (distinct from the .pt file-bytes hash). `named_arrays` = {tensor_name: np.ndarray}."""
    import numpy as np
    if not named_arrays:
        raise ValueError("state_dict is empty")
    h = hashlib.sha256()
    for name in sorted(named_arrays):
        a = np.ascontiguousarray(named_arrays[name])
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)          # normalize endianness for byte-stable hashing
        h.update(name.encode()); h.update(b"|")
        h.update(str(a.dtype.str).encode()); h.update(b"|")
        h.update(repr(tuple(a.shape)).encode()); h.update(b"|")
        h.update(a.tobytes(order="C")); h.update(b";")
    return h.hexdigest()


def check_eligible_subjects(disease, raw_by_cohort, manifest):
    """PURE eligible-subject reconciliation (no FS). `raw_by_cohort` = {cohort: [local sub-* names]} discovered on disk.
    eligible = (all namespaced raw subjects) − excluded. FAIL-CLOSED: every excluded subject must exist in raw; eligible
    count must equal EXACT_ELIGIBLE[disease]; the eligible + per-cohort hashes must match the manifest (so an extra raw
    subject not pinned as excluded, or a missing one, fails). Returns the eligible list."""
    cohorts = list(manifest["dev_cohorts"])
    excluded = set(manifest["excluded_subjects"])
    all_ns = set()
    for c in cohorts:
        all_ns |= {f"{c}/{s}" for s in raw_by_cohort.get(c, [])}
    missing_exc = excluded - all_ns
    if missing_exc:
        raise ValueError(f"{disease}: excluded subjects not present on disk: {sorted(missing_exc)}")
    eligible = sorted(all_ns - excluded)
    if len(eligible) != EXACT_ELIGIBLE[disease]:
        raise ValueError(f"{disease}: eligible count {len(eligible)} != required {EXACT_ELIGIBLE[disease]} "
                         f"(raw {len(all_ns)} − excluded {len(excluded)})")
    if canonical_subject_list_sha256(eligible) != manifest["eligible_subject_list_sha256"]:
        raise ValueError(f"{disease}: eligible_subject_list_sha256 mismatch (raw−excluded != pinned DEV-eligible set)")
    for c in cohorts:
        elig_c = sorted({f"{c}/{s}" for s in raw_by_cohort.get(c, [])} - excluded)
        if canonical_subject_list_sha256(elig_c) != manifest["per_cohort_eligible_subject_list_sha256"][c]:
            raise ValueError(f"{disease}: per_cohort_eligible_subject_list_sha256[{c}] mismatch")
    return eligible


def validate_b1_authorization(auth):
    """FAIL-CLOSED schema check for a B1 training-authorization manifest (pure). Cross-checks vs the input manifest/output
    happen in run_regen_substrate. Returns the auth."""
    if not isinstance(auth, dict):
        raise ValueError("B1 authorization must be a JSON object")
    missing = [f for f in B1_AUTH_FIELDS if f not in auth]
    if missing:
        raise ValueError(f"B1 authorization missing fields: {missing}")
    extra = [f for f in auth if f not in B1_AUTH_FIELDS]
    if extra:
        raise ValueError(f"B1 authorization has unknown extra fields: {extra}")
    if not _is_hex(auth["protocol_commit"], 40):
        raise ValueError("authorization protocol_commit must be 40-hex")
    if auth["disease"] not in DEV_SCOPE:
        raise ValueError(f"authorization disease must be one of {sorted(DEV_SCOPE)}")
    for hf in ("dev_input_manifest_sha256", "env_lock_sha256"):
        if not _is_hex(auth[hf], 64):
            raise ValueError(f"authorization {hf} must be 64-hex")
    for sf in ("output_path", "authorized_by", "authorization_time"):
        if not isinstance(auth[sf], str) or not auth[sf]:
            raise ValueError(f"authorization {sf} must be a non-empty string")
    if auth["statement"] != REQUIRED_AUTH_STATEMENT:
        raise ValueError(f"authorization statement must be EXACTLY: {REQUIRED_AUTH_STATEMENT!r}")
    return auth


def validate_compat_authorization(auth):
    """FAIL-CLOSED schema check for a C1 compatibility-replay authorization manifest (pure). Cross-checks vs the substrate
    manifest/output happen in run_substrate_compatibility. Returns the auth."""
    if not isinstance(auth, dict):
        raise ValueError("compatibility authorization must be a JSON object")
    missing = [f for f in COMPAT_AUTH_FIELDS if f not in auth]
    if missing:
        raise ValueError(f"compatibility authorization missing fields: {missing}")
    extra = [f for f in auth if f not in COMPAT_AUTH_FIELDS]
    if extra:
        raise ValueError(f"compatibility authorization has unknown extra fields: {extra}")
    for cf in ("compatibility_protocol_commit", "substrate_protocol_commit"):
        if not _is_hex(auth[cf], 40):
            raise ValueError(f"authorization {cf} must be 40-hex")
    for hf in ("substrate_manifest_sha256", "env_lock_sha256"):
        if not _is_hex(auth[hf], 64):
            raise ValueError(f"authorization {hf} must be 64-hex")
    for sf in ("output_path", "authorized_by", "authorization_time"):
        if not isinstance(auth[sf], str) or not auth[sf]:
            raise ValueError(f"authorization {sf} must be a non-empty string")
    if auth["statement"] != REQUIRED_COMPAT_STATEMENT:
        raise ValueError(f"authorization statement must be EXACTLY: {REQUIRED_COMPAT_STATEMENT!r}")
    return auth


class SubstrateCompatibilityNotAuthorizedError(RuntimeError):
    """Raised by run_substrate_compatibility.run(). The fixed-candidate DEV compatibility replay re-embeds DEV cohorts with
    the NEW (trained) substrate — it needs the trained artifacts + torch + DEV raw, all gated behind B1. NEVER runs implicitly."""


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in "0123456789abcdef" for c in s)


def _is_int0(x):
    return type(x) is int and x == 0                             # strict: reject bool / "0" / 0.0 / 0.9


def canonical_pipeline_config_sha256():
    """The canonical sha-256 of the frozen DEV pipeline (sorted keys, compact). Manifests must match this exactly."""
    return hashlib.sha256(json.dumps(FROZEN_PIPELINE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_exact_pipeline(cfg):
    if set(cfg) != set(FROZEN_PIPELINE) or cfg != FROZEN_PIPELINE:
        raise ValueError(f"pipeline_config must EXACTLY equal FROZEN_PIPELINE (no extra/missing keys), got {cfg!r}")


def expected_artifact_paths(disease, output_dir):
    """The artifact files the (future, authorized) training will emit for `disease` — pinned now so the contract is fixed
    before any training. Mirrors prepare_external_dump.ENCODER_ARTIFACT_FIELDS."""
    d = disease.upper()
    return {
        "encoder_checkpoint_path": os.path.join(output_dir, f"v4_alldev_encoder_{d}.pt"),
        "source_state_path": os.path.join(output_dir, f"v4_alldev_source_state_{d}.npz"),
        "provenance_path": os.path.join(output_dir, f"v4_alldev_substrate_{d}.provenance.json"),
    }


def validate_substrate_request(disease, dev_cohorts, output_dir, *, seed=0, pipeline_config=None, env_lock_path=None):
    """FAIL-CLOSED validation of an all-DEV substrate-regeneration request (pure; no training, no I/O beyond existence
    checks). Raises on any violation; returns a report dict (incl. expected artifacts + hash schema) on success."""
    if disease not in DEV_SCOPE:
        raise ValueError(f"disease must be one of {sorted(DEV_SCOPE)}, got {disease!r}")
    cohorts = list(dev_cohorts)
    if len(cohorts) != len(set(cohorts)):
        raise ValueError(f"{disease}: duplicate cohort in {cohorts}")
    bad = sorted(set(cohorts) & _EXTERNAL_OR_REJECTED)
    if bad:
        raise ValueError(f"{disease}: external/rejected cohort(s) {bad} must NEVER be in the all-DEV training scope")
    if set(cohorts) != set(DEV_SCOPE[disease]):
        raise ValueError(f"{disease}: dev_cohorts must be EXACTLY {sorted(DEV_SCOPE[disease])}, got {sorted(set(cohorts))}")
    if not _is_int0(seed):
        raise ValueError(f"seed must be the int 0 (no bool/str/float), got {seed!r}")
    _require_exact_pipeline(pipeline_config if pipeline_config is not None else FROZEN_PIPELINE)
    if not output_dir:
        raise ValueError("output_dir required")
    if os.path.exists(output_dir):
        raise ValueError(f"output_dir already exists (no overwrite): {output_dir}")
    if not env_lock_path or not os.path.isfile(env_lock_path):
        raise ValueError(f"env lock not present: {env_lock_path!r} (pin a v4 substrate env lock before training)")
    return {
        "disease": disease, "dev_cohorts": sorted(set(cohorts)), "seed": int(seed),
        "pipeline_config": dict(FROZEN_PIPELINE), "output_dir": output_dir, "env_lock_path": env_lock_path,
        "expected_artifacts": expected_artifact_paths(disease, output_dir),
        "encoder_artifact_fields": list(ENCODER_ARTIFACT_FIELDS),
        "hash_schema": {"encoder_state_dict_sha256": "canonical semantic: sha256(sorted name|dtype|shape|LE-bytes)",
                        "encoder_checkpoint_file_sha256": "sha256(.pt file bytes) — transport/integrity",
                        "source_state_artifact_sha256": "acar.v3 SourceStateArtifact canonical (semantic) hash",
                        "source_state_file_sha256": "sha256(.npz file bytes) — transport/integrity",
                        "provenance_sidecar_sha256": "sha256(<dump>.provenance.json) at external-dump time"},
        "substrate_kind": "NEW all-DEV V4 external representation substrate (NOT a recovered original encoder)",
        "authorized_to_train": False,
    }


def train_all_dev_substrate(disease, dev_cohorts, output_dir, *, seed=0, pipeline_config=None, env_lock_path=None,
                            dry_run=False):
    """Validate the request and (only if explicitly authorized) train. Here training is NEVER authorized: dry_run=True
    returns the validation report; dry_run=False raises SubstrateTrainingNotAuthorizedError. No torch/cmi import occurs."""
    report = validate_substrate_request(disease, dev_cohorts, output_dir, seed=seed, pipeline_config=pipeline_config,
                                        env_lock_path=env_lock_path)
    if dry_run:
        return report
    raise SubstrateTrainingNotAuthorizedError(
        f"{disease}: all-DEV substrate training is NOT authorized. Request is valid (dry_run report available), but real "
        f"training requires explicit B1 sign-off (notes/ACAR_V4_SUBSTRATE_REGEN_PLAN.md). No retrain happens here.")


# ----------------------------------------------------------------------------- compatibility replay (numeric pass-line)

def _disease_absolute_ok(s):
    """Absolute (always-required) per-disease gates for the FIXED candidate under the new substrate."""
    fails = []
    if not s.get("lambda_certified"):
        fails.append("CAL LTT λ* not certified")
    if not (s.get("coverage") is not None and s["coverage"] >= COVERAGE_MIN):
        fails.append(f"coverage<{COVERAGE_MIN}")
    if not (s.get("red") is not None and s["red"] > 0.0):
        fails.append("red<=0")
    if not (s.get("L_harm_all_eval") is not None and s["L_harm_all_eval"] <= BUDGET):
        fails.append(f"L_harm_all>{BUDGET}")
    return fails


def compatibility_replay_pass(per_disease):
    """PRE-REGISTERED numeric pass-line for the fixed-candidate (shift_margin+benefit_ranked+harm_indicator) DEV
    compatibility replay under the NEW all-DEV substrate. NO reselection. Returns (authorized: bool, reason: str).

    Per disease (PD, SCZ), ALL required: CAL LTT λ* certified AND coverage≥0.15 AND red>0 AND EVAL L_harm_all≤0.10 AND
    v2_replay EVALUABLE AND red > v2_replay_red. **v2_replay is a HARD REQUIREMENT — NO waiver** (beating v2 is the V4
    external claim; a not-evaluable v2_replay FAILS the replay). Macro: disease-macro red > disease-macro v2_replay (both
    always evaluable here). authorized iff BOTH diseases pass every gate AND the macro gate."""
    if set(per_disease) != {"PD", "SCZ"}:
        return False, f"per_disease must cover exactly PD and SCZ, got {sorted(per_disease)}"
    reasons, reds, v2s = [], [], []
    for d in ("PD", "SCZ"):
        s = per_disease[d]
        for f in _disease_absolute_ok(s):
            reasons.append(f"{d}: {f}")
        if not s.get("v2_evaluable"):
            reasons.append(f"{d}: v2_replay NOT evaluable (HARD requirement — no waiver)")
            continue
        if not (s.get("red") is not None and s.get("v2_replay_red") is not None and s["red"] > s["v2_replay_red"]):
            reasons.append(f"{d}: red<=v2_replay")
        reds.append(s.get("red")); v2s.append(s.get("v2_replay_red"))
    if len(reds) == 2 and all(x is not None for x in reds + v2s):     # macro only when both diseases' v2 are evaluable
        if not (sum(reds) / 2.0 > sum(v2s) / 2.0):
            reasons.append("disease-macro red <= disease-macro v2_replay")
    if reasons:
        return False, "NOT AUTHORIZED: " + "; ".join(reasons)
    return True, "authorized: fixed candidate meets the pre-registered compatibility minimum (v2-beating) under the new substrate"


# ----------------------------------------------------------------------------- frozen command-contract manifest schemas

def expected_compat_output(output_dir):
    """Files the (future, authorized) compatibility replay will emit — pinned now."""
    return {"result_path": os.path.join(output_dir, "compat_RESULT.json"),
            "manifest_path": os.path.join(output_dir, "compat_manifest.json")}


def validate_regen_manifest(spec):
    """FAIL-CLOSED schema check for a run_regen_substrate input manifest (pure; no I/O). Pins the ONLY admissible training
    inputs: 40-hex protocol_commit, repo_clean_required==True, disease+EXACT DEV cohorts (no external/rejected id),
    source_kind, per-cohort source paths, and the input/pipeline/env provenance hashes. Returns the spec."""
    if not isinstance(spec, dict):
        raise ValueError("regen manifest must be a JSON object")
    if not _is_hex(spec.get("protocol_commit", ""), 40):
        raise ValueError("protocol_commit must be a full 40-char lowercase git SHA-1")
    if spec.get("repo_clean_required") is not True:
        raise ValueError("repo_clean_required must be true")
    disease = spec.get("disease")
    if disease not in DEV_SCOPE:
        raise ValueError(f"disease must be one of {sorted(DEV_SCOPE)}, got {disease!r}")
    cohorts = spec.get("dev_cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != len(set(cohorts)):
        raise ValueError("dev_cohorts must be a list of unique ids")
    bad = sorted(set(cohorts) & _EXTERNAL_OR_REJECTED)
    if bad:
        raise ValueError(f"external/rejected cohort(s) {bad} must NEVER be in the all-DEV training scope")
    if set(cohorts) != set(DEV_SCOPE[disease]):
        raise ValueError(f"{disease}: dev_cohorts must be EXACTLY {sorted(DEV_SCOPE[disease])}")
    if spec.get("source_kind") not in SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {SOURCE_KINDS}")
    sp = spec.get("source_paths")
    if not isinstance(sp, dict) or set(sp) != set(cohorts):
        raise ValueError("source_paths must be a dict keyed by EXACTLY the dev_cohorts")
    for c, p in sp.items():
        if not isinstance(p, str) or not os.path.isabs(p) or not p:
            raise ValueError(f"source_paths[{c!r}] must be a non-empty absolute path")
    if not _is_hex(spec.get("source_file_manifest_sha256", ""), 64):
        raise ValueError("source_file_manifest_sha256 must be a 64-char lowercase sha-256 (raw file-list provenance)")
    pcm = spec.get("per_cohort_source_file_manifest_sha256")
    if not isinstance(pcm, dict) or set(pcm) != set(cohorts):
        raise ValueError("per_cohort_source_file_manifest_sha256 must be a dict keyed by EXACTLY the dev_cohorts")
    for c, h in pcm.items():
        if not _is_hex(h, 64):
            raise ValueError(f"per_cohort_source_file_manifest_sha256[{c!r}] must be a 64-char lowercase sha-256")
    # eligible-subject universe (pinned EXACTLY; excluded raw subjects manifest-pinned + never read)
    if not _is_hex(spec.get("eligible_subject_list_sha256", ""), 64):
        raise ValueError("eligible_subject_list_sha256 must be a 64-char lowercase sha-256")
    pce = spec.get("per_cohort_eligible_subject_list_sha256")
    if not isinstance(pce, dict) or set(pce) != set(cohorts):
        raise ValueError("per_cohort_eligible_subject_list_sha256 must be a dict keyed by EXACTLY the dev_cohorts")
    for c, h in pce.items():
        if not _is_hex(h, 64):
            raise ValueError(f"per_cohort_eligible_subject_list_sha256[{c!r}] must be a 64-char lowercase sha-256")
    if type(spec.get("n_eligible_subjects")) is not int or spec["n_eligible_subjects"] != EXACT_ELIGIBLE[disease]:
        raise ValueError(f"n_eligible_subjects must be the int {EXACT_ELIGIBLE[disease]} for {disease}")
    exc = spec.get("excluded_subjects")
    if not isinstance(exc, dict):
        raise ValueError("excluded_subjects must be a dict {namespaced_subject: reason}")
    for s, r in exc.items():
        if not (isinstance(s, str) and s and "/" in s) or not (isinstance(r, str) and r):
            raise ValueError(f"excluded_subjects entry {s!r} must be 'dsid/sub-xxx' -> non-empty reason")
    if not _is_int0(spec.get("seed", 0)):
        raise ValueError("seed must be the int 0 (no bool/str/float)")
    for hf in ("subject_list_sha256", "diagnosis_label_sha256", "pipeline_config_sha256", "env_lock_sha256"):
        if not _is_hex(spec.get(hf, ""), 64):
            raise ValueError(f"{hf} must be a 64-char lowercase sha-256")
    if spec["pipeline_config_sha256"] != canonical_pipeline_config_sha256():
        raise ValueError("pipeline_config_sha256 must equal the canonical FROZEN_PIPELINE hash")
    if not isinstance(spec.get("env_lock_path"), str) or not spec["env_lock_path"]:
        raise ValueError("env_lock_path must be a non-empty string")
    return spec


def validate_substrate_manifest(spec):
    """FAIL-CLOSED schema check for a run_substrate_compatibility manifest (pure; no I/O). Pins the trained substrate +
    the FIXED candidate (NO reselection) + the pinned operating point. Returns the spec."""
    if not isinstance(spec, dict):
        raise ValueError("substrate manifest must be a JSON object")
    if "protocol_commit" in spec:                                    # C1: the single ambiguous commit field is retired
        raise ValueError("protocol_commit is DEPRECATED for the compat manifest; use substrate_protocol_commit "
                         "(the b99fa4f substrate-generation commit) + compatibility_protocol_commit (the C1 replay commit)")
    for cf in ("substrate_protocol_commit", "compatibility_protocol_commit"):
        if not _is_hex(spec.get(cf, ""), 40):
            raise ValueError(f"{cf} must be a full 40-char lowercase git SHA-1")
    if not isinstance(spec.get("env_lock_path"), str) or not spec["env_lock_path"]:
        raise ValueError("env_lock_path must be a non-empty path (the substrate-generation env lock, re-verified at replay)")
    if spec.get("candidate") != FIXED_CANDIDATE:
        raise ValueError(f"candidate must be EXACTLY the fixed candidate {FIXED_CANDIDATE} (no reselection)")
    for k, v in (("alpha", ALPHA), ("budget", BUDGET), ("coverage_min", COVERAGE_MIN)):
        if spec.get(k) != v:
            raise ValueError(f"{k} must be {v} (pinned operating point), got {spec.get(k)!r}")
    subs = spec.get("substrates")
    if not isinstance(subs, dict) or set(subs) != {"PD", "SCZ"}:
        raise ValueError("substrates must be a dict for EXACTLY PD and SCZ")
    cohorts = spec.get("dev_cohorts")
    if not isinstance(cohorts, dict) or set(cohorts) != {"PD", "SCZ"}:
        raise ValueError("dev_cohorts must be a dict for EXACTLY PD and SCZ")
    for d in ("PD", "SCZ"):
        if set(cohorts[d]) != set(DEV_SCOPE[d]):
            raise ValueError(f"dev_cohorts[{d}] must be EXACTLY {sorted(DEV_SCOPE[d])}")
        sd = subs[d]
        if not isinstance(sd, dict):
            raise ValueError(f"substrates[{d}] must be an object")
        for pf in ("encoder_checkpoint_path", "source_state_path", "encoder_provenance_path",
                   "source_state_provenance_path", "dev_input_manifest_path"):
            if not isinstance(sd.get(pf), str) or not sd[pf]:
                raise ValueError(f"substrates[{d}].{pf} must be a non-empty path")
        if not _is_hex(sd.get("dev_input_manifest_sha256", ""), 64):     # pins the EXACT eligible DEV universe to re-embed
            raise ValueError(f"substrates[{d}].dev_input_manifest_sha256 must be a 64-char lowercase sha-256")
        # C3: the DEV feat-dump metadata (subject_id_te/recording_id_te/window_index_te/y_te) is the alignment SOURCE OF TRUTH
        # for re-embedding — pin one sha-verified dump per cohort so the raw-window↔v3-WindowKey order cannot drift.
        dfp, dfs = sd.get("dev_feat_dump_paths"), sd.get("dev_feat_dump_sha256")
        if not (isinstance(dfp, dict) and set(dfp) == set(cohorts[d]) and all(isinstance(v, str) and v for v in dfp.values())):
            raise ValueError(f"substrates[{d}].dev_feat_dump_paths must be {{cohort: path}} keyed by EXACTLY dev_cohorts[{d}]")
        if not (isinstance(dfs, dict) and set(dfs) == set(cohorts[d]) and all(_is_hex(v, 64) for v in dfs.values())):
            raise ValueError(f"substrates[{d}].dev_feat_dump_sha256 must be {{cohort: 64-hex}} keyed by EXACTLY dev_cohorts[{d}]")
        # C5: the scps CACHE that produced the DEV feat dumps is the WINDOW SOURCE for the replay (keyed by the dump's global
        # window_index = the cache row index). One sha-pinned per-condition cache per disease; the live raw reader is NOT used.
        if not isinstance(sd.get("scps_cache_path"), str) or not sd["scps_cache_path"]:
            raise ValueError(f"substrates[{d}].scps_cache_path must be a non-empty path (the per-condition scps cache .npz)")
        if not _is_hex(sd.get("scps_cache_sha256", ""), 64):
            raise ValueError(f"substrates[{d}].scps_cache_sha256 must be a 64-char lowercase sha-256")
        for legacy in ("encoder_checkpoint_sha256", "source_state_sha256"):   # retired ambiguous (file-vs-semantic) names
            if legacy in sd:
                raise ValueError(f"substrates[{d}].{legacy} is a DEPRECATED ambiguous field; use the unambiguous "
                                 f"encoder_state_dict_sha256/encoder_checkpoint_file_sha256/source_state_artifact_sha256/"
                                 f"source_state_file_sha256")
        for hf in ("encoder_state_dict_sha256", "encoder_checkpoint_file_sha256",
                   "source_state_artifact_sha256", "source_state_file_sha256"):
            if not _is_hex(sd.get(hf, ""), 64):
                raise ValueError(f"substrates[{d}].{hf} must be a 64-char lowercase sha-256")
    if not _is_hex(spec.get("env_lock_sha256", ""), 64):
        raise ValueError("env_lock_sha256 must be a 64-char lowercase sha-256")
    return spec
