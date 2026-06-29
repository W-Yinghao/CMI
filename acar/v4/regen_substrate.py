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
SOURCE_KINDS = ("raw_bids", "canonical_features")


class SubstrateTrainingNotAuthorizedError(RuntimeError):
    """Raised by train_all_dev_substrate(dry_run=False) and run_regen_substrate.run(). Real all-DEV substrate training is
    gated behind explicit B1 sign-off; the skeleton/CLI only validate the request. NEVER trains/regenerates implicitly."""


class SubstrateCompatibilityNotAuthorizedError(RuntimeError):
    """Raised by run_substrate_compatibility.run(). The fixed-candidate DEV compatibility replay re-embeds DEV cohorts with
    the NEW (trained) substrate — it needs the trained artifacts + torch + DEV raw, all gated behind B1. NEVER runs implicitly."""


def _is_hex(s, n):
    return isinstance(s, str) and len(s) == n and all(c in "0123456789abcdef" for c in s)


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
    if int(seed) != 0:
        raise ValueError(f"seed must be 0 (frozen determinism), got {seed!r}")
    cfg = pipeline_config if pipeline_config is not None else FROZEN_PIPELINE
    for k, v in FROZEN_PIPELINE.items():
        if cfg.get(k) != v:
            raise ValueError(f"pipeline_config[{k!r}]={cfg.get(k)!r} != frozen DEV {v!r}")
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
        "hash_schema": {"encoder_checkpoint_sha256": "sha256(canonical little-endian state_dict bytes)",
                        "source_state_sha256": "acar.v3 SourceStateArtifact full-bytes hash",
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
    if int(spec.get("seed", 0)) != 0:
        raise ValueError("seed must be 0 (frozen determinism)")
    for hf in ("subject_list_sha256", "diagnosis_label_sha256", "pipeline_config_sha256", "env_lock_sha256"):
        if not _is_hex(spec.get(hf, ""), 64):
            raise ValueError(f"{hf} must be a 64-char lowercase sha-256")
    if not isinstance(spec.get("env_lock_path"), str) or not spec["env_lock_path"]:
        raise ValueError("env_lock_path must be a non-empty string")
    return spec


def validate_substrate_manifest(spec):
    """FAIL-CLOSED schema check for a run_substrate_compatibility manifest (pure; no I/O). Pins the trained substrate +
    the FIXED candidate (NO reselection) + the pinned operating point. Returns the spec."""
    if not isinstance(spec, dict):
        raise ValueError("substrate manifest must be a JSON object")
    if not _is_hex(spec.get("protocol_commit", ""), 40):
        raise ValueError("protocol_commit must be a full 40-char lowercase git SHA-1")
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
                   "source_state_provenance_path"):
            if not isinstance(sd.get(pf), str) or not sd[pf]:
                raise ValueError(f"substrates[{d}].{pf} must be a non-empty path")
        for hf in ("encoder_checkpoint_sha256", "source_state_sha256"):
            if not _is_hex(sd.get(hf, ""), 64):
                raise ValueError(f"substrates[{d}].{hf} must be a 64-char lowercase sha-256")
    if not _is_hex(spec.get("env_lock_sha256", ""), 64):
        raise ValueError("env_lock_sha256 must be a 64-char lowercase sha-256")
    return spec
