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


class SubstrateTrainingNotAuthorizedError(RuntimeError):
    """Raised by train_all_dev_substrate(dry_run=False). Real all-DEV substrate training is gated behind explicit B1
    sign-off; this skeleton only validates the request (dry_run=True). NEVER trains/regenerates implicitly."""


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

    Per disease (PD, SCZ), ALWAYS required: CAL LTT λ* certified AND coverage≥0.15 AND red>0 AND EVAL L_harm_all≤0.10.
    v2 sub-gate (per disease): if v2_replay is evaluable, require red > v2_replay_red; if NOT evaluable, this sub-gate is
    WAIVED for that disease (documented; the absolute gates still hold). Macro: among diseases with an evaluable v2_replay,
    require disease-macro red > disease-macro v2_replay_red; if NO disease has an evaluable v2_replay, the macro v2 gate is
    WAIVED. authorized iff BOTH diseases pass their absolute+v2 gates AND the macro v2 gate (or its waiver) holds."""
    if set(per_disease) != {"PD", "SCZ"}:
        return False, f"per_disease must cover exactly PD and SCZ, got {sorted(per_disease)}"
    reasons, macro_reds, macro_v2 = [], [], []
    for d in ("PD", "SCZ"):
        s = per_disease[d]
        for f in _disease_absolute_ok(s):
            reasons.append(f"{d}: {f}")
        if s.get("v2_evaluable"):
            if not (s.get("red") is not None and s.get("v2_replay_red") is not None and s["red"] > s["v2_replay_red"]):
                reasons.append(f"{d}: red<=v2_replay")
            macro_reds.append(s["red"]); macro_v2.append(s["v2_replay_red"])
    if macro_reds:                                            # macro v2 gate only among v2-evaluable diseases
        if not (sum(macro_reds) / len(macro_reds) > sum(macro_v2) / len(macro_v2)):
            reasons.append("disease-macro red <= disease-macro v2_replay")
    # else: macro v2 gate WAIVED (no disease had an evaluable v2_replay)
    if reasons:
        return False, "NOT AUTHORIZED: " + "; ".join(reasons)
    return True, "authorized: fixed candidate meets the pre-registered compatibility minimum under the new substrate"
