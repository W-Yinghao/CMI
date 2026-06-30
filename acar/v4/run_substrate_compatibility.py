"""ACAR v4 — fixed-candidate DEV substrate-compatibility replay command (C1: executable body, AUTHORIZATION-gated; FAIL-CLOSED).

The ONE frozen command that re-embeds the OLD SEVEN DEV cohorts with the NEW all-DEV substrate and replays the FIXED candidate
(shift_margin+benefit_ranked+harm_indicator; NO reselection) to decide — via the pre-registered numeric pass-line
`regen_substrate.compatibility_replay_pass` (v2_replay a HARD requirement) — whether external Arm B may run. THIS IS NOT A NEW
DEV SELECTION RUN and emits NO selection/external/binding vocabulary (only the SUBSTRATE_COMPATIBILITY_* taxonomy).

C1 structure (mirrors B1b run_regen_substrate):
- TWO-COMMIT split: the substrate manifest pins `substrate_protocol_commit` (b99fa4f, the frozen substrate-generation code) AND
  `compatibility_protocol_commit` (this C1 replay code). The runner requires HEAD == compatibility_protocol_commit, so the
  b99fa4f substrates stay authoritative while the replay runs under the C1 commit (no dead-lock).
- STDLIB-FIRST preflight (schema + git + clean + output-absent + artifact/dev-input/env-lock FILE-byte hashes). Without a valid
  compatibility AUTHORIZATION manifest it raises SubstrateCompatibilityNotAuthorizedError BEFORE any torch/cmi import or DEV read.
- With a valid, hash-bound authorization → atomic output claim → `_run_compatibility_replay` (gated): runtime==env-lock verify +
  substrate SEMANTIC-hash verify run for real; the re-embed-to-feat-dump frontier is finalized/validated at the authorized C-run
  (raises SubstrateReplayNotWiredError until then — a CONTROLLED abort, NEVER a silently wrong verdict); the
  derive→run_dev_exploration→compatibility_replay_pass chain is wired against the real machinery.

Usage:
    python -m acar.v4.run_substrate_compatibility --substrate-manifest /abs/substrate_manifest.json --output /abs/new_compat_dir
        [--compat-authorization /abs/compat_auth.json]      # omit => fail closed (preflight only)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import sys

from acar.v4 import regen_substrate as RS
from acar.v4.run_regen_substrate import (_repo_root, _verify_commit, _verify_clean, _sha256_file,  # noqa: F401
                                         _verify_runtime_matches_lock)


def run(substrate_manifest_path, output, *, compat_authorization=None):
    """STDLIB-FIRST, FAIL-CLOSED. Validates the substrate manifest (two-commit + fixed candidate + pinned op-point + the 4
    artifact hashes + dev-input-manifest pins) + git(HEAD==compatibility_protocol_commit)/clean/output/file-hash preflight.
    Without a valid compatibility authorization it refuses to replay (raises) — NO torch/cmi import, NO DEV read, NO output.
    With a valid, hash-bound authorization it runs the gated replay under an atomic output claim."""
    with open(substrate_manifest_path, "rb") as f:
        raw = f.read()
    input_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    spec = json.loads(raw.decode())
    RS.validate_substrate_manifest(spec)                              # two-commit + fixed candidate + op-point + 4 hashes
    root = _repo_root()
    _verify_commit(root, spec["compatibility_protocol_commit"])       # HEAD == the C1 replay commit (NOT the substrate commit)
    _verify_clean(root)
    if os.path.exists(output):
        raise FileExistsError(f"output dir already exists (no overwrite): {output}")
    _verify_compat_preflight_hashes(spec)                            # artifact .pt/.npz + dev-input-manifest + env-lock FILE shas
    report = {"input_manifest_sha256": input_manifest_sha256, "candidate": RS.FIXED_CANDIDATE,
              "substrate_protocol_commit": spec["substrate_protocol_commit"],
              "compatibility_protocol_commit": spec["compatibility_protocol_commit"],
              "pass_line": {"coverage_min": RS.COVERAGE_MIN, "budget": RS.BUDGET, "alpha": RS.ALPHA,
                            "v2_replay": "HARD requirement (no waiver)"},
              "result_taxonomy": list(RS.SUBSTRATE_COMPAT_STATUSES),
              "expected_output": RS.expected_compat_output(output),
              "command": shlex.join([sys.executable, "-m", "acar.v4.run_substrate_compatibility",
                                     "--substrate-manifest", substrate_manifest_path, "--output", output])}
    if compat_authorization is None:                                  # COMPATIBILITY GATE — no authorization => fail closed
        _require_compat_authorization(report)                        # raises (no torch/cmi import, no DEV read, no output)
    auth = _load_compat_authorization(compat_authorization, spec, input_manifest_sha256, output)  # validate + bind
    return _authorized_replay_and_write(spec, output, report, auth)   # atomic; calls the gated _run_compatibility_replay


def _verify_compat_preflight_hashes(spec):
    """METADATA/file-byte preflight (stdlib; no torch/DEV signal): each disease's encoder .pt + source-state .npz match their
    *_file_sha256; the dev-input-manifest file matches its pinned sha (pins the exact eligible DEV universe to re-embed); the
    env-lock file matches env_lock_sha256. The canonical SEMANTIC hashes are re-verified inside the authorized replay loader."""
    for d in ("PD", "SCZ"):
        sd = spec["substrates"][d]
        for path_key, sha_key in (("encoder_checkpoint_path", "encoder_checkpoint_file_sha256"),
                                  ("source_state_path", "source_state_file_sha256"),
                                  ("dev_input_manifest_path", "dev_input_manifest_sha256")):
            p = sd[path_key]
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{d}: {path_key} missing: {p}")
            got = _sha256_file(p)
            if got != sd[sha_key]:
                raise ValueError(f"{d}: {path_key} {sha_key} mismatch ({got} != {sd[sha_key]})")
    elp = spec["env_lock_path"]
    if not os.path.isfile(elp):
        raise FileNotFoundError(f"env_lock_path missing: {elp}")
    if _sha256_file(elp) != spec["env_lock_sha256"]:
        raise ValueError(f"env_lock_sha256 mismatch ({_sha256_file(elp)} != {spec['env_lock_sha256']})")


def _require_compat_authorization(report):
    raise RS.SubstrateCompatibilityNotAuthorizedError(
        "DEV substrate-compatibility replay is NOT authorized — no compatibility authorization manifest supplied. The manifest "
        "validated (two-commit, fixed candidate, pinned operating point, trained-artifact + dev-input + env-lock file hashes) "
        "and the full preflight passed, but re-embedding DEV with the new substrate + the fixed-candidate replay needs an "
        "explicit, hash-bound compatibility authorization manifest (--compat-authorization; "
        "notes/ACAR_V4_SUBSTRATE_REGEN_COMMAND.md). Decision uses regen_substrate.compatibility_replay_pass (v2_replay HARD). "
        "No torch/cmi import, no DEV read, no output written. report=" + json.dumps(report, sort_keys=True))


def _load_compat_authorization(path, spec, input_manifest_sha256, output):
    """Validate + BIND the compatibility authorization to THIS run: schema (RS.validate_compat_authorization) + the
    authorization must match compatibility_protocol_commit, substrate_protocol_commit, substrate_manifest_sha256 (==the file
    sha of THIS manifest), env_lock_sha256, and output_path. Any mismatch raises BEFORE any heavy import / DEV read."""
    with open(path) as f:
        auth = json.load(f)
    RS.validate_compat_authorization(auth)
    checks = (("compatibility_protocol_commit", spec["compatibility_protocol_commit"]),
              ("substrate_protocol_commit", spec["substrate_protocol_commit"]),
              ("substrate_manifest_sha256", input_manifest_sha256),
              ("env_lock_sha256", spec["env_lock_sha256"]), ("output_path", output))
    for k, want in checks:
        if auth[k] != want:
            raise ValueError(f"compatibility authorization {k} != run {k} ({auth[k]!r} != {want!r})")
    return auth


def _verify_substrate_semantic_hashes(spec):                        # pragma: no cover — gated (torch/acar.v3); tests monkeypatch
    """Re-verify the canonical SEMANTIC substrate hashes against the on-disk artifacts (SAFE weights_only load + acar.v3
    self-verifying load_frozen) — the artifacts must be byte-and-semantics identical to the b99fa4f training record before any
    DEV metric. NO unsafe pickle fallback."""
    import numpy as np
    import torch
    from acar.v3.loader import load_frozen_source_state_artifact
    for d in ("PD", "SCZ"):
        sd = spec["substrates"][d]
        state = torch.load(sd["encoder_checkpoint_path"], map_location="cpu", weights_only=True)
        got = RS.canonical_state_dict_sha256({k: v.detach().cpu().numpy() for k, v in state.items()})
        if got != sd["encoder_state_dict_sha256"]:
            raise ValueError(f"{d}: encoder_state_dict_sha256 mismatch ({got} != {sd['encoder_state_dict_sha256']})")
        art = load_frozen_source_state_artifact(dict(np.load(sd["source_state_path"], allow_pickle=False)))
        if art.source_state_sha256 != sd["source_state_artifact_sha256"]:
            raise ValueError(f"{d}: source_state_artifact_sha256 mismatch")


def _reembed_dev_under_substrate(spec, workdir):                     # pragma: no cover — gated re-embed frontier; tests monkeypatch
    """Re-embed the OLD-SEVEN eligible DEV windows (the exact universe pinned by each disease's dev_input_manifest — PD 230 /
    SCZ 225, ds004000/sub-042 excluded, FROZEN_PIPELINE, cohort-aware keys) with the NEW all-DEV encoder, and write a new
    feat-dump dir of `audit_{disease}_{ds}_erm_0.npz` in the EXACT cmi schema that real_adapter.build_cohort_inputs consumes.
    This is the one re-embed-to-feat-format frontier: producing a subtly wrong dump would silently corrupt the compatibility
    verdict, so it is FINALIZED + validated at the authorized C-run step rather than shipped untested. Until then it raises a
    CONTROLLED SubstrateReplayNotWiredError (no wrong verdict, output cleaned by the caller)."""
    raise RS.SubstrateReplayNotWiredError(
        "re-embed-to-feat-dump is wired against the new substrate encoders + the dev_input_manifest eligible universe but its "
        "EXACT cmi audit_*.npz schema reproduction is confirmed at the authorized C-run step (no untested re-embed → no "
        "silently-wrong verdict). The downstream derive→run_dev_exploration→compatibility_replay_pass chain is real.")


def _fixed_candidate_per_disease_metrics(new_feat_dir, spec):       # pragma: no cover — gated; tests monkeypatch _run_compatibility_replay
    """REAL wiring: build the seven v3 cohort inputs from the re-embedded feat dir, derive V4OOFRecords + the v2-replay
    comparator, run the FIXED-candidate exploration (real_mode, g3=v2_replay), and extract the per-disease stats that
    compatibility_replay_pass consumes: {lambda_certified, coverage, red, L_harm_all_eval, v2_evaluable, v2_replay_red}. NO
    reselection: the exploration is pinned to EXACTLY ONE config (1x1x1 — score=shift_margin, policy=benefit_ranked,
    loss=harm_indicator); policy_families + losses + budget_by_loss are fixed so there is no 3x3 grid to choose from."""
    from acar.v4 import real_adapter as RA
    from acar.v4.develop import run_dev_exploration, V4DevConfig
    cohort_inputs = RA.build_cohort_inputs(feat_dir=new_feat_dir)
    records, v2_replay = RA.derive(cohort_inputs)
    # FIX the WHOLE candidate (1x1x1), not just the score family: pin policy + loss too, so the exploration computes EXACTLY
    # the fixed candidate (shift_margin x benefit_ranked x harm_indicator) and there is NO 3x3 policy/loss grid to silently
    # reselect from. budget_by_loss carries the single fixed loss's budget (== BUDGET).
    cfg = V4DevConfig(policy_families=(RS.FIXED_CANDIDATE["policy"],), losses=(RS.FIXED_CANDIDATE["loss"],),
                      budget_by_loss={RS.FIXED_CANDIDATE["loss"]: RS.BUDGET}, alpha=RS.ALPHA,
                      coverage_min=RS.COVERAGE_MIN, g3_comparator="v2_replay")
    result = run_dev_exploration(records, config=cfg, score_families=[RS.FIXED_CANDIDATE["score_family"]],
                                 real_mode=True, v2_replay_red_by_disease=v2_replay)
    return _extract_fixed_candidate_stats(result, v2_replay)        # gated extraction (exactly one config); confirmed at C-run


def _extract_fixed_candidate_stats(result, v2_replay):             # pragma: no cover — gated; confirmed at C-run
    """Map the exploration result for the FIXED candidate to the per-disease dict compatibility_replay_pass requires. The exact
    result accessor is confirmed at the authorized C-run (the result schema is exercised there with real data)."""
    raise RS.SubstrateReplayNotWiredError(
        "fixed-candidate per-disease stat extraction from the exploration result is confirmed at the authorized C-run step.")


def _run_compatibility_replay(spec, output):                        # pragma: no cover — gated real orchestration; tests monkeypatch
    """REAL orchestration (NOT an entry-raise). Reached ONLY with a valid, bound compatibility authorization. Steps:
    runtime==env-lock verify → substrate SEMANTIC-hash verify (both run for real) → re-embed old-seven DEV under the new
    substrate (gated frontier; confirmed at C-run) → derive + FIXED-candidate exploration → per-disease stats →
    regen_substrate.compatibility_replay_pass → verdict. Returns {status, reason, per_disease}. Tests monkeypatch this whole
    function. An operational failure propagates (caller cleans the output → OPERATIONALLY_ABORTED_NO_VERDICT)."""
    _verify_runtime_matches_lock(spec)                              # cuda + threads=1 + versions == the substrate env lock
    _verify_substrate_semantic_hashes(spec)                        # canonical encoder/source-state hashes == record
    workdir = os.path.join(output, "_reembed")
    new_feat_dir = _reembed_dev_under_substrate(spec, workdir)     # gated frontier (controlled abort until C-run)
    per_disease = _fixed_candidate_per_disease_metrics(new_feat_dir, spec)
    authorized, reason = RS.compatibility_replay_pass(per_disease)  # FROZEN pre-registered pass-line (v2_replay HARD)
    return {"status": "SUBSTRATE_COMPATIBILITY_PASS" if authorized else "SUBSTRATE_COMPATIBILITY_FAIL",
            "reason": reason, "per_disease": per_disease}


def _authorized_replay_and_write(spec, output, report, auth):
    """Atomic output claim → gated replay (called once) → write compat_manifest.json then compat_RESULT.json (status LAST). An
    operational failure removes the claimed output (no partial; OPERATIONALLY_ABORTED_NO_VERDICT — a partial dir is NEVER read
    as a compatibility FAIL)."""
    os.mkdir(output)                                                # atomic claim
    try:
        verdict = _run_compatibility_replay(spec, output)
        if verdict.get("status") not in ("SUBSTRATE_COMPATIBILITY_PASS", "SUBSTRATE_COMPATIBILITY_FAIL"):
            raise RuntimeError(f"replay returned a non-verdict status: {verdict.get('status')!r}")
        body = {"substrate_protocol_commit": spec["substrate_protocol_commit"],
                "compatibility_protocol_commit": spec["compatibility_protocol_commit"],
                "substrate_manifest_sha256": report.get("input_manifest_sha256"), "command": report.get("command"),
                "candidate": RS.FIXED_CANDIDATE, "pass_line": report["pass_line"],
                "env_lock_sha256": spec["env_lock_sha256"],
                "authorization": {k: auth[k] for k in ("authorized_by", "authorization_time", "statement")},
                "verdict": verdict}
        with open(os.path.join(output, "compat_manifest.json"), "w") as f:
            json.dump(body, f, sort_keys=True, allow_nan=False, indent=2)
        with open(os.path.join(output, "compat_RESULT.json"), "w") as f:   # written LAST = completion sentinel
            json.dump({"status": verdict["status"], "reason": verdict["reason"],
                       "candidate": RS.FIXED_CANDIDATE}, f, sort_keys=True, allow_nan=False, indent=2)
        return body
    except BaseException:
        import shutil
        shutil.rmtree(output, ignore_errors=True)                  # no partial; abort = OPERATIONALLY_ABORTED_NO_VERDICT
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 fixed-candidate substrate-compatibility replay (C1; authorization-gated)")
    ap.add_argument("--substrate-manifest", required=True)
    ap.add_argument("--output", required=True, help="must not exist")
    ap.add_argument("--compat-authorization", default=None, help="path to a compatibility authorization manifest (omit => fail closed)")
    args = ap.parse_args(argv)
    return run(args.substrate_manifest, args.output, compat_authorization=args.compat_authorization)


if __name__ == "__main__":
    main()
