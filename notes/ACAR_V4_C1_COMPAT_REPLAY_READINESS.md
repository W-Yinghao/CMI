# ACAR v4 — C1 compatibility-replay READINESS **(executable, AUTHORIZATION-gated; NO replay run)**

```
STATUS : run_substrate_compatibility.py is upgraded from a PREFLIGHT-ONLY fail-closed command to an EXECUTABLE,
         authorization-gated fixed-candidate DEV substrate-compatibility replay. NO replay was run: without a valid
         compatibility authorization manifest it still fails closed (SubstrateCompatibilityNotAuthorizedError) before any
         torch/cmi import or DEV read. Synthetic-tested only. NO DEV raw read, NO re-embedding, NO held-out/external read,
         NO acar-v4-protocol tag. v2/v3 untouched; lockbox SEALED.
DATE   : 2026-06-30 (machine UTC)
```

## 1. Two-commit split (the key C1 design — avoids the dead-lock)
The substrate-compat manifest now pins TWO commits instead of one ambiguous `protocol_commit`:
```
substrate_protocol_commit     = b99fa4f…   (the FROZEN substrate-generation code that trained the PD/SCZ substrates)
compatibility_protocol_commit = <C1 commit>  (this executable-replay code)
```
The runner requires `HEAD == compatibility_protocol_commit` (NOT the substrate commit), so the b99fa4f substrates stay
authoritative artifacts while the replay runs under the C1 commit. `validate_substrate_manifest` REJECTS the retired bare
`protocol_commit`. The substrates are NOT retrained or modified.

## 2. Compatibility authorization manifest (schema only; NOT created)
`regen_substrate.validate_compat_authorization` + `COMPAT_AUTH_FIELDS` + `REQUIRED_COMPAT_STATEMENT`:
```json
{
  "compatibility_protocol_commit": "<C1 commit; HEAD must == this at run>",
  "substrate_protocol_commit": "b99fa4fcfb83c6ee60996c50dba6828d40561f26",
  "substrate_manifest_sha256": "<sha256 of the substrate-compat manifest FILE bytes>",
  "env_lock_sha256": "61e505b3…",
  "output_path": "<absolute new nonexistent output directory>",
  "authorized_by": "...", "authorization_time": "<ISO-8601 UTC>",
  "statement": "Authorize fixed-candidate DEV substrate compatibility replay exactly under ACAR_V4_SUBSTRATE_REGEN_COMMAND.md"
}
```
`run_substrate_compatibility._load_compat_authorization` BINDS it: compatibility_protocol_commit / substrate_protocol_commit /
substrate_manifest_sha256 (== the file sha of THIS manifest) / env_lock_sha256 / output_path must all match. Missing/invalid →
fail before any torch/cmi import, DEV read, or output claim.

## 3. Substrate-compat manifest schema additions (for the replay data source)
Per disease, in addition to the 4 unambiguous artifact hashes (H5): `dev_input_manifest_path` + `dev_input_manifest_sha256`
(the B1b regen input manifest — pins the EXACT eligible DEV universe to re-embed: PD 230 / SCZ 225, ds004000/sub-042 excluded,
FROZEN_PIPELINE, cohort-aware keys — so the replay can NOT rediscover a different subject universe). Top-level: `env_lock_path`
(the substrate env lock, re-verified at replay). The stdlib preflight verifies the `*_file_sha256`, the dev-input-manifest file
sha, and the env-lock file sha.

## 4. Executable replay body (gated; `_run_compatibility_replay`)
Reached ONLY with a valid, bound authorization, under an atomic `os.mkdir(output)` claim:
```
_verify_runtime_matches_lock(spec)              # cuda + threads=1 + versions == the substrate env lock        (REAL)
_verify_substrate_semantic_hashes(spec)         # canonical encoder_state_dict + source_state_artifact == record (REAL; safe
                                                #   weights_only load + acar.v3 self-verifying load_frozen; no unsafe pickle)
new_feat_dir = _reembed_dev_under_substrate(..)  # re-embed old-seven eligible DEV with the NEW encoder → cmi feat dump   [FRONTIER]
records,v2 = real_adapter.derive(build_cohort_inputs(new_feat_dir))                                              (REAL wiring)
per_disease = run_dev_exploration(records, EXACTLY ONE config 1x1x1 [score=shift_margin, policy=benefit_ranked,            (REAL wiring;
              loss=harm_indicator] via V4DevConfig(policy_families/losses/budget_by_loss pinned), g3=v2_replay) → extract    extract=FRONTIER)
authorized,reason = regen_substrate.compatibility_replay_pass(per_disease)   # FROZEN pre-registered pass-line   (REAL)
status = SUBSTRATE_COMPATIBILITY_PASS | _FAIL
```
- The pass-line is the FROZEN `compatibility_replay_pass` (v2_replay HARD; per disease: CAL LTT λ* certified, coverage≥0.15,
  red>0, EVAL L_harm_all≤0.10, v2 evaluable, red>v2_replay_red; macro red>macro v2). FIXED candidate only — the exploration is
  pinned to EXACTLY ONE config (1×1×1: `score_families=[shift_margin]`, `policy_families=(benefit_ranked,)`,
  `losses=(harm_indicator,)`, `budget_by_loss={harm_indicator:0.10}`), so there is NO 3×3 policy/loss grid to silently
  reselect from — no score/loss/grid/comparator/threshold knobs.
- **Two deliberate FRONTIERS** (`_reembed_dev_under_substrate`, `_extract_fixed_candidate_stats`): producing a subtly wrong
  re-embedded feat dump or mis-reading the exploration result would yield a SILENTLY-WRONG scientific verdict — strictly worse
  than a clean abort. So these are finalized + validated at the authorized C-run step and until then raise a CONTROLLED
  `SubstrateReplayNotWiredError` (the orchestration's runtime + semantic verify run for real; the derive→pass chain is wired
  against the real machinery). No untested re-embed ⇒ no silently-wrong verdict.

## 5. Result taxonomy (no selection/external/binding vocabulary)
`regen_substrate.SUBSTRATE_COMPAT_STATUSES = (SUBSTRATE_COMPATIBILITY_PASS, SUBSTRATE_COMPATIBILITY_FAIL,
OPERATIONALLY_ABORTED_NO_VERDICT)`. PASS/FAIL are written to `compat_RESULT.json` (last; `allow_nan=False`) +
`compat_manifest.json`. An operational failure removes the claimed output (no partial; ABORT is NEVER read as a FAIL). This is
the old-seven-DEV-substrate compatibility check — NOT a new DEV selection run, NOT external evidence; no SELECT / DEV_STOP /
V4_EXTERNAL_CONFIRMED / external-G2 / coverage-theorem vocabulary is emitted.

## 6. Synthetic guards (no real raw/torch/GPU)
two-commit required + retired `protocol_commit` rejected; `env_lock_path` required; per-disease dev-input fields required;
HEAD≠compatibility_protocol_commit → fail; reselection (candidate≠fixed) → fail; op-point drift → fail; missing/invalid compat
authorization → fail before import/read/output; authorization hash mismatch → fail; wrong statement / extra / missing field →
fail; fake replay body called exactly once only after authorization; PASS verdict & FAIL verdict both written; non-verdict
status → abort+cleanup; replay raising → output cleaned; the two inner frontiers raise the controlled
`SubstrateReplayNotWiredError`. Tests monkeypatch `_run_compatibility_replay`.

## Next sequence (separate decisions; NOT started)
1. (if the replay uses torch/GPU) capture/verify a runtime lock for the C1 commit;
2. rebuild the substrate-compat manifest with `compatibility_protocol_commit = <C1 commit>` (+ env_lock_path + per-disease
   dev_input_manifest fields);
3. run the fail-closed compatibility PREFLIGHT at detached C1 WITHOUT authorization → expect SubstrateCompatibilityNotAuthorizedError;
4. record; THEN ask for an explicit compatibility-replay authorization.
Only after that authorization may the old-seven DEV compatibility replay RUN (and the two frontiers be finalized + validated).
A PASS then unlocks ACAR_FROZEN_v4 + the acar-v4-protocol tag; a FAIL → DEV-only / new dated protocol, with NO post-hoc tuning
of candidate/score/loss/grid/comparator/thresholds.
