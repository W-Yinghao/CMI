# ACAR v4 — B1b training authorization template **(the explicit, hash-bound act that unlocks real training)**

```
STATUS : TEMPLATE / NOT AN AUTHORIZATION. Real all-DEV substrate training runs ONLY when a B1 authorization manifest (below)
         is supplied to run_regen_substrate via --b1-authorization AND it binds (by hash) to the exact run. Until a real,
         signed authorization manifest is created, run_regen_substrate fails closed (SubstrateTrainingNotAuthorizedError).
DATE   : 2026-06-30 (machine UTC)
```

## How the gate works (code)
`run_regen_substrate.run(..., b1_authorization=<path>)`:
- no `--b1-authorization` → `_require_b1_authorization` raises (no torch/cmi import, no DEV read, no output);
- with one → `_load_b1_authorization` validates the schema (`regen_substrate.validate_b1_authorization`) AND binds it to the
  run: `protocol_commit`, `disease`, `dev_input_manifest_sha256`, `env_lock_sha256`, `output_path` must all match; the
  `statement` must be EXACTLY the required string. Any mismatch raises before training. Only then does
  `_authorized_train_and_write` claim the output atomically and call the gated `_train_substrate` (which reads DEV raw +
  trains EEGNet + fits source-state — the FIRST step that touches DEV signal).

## Authorization manifest schema (B1_AUTH_FIELDS — all required, exact set)
```json
{
  "protocol_commit": "<H2 — the clean commit the preflight ran at; HEAD must == this at run>",
  "disease": "PD",                                  // or "SCZ" (one authorization per disease)
  "dev_input_manifest_sha256": "<sha256 of the PD/SCZ regen input manifest FILE bytes>",
  "env_lock_sha256": "<sha256 of the regen env-lock FILE bytes (operational, for H2)>",
  "output_path": "<absolute output dir; must not exist>",
  "authorized_by": "<name>",
  "authorization_time": "<ISO-8601 UTC>",
  "statement": "Authorize all-DEV substrate regeneration for this disease exactly under ACAR_V4_SUBSTRATE_REGEN_COMMAND.md"
}
```

## Preconditions BEFORE creating any authorization (all must hold)
```
1. executable training path exists + synthetic-guarded                                  ✓ (this patch)
2. eligible DEV subject universe pinned EXACTLY: PD 230, SCZ 225                          ✓ (EXACT_ELIGIBLE)
3. SCZ extra raw dir resolved before raw read: ds004000/sub-042 EXCLUDED + never read     ✓ (excluded_subjects, check_eligible_subjects)
4. new protocol commit H2 (executable B1b path) is clean                                  ← H2 = this patch's commit
5. H2 regen env lock captured + verified (CAPTURED_AND_VERIFIED, interop=1, versions)     ← recapture for H2 (GPU)
6. H2 PD/SCZ manifests rebuilt (with eligible fields) + pass fail-closed preflight at H2  ← rebuild + preflight at H2
7. this authorization-manifest template is fixed                                          ✓ (this file)
8. all tests green                                                                        ✓ (this patch)
```
Items 4–6 are the "H2 re-sequence" (ACAR_V4_SUBSTRATE_REGEN_COMMAND.md): the B1b-readiness patch CHANGES the runner, so the
046507a env lock + manifests are stale for the new code path and MUST be re-captured/re-built/re-preflighted at H2 before
any authorization.

## Reminder
B1b is a SEPARATE human decision. Even after a successful authorized training run: artifact-hash review → fixed-candidate
compatibility replay (v2-HARD) → only on PASS draft `ACAR_FROZEN_v4.md` with the new substrate hashes → clean run → tag
`acar-v4-protocol` → held-out preprocessing → one external Arm-B run. If the replay fails → Option C (DEV-only).
