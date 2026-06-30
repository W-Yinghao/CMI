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
  `_authorized_train_and_write` FIRST verify the live runtime == env lock (`_verify_runtime_matches_lock`; H4), then claim the
  output atomically and call `_train_substrate` — a **real executable body** (H3): `load_eligible_windows` (per-eligible-subject
  raw open via `cmi.load_cohort(subjects={subject})` — excluded never opened) → `_train_encoder_and_save` (deterministic ERM per
  `RS.TRAINING_SCHEDULE`, cuda-only, non-finite-guarded) → `_fit_and_serialize_source_state` (acar.v3
  `fit_source_state_artifact`→`freeze`→`np.savez`). This is the FIRST step that touches DEV signal.

## Authorization manifest schema (B1_AUTH_FIELDS — all required, exact set)
```json
{
  "protocol_commit": "<H = H3 — the clean commit the preflight ran at; HEAD must == this at run>",
  "disease": "PD",                                  // or "SCZ" (one authorization per disease)
  "dev_input_manifest_sha256": "<sha256 of the PD/SCZ regen input manifest FILE bytes>",
  "env_lock_sha256": "<sha256 of the regen env-lock FILE bytes (operational, for H)>",
  "output_path": "<absolute output dir; must not exist>",
  "authorized_by": "<name>",
  "authorization_time": "<ISO-8601 UTC>",
  "statement": "Authorize all-DEV substrate regeneration for this disease exactly under ACAR_V4_SUBSTRATE_REGEN_COMMAND.md"
}
```

## Preconditions BEFORE creating any authorization (all must hold)
```
1. executable training path is REAL (not placeholder/raise): raw allowlist loader + ERM loop + source-state   ✓ (H3)
2. excluded subjects filtered BEFORE raw open (cmi.load_cohort subjects= allowlist; load_eligible_windows)     ✓ (H3)
3. cohort-aware subject matching (dsid/sub key throughout)                                                     ✓ (H3)
4. source_kind for B1b training = raw_bids only                                                                ✓ (H3)
5. exact training schedule pinned in code + docs (RS.TRAINING_SCHEDULE)                                        ✓ (H3)
6. source-state serialization implemented (delegated to fixed acar.v3 fitter)                                  ✓ (H3)
7. eligible DEV subject universe pinned EXACTLY: PD 230, SCZ 225 (SCZ ds004000/sub-042 EXCLUDED)               ✓ (EXACT_ELIGIBLE)
8. NO silent CPU fallback (require_cuda) — training aborts if cuda unavailable                                 ✓ (H4)
9. live runtime == captured env lock (device_kind=cuda, threads=1, all versions) before any output/raw         ✓ (H4)
10. raw value/label fail-closed (finite windows; labels {0,1}; within-subject identical; both classes; ≥1 win) ✓ (H4)
11. non-finite logits/loss/grad/param/embedding -> abort + cleanup (no NaN encoder/source-state written)       ✓ (H4)
12. encoder/source-state record BOTH canonical semantic sha AND file-bytes sha (no field overload)             ✓ (H4)
13. dual-hash naming UNIFIED pipeline-wide (regen output + substrate-compat + external encoder artifact); ambiguous
    encoder_checkpoint_sha256/source_state_sha256 retired in the substrate-artifact schema; source_kind=raw_bids doc fixed ✓ (H5)
14. this authorization-manifest template is fixed + all tests green                                            ✓ (H5)
15. new protocol commit H = H5 (unified hash schema) is clean                                                  ← H5 = this patch's commit
16. H regen env lock captured + verified (CAPTURED_AND_VERIFIED, cuda, interop=1, versions, protocol_commit=H) ← recapture for H (GPU)
17. H PD/SCZ manifests rebuilt (eligible fields, raw_bids) + pass fail-closed preflight at detached H          ← rebuild + preflight at H
```
Items 15–17 are the "H re-sequence" (ACAR_V4_SUBSTRATE_REGEN_COMMAND.md §7d): each runner/schema change (H2→H3→H4→H5) restales
the 046507a env lock + manifests, which MUST be re-captured/re-built/re-preflighted at the LATEST commit (H5) before any authorization.

The four unambiguous substrate-artifact hash fields (H5): `encoder_state_dict_sha256` (canonical) ·
`encoder_checkpoint_file_sha256` (.pt bytes) · `source_state_artifact_sha256` (acar.v3 canonical) · `source_state_file_sha256`
(.npz bytes).

## Reminder
B1b is a SEPARATE human decision. Even after a successful authorized training run: artifact-hash review → fixed-candidate
compatibility replay (v2-HARD) → only on PASS draft `ACAR_FROZEN_v4.md` with the new substrate hashes → clean run → tag
`acar-v4-protocol` → held-out preprocessing → one external Arm-B run. If the replay fails → Option C (DEV-only).
