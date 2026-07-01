# ACAR V5 — Stage-1B0 real-data-entry hardening / authorization wiring (SYNTHETIC-ONLY, NO DATA)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B0 installs the three locks that must be in place BEFORE the first
real DEV read (Stage-1B). It is wiring + guards only: it validates contracts and path STRINGS, opens nothing, trains nothing.
Code: `acar/v5/substrate/stage1b_authorization.py`, `stage1b_manifest.py`, `stage1_runtime_lock.py` (+ the Stage-1A preflight
patch); guards under `acar/v5/tests/test_stage1b_*.py`.

## The three locks
1. **Final-external ref stays schema-only — ALWAYS.** `stage1b_manifest.assert_final_external_schema_only` fails on ANY
   `final_external_refs` entry with a non-None `source_path`, regardless of authorization; it also enforces role
   `stage5_external_execution`, exactly one ref per disease, diseases == {PD, SCZ}. Wired into BOTH `stage1_preflight.run_preflight`
   and the Stage-1B build gate. ⇒ a "valid Stage-1B auth" can never open a final all-source external substrate build (that is
   Stage-5, post-candidate-fixed, per the frozen protocol).
2. **Structured Stage-1B authorization contract** (`stage1b_authorization.validate_stage1b_authorization`): not a magic
   statement — an auditable run contract binding, exactly:
   ```
   stage=Stage-1B · protocol_tag=acar-v5-protocol · protocol_tag_target_sha ∈ prefixes of 4278435… ·
   implementation_base_sha=40-hex · allowed_ref_type=fold_contained_only · allowed_refs = EXACTLY the 30 fold refs ·
   allowed_seeds={20260711,20260712,20260713} · selection_seed=20260711 ·
   forbid_final_external_refs=forbid_external_sites=forbid_candidate_selection=forbid_external_read=True ·
   run_id=<non-empty> · statement=REQUIRED_STAGE1B_STATEMENT
   ```
   Missing/extra field, wrong stage/tag/sha, allowed_refs ≠ the 30, wrong seeds/selection-seed, any forbid-flag ≠ True → fail.
3. **DEV-source WHITELIST (not blacklist)** (`stage1b_manifest.validate_dev_source_path`): a Stage-1B real path is admissible
   ONLY if it references a disease-matched frozen DEV cohort (PD {ds002778,ds003490,ds004584} / SCZ
   {ds003944,ds003947,ds004000,ds004367}) and NO other-disease cohort, external site (zenodo14808296 / ds007526 /
   zenodo14178398 / ds007020), or prior-artifact/cache marker (/scps/cache, feat_dump_v*, acar_v4_/acar_v3_). String check only —
   opens nothing. (Note: unlike the Stage-1A blanket forbidder, this ALLOWS `/projects/` for a disease-matched DEV cohort, since
   raw DEV lives in the datalake — but still blocks the scps cache + v4/v3 artifacts + all external sites.)

## The build-readiness gate
`stage1_runtime_lock.require_stage1b_ready(plan, authorization, runtime_lock)` — the gate real Stage-1B build code MUST pass
before any DEV read/train. Requires ALL of: (1) the structured authorization, (2) a matching runtime lock
(`validate_runtime_lock`: stage/tag/target-sha/run_id cross-bound to the auth, `status=CAPTURED_AND_VERIFIED`, device cpu|cuda),
and (3) the build manifest DEV whitelist + final-external schema-only. Returns `STAGE1B_READY` with the count of admitted fold
refs (0 for the default plan-only spec). In Stage-1B0 no real lock is captured and no real path is opened.

## Stage-1A preflight patch
`run_preflight` now (a) enforces final-external schema-only always, and (b) requires the STRUCTURED authorization contract (not a
bare statement) for any fold-ref `source_path`; the default plan-only spec still returns `STAGE1A_PREFLIGHT_OK` with 0 real reads.
`Stage1BuildNotAuthorizedError` is now defined centrally in `stage1b_authorization` (re-exported from `stage1_preflight` for
back-compat).

## Guards (synthetic; part of `acar/v5/tests/run_all.py`)
`test_stage1b_authorization_contract` (structured contract; every field bound; allowed_refs == exactly 30) ·
`test_stage1b_final_external_still_forbidden` (final source_path rejected by helper + preflight + build gate, even with auth+lock) ·
`test_stage1b_dev_source_whitelist` (disease-matched DEV ok; other-disease / no-cohort / site / artifact rejected) ·
`test_stage1b_runtime_lock_required` (readiness needs auth AND matching lock; run_id/target cross-bound) ·
`test_stage1b_fold_refs_only` (only authorized fold refs admissible; final-external can't build; default admits 0 real refs).

## Still forbidden in Stage-1B0 (unchanged)
real DEV read · OpenNeuro/Zenodo access · EEGNet/spectral-z training · source-state fitting · embedding dump · candidate
selection · S1/S2/S3 robustness execution · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B substrate build** = the first authorized real DEV read + fold-contained encoder/source-state training + registry
population, run ONLY behind a real (git-HEAD/runtime-verified) instance of this contract + a captured runtime lock, under a
tightly-scoped Stage-1B authorization. Not triggered by any Stage-1A/1B0 code.
