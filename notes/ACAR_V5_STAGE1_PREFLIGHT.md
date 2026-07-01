# ACAR V5 — Stage-1A substrate-build PREFLIGHT (plan / schema / fail-closed; SYNTHETIC-ONLY, NO DATA)

Protocol frozen at tag `acar-v5-protocol` @ 4278435 (notes/ACAR_FROZEN_v5.md + companions). Stage-1A is a **dry-run**: it fixes
the substrate-build PLAN, the artifact-ref SCHEMA, and the fail-closed real-data door — it trains nothing and reads no data. Code:
`acar/v5/substrate/plan.py`, `build_manifest_schema.py`, `stage1_preflight.py`; guards under `acar/v5/tests/test_stage1_*.py`.

## What Stage-1A fixes (all data-free)
1. **Deterministic substrate plan** (`plan.build_substrate_plan()`) — reads no real file; every `source_path` is `None`.
2. **Fold-contained DEV substrate refs** = disease × fold × seed:
   ```
   diseases = {PD, SCZ} · folds = 0..4 · seeds = {20260711, 20260712, 20260713}   →  30 refs
   ```
3. **Seed roles** (`plan.assert_seed_role`): the Stage-2 SELECTION role is allowed ONLY for the canonical selection seed
   **20260711** (→ 10 selection refs = 2×5); seeds **20260712 / 20260713** are **S1-robustness ONLY** and can never carry the
   selection role.
4. **Final external-execution substrate ref schema** (`plan.final_external_refs()`): a DISTINCT ref type
   `external_exec/<disease>/all_source_dev` for the Stage-5 all-source substrate — declared as a FUTURE artifact type only. It is
   NOT built or registered in Stage-1A, and the fold registry / schema REFUSE it in the fold set (so a non-fold final external
   substrate can never be slipped into the fold-contained registry).
5. **Registry roundtrip** is exercised with SYNTHETIC dummy hashes + temp only (no real artifact).
6. **External sites excluded from Stage-1A**: the primary held-outs (SCZ `zenodo14808296`, PD `ds007526`), the provisional
   `zenodo14178398` (ASZED), and the excluded `ds007020` may not appear in any Stage-1A ref or read path.
7. **Fail-closed preflight** (`stage1_preflight.run_preflight`): the default plan is plan-only (0 real reads →
   `STAGE1A_PREFLIGHT_OK`). Any entry that declares a real `source_path` requires an explicit, tag-bound **Stage-1B**
   authorization (never issued in Stage-1A) → `Stage1BuildNotAuthorizedError`; and any path targeting real DEV / v4 artifacts /
   caches / external sites is refused outright (`Stage1ForbiddenTargetError`) regardless of authorization.

## Guards (all synthetic; part of `acar/v5/tests/run_all.py`)
`test_stage1_plan_counts_and_refs` (30 fold / 10 selection / 2 final-external; preflight reads nothing) ·
`test_stage1_no_real_data_paths` (source_path without/with-bad auth rejected; forbidden targets refused even with auth) ·
`test_stage1_registry_roundtrip_synthetic` (fold ref registers+admits with dummy hashes; ref-only rejected) ·
`test_stage1_external_sites_forbidden` (no site token in any ref; site paths are forbidden targets; classification) ·
`test_stage1_s1_seed_roles` (selection only seed 20260711; 12/13 S1-only) ·
`test_stage1_final_external_ref_separate` (final ref distinct type; fold registry/schema refuse it; schema-only in Stage-1A).

## Still forbidden in Stage-1A (unchanged)
real DEV cohort read · OpenNeuro/Zenodo data access · EEGNet/spectral-z substrate training · source-state fitting on real data ·
embedding dump · candidate selection · S1/S2/S3 robustness execution · held-out/external read · lockbox consumption.

## Next gate (separate authorization; NOT started)
**Stage-1B substrate build** = the FIRST authorized real DEV read + fold-contained encoder/source-state training + registry
population, under an explicit tag-bound Stage-1B authorization and its own runtime lock — never triggered by Stage-1A.
