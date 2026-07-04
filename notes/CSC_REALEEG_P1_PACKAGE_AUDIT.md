# CSC-realEEG-P1 — freeze-package build + AUDIT record (DRY-RUN only; no tag, no run)

Status: **P1 package built and audited. DRY-RUN ONLY.** No `csc-realeeg-v1` tag created, no validation bank
run, no certifier executed, no genuine contrast run. Method locks byte-unchanged; synthetic tags `dee8958`
(A) / `0595f64` (B3) untouched. A validation run requires a separate go (tag + run).

## Package (all under `csc/`)
- `csc/mininfo/realeeg_lee2019_cache_manifest.json` — frozen cache provenance + feature pipeline (16-ch
  `SM16_no_FCz`, band-pass 8–30, window 0.5–3.5, resample 128, `normalize=None`, `log_var_time`, label
  {left:0,right:1}, run `EEG_MI_train`). Pins cache/metadata/builder sha256; feasibility_report_commit 223c99e.
- `csc/mininfo/realeeg_bank_manifest.json` — 9-condition semi-synthetic injected bank on real features, each
  with full spec (sessions, real/synthetic labels, label model, held-fixed, injected shift, ground truth,
  gating, routes). `NULL_cov = real S1/S2 split + Y*~pooled P̂(Y|Z)` = primary type-I gate. Gating set =
  {NULL_cov, NULL_label, NULL_cov_plus_label}. Genuine contrast + power = descriptive/non-gating. Seed
  schedule base 20000000 (disjoint from A 900000–1800065 and B3 3000000–14100047).
- `csc/mininfo/realeeg_routeA_manifest.json`, `csc/mininfo/realeeg_b3_manifest.json` — explicit route locks
  (no shared hidden defaults): pinned method hashes, cache hash, seeds, B_subject=2000, B_certifier=200 (B3),
  invalid cap 0.20, α 0.025/0.05, denominator rules, endpoints, PASS/FAIL/INCONCLUSIVE, `gating_flags`
  (R2/R5-2b/genuine-contrast all `is_gating:false`), guardrail. Route A = transfer test (not must-fail).
- `csc/mininfo/run_realeeg_validation.py` — DRY-RUN runner; `--execute` structurally REFUSED (exit 2); no path
  runs an injection/certifier or creates a tag.
- `csc/tests/test_realeeg.py` — fail-closed tests (20).

## Pinned method locks (verified byte-identical to the frozen tags)
B3: `paired_calibrated.py` 26e505ed…, `paired_conditional_test.py` 1263f672…, `paired_certifier.py` 8d97a197…
Route A: `protocol.py` 9c158ea7…, `certificate/certifier.py` ef6b734d…, `atlas.py` aab2e60e…,
`residual_test.py` 5b8496e3…. Builder: `build_lee2019_b3_cache.py` 5bf517d1…. Cache
`LEE2019_B3.npz` 5196b6d6…, metadata 68cd8c95….

## Audit
- JSON valid ×4; `py_compile` OK; **dry-run 52/52 PASS**; `--execute` REFUSED **exit 2**; **fail-closed tests
  20/20 PASS**.
- **Independent red-team verdict: SOUND-AND-SAFE-TO-COMMIT, no blockers.** Verified: no execution leak (runner
  imports only argparse/hashlib/json/os/sys; only `--execute` branch is `refuse_execute→exit 2`); all 7 method
  hashes match disk AND match the versions inside tags `csc-confirmatory-v1`/dee8958 and
  `csc-b3-confirmatory-v1`/0595f64; cache sha256 matches; no `csc-realeeg-v1` tag; synthetic tags intact;
  gaming surfaces (montage swap, feature-family switch, NULL_cov demotion, genuine-contrast/power gating, α/cap
  changes) all fail-closed.
- **6 red-team hardening items (2 major, 4 minor) — all FIXED + re-verified with new tests:**
  1. [major] trap controls (NULL_label, NULL_cov_plus_label) were demotable to non-gating undetected → added
     `bank_trap_controls_gating` + `bank_gating_set_exact` checks + test.
  2. [major] cache-absent branch was an unconditional pass (builder_sha256 unverified) → absent branch now
     verifies the builder sha256, fail-closed + test.
  3. [minor] seed-disjointness walrus bug (effective check was a hardcoded `>14100047`) → rewritten to use
     `FORBIDDEN_SEED_RANGES` + `max(hi)` + test proving it fails on a range extension.
  4. [minor] R2/R5 "not gating" were substring checks → added explicit `gating_flags` booleans + checks + test.
  5. [minor] frozen scalar fields (label_map, feature_name, run) unverified → added equality checks + test.
  6. [minor] Route A PASS rule omitted R3 eligibility → added R3 to Route A endpoints + PASS (parity with B3).

## Authorization / next
STOP per authorization. NOT done (needs a separate go): create tag `csc-realeeg-v1`; run the injected bank or
Route A/B3 certifiers; run the genuine session contrast; use 2b as gating; switch feature family. The next
phase would freeze the package (tag) and then, as a further separate go, run the validation.
