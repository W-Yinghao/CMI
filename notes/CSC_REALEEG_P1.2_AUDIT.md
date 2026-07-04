# CSC-realEEG-P1.2 — executable-but-guarded freeze package + AUDIT record (NO tag, NO run)

Status: **P1.2 package built + red-teamed + hardened. EXECUTABLE but FAIL-CLOSED.** No `csc-realeeg-v1` tag,
no validation run, no genuine contrast. Method locks byte-unchanged; synthetic tags `dee8958`/`0595f64`
untouched. A run needs: (1) this audited package, (2) tag creation, (3) a separate reviewer go.

## What P1.2 adds (on top of P1 / P1.1)
- `csc/mininfo/realeeg_engine.py` — the validation ENGINE: 9-condition semi-synthetic injection on real `Z`
  (`build_cohort`), B3 (`certify_paired_calibrated`) + Route A (`run_frozen_protocol`, session1→session2,
  subject-as-domain) invocations, subject-clustered bootstrap, 3-tier `evaluate_verdict`, and a `smoke()`
  self-test on a toy cache (non-real seed). Hash-pinned in the bank manifest.
- `run_realeeg_validation.py` — now `dry_run` (default) + **guarded `--execute`** (git-frozen tag + clean tree
  + method/cache/engine hashes → else REFUSE exit 2; with no tag it always refuses) + `--smoke` (toy plumbing).
- `run_realeeg_validation.sbatch` — frozen-worktree fail-closed wrapper (A/B3 pattern): HEAD==tag, clean tree,
  cache sha256, stale removal, temp→mv, freshness re-check; infra-fail exit 2; scientific FAIL preserves artifact.
- **Route A label-unit adaptation (reviewer choice A):** SAME byte-frozen A code; `label_unit="trial"` for MI
  (analysis_unit `"subject"` → subject-clustered), guardrail fails closed on `"subject"`. Framed as a
  trial-label TRANSFER DIAGNOSTIC, NOT a subject-label revalidation (manifest `route_A_config` + pre-reg).
- **3-tier verdict:** TIER1 B3 real-feature SAFETY (gating, drives package PASS on the 4 nulls incl.
  `random_label_control`); TIER2 B3 POWER (reported); TIER3 Route A trial-label DIAGNOSTIC (reported).

## Audit
- JSON valid; `py_compile` OK; **dry-run 54/54 PASS**; `--execute` REFUSED **exit 2** (no tag); **tests 32/32
  PASS**; `--smoke` OK (B3 confirms POS_concept, abstains on nulls/genuine; Route A runs+abstains); sbatch
  `bash -n` OK.
- **Independent red-team verdict: ISSUES** (safe as a guarded dry-run — genuinely fail-closed, cannot run/leak
  without a tag; injections match declared ground truth; execution safety solid — but the frozen verdict logic
  had defects). **1 blocker + 4 majors — ALL FIXED + re-verified:**
  1. **[BLOCKER V1]** `evaluate_verdict` counted B3 abstain/invalid states (NEED_MORE_LABELS /
     INVALID_PAIR_STRUCTURE / UNIDENTIFIABLE) as valid non-confirmations → padded the type-I denominator →
     anti-conservative auto-PASS when B3 abstains. FIXED: DECIDED-only denominator; abstain/invalid → invalid
     fraction; INCONCLUSIVE above the 0.20 cap; R4 out-of-5-state check. Tested (all-abstain null → INCONCLUSIVE).
  2. **[MAJOR V2]** gate used binomial Clopper–Pearson over overlapping cohorts, not the pre-registered
     subject-clustered bootstrap (`subject_bootstrap_upper` was dead code). FIXED: bootstrap now drives the R1
     upper bound. Tested.
  3. **[MAJOR H1]** the injection+verdict engine was not hash-pinned. FIXED: `engine_sha256` pinned in the bank
     manifest, verified in BOTH `dry_run` and `execute`. Tested (mismatch fails closed).
  4. **[MAJOR M1]** bank manifest claimed the pooled boundary was a "subject-grouped cross-fit (no leakage)";
     it is an in-sample generative fit. FIXED: wording corrected (null holds because the boundary is
     session-independent by construction, not by cross-fitting).
  5. **[MAJOR A1]** `certify_A` recorded the full `Certificate(...)` repr as the state (so A abstentions were
     miscounted / never matched). FIXED: extract the bare `.state` string. Tested.
  - Minors (documented, non-gating): antisymmetric-rotation wording (m1); INCONCLUSIVE vs FAIL labeling (m2,
    fixed); pure-conditional realized only by rotation degree (m3); `_prior_resample` with-replacement (m4).

## Authorization / next
STOP per authorization. NOT done (each needs a separate go): create tag `csc-realeeg-v1`; run the validation
(injected bank + Route A/B3 certifiers + genuine contrast + subject-clustered bounds); run the genuine
contrast. A final pre-tag red-team of the (now-fixed) engine is advisable before the tag. No clinical/PD claim.
