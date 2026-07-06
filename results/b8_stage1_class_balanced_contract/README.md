# B8.1 class-balanced randomized-audit contract canary (diagnostic-only) — MEETS pre-registered targets on the emulator (modest, masking-dependent), NOT "strong", NOT a statistic-level collider fix

Successor to B8.0 (MIXED). **Single causal change:** the exact-null stratifier moved from the OBSERVED post-treatment
label Y (B8.0's collider) to a **PRE-ASSIGNMENT design_class Dc = the cued class y0**; condition C is balanced-randomized
within `(block, Dc)`; the null resamples `C*` within `(block, Dc)`. Plus a hard **provenance gate** whose operative piece is
**H3** (executed C == hash-pinned registered schedule). The B3 contrast T + studentized/estimability/min_confirm gates are
byte-unchanged from B8.0. 12 worlds × 50 = 600 cohorts, base 420e6, Lee2019 SM16 emulator.

development-only / NOT confirmatory / NOT deployable / **NOT validation (Lee2019 is a real-geometry emulator, not a real
randomized audit)** / no tag. Protocol pre-registered + committed `3109c0f` (sha `08e3d6a8cf6df265`) and design-red-teamed
(`wtp17jn53`, 3/3 MINOR_ISSUE) **before** the run.

## RESULT — MEETS pre-registered targets (decision-level), with material caveats

Red-team `wvfdv8j1c`: accounting/isolation **PASS** (clean, 0 mismatches, disjoint seeds, isolation verified); science
**MINOR_ISSUE** — the decision-level screen is genuinely met, but the initial auto-label "STRONG"/"collider FIXED"
**over-claimed** and is corrected here.

| target | B8.1 | B8.0 | pre-registered screen | honest reading |
|---|---|---|---|---|
| `prior_only` **both-gate** (=ALERT) | **2/50** | 5/50 | ≤3 ✓ (PRIMARY) | CI-overlapping (Fisher p≈0.22) |
| `prior_only` **mean-T-alone** | **5/50** | (5/50) | — | **residual INTACT** — collider MASKED not fixed |
| `cov_plus_prior` (mixed) | **1/50** | 1/50 | ≤3 ✓ | genuinely retained |
| `random_label` | 2/50 | 1/50 | ≤1 pref (got 2) | slightly worse |
| **pooled null** | **6/200** | 7/200 | — | **FLAT** — redistribution, not tightening |
| `POS_boundary` | **8/50** | 4/50 | ≥5 ✓ | Fisher p≈0.36 — **statistically unchanged** vs B8.0 |
| 6 violations | **0 alerts, 50/50 refused** | — | 0-alert ✓ | refused **by construction** (H3) |
| quiet stress cells | **50/50 AUC≤τ, 50/50 provenance-refused, 0 alerts** | (gap) | closed ✓ | provenance catches what AUC misses |

## The honest verdict (do NOT read as "strong" or "fixed")

**B8.1 MEETS the pre-registered decision-level targets on the emulator — a MODEST, MASKING-DEPENDENT positive, not a
statistic-level collider fix.** Specifically:

1. **Collider CONTROLLED (decision-level), not FIXED (statistic-level).** The prior-collider's mean-T signature is
   **intact**: mean-T-alone still fires 5/50 (= B8.0's level; binom vs 0.025 p=0.008). The both-gate ALERT is 2/50 **only
   because the studentized AND-gate did not co-fire** — and that masking is imperfect (2 of 3 studentized fires coincide
   with mean-T, ~6.7× over independence, so the residual leaks into the 2 alerts). The 5→2 both-gate drop is CI-overlapping
   (Fisher p=0.22). **Safety on prior worlds leans on the conservative studentized AND-gate, not a recalibrated statistic.**
2. **Aggregate null control is FLAT.** Pooled decision-level null 7/200 (B8.0) → 6/200 (B8.1); `balanced` 0→1 and
   `random_label` 1→2 got *worse*; only `prior_only` moved — a redistribution within noise, not family-wide tightening.
3. **POS is modest and statistically unchanged vs B8.0.** 8/50 = 16% (84% miss). It separates from the **pooled** null
   floor (p=0.002) but only marginally from the worst single null (`prior_only` 2/50, one-sided p=0.046, CIs overlap) and
   **not** from the mean-T floor (5/50, p=0.28). POS 4→8 vs B8.0 is within noise (p=0.36) — **not demonstrably higher.**
4. **Genuine parts:** mixed control retained (1/50); the **stress-gap closure** — the quiet confound is invisible to AUC
   (50/50 AUC≤τ) yet refused by provenance H3 (50/50), including `quiet_cov_plus_concept` which carries a *real* concept
   (0/50 alerts, refused contract-first). But violation refusal itself is **by construction** (schedule deviation → H3
   fires); the honest contribution is *provenance catches what AUC misses*, not that the emulator's violations are hard.

## Provenance / isolation / scope

Seeds `420e6 + world_index*1e6 + cohort`, all 600 unique, **zero intersection** with all prior CSC runs. Isolation: the
certifier + null take arrays+table+hash only (no world/oracle/B7/observed_T); the null is a pure within-`(block,Dc)`
permutation with no fitted Y|Z or C|Z; the stratifier is the pre-assignment Dc, never observed Y. **Of the 5 provenance
gates, H1/H2/H5 are non-discriminative by construction in this emulator** (H2/H5 always pass) — the canary is an **H3
(adherence) + H4 (support) + D1 (AUC secondary)** test, not a "5-gate" validation. H3 is zero-tolerance; `_schedule_hash`
pins per-`(block,Dc)` counts, not the per-trial assignment. Emulator (Lee2019 SM16 geometry), semi-synthetic, n=50/world,
single base seed.

## Next (reviewer decision, NOT authorized)

The decision-level targets are met but the statistic-level collider and POS remain the open problems. Candidate next steps
(reviewer's call): recalibrate the mean-T gate on prior worlds (so control does not lean on the studentized AND-gate); an
audit/label **budget frontier** to lift POS above the *mean-T* floor (not just the pooled floor); a **multi-seed** replication
to resolve the CI-overlapping 5→2 and 4→8 comparisons; or genuinely randomized-audit data. NOT threshold tuning, NOT a new
statistic, NOT a paper claim. See `b8_stage1_redteam_checks.json`, `b8_stage1_protocol.json`, `notes/b8_1_class_balanced_contract.md`.
Related: `b8_stage0_contract/` (B8.0), `b7_stage1_full_replay/`.
