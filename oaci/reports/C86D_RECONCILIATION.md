# C86D — Reconciliation (same-scope patch; NOT rerun)

Addresses the PM C86D review that **rejected** attempt 902203's scientific result.
All fixes are in the same C86D scope; no new C-number. The corrected code is tested
(32 tests) but **NOT rerun** — a fresh direct `授权 C86D` is required (the prior
authorization was consumed by job 902203, which is preserved as engineering).

## The 10-point patch

1. **A1 entropy sign (hard blocker 1).** `_entropy` now returns the non-negative
   Shannon entropy `Σ p·(−log p)`. A1 is a real Active-Testing/LURE acquisition
   again, not clipped-to-uniform. Test: entropy ≥ 0; A1 score non-uniform on an
   asymmetric pool; symmetric pool degenerates to P0.
2. **Composite plugin = locked C85U methodology (hard blocker 2).** Jeffreys bAcc
   `(C_y+0.5)/(N_y+1)` over the pre-registered class set (a missing class contributes
   0.5, never dropped); 15-bin weighted ECE `Σ_b (w_b/W)|conf_b−acc_b|`;
   LURE-weighted NLL; oriented midrank percentile `(r−1)/(n−1)`; equal-weight
   composite; first-index argmax. **FULL** uses the exact (unsmoothed) construction
   metric. **Positive control (real C85U):** `composite_from_metrics` reproduces
   C85U's `composite_utility` AND `standardized_regret` with **max abs error 0.0**.
3. **Risk scale (hard blocker 3).** Primary risk = held **standardized regret**
   (looked up from C85U per `(context, candidate)`), not the raw composite-utility
   gap. Raw gap is kept separate, used only for the ε / near-optimal geometry. Test:
   the two are not interchangeable.
4. **Selection/evaluation isolation (hard blocker 4).** Split into two processes:
   **D1** (`run_d1.py`) runs the query client/server, persists + SHA-hashes every
   selection freeze (query sequence, per-step q_m, budget-specific LURE weights,
   receipts, per-context estimates, 8 selected indices), and imports **no C85U**;
   **D2** (`run_d2.py`) reads and hash-verifies the D1 freezes, opens/verifies C85U,
   and holds no query-server handle. `execute()` runs them as separate subprocesses.
   Tests: D1 source has no C85U reference; D2 fails closed on a tampered freeze.
5. **Uniform warm start + nested prefixes.** First 4 queries uniform; one full
   acquisition path per (target, method, replicate); budgets 4/8/16/32/FULL are
   prefixes of it with budget-specific LURE weights `v_m^M`. Test: warm-start q_m
   uniform, nested prefixes, active-step LURE weights differ by budget.
6. **Method freeze uses per-cohort taxonomy.** D2 classifies with per-cohort
   mean AND tail (exact CVaR) AND near-opt gates.
7. **C86H registry unchanged.** D2 records `c86h_method_registry = [P0, A1, A2H]`
   regardless of development result; development disposition is a separate object
   (development winner ≠ "C86H only runs P0").
8. **Replicate/target-level results + MC uncertainty** persisted (per-cell std over
   the 8 replicate seeds).
9. **C86L acceptance + C85U identity binding** verified by hashing at run time
   (acceptance manifest, utility-index SHA, 76,464/944, candidate-index canonical
   order), fail-closed.
10. **A2H** query score kept as the registered `Σ_{k<k'} E_π|NLL_k−NLL_k'|`, but its
    action/risk now flow through the corrected plugin + standardized regret.

## Status

Patched + tested (32 C86D tests pass; full collection intact). Attempt 902203 is
preserved as engineering. **No rerun performed.** The corrected development run
(D1→D2) will run only under a fresh direct `授权 C86D`; C86H / C87 / manuscript
remain unauthorized; C86H does not auto-start C87.
