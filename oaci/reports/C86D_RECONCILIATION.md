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

---

## Round 2 — final estimator / seed / order / endpoint reconciliation

Second PM review; same C86D scope; tested (37 tests), NOT rerun (needs fresh
`授权 C86D`).

1. **D2 verifies freezes BEFORE opening C85U.** `run_d2.verify_freezes` fully
   checks path/count/SHA/schema/uniqueness and returns a selected-actions-only
   object; `load_c85u_field` is called only afterward. Test monkeypatches
   `load_c85u_field` to fire if reached — a tampered freeze raises in verification
   first, proving C85U is not opened before selection is verified.
2. **Locked LURE population-total estimator (no self-normalization).** NLL =
   `(1/M) Σ v_m ℓ_m`; Jeffreys bAcc from estimated totals `N̂_y=(N/M)Σ v_m 1{y}`,
   `Ĉ_y=(N/M)Σ v_m 1{y}·correct`, `recall=(Ĉ+0.5)/(N̂+1)`; ECE =
   `Σ_b |(1/M) Σ v_m 1{bin=b}(correct−conf)|`. FULL (M=N, v=1) is exact. Test: the
   estimator is not self-normalized.
3. **Target-bound chain seeds.** `chain_seed = low64(SHA256(C86_ACTIVE_CHAIN_V1|
   dataset|subject|chain))` — different targets get independent streams; the same
   (target, chain) is shared across P0/A1/A2H as paired common random numbers.
4. **Indicator-first near-opt.** Per replicate: `1{ mean_8ctx raw_gap ≤ ε }`;
   target near-opt prob = mean over replicates; cohort = mean over targets — not a
   threshold on the replicate-averaged gap.
5. **Path separation.** C85U identities moved to `c85u_config` (imported only by
   D2 / lazily by `verify_c85u_identity`), so `core` and any D1 process hold no held
   path. D1 is a launcher that starts the sealed server and spawns a **path-blind
   worker** (a separate spawned process holding only the pipe + the client-visible
   pool). Tests: core has no C85U attribute; run_d1 does not import `c85u_config`.
6. **C86L acceptance replay in D1.** The launcher re-hashes every C86L field artifact
   against the accepted content-addressed manifest and binds the acceptance gate
   before any query.
7. **FULL positive control.** The composite pipeline is validated against C85U to
   0.0 error (formula), and D2 checks `full_acquisition_invariant` (P0/A1/A2H select
   identically at FULL). Note: no standalone historical *construction* composite
   artifact exists (C85U is evaluation-side), so the construction FULL reference is
   the exact deterministic recomputation, not a pre-stored table.
8. **Replicate/target-level table persisted.** `C86D_REPLICATE_TABLE.json`
   (target×method×budget×chain rows: std regret, raw gap, near-opt indicator) plus
   paired active−P0 Monte-Carlo SE per budget.
9. **Five-way development taxonomy** (`run_d2._classify`): CROSSED / WEAKENED /
   ACQUISITION_VIEW_NONTRANSPORTABLE / NO_REGISTERED_ACTIVE_GAIN with per-cohort
   mean/tail/near-opt gates + FULL ceiling. POLICY_LIMITED requires a separate
   oracle-acquisition diagnostic (not part of the {P0,A1,A2H} registry), noted in
   the manifest rather than silently emitted.
10. **Old raw-gap `HeldEvaluator` retired** to diagnostic-only (returns
    `target_raw_gap_diagnostic`, explicitly not the primary risk).

37 C86D tests pass. Corrected D1→D2 run performed only under a fresh `授权 C86D`.

---

## Round 3 — budget availability / A1 identity / freeze completeness / taxonomy

Third PM review; same C86D scope; tested (40 tests), NOT rerun.

1. **B32 INPUT_UNAVAILABLE (no min-budget substitution).** `budget_available`: a
   finite budget > physical pool is INPUT_UNAVAILABLE; D1 emits a status freeze with
   no selected action (never disguised as FULL); D2 excludes any budget with an
   unavailable cohort cell from the cross-cohort CROSSED/WEAKENED gate.
2. **A1 = locked mixture expected-NLL.** `s_A1 = Σ_y p̄(y)·(1/K)Σ_k(−log p_k(y))`
   with p̄ the equal-weight candidate mixture — not mean self-entropy. Test: matches
   the mixture formula and exceeds mean self-entropy under confident disagreement.
3. **Complete freeze verifier (before C85U).** `verify_freezes` now checks: exact
   count 3×n_targets×n_chains, each (method,target,chain) once, index==internal
   identity, seed==target-bound chain_seed, full/unique budget set, per available
   freeze exactly 8 canonical contexts, selected∈[0,80], query/q/weight/receipt
   lengths, no duplicate query trials, nested-prefix identity, composite length 81,
   selected==first argmax, and valid INPUT_UNAVAILABLE status.
4. **FULL ceiling gains a near-opt gate** (per cohort mean≤0.05 AND tail≤0.05 AND
   near-opt≥0.90). **Chain-level paired MC SE** (per chain: target-equal active−P0
   effect; SD/√n_chains) — named distinctly from target heterogeneity.
5. **Paired common random numbers.** P0 now uses the same sequential
   without-replacement loop as A1/A2H; a shared (target,chain) seed makes the first
   WARM_START picks identical across P0/A1/A2H. Test confirms it.
6. **FULL positive control per target×context×chain.** D2 reports
   `full_invariant_within_group` and `full_invariant_across_chains`.
7. **C86L acceptance replay requires `acceptance_ok=true` + the exact accepted gate**
   before any query.
8. **bAcc/NLL/ECE component SHA persisted** per context in each D1 freeze (composite
   alone is insufficient to diagnose component failure).

40 C86D tests pass. Corrected D1→D2 run only under a fresh `授权 C86D`.

---

## Round 4 — last-rerun fail-closed preflight

Fourth PM review; same C86D scope; tested (42 tests), NOT rerun.

1. **Truly path-blind worker.** The selection worker moved to
   `c86d/selection_worker.py`, which defines and imports NO oracle/contribution/
   field path. The launcher (`run_d1`) keeps the sealed paths and spawns the worker
   with only the pipe + the client-visible pool. Test: the worker module and its
   namespace contain no sealed path.
2. **Freeze verifier is now externally bound and blocking.** Before any C85U open,
   `verify_freezes` requires: the freeze target set == the accepted C86L target
   registry (derived from `C86L_CONTEXT_INDEX.json`), `n_targets==118`, method set
   `=={P0,A1,A2H}`, chain set `=={0..7}`, exact Cartesian count `2832`, exactly 5
   unique budget rows per freeze, each `q_m` finite in `(0,1]`, each LURE weight
   finite `≥0`, receipts equal to the query sequence with binary labels, FULL length
   == the target's physical pool size, `INPUT_UNAVAILABLE.pool_size` == that size,
   `component_sha_by_context` over exactly the 8 canonical contexts, each
   `target×context×chain` FULL carrying all three methods, and **FULL acquisition
   invariance (within group AND across chains) as a BLOCKING assertion** — any
   failure halts before `load_c85u_field()`.
3. **C86L acceptance count.** Replay now requires `len(output_artifact_hashes) ==
   1891` and `checked == 1891` (in addition to `acceptance_ok=true` + exact gate).

42 C86D tests pass. Expected next gate on acceptance:
`C86D_CORRECTED_LAST_DEVELOPMENT_RERUN_READY`; the corrected D1→D2 run then executes
only under a fresh `授权 C86D`.
