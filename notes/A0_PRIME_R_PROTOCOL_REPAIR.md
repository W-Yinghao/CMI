# A0′-R — PROTOCOL-REPAIR rerun (pre-registered 2026-06-21; repairs ONLY, no new analysis freedom)

The `66ea066` A0′ result (`ROLLBACK_ELIGIBILITY_ONLY`) is **PROVISIONAL — protocol-invalidated by outcome-conditioned
aggregation**. A0′-R repairs the implementation and re-runs. It adds **no** score / generator / severity / direction
/ threshold / admissibility freedom. The `66ea066` outputs are kept as-is; this is a separate artifact.

## P0 — target-label leakage in the deployment-facing batch score (the invalidating bug)
`a0_prime.py` computed the batch rollback score as `post[score][base_correct].mean()`, and `base_correct =
base.argmax(1) == y_target`. So the controller score aggregated ONLY over samples the outcome later showed base got
right — a target-label-conditioned score. It also deleted batches by `len(set(y_target)) < 2` (label-based) instead
of an identity fallback. The frozen protocol requires `S_B = (1/B) Σ_{i∈B} s_i` over the **whole** label-blind
batch, with the scoring API reading only the serialized state + unlabeled target. Using `base_correct` to DEFINE the
`correct→wrong` harm TARGET is legitimate; using it inside the deployed SCORE is not.

## Repairs (exhaustive; nothing else changes)
1. **Whole-batch label-blind score:** `S_B = post[score].mean()` over the entire natural batch. No `base_correct`,
   no `y_target`, anywhere in scoring/batching/fallback.
2. **Two-phase computation:** phase-1 (frozen, y-free) computes scores, `S_B`, batch eligibility, fallback decision;
   phase-2 uses `y_target` ONLY for evaluation endpoints (Δℓ, harm-flip, base/adapt error).
3. **No label-based batch deletion:** identity fallback (adapt := base) on a **label-blind** condition (batch size
   `< 8`); score + outcome are STILL recorded for fallback batches.
4. **Metamorphic guard (hard gate):** permuting `y_target` must leave every score, `S_B`, batch-inclusion, fallback
   flag, and rollback decision **bit-identical**. Asserted in code; failure aborts with no output.
5. **Stable seeds:** SHA-256-derived integer per (cohort, family, severity, realization); every realized seed
   written to the manifest. (No Python `hash()` / `PYTHONHASHSEED` dependence.)
6. **Serialized source state:** frozen probe + source mean/cov + priors (`fit_source_state`); base = probe(z),
   adapted = `pmct_predict_serialized(ref='pooled')`. VERIFY GATE: reproduces `transduct_predict(matched_coral,
   shrink=0.1)` to < 1e-9 per fold (measured bit-exact, |d|=0). No raw source examples at scoring.
7. **Full pre-registered metrics:** Spearman + **C-index** + AUROC[meanΔℓ>0] (batch); harm-flip AUROC + **AUPRC**
   (sample); cohort/family macro; adaptation-specificity (base_err on FULL batch vs harm_flip on base-correct).
8. **Correct 0.03 admissibility:** within EQ=0.03 of the best score **at the same level AND same disease** (not a
   single global best), with consistent direction across PD/SCZ, LOGFO, and the cohort-macro robustness flag.
9. **`cmi` renamed `pr_cmi_proxy`** (prototype-assignment vs readout disagreement × class-distance margin). It is NOT
   `_decoder_cmi_residual` (a label-using source-side statistic). Mechanism claim is scoped accordingly:
   *source-support and the prototype–readout CMI proxy are anti-aligned with adaptation harm* — not all CMI.
10. **Determinism:** the full computation runs TWICE; the canonical summary hash must be identical, else abort.

FROZEN, unchanged from A0′: the four scores + directions, three generators + severity grids, B=32 recording-order
batches, EQ-margin 0.03, the 4-way decision (SINGLE_SCALAR / TWO_LEVEL / POST_ALIGN_ABSTENTION_ONLY /
ROLLBACK_ELIGIBILITY_ONLY / DIAGNOSTIC_ONLY), UNSEEN seed family (now SHA-256), disease-stratified within-disease
LOCO × held-out-generator-family.

## Decision (no re-searching of aggregation/scores under any outcome)
- repaired signal still passes PD, SCZ, LOGFO, cohort-macro → **ROLLBACK_ELIGIBILITY_ONLY (confirmed)** → only then
  pre-register the closed-loop pilot.
- repaired signal disappears or direction unstable → the `66ea066` rollback positive was an outcome-conditioned lens
  artifact → **DIAGNOSTIC_ONLY**.
- Either way: no new aggregation, no new score. Deployment control never uses target labels or source examples;
  TUAB stays sealed until the very end.

## Output (immutable)
```
results/a0_prime_r/<freeze_hash16>/
  a0primer_summary.json   # 4-way decision; metamorphic-guard PASS; serialized-state verify residual; per-disease
                          # within-batch (AUROC+AUPRC) & batch-mean (Spearman+C-index+AUROC) tables; per-family
                          # direction; adaptation-specificity; selective-risk; SHA seed map; double-run hash match
  run_manifest.json
```
