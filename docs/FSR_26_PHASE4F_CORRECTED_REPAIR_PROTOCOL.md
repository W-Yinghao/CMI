# FSR_26 — Phase 4F: Corrected Confirmatory Test of First-Moment Token Neutralization (pre-registration)

**Project FSR — Phase 4F.** A **new confirmatory experiment** (not a re-scoring of Phase 4E, which stays
`repair_claim_level = none`, frozen). Phase 4E surfaced the project's first within-scope, task-safe, non-erasure
repair **signal** (E4 full first-moment mean alignment) but was blocked by a comparator gate that let a
task-destructive ERASE arm veto E4 via netted recovery. Phase 4F **corrects the comparator eligibility rule** and
**re-confirms on fresh token seeds**. CPU-only on the frozen 4B dumps; no GPU/retrain/CMI/fbdualpc/target-fit.
**Frozen before the result run + design-red-teamed.**

## Core question (narrow, by design)
> After correcting comparator eligibility, can **E4 full first-moment mean alignment** be **certified** — on
> **fresh** token seeds — as a **task-safe, target-unlabeled, branch-local** repair primitive for a **controlled
> constant-offset** shortcut?

A pass certifies only: *FSR can verify, localize, attribute, and repair a controlled first-moment branch-local
shortcut using a deployable target-X-only first-moment neutralizer.* It does **not** certify general shortcut
repair (the injection is first-moment by construction; E4 is first-moment; scope is deliberately narrow).

## Fresh confirm seeds (frozen)
```
CONFIRM_SEEDS_4F = [20260721,20260722,20260723,20260724,20260725,20260726,20260727,20260728]  # 8, NEW
```
Phase 4E seeds (0, 20260707/8/9) are **development/support only** and do **not** enter the Phase 4F confirmatory
claim (STOP-1). **8** seeds (raised from 5 per design red-team) to power the leave-one-seed-out test, since 4E's
`E4 > E3` was pool-only and crossed 0 under single-seed removal. `DEV_SEED = 0` for mechanism only.

## Statistics (frozen before the run — design-red-team fix)
The 21 LOSO folds are the **same** `(dataset, target_subject)` clusters repeated across seeds (only the injection
token differs by seed), so an iid bootstrap over seed-rows understates variance ~√(#seeds). **All gate CIs use a
CLUSTERED bootstrap resampling whole `(dataset, target_subject)` folds** (not seed-rows). Gate decisions use the
**clustered CI lower bound**, not the point estimate.

## Arms + primary
Same operators as Phase 4E (FSR_24): E0 exact (oracle bound), **E4 full-space first-moment mean alignment
(PRIMARY, deployable)**, E1 subspace-restricted (SECONDARY: "does source-subspace restriction add anything over
full centering?" — 4E showed E1−E4 < 0), E2 marginalization (exploratory), E3 random-subspace centering
(control), ERASE subspace erasure (control, expected ineligible). All target-X-only + source; target labels
score only. Verdict = CONFIRM-seed aggregate, at source-selected `alpha_star`, **NETTED** (token-specific =
injected-effect − clean-effect).

## α + firewall (unchanged)
`alpha_star` = source-only stress rule (smallest α with source-heldout class-directed logit shift ≥
`FRAC=1.0`×median source margin); full α grid reported as diagnostic, primary claim at `alpha_star`. **Firewall:**
target labels only for final scoring — never token construction, α, or `k/λ` selection. The **comparator veto set
is structural** (`{E1,E3}` vs `{ERASE}`, no target labels); only the one-sided negative-control *diagnostic* reads
final-eval bAcc against frozen thresholds (same category as scoring E4; can only pause, never relax).

## CORRECTED comparator veto set (the Phase 4F fix) — STRUCTURAL, not target-scored
Design-red-team correction (wnhbjp2rt): the veto set is **structural and pre-declared**, so no target-label
decision selects which comparators bind E4.
```
E4 must beat the STRUCTURAL veto set = {E1, E3}   (eligible comparators by construction)
NEGATIVE_CONTROLS = {ERASE}   — excluded by CONSTRUCTION: ERASE erases P_S, i.e. it removes variance and is
                                task-destructive by design; it is a negative control, not a repair competitor.
```
**Provable no-exclusion guarantee.** A *diagnostic* validity test (target FINAL-EVAL bAcc vs frozen thresholds:
`raw_recovery>0` AND `clean_drop≤SAFE_DROP=0.01` AND not `regression_to_floor`) is reported per arm, but it does
**not** decide the veto set. It cannot wrongly exclude a genuinely better repair, because **`clean_drop≤0.01`
implies `cb ≥ orig−0.01 > orig−0.02`, which is the negation of `regression_to_floor` (`cb < orig−0.02`)** — so
`regression_to_floor` is a non-load-bearing sanity flag, and the only operative disqualifiers (`raw≤0`,
`clean_drop>0.01`) are exactly the negation of "valid task-safe repair." Any arm that *is* a valid repair
(including a hypothetical one beating E4) stays in the veto set and can bind E4 — **E1 is in {E1,E3} and its
veto still holds** (removing target-scored eligibility cannot promote E4). E4 is held to the **identical**
`raw>0 + clean_drop≤0.01 + not reg_floor` bar (its task-safety gate); the diagnostic is symmetric.

**One-sided falsification guard.** The target-scored diagnostic is used only to *falsify* the pre-declaration: if
a NEGATIVE_CONTROL (ERASE) scores as a valid repair on the fresh seeds, the verdict is **paused** (`level=none`,
re-open protocol) — the guard can pause, never relax, E4's bar. ERASE's diagnostic is **recomputed** on fresh
seeds (not assumed). This is the same measurement→control discipline as TOS (erasable ≠ beneficial).

**Firewall labeling (honest):** `comparator_veto_set_used_target=false` (the veto set {E1,E3} vs {ERASE} is
structural); `negative_control_diagnostic_uses_target=true` (the one-sided pause guard reads final-eval bAcc,
the same category as scoring E4 itself); `eligibility_rule_preregistered=true`.

## Pass / fail (pre-registered)
All quantities pooled over the 8 CONFIRM seeds × 21 folds, at `alpha_star`, NETTED. Constants `SAFE_DROP=0.01`,
`DELTA=0.02`, `HARM_FLOOR=0.02`.

1. **Harm gate.** Pooled harm ≥ `HARM_FLOOR` AND **clustered** CI lower > 0. Else `none` (`harm_not_established`).
2. **E4 task-safety gate.** `E4 clean_target_drop ≤ 0.01` AND `E4 raw_recovery > 0` AND `bAcc_repaired_injected >
   bAcc_injected` AND E4 not `regression_to_floor` (symmetric with comparators) AND firewall clean. Fail → `none`.
3. **E4 specificity gate.** `E4_netted_gain − E3_netted_gain ≥ DELTA` AND **clustered** CI lower > 0 (bAcc units).
4. **E4 comparator gate.** E4 must beat every member of the structural veto set {E1, E3} on netted gain
   (point > 0). Negative controls {ERASE} are a diagnostic table only, never vetoes.
5. **Falsification guard.** If a NEGATIVE_CONTROL scores as a valid repair on fresh seeds → `none` (re-open).

**Grades** (magnitude thresholds are point estimates per PM; nonzero-ness is the **clustered CI lower bound**):
```
partial_ok := harm established AND E4 task-safe AND E4 beats E3 (>= DELTA AND clustered CI_lo>0)
              AND E4 beats {E1,E3} AND NOT neg_control_falsified
              AND E4 netted recovery > 0.30 AND E4 netted clustered CI_lo > 0
              AND per_dataset sign-consistent (descriptive, N=2) AND drop-anti-harm E4 netted > 0.30
partial:  partial_ok AND NOT strong
strong:   partial_ok AND E4 netted recovery >= 0.50 AND leave-one-seed-out robust
          (EVERY LOSO cut keeps E4-E3 clustered CI_lo > 0 — no point-estimate escape)
none:     any gate above fails.
```
`strong` is a **strict superset** of `partial`. Every claim reports **all-seed pooled + leave-one-seed-out
(clustered, min-cut margin) + per-dataset (2014a/2015, descriptive) + drop-anti-harm**. Naive and clustered CIs
are both reported; gating is on the **clustered** CI.

## Outputs (`results/fsr_phase4f_corrected_repair/`) — requires running BOTH the harness THEN the aggregator
```
# harness (run_phase4f_corrected_repair.py):
phase4f_manifest.csv               # per fold/seed: token_seed(+is_confirm4f), alpha_star, k*, λ*, margin/scale
phase4f_token_centering_results.csv# per fold/seed/alpha: E0/E1/E2/E3/E4/ERASE injected+clean bAcc, raw+netted
phase4f_mechanism_capture.csv      # diagnostic: cos(u_tsub,S), captured_fraction
phase4f_source_heldout_selection.csv # diagnostic: (k,λ) source-heldout netted recovery
phase4f_alpha_rule.csv             # diagnostic: source-heldout shift vs FRAC*margin, alpha_star
phase4f_target_label_firewall.json
# aggregator (aggregate_phase4f_verdict.py):
phase4f_comparator_eligibility.csv # per arm DIAGNOSTIC: raw_recovery, clean_drop, regression_to_floor, valid_repair
phase4f_clean_target_netting.csv   # per fold/seed: E4/E1 on clean vs injected -> token-specific netted
phase4f_random_controls.csv        # E3/ERASE detail
phase4f_leave_one_seed_out.csv     # E4-E3 margin + clustered CI dropping each confirm seed
phase4f_per_dataset_summary.csv    # BNCI2014_001 vs BNCI2015_001 (descriptive, N=2)
phase4f_verdict.json
```
`phase4f_verdict.json` (key fields):
```json
{"fresh_confirm_seeds": [20260721,"...",20260728], "uses_phase4e_seeds_for_claim": false,
 "n_folds_clusters": null, "harm_established": null,
 "injection_harm_denominator": null, "injection_harm_denominator_ci_clustered": [null,null],
 "primary_repair": "E4_full_mean_alignment",
 "e4_task_safe": null, "e4_clean_target_drop": null, "e4_raw_recovery": null, "e4_regression_to_floor": false,
 "e4_netted_recovery": null, "e4_netted_recovery_ci_clustered": [null,null], "e4_netted_ratio_dropped_frac": null,
 "e4_minus_e3_netted_gain": null, "e4_minus_e3_ci_clustered": [null,null], "e4_beats_e3": null,
 "veto_set_structural": ["E1","E3"], "negative_controls": ["ERASE"], "negative_control_falsified": false,
 "e4_beats_eligible_comparators": null, "e1_minus_e4_netted_gain": null,
 "leave_one_seed_out_pass": null, "loso_min_margin": null,
 "per_dataset_sign_consistent": null, "drop_anti_harm_e4_netted": null, "drop_anti_harm_ok": null,
 "comparator_veto_set_used_target": false, "negative_control_diagnostic_uses_target": true,
 "eligibility_rule_preregistered": true, "target_labels_used_for_fit": false,
 "target_labels_used_for_selection": false, "target_labels_used_for_final_eval_only": true,
 "repair_claim_level": "none|partial|strong", "pc2_gpu_gate": "paused|eligible_for_protocol_update|eligible"}
```

## STOP rules
```text
1  Phase 4E seeds (0/20260707/8/9) used for the Phase 4F confirmatory claim (they are dev/support only).
2  target labels used for fit / token / alpha / k-λ selection, OR a TARGET-TUNED comparator veto rule/threshold.
   (The structural veto set {E1,E3} vs {ERASE} uses NO target labels; the one-sided negative-control
   falsification DIAGNOSTIC may read final-eval bAcc against FROZEN thresholds — same category as scoring E4 —
   and can only PAUSE, never relax, E4's bar.)
3  alpha or seed chosen by target bAcc/harm; headline at any alpha other than source-selected alpha_star.
4  a task-DESTRUCTIVE arm (raw_recovery<=0 or clean_drop>SAFE_DROP or regression_to_floor) allowed to VETO E4.
5  E4 credited without the clean-target netting (raw-only claim), or without reporting LOSO + per-dataset.
6  Phase 4F verdict presented as overturning / re-scoring Phase 4E (4E stays none, frozen).
7  GPU / retrain / CMI / fbdualpc / new-model-training / hyperparameter sweep beyond the frozen k∈{1,2,4},λ∈{.5,1}.
8  CLAIM-LOCK: results written as (i) natural shortcut, (ii) DG method / SOTA, (iii) general shortcut repair
   (scope is controlled first-moment constant-offset only), or (iv) "E4 passed" when level < the grade earned.
```

## PC2 (unchanged posture)
`pc2_gpu_gate` starts `paused`. If Phase 4F E4 ≥ **partial** → gate becomes `eligible_for_protocol_update`: a
**PC2-E4 preflight update** (successor to FSR_23, which is bound to the dead Phase-4D D1 primitive) is drafted
for PM review — **no GPU run**. PC2 GPU runs only after the updated PC2-E4 preflight passes PM review.

## Framing (fixed)
Phase 4F is a corrected confirmatory test of a deployable first-moment token neutralizer on an **injected**
constant-offset positive control. Any pass certifies only controlled-first-moment repair, netted against generic
TTA, task-safe, on fresh seeds — never general repair, a DG method, SOTA, or a natural-harm claim. Phase 4E's
`none` is untouched.
