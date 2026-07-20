# A0-PILOT — closed-loop sample-abstention pilot (PRE-REGISTERED, frozen 2026-06-21; run BEFORE any Freeze B)

Tests whether **post-alignment `s_sep` selective abstention** (the only A0′-R survivor) actually reduces deployed
loss vs a matched random-abstention budget — the closed-loop control experiment. NO new gate / score / generator /
severity / coverage / threshold is developed. Multi-seed-block confirmation is EMBEDDED (4 fixed new SHA blocks run
at once), so no open-ended seed sweep. Deployment control uses NO target labels and NO source examples (serialized
state). TUAB / formal P2 / other method dev stay closed.

## Frozen substrate (identical to A0′-R)
`erm:0` = CITA-no-LPC deployment encoder; serialized source-free state (frozen probe + source moments + priors;
bit-exact vs transduct, gated over ALL 7 cohorts). Same CORAL (matched_coral, shrink=0.1, ref=pooled), same score
formulas + directions (`g_unc`, `s_support`, `s_sep`, `pr_cmi_proxy`), B=32 recording-order batches, same generators
(`lowmargin_rot`, `highmargin_cbw`) + severity grids, same controls (`clean`, `covariate_shift_beneficial`).

## Seed blocks (variance estimation ONLY — NOT extra cohorts)
Four FIXED new SHA namespaces, 5 realizations each, ALL run in one invocation (no block added on intermediate
results): `A0PILOT_V1/block_{0,1,2,3}`. Per-cell seed = `sha256("A0PILOT_V1|block_{b}|{cohort}|{fam}|{sev}|{real}")`.
Blocks estimate generator-realization variance; they do NOT increase the independent-cohort count (7).

## Action arms (per batch, after tentative adaptation)
- `BASE` — no adaptation (probe on raw z).
- `ALWAYS_ADAPT` — matched-CORAL adapt, keep all.
- `ADAPT_RANDOM_ABSTAIN` — adapt, abstain a random matched budget.
- `ADAPT_GUNC_ABSTAIN` — adapt, abstain top-k by post-alignment `g_unc`.
- `ADAPT_SSEP_ABSTAIN` — adapt, abstain top-k by post-alignment `s_sep`  (**PRIMARY controller**).
- `ORACLE_HARM_ABSTAIN` — adapt, abstain the actual harmful flips (reference ceiling only; not deployable).
- `BASE_WITH_SSEP_MASK` — diagnostic only: apply the same s_sep mask to BASE predictions (is the mask just picking
  generically-hard samples?). NOT a candidate deployment method.

**Abstention budget (frozen):** within-batch rank, `k_b = floor(0.20 * n_b)`; the SAME `k_b` for every score arm and
the random arm. At B=32 this retains 81.25% → report **achieved coverage**, never "exactly 80%".

**Metamorphic guard (hard gate):** for every NON-oracle arm, permuting `y_target` must leave the abstention mask,
the achieved coverage, and the retained predictions **bit-identical**. Failure aborts with no output. (Scoring and
masking are label-blind; y is used ONLY to evaluate endpoints.)

## Primary endpoints — `ADAPT_SSEP_ABSTAIN` vs `ADAPT_RANDOM_ABSTAIN` at matched coverage
On the RETAINED set (and harm decomposition on retained base-correct samples):
- retained-sample NLL; selective 0–1 risk;
- harmful-flip rate among retained base-correct;
- **prevented** harmful flips (c→w abstained); **lost** beneficial flips (w→c abstained);
- **net protection = prevented − lost**.
Macro-averaged **batch → generator-family → cohort**; PD and SCZ reported **separately**; plus the LOCO breakdown
and the distribution across the four seed blocks.

## Freeze-B decision — enter ONLY IF ALL hold (else `DIAGNOSTIC_ONLY`, no further search)
1. `ADAPT_SSEP_ABSTAIN` improves selective NLL AND 0–1 risk vs `ADAPT_RANDOM_ABSTAIN` on **both** PD and SCZ;
2. neither harm family (`lowmargin_rot`, `highmargin_cbw`) reverses direction;
3. the LOCO breakdown is not driven by a single cohort (improvement holds with each cohort left out);
4. no systematic direction reversal across the four seed blocks;
5. on `clean` and `covariate_shift_beneficial`, s_sep abstention causes **net harm ≤ 0.02/batch** (matched coverage)
   and does not worsen retained-NLL beyond +0.02 vs no-abstain (the no-false-veto tolerance, frozen);
6. `s_sep` is not substantially worse than `g_unc`'s best matched-coverage result (net protection within 0.03).

Pass → freeze **`CITA-no-LPC + post-alignment s_sep selective abstention`** (NOT a pre-adaptation gate, NOT batch
rollback, NOT a CMI/density harm gate). Fail → fix **`DIAGNOSTIC_ONLY`**; do not search score / coverage / seed.

## Mechanism wording (frozen, scoped)
base-error AUROC ~0.52–0.53 ⇒ `pr_cmi_proxy`/`s_support` are **weakly aligned with base difficulty** (NOT "predict
base difficulty"); the robust claim is only that they are **wrong-direction for adaptation harm**, while
post-alignment separability is the single candidate worth this control experiment.

## Output (immutable, committed for verifiability)
```
results/a0_pilot/<freeze_hash16>/
  a0pilot_summary.json   # per-arm per-disease endpoints; matched-coverage tables; net-protection; seed-block & LOCO
                         # distributions; metamorphic-guard PASS; serialized-equiv (all cohorts); Freeze-B gate booleans
  run_manifest.json      # SHA seed map (all 4 blocks), pre-registration hash, dump hashes
```
