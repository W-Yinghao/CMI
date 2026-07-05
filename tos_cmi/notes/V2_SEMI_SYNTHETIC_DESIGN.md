# V2 semi-synthetic acceptance-power benchmark --- DESIGN LOCK

Status: **BLOCKED on a PM decision --- see [V2_ACCEPTANCE_CEILING_FINDING.md](V2_ACCEPTANCE_CEILING_FINDING.md).**
Scaffold written + adversarially reviewed (Worlds B/C, gate reuse, fair router, target-leakage CLEAN), but the
acceptance-power leg (World A) is **not constructible as a gate-ACCEPT test**: across 144 constructions on real
Lee/Cho latents, 0 accept, while 49 have a real target benefit the source-only gate cannot certify. This is a
genuine acceptance-CEILING result, not a bug. World A must be reframed before locking. Thresholds stay frozen.

## Why V2 exists

Track B + Phase 2 established **rejection/abstention power** on real EEG: the source-only gate refuses every
useless/harmful erasure (0 false-accepts, 8/8 harms prevented; task-preserving erasure preserves the deployed
task decision but is transfer-flat). What is still **untested is acceptance power**: *when a genuine beneficial
nuisance actually exists, does the same source-only gate ACCEPT it?* Real EEG has no positive, so we build a
semi-synthetic benchmark on **real EEG latents** with an **injected, ground-truth-controlled nuisance**, and
check the gate's decisions against known truth in three worlds:

* **World A** beneficial nuisance  -> gate should **ACCEPT**  (acceptance power)
* **World B** task-entangled nuisance -> gate should **REJECT** (safety)
* **World C** removable-but-useless -> gate should **ABSTAIN/REJECT** (mirrors real EEG; proves *domain-gain is
  evidence of erasure, not of benefit*)

## Key design decision: the erased domain D is the INJECTED nuisance z, not the real subject

We start from a frozen LOSO dump (real Z_source, y_source, real subject, held-out Z_target, y_target),
z-score the real block, and **append an m-dim nuisance block** `N = alpha * (2 z_i - 1) * u` (u a fixed unit
pattern) driven by an injected binary nuisance variable `z_i`. **The eraser erases D = z** (we pass z as the
"subject/domain" array to the eraser family). This is what makes a *source-LOSO-detectable* beneficial nuisance
constructible under a linear gate: z is linearly decodable (so LEACE/conditional/etc. can remove it), and we
control z's relationship to y and to the source->target shift.

Real subjects are still used for the **LOSO benefit grouping** (pseudo-target = a held-out source subject),
which is SEPARATE from the erased domain D=z. (The driver therefore threads two arrays: `z_dom` to erase,
`grp_subj` for leave-one-source-subject-out.)

## The three worlds (exact construction)

Let `sy = 2*y - 1`. Source subjects are split by reliability; `u = ones(m)/sqrt(m)`; isotropic noise 0.1.

### World A --- beneficial spurious nuisance (target should ACCEPT)
* A fraction `phi` of source subjects are **reversed**, the rest **aligned**:
  aligned subject: `z_i = y_i`; reversed subject: `z_i = 1 - y_i`.
  => pooled source corr(z, y) = (1 - 2*phi) > 0, so a linear head trained on `[Z, N]` **uses** N.
* **Target**: `z_i = 1 - y_i` (reversed) -> N is **misleading** at deployment.
* Ground truth: **beneficial** -- removing z (hence N) recovers the honest real-Z task signal on
  held-out/reversed subjects and on target.
* Why the gate can see it: LOSO over real subjects holds out reversed subjects where N mispredicts -> erasing
  z gives a positive held-out gain. Aligned subjects give a small loss; with moderate `phi` the **net**
  source-LOSO benefit LCB can exceed +0.01. `alpha` is tuned so within-source safety drop stays <= 0.02
  (real-Z carries the honest task; N is a modest add-on). At large `alpha`, N dominates in-distribution ->
  erasing hurts source -> gate correctly REJECTs (expected accept-rate falls with alpha).
* **Expected**: principled eraser (LEACE / fair-conditional) -> ACCEPT with actual target dbAcc LCB > +0.01.

### World B --- task-entangled nuisance (target should REJECT)
* `z_i = y_i` for ALL subjects and target (nuisance is **confounded with the true label**).
  Erasing z = erasing every direction linearly predictive of y -> **destroys the real-Z task carrier too**.
* Ground truth: **unsafe** -- source task collapses.
* **Expected**: every z-removing eraser -> source task-drop UCB > 0.02 -> **REJECT**; unsafe-accept must be 0.

### World C --- removable but useless (target should ABSTAIN/REJECT)
* `z_i` is **independent of y** (random Bernoulli), same in source and target.
  Erasing z removes N (high domain-gain / z-decode collapse) but N ⟂ y and real Z is untouched.
* Ground truth: **neutral** -- removable, no task benefit. Mirrors the real-EEG main result.
* **Expected**: safe (task-drop ~ 0) but source-LOSO benefit LCB <= +0.01 -> **ABSTAIN** (or REJECT via
  random-noise); **no ACCEPT**. Demonstrates *domain-gain is evidence of erasure, not of benefit*.

## Interventions (each erases D = z; uniform factory(Zf,yf,z_f,n_cls,seed)->apply(X))
identity, leace_baseline, tos_vd, rlace, inlp, tp_leace, alpha_leace(alpha_grid),
`fair_conditional_leace_disjoint_router`, random_k.
**`cc_leace_predicted_route_deployable` is NOT a clean deployable** (known tautology, Phase 2); it may appear
only as a labeled control, never as the acceptance-power method.

### fair_conditional_leace_disjoint_router (replaces the tautological cc-predicted)
Breaks the "probe re-learns the router" tautology by making the router **data-disjoint AND architecture-different**
from the evaluation probe:
1. split the source subset into disjoint halves R (router) and H (per-class erasers + downstream head);
2. router = a DIFFERENT estimator (LDA) fit on R only; per-class LEACE fit on R only (true R-labels);
3. apply(X) routes any X (source-H, target) by the R-trained LDA -> per-class LEACE; target true labels NEVER
   used for routing;
4. report **router accuracy** and **route-vs-probe disagreement** so the residual coupling is visible (a
   nonzero disagreement is what lets the transform move argmax and thus show benefit when one truly exists).

## Fixed thresholds (frozen; identical to Track B / Phase 2)
safety task-drop UCB <= 0.02 -> REJECT; source-LOSO benefit LCB > +0.01 -> ACCEPT; else ABSTAIN.
domain-gain = diagnostic only. target = audit only (enters ONLY the post-hoc scoring). NaN safety -> ABSTAIN.

## Correctness scoring
* World A: `correct` = principled eraser ACCEPT **and** actual target dbAcc LCB > +0.01; report
  **acceptance power** = accept-rate of the principled eraser vs n_source and alpha.
* World B: `correct` = no unsafe ACCEPT (every z-removing eraser REJECT); report **unsafe-accept rate**.
* World C: `correct` = no ACCEPT; report accept-rate (~0) with high domain-gain.

## Honest limitations (disclosed up front)
* This is **semi-synthetic**: real latents, injected nuisance. It tests the *decision machinery*, not a new
  real-EEG claim. It is method-deepening evidence, **not** a main-paper result.
* Under a strictly **linear** gate + head, a source-LOSO-detectable beneficial nuisance requires the nuisance's
  reliability to vary across source subjects (World A `phi`); a nuisance that is uniformly reliable across all
  source subjects but fails only at target is, by construction, **undetectable source-only** -- and the gate
  correctly ABSTAINs there (not a failure; the honest ceiling of source-only certification).
* If World A cannot produce any ACCEPT in smoke, we tune the **world generator** (`phi`, `alpha`, `m`,
  n_source), **never** the gate thresholds.

## Smoke test (Step 1; requires PM go)
`Lee2019_MI` & `Cho2017`, EEGNet, seed0, **first 5 folds**, all three worlds, all interventions, alpha_grid.
Pass iff: World A shows >= 1 ACCEPT; World B shows 0 unsafe-accept; World C shows 0 ACCEPT.
Outputs -> `tos_cmi/results/method_deepen/v2/v2_smoke_*`.

## Full benchmark (Step 2; requires PM go AFTER smoke passes)
datasets x backbones(EEGNet,TSMNet) x seeds(0,1,2) x n_source(8,16,32,all) x alpha_grid x interventions,
all folds. Main table + acceptance-power / unsafe-accept / target-gain figures.
