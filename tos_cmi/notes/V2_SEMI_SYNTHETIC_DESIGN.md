# V2 semi-synthetic source-only acceptance ceiling --- DESIGN LOCK

Status: **LOCKED after Option 1.** V2 does **not** claim the gate can accept a beneficial erasure; it
demonstrates a source-only **ceiling**: target-beneficial deployment-shift erasures can exist, but when the
benefit is absent from source-domain evidence, a strict source-only gate cannot safely certify ACCEPT -- the
correct action stays reject/abstain (see [V2_ACCEPTANCE_CEILING_FINDING.md](V2_ACCEPTANCE_CEILING_FINDING.md)
for the 144-cell search that established this). Gate thresholds frozen; only world-generation params tune in smoke.

* **World A** target-beneficial but source-uncertifiable -> expected **REJECT/ABSTAIN** (the ceiling; NOT accept).
* **World B** task-entangled unsafe -> expected **REJECT** (safety).
* **World C** removable but useless -> expected **REJECT/ABSTAIN** (domain-gain is evidence of erasure, not benefit).

## Why V2 exists

Track B + Phase 2 established **rejection/abstention power** on real EEG: the source-only gate refuses every
useless/harmful erasure (0 false-accepts, 8/8 harms prevented; task-preserving erasure preserves the deployed
task decision but is transfer-flat). V2 asks the complementary question and answers it as a **limit result**:
*even when a genuine target-beneficial erasure exists, can a source-only gate certify it?* No -- when the
benefit is created by a source->target shift not represented in the source domains, source-only evidence is
insufficient (Proposition below). We show this on a semi-synthetic benchmark built on **real EEG latents** with
an **injected, ground-truth-controlled nuisance**, across three worlds:

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

### World A --- target-beneficial but SOURCE-UNCERTIFIABLE (the ceiling; gate should NOT accept)
Reframed after the design review + 144-cell search proved no injected nuisance yields a source-LOSO ACCEPT
(see [V2_ACCEPTANCE_CEILING_FINDING.md](V2_ACCEPTANCE_CEILING_FINDING.md)). World A now DEMONSTRATES the ceiling.
* Variant `aligned_noise_flip`: a MINORITY (`f_align`, default 0.15) of source subjects carry the spurious
  shortcut `z_i = y_i`; for the MAJORITY `z_i` is **noise** (independent of y). Pooled corr(z,y)=`f_align`>0,
  so the head USES the nuisance direction u; for the noise-majority the head injects `w_u*noise`.
* **Target**: `z_i = 1 - y_i` (reversed) -> N is **misleading** at deployment. An ORACLE nuisance eraser
  (zeros the injected block) recovers the honest real-Z signal -> **real positive target gain**
  (empirically target dbAcc +0.06..+0.10 in the safe regime, up to +0.46 in the unsafe regime).
* **Ground truth: target-beneficial but source-uncertifiable.** The benefit lives in the source->target shift.
  Held-out SOURCE subjects share the source shortcut, so the source-LOSO benefit LCB is <= 0 (empirically
  negative, e.g. -0.05) even for the SAFE cells. The source-only gate therefore correctly does **not** ACCEPT.
* **Expected**: principled erasers -> REJECT (unsafe cells) or ABSTAIN (safe cells); ACCEPT count = 0. The
  oracle diagnostic shows the target gain exists; random-k does not reproduce it. This is a CEILING, not a pass.

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

## Proposition (source-only non-identifiability of deployment-shift benefit)
Let a gate `g` depend only on source data `S` (drawn from source subjects/domains `P_S(Z,Y,D)`). Consider two
worlds `W+` and `W-` with **identical source distribution** `P_S(Z,Y,D)` -- hence identical gate inputs and
identical source-LOSO statistics -- but different target laws: in `W+`, erasing nuisance `z` **reduces** target
risk; in `W-`, the same erasure is **neutral or harmful** on target. Because `g(S)` sees the same inputs in
both, any source-only rule that ACCEPTs in `W+` also ACCEPTs in `W-` (a false/unsafe accept); any rule that
avoids false accepts across both must REJECT/ABSTAIN in `W+`.

**Therefore: target-beneficial deployment-shift erasure is NOT source-only identifiable** -- *without source
domains that already represent the target shift, and without target information*, source-only certification
cannot license deployment-shift benefit. (This is a conditional non-identifiability statement, not a claim that
source-only benefit is impossible in general; if the source domains encode the shift, or target information is
admitted, the ceiling can be crossed -- see the parked target-informed branch.) World A is the empirical
witness: real target gain exists (oracle), yet the source-only inputs are identical to a no-benefit world, so
the gate abstains/rejects.

## Interventions (each erases D = z; uniform factory(Zf,yf,z_f,n_cls,seed)->apply(X))
identity, random_k, leace_baseline, rlace, tos_vd, inlp, tp_leace, alpha_leace,
`fair_conditional_leace_disjoint_router`, and `oracle_nuisance_eraser_DIAGNOSTIC_ONLY` (removes the injected
nuisance block via ground-truth dims; **NOT deployable** -- shows the target gain exists).
**`cc_leace_predicted_route_deployable` is EXCLUDED** (known tautology, Phase 2). `oracle_label_router` (Phase 2
true-label routing) is parked, not run in V2. Oracle paths are DIAGNOSTIC and never counted as a method result.

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

## Correctness scoring (ceiling framing; NO world expects ACCEPT)
* World A: `correct` = a SAFE cell with real target gain (target dbAcc LCB>+0.01) exists that the gate does
  NOT accept, its source-LOSO benefit LCB<=+0.01, and random-k does not reproduce the oracle target gain.
* World B: `correct` = no unsafe ACCEPT (every z-removing eraser REJECT); report **unsafe-accept rate** (want 0).
* World C: `correct` = no ACCEPT and a high-domain-gain-but-useless cell exists (domain-gain != benefit).

## Honest limitations (disclosed up front)
* This is **semi-synthetic**: real latents, injected nuisance. It tests the *decision machinery* / the source-
  only ceiling, not a new real-EEG claim. Method-deepening evidence, **not** a main-paper result.
* The ceiling is a **conditional** non-identifiability statement (see Proposition): it holds when the source
  domains do not encode the target shift and no target information is admitted. It is NOT a claim that erasure
  benefit is impossible in every setting.
* Gate thresholds are **never** tuned; only world-generation params (`variantA`, `f_align`, `alpha`, `m`,
  n_source) may be adjusted in smoke.

## Smoke test (Step 1; requires PM go) --- CEILING smoke
`Lee2019_MI` & `Cho2017`, EEGNet, seed0, **first 5 folds**, all three worlds, all interventions, alpha_grid.
Pass iff (per the ceiling scoring above): World A has a safe target-beneficial cell the gate does NOT accept
(source benefit LCB<=+0.01, random-k does not reproduce oracle gain); World B 0 unsafe-accept; World C 0 accept
with a high-domain-gain-useless cell. Outputs -> `tos_cmi/results/method_deepen/v2/v2_smoke_*` (+ scatter PNG).

## Full benchmark (Step 2; requires PM go AFTER smoke passes)
datasets x backbones(EEGNet,TSMNet) x seeds(0,1,2) x n_source(8,16,32,all) x alpha_grid x interventions,
all folds. Main table + the ceiling scatter (source-LOSO benefit LCB vs actual target dbAcc LCB, colored by
gate action) + naive-controller table.

## Artifact safety (NOT reviewer-safe as-is)
The method-branch sbatch scripts (`scripts/tos_v2_*.sbatch`, `scripts/tos_phase2*.sbatch`,
`scripts/tos_trackB.sbatch`, etc.) contain **local absolute paths and identity** (`/home/infres/yinwang/...`,
`/projects/...`, a conda-env path). This is acceptable on the internal method branch but is **NOT artifact-safe**
-- **scrub before any reviewer-facing package** (anonymous artifact = aggregates-only tarball; never these
scripts, never a GitHub URL, never identity/PDF-metadata leaks).

## Parked branch (NOT part of the present V2 claim): target-informed acceptance
A target-informed branch could admit **unlabeled target** (does erasing reduce source-target mismatch while
preserving source task?), a **few labeled target** trials (few-shot benefit probe), or target calibration data
to license erasure, and quantify *how much target information crosses the source-only ceiling*. That is not
strict source-only DG and is deferred; it connects to EA-transductive / SCPS. Not implemented now.
