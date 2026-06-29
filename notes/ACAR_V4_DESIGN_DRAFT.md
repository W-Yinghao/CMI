# ACAR v4 — DESIGN DRAFT (CURB: Control-First Utility/Risk-Budgeted Adaptation Routing)

```
STATUS         : NON-BINDING
LINEAGE        : POST-V3 DEV_STOP
LOCKBOX        : NO LOCKBOX ACCESS
EXTERNAL ARM   : NO EXTERNAL ARM (Arm B not authorized)
THIS FILE      : NOT ACAR_FROZEN_v4.md  (design draft only; freezes nothing, selects nothing)
DATE           : 2026-06-29  (post-v3 hypothesis generation)
```

This is a **post-v3 design draft**, not an amendment to v3 and not a re-threshold of v3. ACAR v3 returned a
**terminal** development result `DEV_STOP / NO_LOCKBOX_CONSUMED` (`acar-v3-dev-design-v1 @ 817b04f`; result `9f4e83f`,
tag `acar-v3-dev-run002-dev-stop`). v4 reuses v3's engineering substrate (loader, leakage/provenance machinery,
execution cache, subject-split logic, the frozen estimand and action set) but **must not** modify any v3 module, the v3
protocol commit/tag, or any v3 result. Nothing here consumes the held-out lockbox or approaches external Arm B; the old
seven cohorts are **development-only** and any v4 number computed on them is exploratory / model-selection, never
external validation.

---

## 1. Status and lineage

| stage | tag / commit | status | one-line |
|-------|--------------|--------|----------|
| A0 / A0′ gate-falsification | (exp/lpc-cmi) | DIAGNOSTIC_ONLY (closed) | no source-free harm controller reduced deployed loss; density/CMI wrong-signed; rollback was label leakage |
| ACAR v2 | `acar-v2-protocol @ 9b2f0c1`; result `1528a94` | MEASUREMENT_ONLY | label-free action-conditional features predict negative transfer (G1✓); router not deployable (G2✗) |
| ACAR v3 (HSCR) | `acar-v3-dev-design-v1 @ 817b04f`; result `9f4e83f` | DEV_STOP / NO_LOCKBOX_CONSUMED | stricter pre-registered redesign fails the development S2/S4 gate (coverage collapse + weak PD center) |
| **ACAR v4 (CURB)** | (this draft — no commit/tag yet) | **NON-BINDING DESIGN DRAFT** | control-first: calibrate the *deployed policy's* risk directly instead of upper-bounding every action |

v4 is **new, post-v3, non-binding**. v2 is `MEASUREMENT_ONLY`; v3 is `DEV_STOP / NO_LOCKBOX_CONSUMED`. v4 is not an
amendment to v3. Old seven cohorts = DEV only. Held-out lockbox remains sealed. External Arm B is not authorized.

---

## 2. Scientific question

> **Can we learn and calibrate a selective deployment policy that adapts as much as possible *within a risk budget* —
> rather than upper-bounding the incremental risk of every candidate action simultaneously?**

v3's failure mechanism is now understood (`notes/ACAR_V3_DEV_RUN_002_RESULT.md`): to cover **every** batch × **every**
action simultaneously, the subject-clustered joint conformal radius `q` is pushed so large relative to `|ΔR|` that the
admit rule `U_a = point + q < −δ` almost never fires. Result: adaptation coverage ~0.6–1.1 % (≪ the 15 % floor), the
router abstains to identity, and on PD the center is a weak harm predictor (center-AUROC 0.525–0.570). The harm *signal*
is real (SCZ harm-AUROC 0.68–0.74), but it does not become usable coverage.

v4's hypothesis: the problem is the **control object**, not (only) the predictor. We replace the simultaneous-upper-bound
router with a **risk-budgeted selective policy** and calibrate the risk of the policy *that is actually executed*. We
also build an **information-limit audit** that decomposes any residual failure into an information gap, a policy-learning
gap, and a calibration gap — so a v4 negative result is *diagnosed*, not just declared.

---

## 3. Fixed substrate (inherited from v2/v3, unchanged)

- **Estimand (unchanged).** Action-conditional paired incremental risk
  `ΔR_a(B) = R_B(f_a) − R_B(f_0)`, NLL risk, paired pre→post on the same batch `B`, label-free at deployment.
  Sign convention (frozen): **ΔR_a(B) < 0 means action `a` reduced risk on `B` (good)**; identity is `f_0` with
  `ΔR_identity ≡ 0`. Deployed **reduction** of a policy `π` is `red(π) = −mean_B ΔR_{π(B)}(B)` (positive = good), with
  identity / fallback batches contributing 0 and **included in the denominator** (coverage and red share the same
  denominator — fallback never inflates utility).
- **Actions (unchanged).** `ACTIONS = [identity, matched_coral, spdim, t3a]`, `NON_IDENTITY = [matched_coral, spdim,
  t3a]`. `identity` is always the `f_0` reference.
- **Cohorts (DEV-only).** `DISEASE = {PD: ds002778/ds003490/ds004584 (230 subj), SCZ:
  ds003944/ds003947/ds004000/ds004367 (225 subj)}`. Seven cohorts, all **development data**. No held-out / external
  cohort is read by any v4 DEV step.
- **Batching / fallback (unchanged).** Recording-ordered batches of `B=32`; batches below `MIN_BATCH=8` are **retained**
  but forced to identity (label-blind), counted in the denominator with ΔR 0.
- **Subject as the calibration unit (unchanged).** Subject/recording cluster is the independent unit; subject-disjoint
  outer folds; every subject is EVAL out-of-fold exactly once; permutation-independent canonical-SubjectKey hashing.
- **Leakage firewall / provenance (unchanged).** Field-separated dump hashes, immutable source-state bytes artifact,
  single-execution batch-action records, env lock, atomic frozen runner. v4 reuses these **as-is** via the v3 loader.

What v4 changes is **only** the control object (Section 4), the calibration target (Section 5), and the diagnostic layer
(Section 6). It does **not** change the estimand, actions, cohorts, batching, or the subject calibration unit.

---

## 4. Direction A (primary) — policy-risk budgeting

Stop asking the v3 question `ΔR_a(B) ≤ U_a(B)  ∀ a, B?`. Instead define a one-parameter family of **label-free
selective policies** and calibrate the risk of the policy that is *actually executed*.

**Per (batch, action) the predictor emits two label-free scores** (cross-fit, no target labels):
- a **harm score** `ĥ_a(B)` — a conservative (upper) estimate of `ΔR_a(B)` (higher ⇒ more likely harmful), and
- a **benefit score** `b̂_a(B)` — a center estimate of `ΔR_a(B)` (lower ⇒ more expected reduction).

Both are "lower is safer/better" in ΔR units.

### A1 — nested safe-action-set policy (primary family)

```
Γ_λ(B) = { a ∈ NON_IDENTITY : ĥ_a(B) ≤ λ }            # actions whose harm bound is within budget λ
π_λ(B) = identity                       if Γ_λ(B) = ∅            (fail-safe: no admissible action)
         argmin_{a ∈ Γ_λ(B)} b̂_a(B)    otherwise               (most expected reduction among the safe set)
```

The budget `λ` separates **safety screening** (which actions are admissible) from **utility selection** (which
admissible action to take). Smaller `λ` ⇒ smaller safe set ⇒ lower coverage, lower harm exposure; larger `λ` ⇒ more
coverage, more potential benefit *and* harm. (Optional, recorded as an ablation switch, default OFF for the primary
family: require `b̂_{a*}(B) < 0` before adapting — abstain when even the best admissible action is predicted not to
help.) v3 is the degenerate point where the only admissible-action test is the joint conformal upper bound and the
budget is `−δ`.

### A2 — risk-control calibration (calibrate the executed policy, not all-action coverage)

On CAL subjects, do **not** build an all-action joint conformal score. Build the **deployed-policy subject risk** of the
policy you will actually run:

```
L_s(λ) = (1/|B(s)|) Σ_{B ∈ B(s)} ℓ( ΔR_{π_λ(B)}(B) , π_λ(B) )
```

Candidate per-batch losses (frozen choice happens at the v4 DEV-design lock, not here):

```
ℓ_mean(B)   = ΔR_{π_λ(B)}(B)                                            # raw deployed incremental risk
ℓ_pos(B)    = max( ΔR_{π_λ(B)}(B) , 0 )                                 # one-sided harm only
ℓ_harm(B)   = 1[ π_λ(B) ≠ identity  AND  ΔR_{π_λ(B)}(B) > 0 ]           # harmful-adaptation indicator
```

Choose the **most aggressive** (largest-coverage) `λ` whose CAL risk still satisfies a pre-registered budget. This is
strictly closer to control than v3: we calibrate **only the policy that is executed**, never the upper bounds of
unexecuted actions.

**Calibration machinery (candidate, to be frozen later).** Treat `λ` on a finite grid and use a Learn-Then-Test (LTT) /
RCPS-style multiple-testing procedure over the grid to select the most aggressive policy whose subject-level CAL risk
controls the chosen loss at a pre-registered level (subjects = exchangeable units). **Monotonicity caveat (binding for
the freeze):** deployed-policy risk need not be monotone in `λ`, so we do **not** assume a monotone conformal-risk-control
theorem. Either (i) restrict to a finite `λ` grid + LTT family-wise control (no monotonicity needed), or (ii) if a
non-monotonic conformal-risk-control guarantee is used, its conditions and implementation must be written into
`ACAR_FROZEN_v4.md` **before** any confirmatory run — never reconstructed after the fact. References (background only,
not yet adopted): RCPS (distribution-free risk-controlling prediction sets), conformal risk control (monotone losses),
and non-monotonic-loss extensions; adoption requires the proof conditions be pinned in the freeze.

### A3 — Direction-A DEV gate (replaces v3's center-AUROC gate)

v3's PD center-AUROC ≥ 0.60 criterion was tied to its risk-*regression* logic. v4 gates the **deployed policy**
directly (see Section 8 for the full G0–G6). The Direction-A non-vacuity check is, per disease and disease-macro:
deployed adaptation coverage ≥ 0.15; deployed NLL reduction > 0 and > C0/v2 replay; harmful-adapted-batch rate passes
the pre-registered CAL risk rule; fallback identity batches in the denominator; **both** diseases non-vacuous.

---

## 5. Direction B — hierarchical / policy-level calibration

This is the statistical-structure direction: keep the subject as the independent calibration unit, but **stop requiring
the all-action joint-max coverage event**.

v3's coverage event `∀ B ∈ B(s), ∀ a: ΔR_a(B) ≤ U_a(B)` is too strong for safe routing, because the deployed policy
executes only one action per batch — the upper bounds of unexecuted actions do not enter deployed loss. v4-B calibrates
the **policy-level subject risk** instead:

```
Z_s(λ)   = (1/|B(s)|) Σ_{B ∈ B(s)} ΔR_{π_λ(B)}(B)                # deployed subject risk
Z_s^+(λ) = (1/|B(s)|) Σ_{B ∈ B(s)} max( ΔR_{π_λ(B)}(B) , 0 )     # positive-harm version
```

still with subjects as independent units, but without the simultaneous max over actions.

### B1 — three calibration variants (compared in DEV)

```
B0 : v3-style all-action joint-max conformal           (legacy baseline / comparator)
B1 : policy-only subject-aggregate conformal           (calibrate Z_s of the executed policy)
B2 : hierarchical batch→subject risk control           (nest batch variability inside the subject unit)
```

The point of B1/B2 is **not** merely a smaller `q` — it is placing the calibrated quantity on the **right object**
(deployed policy risk), so the budget is spent where it changes deployed loss.

### B2 — claim wording (binding for any future external write-up)

If B1/B2 ever reach external Arm B, the coverage/risk claim must read:

> For exchangeable subjects from the same site, the frozen policy calibrated on site-local CAL subjects controls the
> pre-specified deployed-risk functional.

It must **not** read "all action risks are upper-bounded for all future batches." This is intentionally **weaker** than
v3's theorem but matches the router's true use: only the executed action's risk matters.

---

## 6. Direction C — information-limit audit (risk–coverage frontiers)

This is the explanatory direction and the paper's insurance: it decides whether a v4 negative is an **information**
limit, a **policy-learning** limit, or a **calibration** limit. On the seven DEV cohorts, all curves computed on
subject-disjoint OOF cells (no calibration leakage), axes: x = adaptation coverage, y = deployed NLL reduction
`−mean ΔR`, plus a harm axis `P(ΔR_{π(B)} > 0 | π(B) ≠ identity)`.

```
F_true_oracle  : choose batches & actions by TRUE ΔR (best action per batch, adapt the most-reducing batches up to
                 coverage c). Upper bound of what batch-level action selection can ever achieve.
F_score_oracle : the UPPER ENVELOPE over a pre-listed set of single-score frontiers — each single-score frontier
                 chooses batches & actions by ONE label-free score (coverage level on the x-axis; evaluate with true
                 ΔR). The envelope is the information ceiling of the listed observables; a single score is only one
                 rule, not the ceiling (code: frontier_single_score_oracle vs frontier_score_oracle_union).
F_policy_family: the v4-A/B candidate policy family π_λ on OOF (fitted predictors, no oracle thresholds).
                 Includes policy-learning error.
F_calibrated   : the actually-calibrated frozen policy points (after CAL risk control). v2 router and v3 C1/C2/C3
                 land here as reference points.
```

### Interpretation matrix

```
F_true_oracle high, F_score_oracle low   ⇒ INFORMATION gap   (label-free observables lack the info; no honest router can win)
F_score_oracle high, F_policy_family low ⇒ POLICY-LEARNING gap (model/policy fails to exploit available info)
F_policy_family high, F_calibrated low   ⇒ CALIBRATION gap    (calibration structure too conservative; v3 lives here)
```

This refines the project's "measurement→control gap" into three named, *separately attributable* gaps — a stronger and
more honest story than v2/v3's single "risk-regression + conformal q" account, regardless of whether any v4 policy
passes. The decomposition has two parallel outputs: the **ceiling** gaps (main diagnostic, exact telescoping identity
`info+policy+calibration = ceiling(true) − red(calibrated)`; `info_gap ≥ 0` guaranteed) and the **AUC** gaps
(descriptive area-over-coverage summary on the Pareto envelope, never used for pass/fail). All frontier and accounting
primitives are subject-macro weighted (pass `subject_macro_weights(subject_ids)`; fallback rows stay in the weighted
denominator) and fail-closed (non-finite / out-of-range / bad-weight inputs raise).

---

## 7. DEV exploration plan (shared substrate, no external)

- **Shared splits.** Reuse v3's subject-disjoint outer folds + FIT/CAL/TRAIN/VAL nesting and canonical-SubjectKey
  hashing, unchanged, so A/B/C are evaluated on identical cells.
- **Shared execution cache.** Reuse v3's single-execution `BatchActionExecutionRecord` / disease execution cache — the
  per-batch per-action embeddings and `ΔR_a(B)` are computed **once** and shared across all v4 directions (no re-fit
  drift, no re-leak).
- **Subject-balanced metrics.** All red / coverage / harm-rate / AUROC are subject-macro (subject-equal-weighted),
  matching v3's S2/S4 accounting; fallback identity batches included in denominators.
- **No external.** Every v4 DEV number is exploratory / model-selection on the seven DEV cohorts. No held-out, no
  lockbox, no Arm B. Detailed protocol: `notes/ACAR_V4_DEV_EXPLORATION_PLAN.md`. Boundary: see Section 9 and
  `notes/ACAR_V4_LOCKBOX_BOUNDARY.md` (to be written before any freeze).

---

## 8. Candidate gate for a possible v4 freeze (NON-BINDING here)

A v4 freeze is considered **only** if Phase-1 exploration produces a candidate passing all of:

```
G0  all provenance / leakage / split guards pass (inherited from v3)
G1  per-disease adaptation coverage ≥ 0.15
G2  per-disease deployed NLL reduction > 0
G3  disease-macro deployed NLL reduction > C0/v2 replay
G4  harmful adapted-batch rate controlled by the pre-registered CAL risk rule
G5  fallback identity batches included in the denominator (accounting invariant)
G6  PD and SCZ both non-vacuous; no disease has zero adaptation
```

Direction-C frontiers are reported as **explanatory output**, not pass/fail:
`F_true_oracle`, `F_score_oracle`, `F_policy_family`, and the v2/v3 calibrated points. If A/B do **not** pass but C
shows a high true-oracle frontier with a low score/policy frontier, the paper's conclusion is strong: label-free
closed-loop control may be **information-limited** on PD, not an implementation failure.

These thresholds are **draft**. The binding numbers, the single primary policy family, the single calibration method,
the single `λ` grid, the single selection scalar, and the loss `ℓ` are fixed only in `ACAR_FROZEN_v4.md` at the v4
DEV-design lock — never adjusted after a DEV read.

---

## 9. Provenance and guards (branch/tag discipline)

- **No v3 mutation.** v4 adds files under `acar/v4/` and `notes/ACAR_V4_*`; it never edits `acar/v3/*`, `acar/*` (v2),
  the v3 protocol commit/tag (`817b04f` / `acar-v3-dev-design-v1`), or any v3 result. The v3 lock lives in its own
  detached worktree (`ACAR_V3_LOCKED_RUN_817b04f`) and is read-only.
- **Reuse, don't fork, the firewall.** v4 imports the v3 loader, env lock, execution cache, and split logic; it does
  not reimplement provenance. Record-level field-separated hashes and the label firewall are inherited unchanged.
- **Branch / tag.** v4 work lands on the `acar` branch on top of `603a817`. A v4 protocol tag (`acar-v4-protocol` or
  `acar-v4-dev-design-v1`) is created **only** at a future freeze, never on this draft.
- **NON-BINDING here.** This draft and the first v4 code (synthetic-guard-only `frontiers.py` / `policies.py`) select
  nothing, freeze nothing, and read no real cohort.

---

## 10. Stop rules

```
No candidate passes G0–G6 in Phase-1 DEV   ⇒  V4-DEV-NEGATIVE / NO LOCKBOX CONSUMED
                                              (record the frontier decomposition: information / policy / calibration gap)
A candidate passes G0–G6                    ⇒  write ACAR_FROZEN_v4.md, freeze ONE policy family + ONE calibration
                                              method + ONE λ grid + ONE selection scalar + ONE loss + the external
                                              cohort list + the Arm-B site-local split rule; tag acar-v4-protocol;
                                              only THEN may external Arm B be approached (separately authorized).
```

No threshold / seed / `λ` / loss / candidate search to chase a pass. A killed or partial run is **operationally
aborted**, never a scientific verdict. Any continuation past v4 is again a NEW dated, separately-tagged protocol.
