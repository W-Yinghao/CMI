# Fork 1 — Target-information frontier (DESIGN-LOCK; NO experiments yet)

**Branch `science-target-info-v1`** (from `science-source-rich-v1` @ b2edb1d). This is a design-lock document only:
no driver, no runs. The successor question after the source-only line was pushed to depth (refusal power = Track B;
acceptance ceiling = V2; source-rich partial positive + high-dim discovery limitation = Fork 2, see
[[../notes/SOURCE_RICH_FINAL_VERDICT.md]]).

Standing discipline inherited: target labels are the object of study here (this is explicitly NOT strict source-only
DG), but they must NEVER be used to both DECIDE and EVALUATE; thresholds frozen at safety≤0.02 / benefit>0.01;
semi-synthetic first, real EEG second; report-then-go; no paper edits (uploaded PDF is a stale 2a-only snapshot).

---

## 1. Core scientific question

```
Strict source-only gates refuse when benefit is source-invisible.
Source-rich environments can help only when the relevant shift is represented and discoverable.
How much target information is needed to safely license accept when source-only evidence cannot?
```
The variable of study is a TARGET-INFORMATION BUDGET. We measure, as the budget grows, whether the gate transitions
from refuse/abstain to safe ACCEPT, and at what budget the true-accept curve approaches the oracle upper bound while
false accepts stay near zero.

---

## 2. Target-information budgets (pre-registered ladder)

```
B0 : source-only only                                  (baseline; the ceiling — expected refuse/abstain)
B1 : unlabeled target Z_T / X_T                        (TRIAGE only — see discipline)
B2 : k labeled target trials per class, k in {1,2,4,8,16}   (the real crossing path)
B3 : active calibration / sequential labels            (the real crossing path — budget efficiency)
B4 : oracle target-informed selector  DIAGNOSTIC_ONLY  (upper bound; NOT a method)
```

Discipline per budget:
- **B1 (unlabeled target):** must NOT be used as an accept certificate in the first design. Unlabeled target
  distribution-mismatch is a SCREEN / needs-labels trigger, not evidence of benefit. Allowed actions from
  unlabeled-only signal: `reject` / `abstain` / `request_labels`. `accept` is FORBIDDEN from unlabeled-only signal.
  (Prevents over-claiming unsupervised target adaptation.)
- **B2 (k labels/class):** the primary path expected to cross the ceiling. Sweep k to trace the label-budget curve.
- **B3 (active calibration):** sequential/active label acquisition; question = does it reach the same accept
  decisions as B2 at a SMALLER label budget?
- **B4 (oracle selector):** uses target labels on the audit-equivalent data to pick the best intervention — an
  UPPER BOUND on what any target-informed selector could achieve. Reported for calibration of the curve; it is
  explicitly DIAGNOSTIC and never presented as a deployable method.

---

## 3. Clean evaluation split (THE key design constraint)

Target labels cannot both decide the gate and score the final evaluation. For each target subject:

```
target calibration split :  used by the target-informed gate (B2/B3/B4 decisions)
target audit split        :  final evaluation ONLY (never seen by the gate)
```

Because target trials are scarce, use repeated split / cross-fitting rather than one split:

```
split each target subject's trials into calibration and audit (stratified by class)
repeat R times with different splits (R pre-registered, e.g. R=10)
the gate decides on calibration; ΔbAcc is scored on the held-out audit split
report a PAIRED CI over (subject x split) with subject-clustered bootstrap
```

Guardrails:
- The k labeled trials of budget B2 are drawn ONLY from the calibration split; the audit split is untouched by the
  gate, hyperparameter, calibration, or intervention selection.
- No leakage of audit-split labels into eraser fit, head fit, threshold, or intervention choice.
- A structural pre-run check (mirroring V2's `target_leak_structural_check`) must assert calibration∩audit=∅ and that
  no audit label is read before scoring, emitting a PASS token or halting.

---

## 4. Gate design (reuse the existing controller structure)

```
source safety gate :
  reject if source task-drop UCB > 0.02                       (unchanged, frozen)

target benefit gate :
  accept only if target-CALIBRATION ΔbAcc LCB > +0.01         (target-informed; on calibration split only)

domain gain :
  diagnostic only                                             (never a decision input)

random / compression control :
  an accepted intervention must beat the same-k random / compression baseline,
  or be flagged non-specific (regularization, not domain-removal)
```

For unlabeled target (B1):
```
use ONLY as triage:
  accept is FORBIDDEN from unlabeled-only signal in this first design
  allowed actions: reject / abstain / request_labels
```

Decision logic (per intervention, per target subject, per split):
1. source safety gate must pass (else reject);
2. for B0/B1: no accept path (abstain or, for B1, request_labels);
3. for B2/B3: accept iff target-calibration ΔbAcc LCB > +0.01 AND beats same-k random control;
4. B4 oracle: pick argmax audit-equivalent ΔbAcc (diagnostic ceiling).

---

## 5. Main experiment objects (two tiers; semi-synthetic FIRST)

### Tier 1 — semi-synthetic (development; has KNOWN target-beneficial cells from V2 / Fork 2)
```
datasets  : Lee2019_MI, Cho2017
backbones : EEGNet
worlds    : V2 World A (source-INVISIBLE target-beneficial) + source-rich World A (source-visible)
budgets   : B0, B1(unlabeled), B2(k labels/class), B3(active), B4(oracle)
```
Objective: as k grows, does the gate move from abstain → ACCEPT on the genuinely target-beneficial cells, while the
false-accept rate on harmful/useless cells stays near 0, and does the label-budget curve approach the B4 oracle?

### Tier 2 — real EEG (second; NO known target-beneficial erasure)
```
datasets  : (real EEG cells from the frozen dumps, as scoped later)
expectation: with target labels, the gate should STILL reject/abstain;
             if it accepts, audit whether the accepted target gain is real or overfit.
```
This tier is a FALSE-ACCEPT-CONTROL test, not an accept-power demonstration: real EEG currently has no
target-beneficial erasure (C12/C13), so a well-behaved target-informed gate should not spuriously accept.

---

## 6. Pass / fail (pre-registered)

### Success
```
Semi-synthetic target-beneficial worlds:
  target-label budget INCREASES the true accept rate
  false accept remains near 0
  accepted interventions have POSITIVE held-out (audit-split) target gain
  the label-budget curve APPROACHES the oracle (B4)

Real EEG:
  the gate does NOT spuriously accept useless/harmful erasures
```

### Failure
```
few-shot target gate OVERFITS and false-accepts
unlabeled target screen accepts harmful cells
active calibration FAILS to reduce the label budget vs passive B2
```

---

## 7. Do NOT (until explicit PM go)
```
no driver implementation run ; no experiments ; no target-label runs ;
no Track E end-to-end training ; no paper edits ; no new datasets ;
no B1-as-accept-certificate ; no audit-split leakage into any decision
```
This document + `eeg/configs/target_info_frontier_fixed.yaml` are the design-lock. Freeze the config hash AFTER PM
review (the config is DRAFT-for-review until then). Next report = design-lock deliverable checklist, then hold.

See [[tos-cmi-method-deepen-v2]], [[source_only_acceptance_ceiling_theory]] (C15 P3 = the target-label sample-
complexity bridge this frontier instantiates), [[canonical-yaml-and-report-then-go]], [[cmi-survivor-audit]].
