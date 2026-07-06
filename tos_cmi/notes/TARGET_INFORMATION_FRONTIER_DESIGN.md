# Fork 1 — Target-information frontier (DESIGN-LOCK; NO experiments)

Branch `science-target-info-v1` (from `science-source-rich-v1`).
This is a design-lock document only: no driver, no runs.

It succeeds the source-only line, whose final readout is:
- Real EEG: the source-only gate mainly provides **refusal power** (Track B, C12/C13).
- **V2:** a source-only **acceptance ceiling** (benefit that is source-invisible cannot be certified).
- **Fork 2:** a **source-rich partial positive** with caveats — EEGNet semi-synthetic Lee→Cho positive,
  but TSMNet discovery fails / oracle fragile (see `SOURCE_RICH_FINAL_VERDICT.md`).

Config: `eeg/configs/target_info_frontier_fixed.yaml`.

---

## 1. Scientific question

> How much target information is needed to safely license accept when source-only evidence cannot?

Strict source-only gates refuse when benefit is source-invisible.
Source-rich environments help only when the relevant shift is represented and discoverable.
The variable of study here is a **target-information budget**; we measure whether, as the budget grows,
the gate transitions from refuse/abstain to safe ACCEPT, and at what budget the true-accept curve
approaches the oracle upper bound while false accepts stay near zero.

This line is explicitly **NOT strict source-only DG** — it studies target information on purpose.

---

## 2. Budget ladder

```
B0 source-only              baseline; the ceiling — expect refuse/abstain
B1 unlabeled target         TRIAGE only — no accept
B2 k labels/class           main crossing path
B3 sequential calibration   early stopping over the B2 budget
B4 oracle selector          diagnostic upper bound (not a method)
```

- **B0** — source-only evidence only.
- **B1 (unlabeled target Z_T / X_T)** — a screen / needs-labels trigger, NOT an accept certificate.
  Allowed actions from unlabeled-only signal: `reject` / `abstain` / `request_labels`.
  `accept` is FORBIDDEN from unlabeled-only signal (prevents over-claiming unsupervised target adaptation).
- **B2 (k labeled target trials per class, k ∈ {1,2,4,8,16})** — the primary path expected to cross the
  ceiling; sweep k to trace the label-budget curve.
- **B3 (sequential calibration)** — reveal the same per-class labels sequentially and stop early once an
  accept/reject condition is certified. True active acquisition is DEFERRED (see §7).
- **B4 (oracle selector, DIAGNOSTIC only)** — uses target labels to pick the best intervention: an upper
  bound on any target-informed selector. Reported to calibrate the curve; never a deployable method.

---

## 3. Calibration / audit split

Target labels cannot both decide the gate and score the final evaluation. For each target subject:

```
target calibration split   used by the target-informed gate (B2/B3/B4 decisions)
target audit split          final evaluation ONLY (never seen by the gate)
```

Because target trials are scarce, use repeated split / cross-fitting:

```
split each target subject's trials into calibration and audit, stratified by class
repeat R = 10 times with different splits
the gate decides on calibration; ΔbAcc is scored on the held-out audit split
report a paired CI over (subject × split) with a subject-clustered bootstrap
```

k-availability rule (no silent label reuse):

```
if a target subject lacks enough calibration trials for a requested k,
mark that k UNAVAILABLE for that subject/split — do NOT silently reuse audit labels
report the effective n per class after splitting
```

Guardrails:
- The k labeled trials (B2/B3) are drawn ONLY from the calibration split; the audit split is untouched by
  gate, hyperparameter, calibration, or intervention selection.
- A structural pre-run check asserts calibration ∩ audit = ∅ and that no audit label is read before scoring,
  emitting `TARGET_LEAK_STRUCTURAL_PASS` or halting.
- R = 10 is the first-version default. If CIs are unstable, an R = 20 sensitivity check is a later option,
  NOT a default of this design-lock.

---

## 4. Gate design

Reuse the existing controller structure; thresholds frozen.

```
source safety gate     reject if source task-drop UCB > 0.02
target benefit gate     accept only if target-CALIBRATION ΔbAcc LCB > +0.01
domain gain             diagnostic only (never a decision input)
specificity control     accepted intervention must beat the same-k random / compression
                        baseline, or be flagged non-specific
unlabeled target        triage only; accept forbidden from unlabeled-only signal
audit split             never used for any decision
```

Decision logic (per intervention, per target subject, per split):
1. source safety gate must pass (else reject);
2. B0 / B1: no accept path (abstain; B1 may `request_labels`);
3. B2 / B3: accept iff target-calibration ΔbAcc LCB > +0.01 AND beats the same-k random control;
4. B4 oracle: pick argmax audit-equivalent ΔbAcc (diagnostic ceiling).

---

## 5. Tier 1 — semi-synthetic experiment (development, FIRST)

Has KNOWN target-beneficial cells from V2 / Fork 2.

```
datasets    Lee2019_MI, Cho2017
backbones   EEGNet
worlds      V2 World A (source-INVISIBLE target-beneficial)
            source-rich World A (source-VISIBLE)
budgets     B0, B1, B2, B3, B4
```

Objective: as k grows, does the gate move from abstain → ACCEPT on the genuinely target-beneficial cells
while the false-accept rate on harmful/useless cells stays near 0, and does the label-budget curve
approach the B4 oracle?

---

## 6. Tier 2 — real EEG false-accept-control experiment (SECOND)

Real EEG currently has NO target-beneficial erasure (C12/C13), so this tier is a **false-accept-control**
test, not an accept-power demonstration.

```
datasets    Lee2019_MI, Cho2017
backbones   EEGNet, TSMNet
seeds       0, 1, 2
folds       first 15 (initial Tier-2 smoke / pilot)
secondary   High-Gamma (4-class stress test, after Lee/Cho)
```

2a / 2b are NOT in the first Fork-1 round (small-N / special montage; appendix sanity only).

Expectation: with target labels the gate should STILL reject/abstain; if it accepts, audit whether the
accepted target gain is real or overfit.

---

## 7. B3 — sequential calibration; true active deferred

The first executable design treats B3 as **sequential calibration, not full active learning**:

```
B3 sequential calibration
  use the same per-class calibration pool as B2
  reveal labels sequentially at k = 1, 2, 4, 8, 16
  stop early if the accept/reject condition is certified
  otherwise request more labels

active acquisition (uncertainty sampling from an unlabeled target pool) is DEFERRED
  to a later sub-branch: B3_active_query_policy_v2
```

Do NOT design uncertainty sampling now — that would turn the first target-info frontier into an
active-learning paper. The first-round question is only:

> does sequentially increasing the target-label budget push the gate from abstain to safe accept?

---

## 8. Pass / fail criteria

### Success
```
Semi-synthetic target-beneficial worlds:
  target-label budget increases the true accept rate
  false accept remains near 0
  accepted interventions have positive held-out (audit-split) target gain
  the label-budget curve approaches the oracle (B4)

Real EEG:
  the gate does NOT spuriously accept useless/harmful erasures
```

### Failure
```
few-shot target gate overfits and false-accepts
unlabeled target screen accepts harmful cells
sequential calibration fails to reduce the label budget vs passive B2
```

---

## 9. Forbidden claims

```
unsupervised target adaptation from unlabeled-only signal
active-learning result in the first round (B3 is sequential calibration, not active acquisition)
real EEG target-gain result
"source-rich environments solve source-only acceptance"
using the audit split for any decision
```

The uploaded PDF is a stale BCI-IV-2a-only snapshot (its limitations still say single dataset); it is NOT
the writing baseline for this branch.

---

## 10. Do-not-run discipline (until explicit PM go)

```
no driver implementation
no experiments / no runs (semi-synthetic or real EEG)
no target-label runs
no Track E end-to-end training
no paper edits
no new datasets beyond the scoped ones
no true active-learning acquisition
no B1-as-accept-certificate
no audit-split leakage into any decision
```

This document + `eeg/configs/target_info_frontier_fixed.yaml` are the design-lock.
Freeze the config hash only AFTER PM review. After freeze, the next step is designing a Tier-1 smoke
driver — still not running experiments.

See `SOURCE_RICH_FINAL_VERDICT.md`, `CEILING_THEORY.md` (C15 P3 = the target-label sample-complexity
bridge this frontier instantiates), `CLAIMS_LEDGER.md`.
