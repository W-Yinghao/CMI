# Tier-1 smoke driver — DESIGN ONLY (no executable driver, no runs)

Branch `science-target-info-v1`. This is a **design document only**: no executable `run_*.py`, no experiments,
no target-label runs. Pseudocode / interface sketches below are illustrative, NOT to be implemented until a
separate PM go. The frozen design-lock is `TARGET_INFORMATION_FRONTIER_DESIGN.md` (status DESIGN LOCK FROZEN,
approved content hash `3ad4ef312e325fa6`) + `eeg/configs/target_info_frontier_fixed.yaml`
(`experiments_allowed: false`, `driver_allowed: false`).

## Question this driver would answer

> In semi-synthetic target-beneficial worlds, as the target-information budget grows from 0 to
> few-label / sequential calibration, does the controller move from abstain to **safe** accept while the
> false-accept rate stays near 0?

Contrast against the source-only baseline (B0): the V2 ceiling shows source-only must abstain on
source-INVISIBLE benefit; Fork 2 shows a source-rich discovered-env accept only on low-dim EEGNet. Tier-1
asks whether TARGET LABELS are the reliable lever that source-only evidence cannot be.

---

## 1. Fixed Tier-1 smoke scope (locked)

```
datasets              Lee2019_MI, Cho2017
backbone              EEGNet only
worlds                V2 source-invisible World A ; source-rich source-visible World A
seeds                 seed0
folds                 first5
target split repeats  R = 10
budgets               B0 source-only
                      B1 unlabeled target (triage-only)
                      B2 k labels/class, k = {1,2,4,8,16}
                      B3 sequential calibration over the same k-grid
                      B4 oracle diagnostic
```

Do NOT include: TSMNet · real-EEG Tier-2 · active acquisition beyond sequential calibration ·
B1-unlabeled-as-accept-certificate.

---

## 2. Data flow (per target subject / fold / split)

```
1. Source data fits eraser / head / source-safety components   (source-only; no target labels)
2. Target trials are split into calibration and audit          (stratified by class, R=10 repeats)
3. Calibration labels may be used ONLY by B2/B3/B4 gate decisions
4. Audit labels are used ONLY for final evaluation
5. The same target label can never both decide and evaluate
```

The eraser and any head used inside the safety/benefit machinery are fit on SOURCE only (semi-synthetic World A
injects the known nuisance `D_nuis=z`; the eraser erases `D_nuis`, exactly as in V2 / Fork 2). Target
calibration labels enter ONLY the target-benefit gate, never the eraser fit.

---

## 3. Structural check (must run before ANY task, if ever implemented)

```
TARGET_LEAK_STRUCTURAL_PASS asserts, per (subject, fold, split):
  calibration ∩ audit = ∅                         (index-disjoint)
  audit labels are not read before final scoring   (audit y accessed only in the eval function)
  B1 cannot return ACCEPT                           (B1 action ∈ {reject, abstain, request_labels})
  B4 oracle is DIAGNOSTIC_ONLY                      (never enters a deployable accept path)
emit TARGET_LEAK_STRUCTURAL_PASS or HALT.
```

This mirrors V2's pre-run `target_leak_structural_check` (which emitted `TARGET_LEAK_STRUCTURAL_PASS` or halted).
It is a hard gate: no per-task compute runs until it passes.

---

## 4. Budget behavior

### B0 — source-only
```
Uses only the existing source-only gate (source safety UCB + source-LOEO / source-visible benefit).
Expected: abstain/reject on source-invisible target benefit (the V2 ceiling).
```

### B1 — unlabeled target (triage-only)
```
Uses target X/Z only (no target labels).
ACCEPT is FORBIDDEN.
Allowed actions: reject / abstain / request_labels.
Signal = source↔target distribution mismatch on Z (e.g. covariate-shift / MMD-style screen) used ONLY to
decide reject vs abstain vs request_labels. This is triage, NOT an unsupervised adaptation claim.
```

### B2 — k labels per class
```
Use k labeled CALIBRATION trials per class (k ∈ {1,2,4,8,16}).
Accept iff  source safety passes  AND  target-calibration ΔbAcc LCB > +0.01  AND  beats same-k random.
Report effective n_per_class; if a subject lacks enough calibration trials for a requested k:
  mark that k UNAVAILABLE for that subject/split ; never reuse audit labels.
```

### B3 — sequential calibration
```
Reveal labels sequentially at k = 1,2,4,8,16.
Stop early if accept OR reject can be certified at the current k.
Otherwise request more labels (advance to the next k).
Record the label budget used before the decision.
NOT full active learning — reveal order is fixed/random, no uncertainty sampling. Active acquisition deferred.
```

### B4 — oracle diagnostic
```
Uses ALL target labels only as an upper bound on the best achievable selection.
DIAGNOSTIC_ONLY ; never presented as deployable ; reported to calibrate the oracle gap.
```

---

## 5. Metrics the eventual driver would output

```
true accept rate            false accept rate            abstain rate            reject rate
request-labels rate (B1/B3) accepted target ΔbAcc        accepted target ΔNLL    label budget used before decision
oracle gap (B4 − budget)    random-k specificity check
```

For EVERY accepted intervention:
```
source task safety must pass
target-CALIBRATION benefit must pass (LCB > +0.01)
audit target benefit reported SEPARATELY (held-out; the honest number)
same-k random must NOT reproduce the gain, else the accept is flagged NON-SPECIFIC
```

"Target-beneficial" ground truth per cell is known by construction (World A) and via the audit split; false
accept = ACCEPT on a cell whose held-out audit ΔbAcc hi < −0.01 (harmful) or whose gain is same-k-random-explained.

---

## 6. Pass / fail (for the FUTURE Tier-1 smoke run, not now)

### Pass
```
increasing k raises the true accept rate
false accept remains near 0
accepted interventions have positive AUDIT target ΔbAcc
B3 sequential uses fewer labels than fixed k=16 where possible
B1 never accepts
B4 oracle upper bound is higher than deployable budgets
```

### Fail
```
B2/B3 false-accept harmful cells
calibration gain does not transfer to audit (overfit)
B1 accepts from unlabeled target
same-k random explains the accepted gain
oracle and deployable paths are confused
```

---

## 7. Interface sketch (illustrative pseudocode — DO NOT IMPLEMENT YET)

Intended future module (NOT created): `tos_cmi/eeg/run_target_info_tier1_smoke.py`.
Would REUSE existing components (no new estimators invented):
- `run_v2_certificate`: `_stratified_split`, `_task_scores`, `_nuisance_m`
- `source_ood_benefit_gate`: `_bacc`, `_subj_acc`, `_boot_bound`, `gate_action`, `SAFETY_EPS`, `BENEFIT_LCB`
- `source_rich_worlds`: `inject_source_rich` (source-visible World A)
- `v2_worlds`: `FACTORIES`, `oracle_nuisance_eraser_factory`, plus the V2 source-INVISIBLE World A builder
- `erasure_baselines`: `_ids`

```
def structural_check(cal_idx, aud_idx, budget):        # HARD gate, runs first
    assert set(cal_idx).isdisjoint(aud_idx)
    assert budget != "B1" or ACTIONS[budget] <= {reject, abstain, request_labels}
    assert budget != "B4" or DIAGNOSTIC_ONLY[budget]
    return "TARGET_LEAK_STRUCTURAL_PASS"

def target_calibration_benefit(F, Zs, ys, Zt, yt, cal_idx, k, seed):
    # k labeled trials/class from calibration ONLY; ΔbAcc(erased − full) with subject/split bootstrap LCB.
    # audit indices are NEVER passed here.
    ...
    return dbacc_cal, lcb_cal, n_per_class_effective   # UNAVAILABLE if calibration lacks k/class

def decide(budget, source_safety, source_benefit, cal_benefit, unlabeled_triage, same_k_random):
    if not source_safety.pass: return "reject"
    if budget == "B0": return gate_action(source_safety.ucb, source_benefit.lcb, SAFETY_EPS, BENEFIT_LCB)
    if budget == "B1": return unlabeled_triage           # ∈ {reject, abstain, request_labels}; never accept
    if budget in ("B2","B3"):
        if cal_benefit.lcb > BENEFIT_LCB and beats(same_k_random): return "accept"
        return "abstain"                                  # B3 loops k until accept/reject certified
    if budget == "B4": return "DIAGNOSTIC(argmax audit ΔbAcc)"

def audit_eval(F_accepted, Zt, yt, aud_idx):              # held-out; only place audit y is read
    return dbacc_audit, dnll_audit
```

Design risks to resolve AT implementation time (flagged, not solved here):
- small-k statistical fragility (k=1,2 per class → wide LCB; the gate should ABSTAIN not false-accept — verify
  the bootstrap LCB is conservative at small k).
- calibration→audit transfer gap (report both; the honest number is audit).
- B1 triage statistic choice (must be decision-only, never a benefit magnitude).

---

## 8. Do-not (still forbidden until a separate PM go)
```
writing executable driver ; running Tier-1 ; running Tier-2 ; using target labels in any experiment ;
opening Track E ; editing manuscript .tex ; implementing active acquisition ; using B1 to accept
```
Ledger status for this artifact = `design_only` (see `CLAIMS_LEDGER.md: target_information_tier1_smoke_driver`).
The uploaded PDF is a stale 2a-only snapshot and is NOT the writing baseline for this branch.
