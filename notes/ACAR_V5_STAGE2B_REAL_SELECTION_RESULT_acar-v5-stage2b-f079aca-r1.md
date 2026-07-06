# ACAR V5 — Stage-2B real DEV candidate selection (RESULT: FAIL)

```
status: STAGE2B_FAIL

run_id:                    acar-v5-stage2b-f079aca-r1
implementation_base_sha:   f079aca9570fef47b333e34cc238376e29fb6cc8
stage1b_run_id:            acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:   2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:   4278435975a72b1127803dd2cffab420c083e430

selection report written:  NO
selected_candidate_id:     NONE (run crashed in Pass-1, first candidate)
```

The binding run failed with an unhandled numerical exception inside the Stage-2B2 stable `matched_coral` path on a **single-window
(n=1) tail batch**. Per the Stage-2B authorization, the run was stopped immediately: NOT patched, NOT rerun, NOT tuned, NOT
resumed, no candidate skipped. This note reports the failure; the fix is a method/code decision requiring a NEW authorization.

## Pre-flight (both passed — the failure is a numerical hole, not a guard/pin/label problem)

- No-label `--guard` PASS: worktree HEAD == f079aca, package ADMITTED, registry sha matched, exactly the 10 seed-20260711 refs +
  22 frozen candidates, S1 seeds not opened, no forbidden site token, auth valid+bound, labels not read.
- Adversarial 4-lens launcher review (pin-correctness / label-firewall / auth-gate / discipline): 0 findings (verified the agents
  actually inspected the launcher). NOTE: the review was scoped to the LAUNCHER; it was explicitly told not to re-audit the frozen
  engine / stable_matched_coral internals — so it could not have caught this engine-path numerical hole.

## Failure

```
first failing module:      acar.v5.stage2_stable_coral.transport_operator
                           (via real_action_provider "matched_coral" -> stable_matched_coral_v1
                            -> transport_operator -> np.cov(Z) -> _svd_cap -> np.linalg.svd)
first failing candidate:   manifest[0]  (Pass-1, first candidate x first disease)
first failing disease:     PD
first failing fold:        fold0
first failing split:       CAL   (traceback at stage2_selection_engine.py:83; CAL is evaluated before EVAL)
first failing subject:     PD/ds004584/sub-076   (33 windows -> window_batches -> [32, 1]; the size-1 tail is the crash)
exception class:           numpy.linalg.LinAlgError
exception message:         SVD did not converge
preceding warnings:        "Degrees of freedom <= 0 for slice" (np.cov at n=1) ; divide-by-zero / invalid-value in np.cov
report written:            NO (crashed before build_selection_report; no <RUN_ID>.result.json)
selected_candidate_id:     NONE
node:                      nodecpu04   (SLURM job 883822)   elapsed ~ under 3 min before the crash
```

Log tail:
```
stage2_stable_coral.py:70: RuntimeWarning: Degrees of freedom <= 0 for slice
  C_T = np.cov(Z, rowvar=False)
numpy .../_function_base_impl.py:2888: RuntimeWarning: divide by zero encountered in divide
numpy .../_function_base_impl.py:2888: RuntimeWarning: invalid value encountered in multiply
Traceback (most recent call last):
  ... run_selection -> _evaluate_candidate_on_disease -> PE.evaluate_candidate_disease (CAL, line 83)
      -> AR.subject_action_outputs -> real_action_provider("matched_coral")
      -> stable_matched_coral_v1 -> transport_operator -> _svd_cap -> np.linalg.svd
numpy.linalg.LinAlgError: SVD did not converge
```

## Root cause (definitive, real-data-grounded)

`stable_matched_coral_v1` / `transport_operator` have **no guard for n < 2 windows**. At n=1 the target-batch covariance
`np.cov(Z, rowvar=False)` is undefined (DOF = n−1 = 0 → divide-by-zero → non-finite `C_T`), so `_shrink` → `_psd_power` →
`_svd_cap` receives a non-finite operator and `np.linalg.svd` raises `SVD did not converge`. (Internal inconsistency: line 96
already special-cases `n >= 2` for `se`, but the unconditional `transport_operator` call at line 91 hits the undefined covariance
first.)

Why the selection engine reaches an n=1 batch (while Stage-2B2P did not):
- `stage2_policy_eval.window_batches` splits a subject's windows into consecutive 32-window batches; a subject whose window count
  ≡ 1 (mod 32) yields a final batch of exactly **1** window.
- `stage2_policy_eval.evaluate_candidate_disease` runs `subject_action_outputs` on **every** batch — including sub-MIN_BATCH
  forced tails — BEFORE the forced-identity routing check, because the harmful-oracle `red_upper` term needs every action's ΔR on
  every batch (`upper_drs.append(min(0.0, min(dr.values())))`). So a forced n=1 tail is still pushed through `matched_coral`.
- Stage-2B2P (by its own authorized spec) **excluded tails < 32** and tested only full 32-window batches, so the n<2 path was
  never exercised on the stable action. That PASS was real but scoped to full batches; it did not cover what the engine routes.

Real-data extent (label-free window-count scan, SLURM job 883832 — no labels / no actions / no scoring):
```
subjects scanned                    2280
subjects with an n=1 tail batch       55   (across ALL 10 selection refs)
total batches                      11175
n=1 tail batches (crash trigger)      55
forced tails (size < 8)              405   (only the size-1 subset crashes; size 2..7 have DOF >= 1 and do not)
PD folds with an n=1 CAL subject      5 of 5     (fold0 first-crash subject: PD/ds004584/sub-076)
SCZ folds with an n=1 CAL subject     0 of 5     (SCZ n=1 tails fall only in fit/eval)
```

## Forbidden-stage confirmation

```
S1 seeds 20260712 / 20260713 opened:  NO (only seed20260711 refs loaded)
S1 / S2 / S3 robustness run:          NO (crashed in Pass-1, first candidate)
external / held-out read:             NO
ASZED read:                           NO
lockbox touched:                      NO
substrate rebuilt:                    NO
code changed / candidate space / batch size / thresholds / CAL-EVAL interp / stable-CORAL caps altered:  NO
```

## Status and what is needed (method decision — NOT taken here)

Stage-2B real DEV candidate selection is **BLOCKED** on this numerical hole. Resolving it is a method/code decision that requires a
NEW authorization (it is a code change to the stable action path, which the current authorization forbids mid-run). The decision
space (for the user to choose — not chosen here):
- define `stable_matched_coral_v1` behavior at n < 2 (target covariance undefined): e.g. reliability `alpha_eff = 0` ⇒ `z_post = Z`
  (no adaptation, identity-equivalent), or a different fail-closed contract; and/or
- exclude n < 2 (or all forced sub-MIN_BATCH) tails from the `red_upper` all-action oracle in `stage2_policy_eval`, so forced
  tails are never routed through any adapting action; and
- extend the Stage-2B2P-style stress to the SAME batch set the selection engine routes (all batches incl. forced n=1 tails), not
  only full 32-window batches, before re-authorizing Stage-2B.

Until a new authorization pins the amendment, the a5c44c3 and f079aca Stage-2B real-run lines both remain unrunnable for selection.
No further action taken.
