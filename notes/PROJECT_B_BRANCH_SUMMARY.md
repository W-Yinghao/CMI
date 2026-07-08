# Project B Branch Summary

## 1. Branch and commit range
Branch `project-b-refusal-router` off `main` (merge-base `c8fce202b8bc` .. HEAD `cc1723df67d1`): 16 commits,
39 files, all additive (no `main` file modified). Worktree
`/home/infres/yinwang/CMI_AAAI_projectB`.

## 2. Problem and deployment posture
At deployment an unlabelled EEG target may or may not benefit from test-time adaptation (TTA), and
source data alone cannot tell which. Project B is a **refusal-first deployment router**, **not a new EEG
TTA optimizer**. For each target it chooses among `REFUSE / IDENTITY / OFFLINE_TTA / ONLINE_TTA` and
emits auditable OACI reason codes. Target labels are used **only post-hoc** for evaluation.

## 3. What was added
Router core (`h2cmi/router/*`), an integration harness (`h2cmi/eval/router_harness.py`), a label-safe
MOABB bridge (`h2cmi/data/real_eeg_bridge.py`), two unit/smoke tests, the Step-2..Step-4 experiment and
packaging scripts, and the `notes/PROJECT_B_*.md` evidence/paper docs. No `h2cmi/**` core model or
`cmi/**` file was modified outside the new router/bridge/harness/test files.

## 4. Router architecture
Action-specific blockers: support / stability / diagnostic failures block **every** action (including
IDENTITY); TTA-evidence and ACAR-harm failures block only the TTA actions. Selection is
**safe-beneficial-then-identity**: a beneficial admissible TTA can win; else a support-valid IDENTITY;
else REFUSE. Support is source-only (q95 in-source target-prior NLL; nested variant adds a
scale-normalised held-out-unit excess). ACAR gives per-action split-conformal error/harm bounds with an
explicit `available / degenerate / unavailable` state; when harm is degenerate/unavailable, TTA is
blocked and no harm bound is fabricated.

## 5. Evidence package
Synthetic frozen worlds (Step-2G), a real-EEG bridge smoke (Step-3A), a bounded real benchmark
expansion (Step-3C), and a merged paper draft + claim boundary + reviewer checklist (Step-3D). All are
regenerable from the recorded commands; none is re-run by this packaging step.

## 6. Synthetic results
`R2` nested support fixes the Step-2E all-refuse over-refusal (coverage 0 -> 0.83, accepted bAcc 0.880).
`HF3` / `H-OOD` show harmful TTA and the router's limits (concept-degraded identity can be support-valid;
LOW_ESS remains the active blocker after nested widening). **No beneficial-TTA recovery** under
degenerate ACAR-harm — the forgone R2 benefit is a knowing missed benefit.

## 7. Real EEG results
BNCI2014_004 bounded expansion (4 targets, subject + session, both support modes):
- raw offline TTA harmful, mean d_bAcc = -0.140;
- OFFLINE_TTA never selected (rate 0.00);
- subject-level coverage 0.50; session-level coverage 0.40;
- SUPPORT_MISMATCH dominates refusal;
- LOW_ESS inactive in Step-3C;
- nested support inert (nested excess ~ 0 -> nested == baseline).

## 8. Label-safety and leakage-safety
Target labels are post-hoc only; support thresholds are source-only; no target-label threshold tuning.
`cmi_residual` is not read at route time (deployment emits `OACI_LEAKAGE_RESIDUAL_UNAVAILABLE`). The
harness fails loudly if OFFLINE_TTA is ever selected while ACAR-harm is degenerate/unavailable.

## 9. Claim boundary
Claimable (7 items) and NOT-claimable (8 items) are frozen in
`notes/PROJECT_B_CLAIM_BOUNDARY.md` / `claim_boundary_final.json`. Headline: the router blocks unsafe TTA
and routes support-valid IDENTITY vs REFUSE with an OACI audit; it does **not** improve accuracy over
identity, recover beneficial TTA, solve concept shift, or constitute a full benchmark.

## 10. Known limitations
Source-only ACAR-harm is frequently degenerate; support can't detect concept-shift accuracy loss; the
real benchmark ran in a harmful-TTA regime (so it demonstrates harm avoidance, not accuracy gain); the
real evidence is a bridge smoke plus a bounded expansion, **not a full benchmark**.

## 11. Tests and validation
Eight fast/medium contract + self-tests pass (see `test_matrix.csv`); the Step-2G / Step-3D / Step-4
generators validate their own outputs (overclaim guard, required mentions, TTA-blocker
identity-count-0 invariant, no `h2cmi/**` / `cmi/**` diff).

## 12. Files changed
39 additive files across router-core / router-harness / real-eeg-bridge / tests /
experiment-scripts / documentation. Full list in `file_inventory.csv`.

## 13. Recommended next step after merge
Prepare the actual PR into `main`, then (separately scoped) Step-3E real-dataset expansion
(BNCI2014_001 / Lee2019_MI, GPU, more targets, session-level routing, no target-label tuning) on an
expansion branch.
