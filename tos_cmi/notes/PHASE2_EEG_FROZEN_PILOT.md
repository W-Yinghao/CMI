# Phase 2 — Frozen-feature EEG pilot (measurement / diagnostic only)

**Boundary:** measurement only. NO TOS-CMI end-to-end training, NO selective-penalty training, NO
`task_protect`, NO deletion promise. The certified gate's role here is a **safety diagnostic +
refuse-to-delete** module (the task-protected certification line closed as an honest negative —
[PHASE131_CERTIFICATION.md](PHASE131_CERTIFICATION.md)).

## Setup
- Data: **BNCI2014_001 / BCI-IV-2a** (9 subj, 4-class feet/left/right/tongue, 2 sessions, 22ch,
  250→128 Hz), offline via `cmi.paths.configure_offline_moabb`.
- Backbone: **TSMNet**, frozen **LogEig tangent latent Z** (z_dim=210). Trained per LOSO fold on
  source (all non-target subjects); target subject dumped for REPORT only (never in selector/gate).
- Methods dumped: **ERM** (`erm:0`) + **global LPC** (`lpc_prior:λ`), λ ∈ {0.03,0.1,0.3,1.0,3.0}.
- Standalone runner `tos_cmi/run_eeg_frozen_pilot.py` (imports cmi/ stack, NO trainer mod);
  diagnostic `tos_cmi/eeg/report.py` (CPU). Env `icml`, GPU via `scripts/tos_eeg_frozen_pilot.sbatch`.
- domain = source **subject**. Artifacts: `results/tos_cmi_eeg_frozen/<dataset>_<backbone>_LOSO/`.

## The five questions this pilot answers
1. Does TSMNet global-LPC collapse reproduce (and is it λ-fragile)?
2. Is the score-Fisher spectrum stable on real frozen ERM features?
3. Is there a domain-rich / task-light candidate subspace?
4. Does the certified gate delete, or refuse (identity)? Why?
5. Do the TOS diagnostics EXPLAIN the global-LPC collapse / λ-sensitivity?

## PRELIMINARY (subj1 only, 40 epochs — plumbing validation, NOT the result)
```
config         tgt_bAcc  src_fit  labelP  domP_adv  score-Fisher (trial-level)
ERM            0.561     0.687    0.66    +0.87     TASK_RISK_UCB k=0  (domain dirs task-entangled)
LPC λ=0.3      0.557     0.693    0.65    +0.87     NO_CANDIDATE  k=0
LPC λ=1.0      0.460     0.418    0.49    +0.86     ACCEPTED k=2 (on a COLLAPSED model)
```
Early signal supports the hypothesis: on ERM features the domain-rich directions are TASK-ENTANGLED
(k1 task_ucb 0.031 > δ_Y → gate REFUSES, TASK_RISK_UCB), and global LPC at λ=1.0 COLLAPSES the label
task (src fit 0.69→0.42, labelP 0.66→0.49). The gate refusing is the POSITIVE diagnostic. Strong
subject structure (domain-probe advantage +0.87). NOTE: 40 epochs is under-trained; one fold only.

Certification caveat (recorded): with domain==cluster==subject under LOSO, group-aware folds cannot
cover all subjects per fold → FOLD_COVERAGE_FAILURE → certified (group-aware) selective deletion is
infeasible in this regime; the exploratory diagnostic uses trial-level folds.

## RUNNING: full first round (seed 0, all 9 LOSO folds, λ sweep, 300 epochs)
jobs 875421-875429. To fill once complete: collapse-vs-λ curve, ERM score-Fisher decision across
folds, ERM-vs-LPC feature geometry, then 3-seed stability. Pass/stop conditions in the Phase-2 plan.
