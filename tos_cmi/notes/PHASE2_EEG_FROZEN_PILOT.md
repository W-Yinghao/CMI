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

## SEED-0 RESULTS (9 LOSO folds, 300 epochs) — coherent honest result

### (Q1) Global-LPC λ-fragile COLLAPSE reproduces decisively
```
config       tgt_bAcc src_fit labelP domAdv effrk   score-Fisher gate (9 folds, trial-level)
ERM          0.403    0.814   0.74   0.87   169     TASK_RISK_UCB 4 / ACCEPTED 5  (task_ucb~0.035~δ_Y)
LPC λ=0.03   0.403    0.816   0.74   0.87   169     ACCEPTED 6 / TASK_RISK_UCB 2 / NO_CAND 1
LPC λ=0.1    0.402    0.823   0.75   0.87   169     ACCEPTED 6 / TASK_RISK_UCB 3
LPC λ=0.3    0.402    0.821   0.75   0.87   169     ACCEPTED 6 / TASK_RISK_UCB 3
LPC λ=1.0    0.252    0.251   0.25  -0.01   159     DOMAIN_GATE_CLOSED 9
LPC λ=3.0    0.250    0.250   0.25  -0.01   164     DOMAIN_GATE_CLOSED 9
```
Sharp transition λ=0.3→1.0: at λ≥1.0 EVERYTHING collapses to chance (target & source label task &
domain all → 0.25/−0.01). The TSMNet/2a global-LPC λ-fragility counterexample is reproduced.

### (Q3,Q5) Projection ablation — domain leakage is NOT low-rank removable (the decisive test)
ERM (16 dumps, 2 seeds): `task Z=0.75 RZ=0.74 PNZ=0.35 | domain Z=1.00 RZ=0.96 PNZ=0.82 randR=1.00`.
Deleting the domain-rich/label-light V_D PRESERVES the task (Δtask≈0, PNZ≈chance → V_D label-light
out-of-sample) but BARELY removes subject leakage (1.00→0.96, ≈ removing RANDOM directions).
Subject info is high-dim & REDUNDANT (0.82-decodable from V_D AND 0.96 from its complement). So
NO low-rank removable nuisance subspace exists.

### (Q2,Q4) Score-Fisher gate: borderline + (per ablation) vacuous
ERM task_ucb mean ≈ δ_Y → 5/9 ACCEPT, 4/9 refuse (unstable, like the synthetic cert). The 5 accepts
are `diagnostic_accept` (NO EEG exact-scope power certificate → certified_accept=False), and the
ablation shows they do NOT actually remove domain → vacuous. At λ≥1.0 the gate correctly returns
DOMAIN_GATE_CLOSED (no structure in the collapsed rep).

### Synthesis (Q5)
Domain leakage on real ERM TSMNet features is DISTRIBUTED & REDUNDANT, not concentrated in a
low-rank task-light subspace. => low-rank selective deletion CANNOT remove it; global LPC removes it
only by collapsing the whole representation (killing the task) — the observed λ-fragility. The TOS
measurement framework correctly diagnoses this: borderline/vacuous selective candidates + correct
abstention, explaining WHY global LPC collapses. This is a measurement-to-control NEGATIVE for
deployable selective deletion here, and a positive for the diagnostic/refuse-to-delete role.

### 3-SEED STABILITY (seeds 0,1,2) + adversarial workflow verification — FINAL (corrected)

Stability (all 3 seeds): ERM tgt_bAcc ~0.39-0.40 / labelP ~0.75 / domAdv +0.87; LPC λ≥1 collapse to
chance; ablation domain_RZ ~0.95-0.96; gate ERM 5-7/9 "accept"; subject decode 1.00 ≫ session 0.90.
All four findings reproduce across seeds.

A 4-agent adversarial workflow (independent re-derivation from raw npz/code) produced TWO CORRECTIONS:

1. **Global-LPC λ-collapse is an OPTIMIZATION OBJECTIVE-SCALING failure, not a representation-geometry
   failure.** [SETTLED in Phase 2.1 — see PHASE21_LPC_COLLAPSE_MECHANISM.md.] Phase 2.0 inferred this
   indirectly from bimodality (26/27 collapse at λ=1). Phase 2.1 confirmed it with per-epoch curves +
   a 4-agent adversarial verification: at λ≥1 the objective bifurcates (sharp λ-cliff, deterministic)
   to a degenerate trivial minimizer = **feature-norm collapse to the ORIGIN** (feat_norm 1.09→0.0000,
   top-1 SV→0.001, penalty→~0, CE→ln4=chance). It is NOT a gradient explosion (abs peak grad at
   collapse is ~10× SMALLER than healthy training; "50–300×" was an artifact of dividing by the
   post-collapse near-zero median; 0/36 nonfinite) and "eff_rank stays high" is NON-PROBATIVE
   (scale-invariant metric). Mechanism = sharp optimization bifurcation that PRODUCES geometric
   compression to a point; directly-opposed task/LPC gradients (cos=−0.99 at λ=3).

2. **V_D removal DENTS but does not REMOVE domain (and it DOES beat random).** Correcting the earlier
   "≈ random": deleting V_D drops subject decode 0.997→0.955 (MLP) / 0.998→0.914 (linear), while
   removing k RANDOM dirs drops it ~0 (→0.997) — so V_D PREFERENTIALLY captures domain (nDcand 2-5,
   real domain energy). BUT the drop is small (domain stays ~0.95) and task is fully preserved
   (task_RZ≈task_Z, Δ≈0) ⇒ subject leakage is high-dim/REDUNDANT; low-rank deletion is INSUFFICIENT
   to remove it. (robust, SUPPORTED.)

3. **No TARGET leakage** (SUPPORTED): held-out target data never enters M/Fisher/candidate/gate/probe
   — only tgt_bacc/tgt_nll read target keys (report.py). (The `target` param name in score_fisher is
   an unrelated probe-label overload, not a leak.)

4. **V_D is effectively label-light** out-of-sample: PNZ task ≈ chance+0.04 (≈ random-k level) under
   linear AND MLP (0.36 vs full 0.75); deletion preserves task. ("exactly chance" technically
   refuted; practically label-light.)

### FINAL Phase 2.0 conclusion (honest, verified)
On real TSMNet/2a frozen features: the score-Fisher selector DOES find a genuine domain-preferential,
~label-light low-rank subspace, but deleting it only DENTS the subject leakage (high-dim/redundant)
→ low-rank selective deletion cannot REMOVE the leakage; and global LPC "removes" it only by a
λ-fragile OPTIMIZATION objective-scaling collapse to the trivial Z→0 solution (Phase 2.1 confirmed:
feature-norm collapse to the origin via a sharp λ-bifurcation, NOT gradient explosion, NOT smooth
geometric compression). So: measurement-to-control POSITIVE for diagnosis (the framework localizes the
leakage subspace + correctly shows low-rank deletion is insufficient + the certified gate abstains),
NEGATIVE for BOTH deployable knobs available here (global LPC = objective-scaling collapse; low-rank
selective deletion = insufficient). NOT a frozen-projection pilot. task_protect + power_floor stay OFF;
gate = diagnostic/refuse-to-delete. **Phase 2 (2.0 + 2.1) COMPLETE.**
