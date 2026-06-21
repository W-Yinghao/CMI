# OACI — experimental design

Strict DG throughout: source-only model selection, no target data, no target calibration
(Protocols A–C from the top-level [`README.md`](../README.md)). The headline experiment is
**not** the main table — it is the **controlled missing-cell stress test**, because the
contribution is about *support mismatch*.

## Datasets (unified preprocessing, common montage map)

| Dataset | role | domain factor(s) | classes |
|---|---|---|---|
| BNCI2014-001 (2a) | core MI, LOSO + cross-session | subject, session | 4 |
| BNCI2014-004 (2b) | core MI, LOSO + cross-session | subject, session | 2 |
| Cho2017 | cross-dataset binary MI | subject (single session) | 2 (L/R) |
| Lee2019 (OpenBMI) | large-scale LOSO + cross-session | subject, session | 2 |
| SEED | affective, cross-subject | subject, session | 3 |
| SCZ cross-site | clinical, cross-site | site, subject | 2 |
| PD cross-site | clinical, cross-site | site, subject | 2 |

The clinical SCZ / PD cross-site pairs are where natural **site×class support mismatch**
occurs (some sites short on a class); MI sets give a controlled baseline with full support.

## Main experiment — controlled missing-cell stress test

Driven by [`data/missing_cell.py`](data/missing_cell.py) (`make_schedule`): start from a
fully-supported configuration, then **systematically delete site×class cells** (`(d,y)`
pairs) to sweep the support graph from connected → fragmented. The harness fixes the cell
mask, reference weights, group IDs and deletion schedule that all four methods share. At each
deletion level, compare on identical splits/seeds:

1. **ERM** (no invariance penalty),
2. **global LPC** (conditional invariance over *all* cells, smoothing the missing ones — the
   failure mode),
3. **uniform-prior** alignment (`π_y` = uniform over domains),
4. **support-aware OACI** (this method — penalty restricted to comparable cells, risk-feasible).

Hypothesis: OACI ≈ ERM/global-LPC when support is full, and **strictly degrades less** (in
worst-domain accuracy / calibration / label separability) as cells are deleted and the
support graph fragments — because global/uniform routes erase `Y` on the unsupported cells
(THEORY §3) while OACI leaves them free.

Report the deletion level at which the support graph first **fragments**
(`MissingCellSchedule.first_fragmentation_level()`) and align it with where global/uniform
start to hurt worst-domain accuracy / calibration.

## Estimand (fixed across the sweep)

Report the **primary** ``L_abs = Σ_{y∈C_cmp} p_ref(y) L_y`` paired with the identifiable mass
fraction ``Σ_{y∈C_cmp} p_ref(y)``, both under a **fixed** reference prior ``p_ref`` computed
once on the full pre-deletion configuration (so the estimand does not drift as cells are
deleted). ``L_cond`` (renormalised over comparable classes) is **diagnostic only** — its
weights move with support fragmentation. As cells are deleted, ``L_abs`` falls because
comparable mass leaves the sum; that drop is the honest consequence of non-identifiability,
reported, not renormalised away. (THEORY §Estimand.)

## Metrics

* **Grouped capacity-sup extractable leakage ``L_Q^ov``** — recording/subject-grouped,
  capacity sup over the probe family `Q`, cross-fit, permutation null. This is the operational
  *lower bound* on `I_ov` (THEORY §4); the optimization target is ``UCB_{1-α}[L_Q^ov]`` with
  capacity selection inside each bootstrap resample. (Never reported as "precise CMI".)
  Implemented in [`leakage/`](leakage/) — `estimate_extractable_leakage` (point `L_abs`/`L_cond`)
  and `bootstrap_ucb` (`bootstrap_ucl`, basic one-sided; `percentile_ucl` as sensitivity).
* **Mean** and **worst-domain** balanced accuracy. Worst-domain is a primary endpoint.
* **ECE / NLL** — calibration, source-only (no target temperature). Watch for the LPC trap:
  a calibration "win" that is just global confidence rescaling (oracle-T deconfound).
* **Domain×class cell-wise label separability** — linear probe per observed cell; the §3
  preservation check (does OACI keep `Y` where global/uniform erase it?).
* **Noninferiority CI vs ERM** — paired, domain/subject-clustered bootstrap CI on the
  source-risk gap (the §2 realized-`ε` check) and on the target balanced-accuracy delta.

## Ablations

* support threshold `m` sweep (cell estimability vs coverage trade-off);
* point-estimate leakage vs UCB-minimization (does minimizing the *bound* matter?);
* clustered vs i.i.d. bootstrap (does the dependence unit change the UCB / conclusion?);
* `ε` sweep (risk-feasibility slack);
* rare-cell batch sampler on/off (does keeping comparable cells estimable per step matter?).

## Kill / termination criteria (registered BEFORE tuning)

Declare the conditional-invariance angle **not a downstream benefit** and stop if either:

* **K1 — leakage collapses under honest grouping.** After the recording-grouped max-probe
  cross-fit probe, the leakage *difference* between OACI and ERM is within the permutation
  null band (i.e. there was little extractable grouped domain info to remove). Then there is
  nothing to control.
* **K2 — no downstream payoff.** Reducing leakage yields no reproducible improvement (CI
  excludes/touches 0 across seeds) in **either** worst-domain balanced accuracy **or**
  calibration, across the stress-test sweep.

Additional collapse guard (carried from LPC P1.5): if a leakage reduction co-occurs with
representation eff-rank / source-utility dropping past pre-set gates, the reduction is
**via-collapse** and does not count as invariance.

## Compute

conda env `icml` (`/home/infres/yinwang/anaconda3/envs/icml/bin/python`); SLURM V100/A100
via `sbatch` (login node has no GPU). Data root read-only datalake (see top-level README).
Save `.preds.npz` per run so metrics/leakage can be recomputed without GPU retrain.
