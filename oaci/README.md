# OACI — Overlap-Aware Risk-Feasible Conditional Invariance

**Working title:** *Partial Conditional Invariance under Domain–Class Support Mismatch.*
**Setting:** strict domain generalization (DG) — source-only model selection, **no** target
data, no target calibration. Target: a complete strict-DG paper for EEG clinical cross-site.

This package is **isolated** from `cmi/` (the CLOSED LPC line — see top-level
[`CLOSEOUT.md`](../CLOSEOUT.md) / [`notes/EVIDENCE_LEDGER.md`](../notes/EVIDENCE_LEDGER.md))
and from `h2cmi/`. Nothing here imports either. It is a fresh direction, designed to *not*
repeat the three documented failure modes of the LPC line.

---

## The problem

The global conditional-invariance objective `min_θ I(Z;D|Y)` quietly assumes every
`p(z|y,d)` we want to compare has enough support to estimate. Clinical EEG routinely
violates this:

* some sites have no examples of a class;
* a class has only a handful of samples in some domains;
* subject ↔ label is nearly one-to-one;
* the sampler distorts the effective `p(D|Y)`.

So the cells we'd compare are **partially observed**. Smoothing a zero-/tiny-sample cell
pretends a comparison exists where the data cannot support one — which is exactly how the
LPC line produced *via-collapse* "invariance" and forced *Y-erasure* on unsupported cells.

## The idea: only compare what the data can estimate

Let `n_{d,y}` be the effective sample count and fix a support threshold `m`. Define the
per-class **support set**

```
S_y = { d : n_{d,y} >= m }.
```

Define leakage **only on observed cells**, the overlap-aware conditional MI:

```
I_ov(Z;D|Y) = Σ_y p(y) · I( Z ; D_{S_y} | Y=y, D ∈ S_y ).
```

Cells with zero / too-low effective sample size are **not** smoothed into existence — they
are explicitly marked **ineligible** (a finite-sample *estimator-eligibility* flag, not a
population claim — see THEORY §0). This bookkeeping is the [`support_graph`](support_graph.py)
module (implemented + tested): it builds `S_y`, the comparable classes (`|S_y| ≥ 2`), the
**per-class** identifiable pairs (`is_identifiable_pair(d,d',y)` — both in `S_y`; there is
**no** cross-class transitive reach), and the domain–domain **coupling** components — which
describe only how the constraint system *decomposes*, NOT which equalities are identifiable
(THEORY §1).

## The objective: risk-feasible, honestly-bounded

Two-stage **lexicographic constrained** optimization. First fit ERM for the source-risk
lower bound `R_ERM`; then

```
min_θ   UCB_{1-α}[ L_Q^ov ]          (NOT UCB[I_ov] — see below)
s.t.    R_src(θ) <= R_ERM + ε.
```

* **The target is the probe-class-*extractable* leakage `L_Q^ov`, not the true `I_ov`.** A
  finite probe `q` gives `E[-log q(D|Z,Y)] = H(D|Z,Y) + E·KL(p‖q)`, so the probe entropy-gap
  is a *lower* bound on `I_ov` (THEORY §4). We minimise/bound `L_Q^ov = sup_{q∈Q}(Ĥ(D|Y) −
  E[-log q])` — "extractable conditional domain information", the project's binding name.
* **`UCB_{1-α}[L_Q^ov]` is a real upper confidence bound on that functional** — cross-fit
  critic + domain/recording-clustered bootstrap + a capacity sup **selected inside each
  resample** — **not** a posterior-KL relabelled "upper bound" (the precise LPC error; see
  [`notes/EVIDENCE_LEDGER.md`](../notes/EVIDENCE_LEDGER.md)).
* **`λ` is a primal–dual multiplier** that enforces the risk constraint automatically. It
  is a Lagrange knob, **not** part of the model's meaning.
* The constraint is **noninferiority** w.r.t. ERM: we never trade away source accuracy past
  `ε` to chase a leakage number.

## Why this is new (and not just "ERM-feasible DG penalty")

Adding a risk constraint to a DG penalty already exists. The novelty is the combination:

1. **Partial identifiability under a domain–class support graph** — which conditional
   invariances are estimable at all (disconnected support ⇒ untestable cross-component
   invariance).
2. **Support-aware conditional MI** — the objective is defined on observed cells only.
3. **Estimator uncertainty as the thing minimized** — an honest UCB, not a point proxy.
4. **EEG rare-cell batching** — sampling that keeps comparable cells estimable per step.
5. **Worst-domain + calibration** outcomes beyond mean-risk noninferiority.

## Three theory targets (see [`THEORY.md`](THEORY.md))

1. **Support theorem** — the equality `p(z|y,d)=p(z|y,d')` is identifiable iff both cells are
   eligible for that **same** class `y` (`d,d'∈S_y`); there is no cross-class transitive
   reach, and graph connectivity is only *decomposability*, not identifiability (THEORY §1).
2. **No-excess empirical risk** — at an exact feasible solution stage 2 adds at most `ε` to
   source risk over ERM (reported with the realized gap; the optimality+penalty pairing is
   the safety floor, not the novelty).
3. **Label-preservation proposition** — not aligning ineligible cells avoids the *explicit*
   forced `Y`-erasure of the uniform/marginal/chain routes; the stronger "representation left
   free" holds only under a cell-separable parametrization (shared encoders still couple via
   gradients — THEORY §3).

## Component map (proposal → module)

| Concept | Status | Module |
|---|---|---|
| Support graph: `S_y`, eligibility vs presence, **per-class** identifiable pairs, coupling components, fixed-prior `L_abs`/`L_cond` | **implemented + tested** | [`support_graph.py`](support_graph.py) |
| Config: support threshold, UCB on `L_Q^ov` (cross-fit/bootstrap/probe), risk constraint | **implemented** | [`config.py`](config.py) |
| Controlled missing-cell harness — deletion schedule, fixed cell mask / reference weights / group IDs | **implemented + tested** | [`data/missing_cell.py`](data/missing_cell.py) |
| Per-class conditional-domain **probe** (frozen Z, label space `S_y`, in-fold preprocessing, capacity family) | **implemented + tested** | [`leakage/critic.py`](leakage/critic.py) |
| Strict recording/subject-grouped **cross-fit** (domain-stratified folds, feasibility, OOF NLL) | **implemented + tested** | [`leakage/crossfit.py`](leakage/crossfit.py) |
| **Point estimate** `L_abs`/`L_cond` (`Ĥ_y−NLL^OOF`, capacity sup after aggregation) | **implemented + tested** | [`leakage/estimate.py`](leakage/estimate.py) |
| Recording-clustered **bootstrap UCB** (within-domain resample, in-replicate capacity reselection, basic one-sided + percentile) | **implemented + tested** | [`leakage/ucb.py`](leakage/ucb.py) |
| **Risk-feasible trainer** — PyTorch conditional adversary `C_D`, primal–dual `min(H_ref−C_D)+λ(R_src−τ)`, dual on the risk constraint, feasibility selector + byte-exact ERM fallback | **implemented + tested** | [`train/`](train/) |
| **Rare-cell paired-stream sampler** — task stream (incl. ineligible cells) + adversary stream (eligible cells, covers all per logical step); importance weights restore fixed `p(d\|y)`/`p(y)`; microbatch accumulation normalised by fixed `N_ov` | **implemented + tested** | [`data/sampler.py`](data/sampler.py), [`data/batch.py`](data/batch.py) |
| **Eval** — fixed-estimand pooled/mean/worst-domain bAcc + worst-paired-Δ, NLL/ECE (no target calibration), paired clustered bootstrap (one plan reused; whole-group, no row fallback; invalid-rate; too-few-clusters→non-estimable), noninferiority rules, missing-cell sweep scalar `ΔA_post` | **implemented + tested** | [`eval/`](eval/) |

The sampler never redefines eligibility/`S_y`/`p_ref`/`n_{d,y}` (fixed full-data support graph);
a batch only guarantees eligible-cell **coverage**. `w^adv=n_{d,y}/m` restores the fixed empirical
`p(d|y, d∈S_y)` (not the sampler's near-uniform per-cell draw); `w^task=n_y/m_y` restores `p(y)`.
The trainer integrates it via [`train_risk_feasible(..., sampler=...)`](train/primal_dual.py).

The trainer's inner game uses a **PyTorch** conditional-domain adversary (`train/adversary.py`),
deliberately distinct from the non-differentiable sklearn `extractable_LQ_ov` estimator in
`leakage/` (used only as an injected outer `score_fn` on frozen representations — never
back-propagated through).

## Run what exists

```bash
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY -m oaci.support_graph                 # worked disconnected-support example
$PY -m oaci.data.missing_cell             # missing-cell deletion sweep
$PY -m oaci.leakage.synthetic             # leakage demo: null (~0) vs perfect (≈Ĥ) extractable_LQ_ov
$PY -m oaci.tests.test_support_graph      # support-graph tests
$PY -m oaci.tests.test_missing_cell       # missing-cell harness tests
$PY -m oaci.tests.test_leakage_estimate   # estimator: null / perfect / exclusion / capacity / no-trunc
$PY -m oaci.tests.test_leakage_crossfit   # grouped cross-fit: group-memorisation, feasibility
$PY -m oaci.tests.test_leakage_ucb        # bootstrap UCB: reselection, formulas, reproducibility
$PY -m oaci.train.synthetic               # risk-feasible trainer acceptance report (feasible + leakage↓)
$PY -m oaci.data.sampler_demo             # rare-cell sampler report (50:1 imbalance; weighted p(d|y) restored)
$PY -m oaci.eval.synthetic                # eval panel: pooled looks fine but a small domain is harmed
$PY -m oaci.tests.test_rare_cell_sampler  # sampler: coverage, exact priors, microbatch invariance, fixes
$PY -m oaci.tests.test_eval               # eval: estimands, in-bootstrap worst-domain, NI rules, fixed pop
# all of the above, in parallel on a CPU compute node (off the login node):
sbatch oaci/slurm_ci.sh
$PY -m oaci.tests.test_train_risk         # primal metric: balanced_ce partition-invariant, balanced_err rejected
$PY -m oaci.tests.test_train_adversary    # adversary: grad signs, ineligible no-grad, fixed p_ref, no-op
$PY -m oaci.tests.test_train_primal_dual  # dual direction, ERM immutability, seed reproducibility
$PY -m oaci.tests.test_train_selector     # feasibility selection + byte-exact ERM fallback
$PY -m oaci.config                        # config validation
```

## Kill / termination criteria (pre-registered, before any tuning)

Stop and **do not** claim conditional invariance as a downstream benefit if, after the
recording-grouped probe, **leakage differences essentially vanish**, OR if reducing leakage
leaves **worst-domain accuracy and calibration with no reproducible improvement**. See
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the exact thresholds. This package inherits the LPC
line's culture: register kill criteria first, surface what is non-identifiable, never
relabel a proxy as a bound.

## Status / honesty

The full pipeline is **implemented and unit-tested on synthetic data** (support graph,
missing-cell harness, extractable-leakage estimator + bootstrap UCB, rare-cell paired-stream
sampler, risk-feasible primal–dual trainer, and the evaluation stack). It is a *research
implementation*: every component is correct, deterministic, and composes, and the synthetic
demos exercise the intended behaviours (e.g. the trainer stays risk-feasible; the eval panel
exposes a small-domain harm that pooled accuracy hides). It is **not** a real-EEG result — that
requires the confirmatory protocol in [`EXPERIMENTS.md`](EXPERIMENTS.md) (unified preprocessing,
5–10 seeds, recording/subject-clustered inference, the controlled missing-cell stress test) and
the pre-registered kill criteria K1/K2. Run everything off the login node with
`sbatch oaci/slurm_ci.sh`.
