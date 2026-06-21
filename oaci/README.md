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
are explicitly marked **non-identifiable**. This bookkeeping is the [`support_graph`](support_graph.py)
module (already implemented + tested): it builds `S_y`, the comparable classes
(`|S_y| ≥ 2`), the domain–domain support graph, and its connected components.

## The objective: risk-feasible, honestly-bounded

Two-stage **lexicographic constrained** optimization. First fit ERM for the source-risk
lower bound `R_ERM`; then

```
min_θ   UCB_{1-α}[ I_ov(Z;D|Y) ]
s.t.    R_src(θ) <= R_ERM + ε.
```

* **`UCB_{1-α}[·]` is a real statistical upper confidence bound** — cross-fit critic +
  domain/class-clustered bootstrap + multi-capacity probe — **not** a posterior-KL
  relabelled "upper bound" (the precise error the LPC line made; see the naming
  corrections in [`notes/EVIDENCE_LEDGER.md`](../notes/EVIDENCE_LEDGER.md)).
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

1. **Support theorem** — only invariance on observed `(d,y)` cells is data-identifiable;
   when the support graph is disconnected, cross-component global invariance requires extra
   untestable assumptions.
2. **No-excess empirical risk** — at an exact feasible solution, stage 2 adds at most `ε`
   to source risk over ERM.
3. **Label-preservation proposition** — not aligning unsupported cells avoids the forced
   `Y`-erasure of the uniform / marginal / chain routes.

## Component map (proposal → module)

| Concept | Status | Module |
|---|---|---|
| Domain–class support graph, `S_y`, components, non-identifiable cells | **implemented + tested** | [`support_graph.py`](support_graph.py) |
| Config: support threshold, UCB (cross-fit/bootstrap/probe), risk constraint | **implemented** | [`config.py`](config.py) |
| Support-aware conditional-MI critic + cross-fit point estimate | TODO | `leakage/` |
| Clustered-bootstrap + multi-capacity UCB | TODO | `leakage/ucb.py` |
| Lexicographic / primal–dual risk-feasible trainer | TODO | `train/` |
| Rare-cell batch sampler (keeps comparable cells estimable) | TODO | `data/` |
| Controlled missing-cell stress test (delete site×class cells) | TODO | `data/missing_cell.py` |
| Eval: grouped max-probe leakage, mean/worst bAcc, ECE/NLL, noninferiority CI | TODO | `eval/` |

## Run what exists

```bash
PY=/home/infres/yinwang/anaconda3/envs/icml/bin/python
$PY -m oaci.support_graph                 # worked disconnected-support example
$PY -m oaci.tests.test_support_graph      # 9 support-graph tests (standalone / pytest)
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

Day-0 scaffold. The **support-graph identifiability core is real and tested**; everything
downstream (critic, UCB, trainer, stress test, eval) is specified but not yet implemented.
No empirical claim is made yet.
