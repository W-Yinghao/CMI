# CMI-Trace Task 7 — cross-backbone AMOUNT vs USE — RESULTS (neutral, adversarially verified)

SLURM `907879` (CPU, 8-way joblib, ~3.4 h). Coverage **72/72** probe cells (3 datasets × 2 backbones × first-12
LOSO folds) + 36 TSMNet exact-head validation dumps; **firewall 0 fail, head-replay 0 fail**, n_perm=100,
k_spec=16, n_random=50, seed=0. Fold-cluster bootstrap n_boot=10000. Pre-reg: `CMI_TRACE_TASK7_AMOUNT_USE_FROZEN.md`.
Aggregation independently re-derived + adversarially stress-tested by a 5-agent verification workflow
(`wf_e9169a38-caf`, 4 skeptics + synthesis) before this write-up.

## Endpoints (per backbone, pooled over datasets; fold = cluster unit)

| cell | E7.1 λ_frac_sig | E7.2 τ−random | E7.2 \|τ\| (nats) | mean λ_excess | E7.3 corr(λ,τ) RAW | E7.3 partial(λ,τ\|energy) |
|---|---|---|---|---|---|---|
| **EEGNet** (36 folds) | +0.997 [.991,1.000] | −0.006 [−.006,−.005] | +0.0011 [.001,.001] | +0.076 | +0.076 [.046,.109] | **−0.325 [−.403,−.256]** |
| **TSMNet** (36 folds) | +1.000 [1,1] | −0.001 [−.005,+.002] | +0.0121 [.010,.014] | +0.181 | +0.116 [.072,.158] | **−0.032 [−.091,+.029]** |

Per-cell partial(λ,τ\|energy): Lee/EEGNet −0.504, Cho/EEGNet −0.502, BNCI2015/EEGNet −0.145 (all neg, CI excl 0);
Lee/TSMNet −0.065 (null), Cho/TSMNet +0.173, BNCI2015/TSMNet −0.240. **E7.5 (TSMNet probe-vs-exact):**
corr(τ_probe,τ_exact) = **+0.994 [.993,.996]**, mean\|Δτ\| = 0.0013.

## Headline (under-claimed)
Across two independent backbones (EEGNet, TSMNet) the four-object separation **replicates**: subject
information is **encoded** — per-direction λ-significance is label-dependent (λ-excess ≈0.076 / 0.181 nats) —
while its functional **Use is negligible**: **\|τ\| ≤ 0.012 nats** on both backbones, and on both, subject
directions are used **no more than random directions** (τ−random = −0.006 / −0.001, i.e. **below-random Use**).
Where the deployed head is linearly checkable (TSMNet), the source-fit probe tracks it at corr = 0.994.
**We claim negligible, below-random Use — NOT perfect decoupling:** the raw λ–τ correlation is nonzero
(+0.08 / +0.12) but is an **energy-rank confound** that reverses to strongly negative (EEGNet, −0.33) or
vanishes (TSMNet, null) once subject-direction energy is controlled — which if anything reinforces
"encoded-but-not-used." (Conditional on energy, higher-Amount directions are NOT more Used.)

## Adversarial verification verdicts (`wf_e9169a38-caf`, 5 agents, Opus-4.8)
- **V1 arithmetic — CONFIRMED.** Independent recompute (no `aggregate()` import) matches every reported figure
  to 10 decimals; coverage genuinely 72/72 unique (ds,bb,fold), fold==sub, 16 dirs/fold. Note: E7.3/E7.5 point
  estimates are **pooled Pearson over 576 concatenated directions** (only the bootstrap CI resamples folds);
  E7.2/mean_λ equal grand direction means only because every fold has exactly 16 directions.
- **V4 determinism/firewall/tautology — CONFIRMED.** Bit-identical self-replay (max\|Δ\|=0); firewall fails
  loud on corrupt target-tag / missing target_indices (has teeth); probe vs exact heads are **distinct**
  (\|cos\|≈0.997, not 1.0) so 0.994 is not a construction tautology — but it is a **soft, TSMNet-only** check
  (binary near-logistic head); it does **not** validate the EEGNet probe (EEGNet deployed head genuinely
  nonlinear, lstsq replay fails 1.76–6.82 logit units / +0.04–0.09 CE nats → probe-only by necessity).
- **V3 corr-confound — sub-claim REFUTED (the important catch).** λ is largely an energy-rank proxy
  (r(λ,rank)=−0.905 EEGNet / −0.517 TSMNet). The raw positive corr(λ,τ) is reproducible and **not** outlier-
  driven, but under the correct energy control it **reverses to −0.44 (EEGNet)** / **null (TSMNet)**. The
  "weak positive coupling" is not real; report the partial corr alongside the raw, always.
- **V2 λ-null — PLAUSIBLE (not CONFIRMED).** The null does **not** over-fire: label-shuffled subject ids give
  frac_sig 0.00 / p 0.607 (significance is label-dependent). BUT (i) that control is **smoke-only** (n_perm=20,
  K=4, one EEGNet fold; TSMNet untested); (ii) a **selection double-dip** surfaced — top subject-scatter
  directions are selected using rows that include the eval split the ruler scores (pure-null control fired ~0.50).

## Caveats that MUST survive into any manuscript use
1. **E7.3 is an energy-rank confound, not a coupling** — always report raw (+0.08/+0.12) *with* the partial
   (−0.33 EEGNet / null TSMNet). Never present the raw corr as a standalone Amount→Use result.
2. **Use is below-random, not merely small** — \|τ\| ≈ 0.001–0.012 nats AND τ < random on both backbones.
3. **"Perfect decoupling" is NOT claimed** — the raw λ–τ correlation is nonzero; the residual is an energy
   artifact, not evidence of zero relationship.
4. **E7.5 r=0.994 is a soft, TSMNet-only check** — reflects binary near-logistic head geometry; does NOT
   validate the EEGNet probe.
5. **"Pervasive" encoding is PROVISIONAL** — say **"label-dependent encoding"** until the full K=12/n_perm=100
   negative controls run on BOTH backbones AND a selection-honest control (directions fit on train-only rows,
   eval held out of selection) close.
6. **Disclose the selection double-dip** and control it before asserting genuine pervasiveness.
7. **Do NOT use "random directions also fire ⇒ artifact"** — random directions fire because they carry real
   subject structure (they die under label-shuffle); the correct null is the label-shuffled control.
8. **Report the pooling convention** — E7.3/E7.5 pooled over 576 directions; CIs resample folds; E7.2/mean_λ
   equal grand means only under the exactly-16-dirs/fold design.

## Open follow-ups (to upgrade "label-dependent" → "pervasive")
- Full negative-control fleet: label-shuffled + random-direction controls at K=12, n_perm=100, on **both**
  backbones × 3 datasets (not smoke).
- Selection-honest spectrum: fit the subject-scatter directions on a train split disjoint from the ruler's
  eval rows; re-report λ-significance.
Neither changes the **Use** result (E7.2/E7.3/E7.5 stand); they only bear on the strength of the **Amount** word.
