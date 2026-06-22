# H²-CMI — Stage V TERMINAL (read this first)

> **⛔ This package reached a frozen terminal state. The README below the line is the original
> design narrative and is RETAINED FOR CODE-MAP PURPOSES ONLY — several of its components
> (target-prior EM TTA, the source-only safety gate, online transform) were FALSIFIED or frozen
> during the program. Do not read it as the method's claims. The authoritative status is the
> result index.**

**Terminal result index** (`results/`, each with its own frozen pre-registration + sha256):
[`METHOD_FREEZE.md`](results/METHOD_FREEZE.md) ·
[`B1A_RESULTS.md`](results/B1A_RESULTS.md) ·
[`B2A_TERMINAL.md`](results/B2A_TERMINAL.md) ·
[`B2B_TERMINAL.md`](results/B2B_TERMINAL.md) ·
[`V2_FROZEN.md`](results/V2_FROZEN.md) · [`V2_RESULTS.md`](results/V2_RESULTS.md) ·
[`V2P_FROZEN.md`](results/V2P_FROZEN.md) · [`V2P_RESULTS.md`](results/V2P_RESULTS.md).
Immutable tags: `H2CMI_METHOD_FREEZE`, `h2cmi-stage-v-terminal` (+ `h2cmi-b1a-astar-terminal`,
`h2cmi-b2a-terminal`, `h2cmi-b2b-source-power-fail`).

**What the program actually found (one paragraph).** The contribution is a *measurement→control
gap*, not a winning adaptation method. (1) The factorial decomposition falsifies "CMI improves
TTA"; the joint EM's harm is the **prior M-step**, so geometry-only / fixed-prior beats the joint
(V1 claims, confirmed on fresh seeds 100–119; **the joint's harm replicates on real EEG**). (2) The
**deployed policy is metadata-only operator routing + identity-by-default** — NO target-statistic
eligibility gate (the source-calibrated gate's evidence score lacks transferable power: B2B ROC
ceiling < 0.25 at 10% false-adaptation). (3) On **real EEG** (V2, offline MOABB L/R): the
operator-support **safety** half holds — metadata abstains under acquisition mismatch (18/18
identity, exact equivalence) — but the **utility** half does NOT reach the bar (supported-regime
paired Δ +0.001, CI includes 0). (4) The controlled prevalence intervention (V2P) shows even
**fixed-prior CC is not prevalence-invariant** (occupancy slope −0.020, CI excludes 0; pooled −0.062,
joint −0.042) — supporting the revised bias theorem. Net: we can correctly *detect when not to adapt*;
unlabeled source-calibrated adaptation is *safe but marginal*. **Experiment search is closed.**

---

# H²-CMI — Hierarchical Class-Conditional Mutual-Information Adaptation for EEG  *(original design narrative — code map only)*

A self-contained implementation of the post-review (ICLR-direction) redesign. It is
**isolated** from the AAAI `cmi/` package — nothing here imports-with-side-effects or
mutates `cmi`; the two coexist in the repo. The whole pipeline runs without real EEG via a
controllable mechanism simulator, so every component is exercised by one smoke test.

```
# fast correctness unit tests (standalone; pytest-compatible too):
for t in test_source_dag_remap test_cmi_null_and_power test_cmi_gradient_sign \
         test_tta test_leakage_group_split; do
  conda run -n icml python -m h2cmi.tests.$t
done
conda run -n icml python -m h2cmi.tests.test_smoke          # slow end-to-end (integration)
conda run -n icml python -m h2cmi.run_synthetic --epochs 20 # report on a held-out site
```

See [`THEORY.md`](THEORY.md) for the four corrected derivations (P0-2…P0-5) and
[`CHANGES.md`](CHANGES.md) for the correctness fixes from the second code review.

## Why this exists (the review's verdict, in one line)

The AAAI work established three boundaries through negative results — invariance ≠ accuracy,
pooled alignment harms under label/concept shift, single-class targets are unidentifiable.
H²-CMI **encodes those boundaries into the algorithm** instead of trying to penalise around
them, and cleanly separates *strict DG* from *transductive/online TTA*.

## Component map (review section → module)

| Review | Component | Module |
|---|---|---|
| 5.1 / 5.4 / 5.5 | Domain-factor **DAG** (site→subject→session, crossed factors), per-factor handling + leakage budgets | [`domains/dag.py`](domains/dag.py) |
| R0 / 10.1 | **EEG mechanism simulator** — orthogonal cov / prior / concept / montage / noise / label-mechanism shift | [`data/eeg_simulator.py`](data/eeg_simulator.py) |
| 5.2 | **EEG encoder**: temporal filterbank + SPD/covariance + electrode-graph branches + fusion + constrained canonicaliser → `(z_c, z_n)` | [`models/encoder.py`](models/encoder.py) |
| 5.3 | **Class-conditional density** `p_φ(z_c\|y)` (Student-t mixture, low-rank+diag) + Bayes generative classifier + discriminative/density/JS **hybrid** head | [`density/student_t_mixture.py`](density/student_t_mixture.py) |
| 5.4 / **P0-2** | **Hierarchical CMI** via `H(D\|Y)−H(D\|Z,Y)`; per-factor critics, chain-rule decomposition, **primal-dual** leakage budgets | [`cmi/hierarchical.py`](cmi/hierarchical.py) |
| 4 / **P0-4 / P0-5** | **Reference-prior marginal alignment** + corrected **GLS** reference distribution | [`align/reference_marginal.py`](align/reference_marginal.py), [`align/distances.py`](align/distances.py) |
| 5.6 | **Disentanglement** of `z_c`/`z_n` (HSIC / cross-cov / nuisance & task probes) | [`disentangle/penalties.py`](disentangle/penalties.py) |
| 5.2 / 5.6 | **SSL aux** (masked reconstruction + VICReg) — anti-collapse | [`ssl/aux.py`](ssl/aux.py) |
| 6 | **Selective class-conditional probabilistic TTA** — constrained near-identity transform + target-prior **EM**, offline & online, identity fallback | [`tta/class_conditional.py`](tta/class_conditional.py) |
| 7 | **Source-only learned safety gate** — gain/harm predictor on inner LOSO diagnostics | [`gate/safety_gate.py`](gate/safety_gate.py) |
| 8 | **Site label mechanism** — latent `Y*`, shrinkage confusion matrix, EM | [`label/site_mechanism.py`](label/site_mechanism.py) |
| 3 / 10 | **Three-setting harness** (strict DG / offline TTA / online TTA), full metric panel, **cross-fitted signed leakage** + permutation null, **domain-clustered bootstrap** | [`eval/harness.py`](eval/harness.py), [`eval/metrics.py`](eval/metrics.py), [`eval/leakage.py`](eval/leakage.py) |
| — | Trainer wiring it all (two-step alternation, seeds-before-build, no `drop_last`) | [`train/trainer.py`](train/trainer.py) |
| — | Config dataclasses | [`config.py`](config.py) |

## Pipeline

```
EEG x ──► H2Encoder ──► (z_c, z_n)
                          │   │
        hybrid head ◄─────┘   └──► nuisance/domain probes (disentangle)
   (CE + density NLL + JS)
        + Σ_j λ_j · Î_j            (hierarchical CMI budgets, P0-2)
        + reference-prior align    (P0-4/P0-5)
        + disentangle + SSL
                          │
                 ┌────────┴─────────────── strict DG (no target data)
   test target ──┤
                 ├── offline TTA: EM(transform A,b ; prior π_T) at p_φ(z_c|y) geometry
                 │      └── gated by source-only safety gate (adapt / identity)
                 └── online TTA: streaming EMA prior + diagonal transform
```

## Each component is independently runnable / self-tested

```
python -m h2cmi.domains.dag                 # (imported; DAG validation in package)
python -m h2cmi.data.eeg_simulator          # simulate + split
python -m h2cmi.models.encoder              # (z_c, z_n) shapes + grad
python -m h2cmi.density.student_t_mixture   # density + hybrid head
python -m h2cmi.cmi.hierarchical            # H_ref, signed Î_j, dual step
python -m h2cmi.align.distances             # SW / energy / Bures-W2 / shrinkage cov
python -m h2cmi.align.reference_marginal    # corrected GLS ref + class-cond alignment
python -m h2cmi.disentangle.penalties       # HSIC / cross-cov / DisentangleLoss
python -m h2cmi.ssl.aux                     # VICReg + masked reconstruction
python -m h2cmi.tta.class_conditional       # offline + online TTA, recovers π_T
python -m h2cmi.gate.safety_gate            # harm AUROC on synthetic signal
python -m h2cmi.label.site_mechanism        # confusion-matrix EM recovery
```

## Minimal trustworthy core (default)

After the second review, `run_synthetic` defaults to `core_config` (`config.py`): encoder +
`p_phi(z_c|y)` + hierarchical CMI + offline **diagonal** TTA. Everything whose optimisation
direction or evaluation protocol still needs work is **off by default** and re-enabled only
once validated piece by piece:

| deferred | why (review) | flag |
|---|---|---|
| disentanglement | min-min adversary surrogate needs alternating Step A/B | `--full` |
| SSL reconstruction | `z_c`→raw-EEG fights the CMI objective; mask is on output not input | `--full` |
| source canonicaliser | absorbable by the fusion layer (not identifiable) | `--full` |
| safety gate | not yet a truly nested inner-LOSO (pseudo-targets saw training) | `--full` |
| online transform | deferred; only prior-only streaming is causal today | — |
| reference alignment | needs domain-class-balanced batches / LOO reference | `--full` |

## Example result (minimal core, recoverable covariance shift, deterministic target site)

`python -m h2cmi.run_synthetic --epochs 20 --concept 0.0 --target_site 0`

* strict-DG balanced acc **~0.73** (3-class), worst-domain **0.67**
* offline diagonal TTA **Δ bAcc ≈ −0.04** here (coverage 1.0, gate off): the cross-fit
  *evidence* check passes but *accuracy* does not improve — an honest demonstration that
  density-evidence gain ≠ accuracy gain, which is exactly what the **safety gate** (on true
  held-out gain) is for. This is one underpowered run (3 target subjects), **not** the
  experiment — use leave-one-site-out × 10 seeds (review §10).
* cross-fitted **grouped** signed leakage (refit-under-permutation null) flags **site** &
  **subject** above the null (excess > 0), discounts **session** (excess ≈ 0)

Switch on concept shift (`--concept 0.6 --concept_frac 0.5`) for the harm study, or `--full`
to enable every module.

## Scaling to real EEG

The encoder/density/CMI/TTA modules take ordinary `(X[n,chans,times], y, DomainLabels)`
arrays — point them at the AAAI loaders (`cmi/data/*.py` produce exactly that interface)
to run on MOABB / cross-site clinical data. Increase `H2Config` capacity (`z_c_dim`,
`spd_rank`, `density.n_components`, `train.epochs`) accordingly; the smoke defaults are
sized for CPU.

## Status / honesty  *(superseded — see the terminal banner at the top)*

The confirmatory protocol this section once deferred to **was carried out** (Stage V): fresh-seed
confirmation (seeds 100–119) and a frozen real-EEG external-validation block (offline MOABB cross-
dataset + cross-session L/R). Outcome, in brief: the safety/abstention behaviour holds on real EEG
but unlabeled source-calibrated adaptation yields no reliable utility, and the joint EM's prior-M-step
harm replicates. The deployed policy is **metadata-only routing + identity-by-default** (no target-
only eligibility gate). See [`results/V2_RESULTS.md`](results/V2_RESULTS.md),
[`results/V2P_RESULTS.md`](results/V2P_RESULTS.md), and [`results/METHOD_FREEZE.md`](results/METHOD_FREEZE.md).
The target-prior EM TTA, the source-only safety gate, and the online transform listed below are
**diagnostic / falsified**, not deployed.
