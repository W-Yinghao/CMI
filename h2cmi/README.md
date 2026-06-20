# H²-CMI — Hierarchical Class-Conditional Mutual-Information Adaptation for EEG

A self-contained implementation of the post-review (ICLR-direction) redesign. It is
**isolated** from the AAAI `cmi/` package — nothing here imports-with-side-effects or
mutates `cmi`; the two coexist in the repo. The whole pipeline runs without real EEG via a
controllable mechanism simulator, so every component is exercised by one smoke test.

```
conda run -n icml python -m h2cmi.tests.test_smoke          # end-to-end smoke test
conda run -n icml python -m h2cmi.run_synthetic --epochs 20 # full report on a held-out site
```

See [`THEORY.md`](THEORY.md) for the four corrected derivations (P0-2…P0-5).

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

## Example result (recoverable covariance shift, unseen target site)

`python -m h2cmi.run_synthetic --epochs 20 --concept 0.0`

* strict-DG balanced acc **0.83** (3-class), worst-domain **0.74**
* offline TTA **Δ bAcc +0.12**, ΔNLL/ΔBrier/ΔECE all improve; domain-clustered bootstrap
  CI **[0.04, 0.27]**, `P(Δ>0)=1.0`
* cross-fitted signed leakage flags **site** & **subject** above the permutation null,
  discounts **session** (excess ≈ 0)

Switch on concept shift (`--concept 0.6 --concept_frac 0.5`) to drive the harm-gate study.

## Scaling to real EEG

The encoder/density/CMI/TTA modules take ordinary `(X[n,chans,times], y, DomainLabels)`
arrays — point them at the AAAI loaders (`cmi/data/*.py` produce exactly that interface)
to run on MOABB / cross-site clinical data. Increase `H2Config` capacity (`z_c_dim`,
`spd_rank`, `density.n_components`, `train.epochs`) accordingly; the smoke defaults are
sized for CPU.

## Status / honesty

This is a **research implementation** of the proposed method on a **simulator**: it
demonstrates that every piece is correct, differentiable, and composes end-to-end, and that
the corrected theory (P0-2…P0-5) is what the code computes. It is **not** a claim of
real-EEG SOTA — that requires the confirmatory protocol in review §10.4 (unified
preprocessing, 5–10 seeds, subject/site-clustered inference, external lockbox).
