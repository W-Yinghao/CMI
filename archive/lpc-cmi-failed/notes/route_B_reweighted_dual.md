# Route B — Reweighted-Dual Variational Decoupling

Status: implemented, CPU-smoke-tested, validation jobs submitted (ADFTD, MUMTAZ).
Add-only and backward compatible: naive `dual`, `balance`, `label_correct` semantics are
unchanged; existing running jobs are unaffected. New behaviour is gated behind the
`--reweight_dual` flag and only activates for method `dual`.

## Motivation (the tension)

The Dual-CMI objective co-minimizes the encoder CMI `I(Z;D|Y)` (covariate-shift /
representation invariance) and the decoder CMI `I(Y;D|Z)` (concept-shift / predictor
invariance). The verified identity is

    I(Z;D|Y) - I(Y;D|Z) = I(Z;D) - I(Y;D).

The TENSION THEOREM: with encoder invariance `I(Z;D|Y)=0`, the identity gives
`I(Y;D|Z) = I(Y;D) - I(Z;D)`. Under label shift `I(Y;D) > 0` the decoder term is forced
positive unless `Y` is a deterministic function of `Z` (zero Bayes error). So the naive
dual objective cannot drive BOTH terms to zero whenever the per-domain label prior differs.

RESOLUTION (GLS reweighting): reweight each domain `d` to a common reference prior `pi*`
via the per-sample importance weight

    w_i = pi*(y_i) / pi_{d_i}(y_i),    pi* = uniform over classes (1/n_cls),

where `pi_d(y) = p(Y=y | D=d)` is the Laplace-smoothed per-domain class frequency
(`_label_shift_weights` in `cmi/train/trainer.py`). Under the reweighted measure
`I~(Y;D) = 0`, so the identity reduces to `I~(Z;D|Y) = I~(Y;D|Z) + I~(Z;D)` with the
label-shift coupling removed; the two CMIs can now be driven toward zero independently.
Naive `dual` applies `w_i` (when `--label_correct`) only to the CE / `H(Y|Z)` term — that
is partial and was empirically inert. Route B applies `w_i` to BOTH CMI estimators.

## What Route B changes (exactly)

`--reweight_dual` is honoured ONLY for `method == "dual"` (`rw_dual = reweight_dual and
is_dual`). When off, every code path reduces EXACTLY to the existing naive `dual` (the
weighted reductions collapse to the plain `F.cross_entropy` / mean when `w == 1`, and the
reference falls back to the empirical `pi_y`). The GLS weights `w_i` are now computed
whenever `--label_correct` OR `rw_dual` is set (previously only for `--label_correct`).

Three estimators are reweighted (all reductions are `sum_i w_i * loss_i / sum_i w_i`):

1. **Step A posterior fit** (`DomainPosteriors.posterior_loss(..., weight=wb)` and
   `iib_ce_h(..., weight=wb)`): `q(D|Z,Y)`, `q(D|Z)`, `q(S|Z)` and the decoder predictor
   `h(Y|Z,D)` are fit on the REWEIGHTED measure, so the Step-B KL is a valid variational
   upper bound on the *reweighted* CMI (otherwise we'd KL a posterior fit to the raw
   distribution against a reweighted reference — inconsistent).

2. **Decoder term** `r_dec = ce_q_w - H(Y|Z,D)_w`:
   - `ce_q_w` = per-sample weighted CE of the task head (`H(Y|Z)` half), weighted by `w_i`
     (and `ce_weight` if `--balance`). This is the half naive dual already weighted only
     under `--label_correct`; Route B makes it consistent for the whole decoder term.
   - `H(Y|Z,D)_w` = `post.iib_ce_h(z, y, d, weight=wb)`, the NEW per-sample weighted CE of
     `h(Y|Z,D)` (the half that naive dual / label_correct left unweighted — the inert part).
   So `r_dec` estimates the reweighted `I~(Y;D|Z) = H_w(Y|Z) - H_w(Y|Z,D)`.

3. **Encoder term** `r_enc = E_w[ KL( q(D|Z,Y) || pi* ) ]`:
   - WEIGHTED batch mean `sum_i w_i KL_i / sum_i w_i` (new `weight` arg of `kl_to_prior`).
   - Reference is the UNIFORM-over-present-domains `pi*` (`reference="uniform"` ->
     `log_unif`), consistent with the reweighted measure, instead of the empirical
     `pi_y(D)`. (In LOSO, domains are remapped contiguous 0..K-1, so "present" = all K.)
   So `r_enc` estimates the reweighted `I~(Z;D|Y)` against the decoupled reference.

The total objective is unchanged in form: `loss = CE + lam_t * r_enc + gamma_t * r_dec`,
with the same warmup on `lam_t`, `gamma_t`.

## Files touched (add-only)

- `cmi/methods/regularizers.py`
  - `kl_to_prior(logits, log_prior, weight=None)` — optional per-sample-weighted mean.
  - `DomainPosteriors.posterior_loss(..., weight=None)` — weighted Step-A fit.
  - `DomainPosteriors.iib_ce_h(..., weight=None)` — weighted decoder CE.
  - `DomainPosteriors.reg(..., weight=None, reference="prior")` — weighted KL + uniform ref.
- `cmi/train/trainer.py`
  - `train_model(..., reweight_dual=False, ...)` new kwarg; `rw_dual = reweight_dual and
    is_dual`; GLS weights computed when `label_correct or rw_dual`; reweighted Step A and
    Step B dual branches gated on `rw_dual`.
- `cmi/run_loso.py`, `cmi/run_scps_crossdataset.py` — new `--reweight_dual` CLI flag,
  passed through to `train_model`.

## Smoke test (CPU, random data)

EEGNet, n_cls=2, n_dom=4, induced per-domain label shift, 6 epochs:
- naive `dual` and reweighted `dual` both train and `predict()` returns finite probs.
- reweighted-dual `r_enc` diagnostic differs from naive (0.0078 vs 0.0058) — the reweight
  is active.
- `--reweight_dual` on a non-dual method (`iib`) is a no-op (gated on `is_dual`).
- naive `dual` is bit-identical with/without `reweight_dual=False` -> backward compatible.

## Validation jobs (where decoupling should matter most)

Comparing `dual` vs reweighted-`dual`:
- ADFTD (dementia, REAL concept shift `I(Y;D|Z) ~ 0.20`): decoupling should help / at least
  not raise `I(Y;D|Z)` the way the naive encoder penalty does (0.20 -> 0.30).
- MUMTAZ (no concept shift, `I(Y;D|Z) ~ 0.005`): decoupling should be inert (sanity null).

    sbatch -p A40  scripts/run.slurm --dataset ADFTD  --backbone EEGNet \
        --configs erm:0 dual:0.3:0.3 --reweight_dual --epochs 80  --resample 128 \
        --out results/rwdual_ADFTD.json
    sbatch -p V100 scripts/run.slurm --dataset MUMTAZ --backbone EEGNet \
        --configs erm:0 dual:0.5:0.5 --reweight_dual --epochs 100 --resample 128 \
        --out results/rwdual_MUMTAZ.json

Read-outs to compare across `dual` vs rw-dual: `decoder_cmi` (held-out `I(Y;D|Z)`),
`leakage_kl` (held-out `I(Z;D|Y)`), pooled / subject balanced accuracy, ECE/NLL from the
`.preds.npz` sidecar.
