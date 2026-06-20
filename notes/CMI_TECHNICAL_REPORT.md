# CMI Technical Report — LPC-CMI (Single) and Dual-CMI for EEG Domain Generalization

**Audience.** A researcher joining the project who needs the full technical picture:
what the original method is, why one conditional-MI term is not the whole story, what
the dual-CMI extension actually buys (and does not), and where the project is headed.

**Scope / honesty mandate.** Every number below is harvested from the verified theory
checks (`notes/theory/verify_*.py`), the results JSONs in `results/`, and the synthetic
validators. Where the empirics say "parity, not a win," this document says so. The
leakage pillar is rock-solid; the accuracy claims are **parity**; the decoder term is a
**diagnostic** more than a lever. Read this as a working scientific record, not a sales
deck.

Notation throughout: `Z = g(X)` learned representation, `Y` label, `D` domain
(subject/site/cohort). All information quantities are in nats. `I(Z;D|Y)` = **encoder /
covariate leakage**; `I(Y;D|Z)` = **decoder / concept leakage**; `π_y(D) = p(D|Y=y)`,
`π_d(y) = p(Y=y|D=d)`.

---

# PART I — SINGLE-CMI (LPC-CMI): the original method

## I.1 The problem — calibration-free EEG domain generalization

EEG decoders trained on a pool of subjects/sites and deployed on an **unseen** subject/site
(leave-one-subject-out, LOSO; multi-source DG, no target data at train time) systematically
exploit **subject/site-identity shortcuts**: idiosyncratic montage, impedance, anatomy, and
acquisition artifacts that correlate with the domain label `D` but not with the task label
`Y`. These shortcuts inflate in-distribution fit and produce **overconfident, miscalibrated**
predictions on new domains. The goal of LPC-CMI ("Label-conditional Posterior-Counted CMI")
is to remove the *training-induced, label-irrelevant* domain information from `Z` while
**preserving the label** — and to do so with a tractable, trustworthy estimator.

## I.2 The objective — conditional domain leakage `I(Z;D|Y)`

We minimize the **conditional** mutual information between the representation and the domain
*given the label*:

```
I(Z;D|Y) = E_{z,y} KL( p(D | z, y) || p(D | y) ) = E_{z,y} KL( p(D | z, y) || π_y(D) ).
```

The conditioning on `Y` is the whole point. Marginal invariance `I(Z;D)=0` is provably
**harmful** under label shift (Zhao et al. ICML 2019, see Part II); the *conditional* term
side-steps that trap — it removes domain identity *within each class*, which is compatible
with a perfect classifier. This is exactly the Generalized Label Shift (GLS) feature
condition `D_S(Z|Y) = D_T(Z|Y)` in information-theoretic clothing.

## I.3 The variational estimator — `q_ψ(D|Z,Y)` against a counted prior `π_y(D)`

The true posterior `p(D|z,y)` is intractable; replace it with a learned MLP classifier
`q_ψ(D|z,y)`. By Barber–Agakov, **for any** `q_ψ`:

```
E_{z,y} KL( q_ψ(D|z,y) || π_y(D) ) = I(Z;D|Y) + E KL( p(D|z,y) || q_ψ(D|z,y) ) ≥ I(Z;D|Y).
```

So `L_enc := E KL(q_ψ || π_y)` is an **upper bound**, tight iff `q_ψ = p(D|z,y)`. Two design
choices make it honest:

- **Learned `q_ψ`, counted `π_y(D)`.** Only the posterior is learned; the prior `π_y(D)` is a
  plug-in count (Laplace-smoothed empirical / subject / effective frequencies). Putting the
  learned object in the numerator of the KL is what makes the bound *upper* (safe to minimize:
  driving `L_enc → 0` forces the true CMI to 0).
- **Bias direction is conservative.** An under-fit `q_ψ` *inflates* the bound rather than
  hiding leakage — so the estimator never reports a falsely-clean encoder, provided Step A is
  run to near-convergence. Numpy check: optimal `q_ψ` gives `E KL = I(Z;D|Y) = 0.194167`
  exactly; perturbing `q_ψ` only raises it.

**Estimator trustworthiness (audit).** Permutation-null ≈ 0 (−0.005..+0.018) everywhere; all
six independent probes (linear/MLP/RF/HGBM/kNN) agree on the leakage ranking; proxy-validated
`r = 0.85` vs an independent kNN `Î(Z;D|Y)`. The estimator is not a single-probe artifact.

## I.4 The two-step trainer

Per minibatch (`n_inner` inner repeats for Step A), **non-adversarial alternation** (not
DANN/GRL):

- **Step A — fit `q_ψ` on detached `Z`** (no encoder gradient): `min_ψ CE(q_ψ(D|z.detach(),y), d)`.
  Detaching is essential: Step A must *maximally* read `D` off `(Z,Y)` so the bound is honest;
  if the encoder could move here it would cheat by hiding `D` from a deliberately-weak `q`.
- **Step B — encoder + task head update, `q_ψ` frozen:**
  `L = CE(Y, ŷ) + λ_t · KL(q_ψ(D|z,y) || π_y[y])`, `z = g(x)` with gradient.
- **Warm-up:** `λ_t = λ · min(1, ep/warmup)` ramps the penalty so the encoder first learns a
  usable representation before invariance pressure (prevents `Z ⊥ everything` collapse).

Code: `cmi/methods/regularizers.py` (estimator), `cmi/train/trainer.py` (trainer).

## I.5 Results — honest summary

### The leakage pillar is rock-solid

`lpc_prior` cuts conditional leakage **10–100×** on every Euclidean backbone, every task, and
**beats both competitors** (CDANN adversarial; cHSIC kernel) at it, *while preserving the
label* (which CDANN/marginal alignment damage). This is the one result that reproduces
universally.

### Accuracy is parity, not a fixed-λ win

On balanced MCPS (motor-imagery, emotion) the mean balanced accuracy is **parity with ERM**
— the DomainBed-style null that the entire DG field reproduces. A pure regularizer does **not**
beat ERM on balanced-task mean accuracy. The honest framing is a **leakage–accuracy Pareto**:
"no accuracy cost at proper (small) λ," not "beats ERM."

| Task (backbone zoo, MI 2a 4-cls / 2b binary) | ERM acc / leakKL | lpc_prior acc / leakKL | best baseline |
|---|---|---|---|
| 2a (4-class) | 42.1 / 1.18 | 39.1 / **0.08** | IIB 43.5 / 0.82 |
| 2b (binary)  | 64.8 / 0.54 | 64.9 / **0.02** | IIB 65.6 / 0.41 |

| λ-sweep (2b @250Hz) | acc / leakKL |
|---|---|
| ERM | 68.6 / 0.53 |
| lpc_prior 0.05 | **69.2 / 0.08** (beats ERM, 6× less leak) |
| lpc_prior 0.3 | 67.7 / 0.03 |
| lpc_prior 1.0 | 67.8 / 0.01 |

Multi-seed MCPS (×3): 2a 51.8±0.4 (ERM) vs 49.3±1.7 (lpc:0.3); 2b 69.0±0.4 vs 68.0±0.6; SEED
54.6±0.4 vs 53.8±0.1 — within noise, λ=0.3 slightly over-regularizes.

### SCPS (clinical) — multi-seed parity, single-seed wins were seed luck

The earlier headline "+3.8 on MUMTAZ, +2.5 on TUAB, +7.5 ADFTD-Deep4" were **single-seed**
results. Multi-seed (`results/ladder_*.json`) corrects them to **parity**:

| Dataset (subject bAcc, multi-seed) | ERM | lpc_prior | leakKL ERM→lpc |
|---|---|---|---|
| MUMTAZ (depression, ×3) | 86.6±0.8 | 85.5±1.5 | 1.53 → 0.020 (75×) |
| TUAB (×2) | 55.6±1.9 | 57.5±5.0 | 1.48 → 0.047 (31×) |
| ADFTD (3-cls dementia, ×3) | 57.1±3.8 | 58.7±6.6 | 1.33 → 0.169 (8×) |

ADFTD is **seed-sensitive** (seed0 +3.7, seed1 −5.0, seed2 −0.3 in the original single-seed
sweep → parity across seeds). The honest read: SCPS is a **λ-tuning story at parity** with
strong leakage removal and (below) a calibration win — not a mean-accuracy win.

### Calibration is a concrete downstream WIN

Removing subject shortcuts makes the model **less overconfident even where mean accuracy is
parity**. From saved `*.preds.npz` (no retrain): `lpc_prior` calibration ≤ ERM (ΔECE ≤ 0.2) on
**27/33** datasets, often dramatically:

| dataset | ΔECE | ΔNLL |
|---|---|---|
| ADFTD (EEGNet, classbal) | −9.4 (32.3→22.8) | −0.67 (2.18→1.51) |
| TUAB | −4.5 (29.3→24.8) | −0.48 (1.68→1.20) |
| DEAP-quadrant | **−18.0** (34.5→16.4) | −0.80 |
| DEAP-arousal | −9.6 (28.0→18.4) | −0.26 |
| LogCov-2a (none-align) | −9.0 | −4.41 (8.08→3.68) |

### EA alignment is TRANSDUCTIVE, not a CMI win

Euclidean Alignment helps (2a none 43.2 → ea 48.8, +5.6) but **ea_strict (source-stats only)
41.8 ≤ none** — the entire EA gain comes from using the *target's unlabeled trials* (zero-label
calibration), not strict DG. On top of EA, CMI adds **worst-case robustness, not mean**. EA's
boost is dataset-dependent (large when source is small/few-subject: 2a +5.6; small when source
is large/diverse: Lee2019 +1.3). RA≈EA; **HA hurts**.

### What did NOT work

- **Constrained-λ source-selection:** picking λ by a DG model-selection criterion is
  **unreliable** — it dropped the model *below* ERM. DG model selection without target data is
  the field's open problem; we did not solve it.
- **GNN (GraphCMINet) parity:** node/edge conditional domain-MI is built and lit-novel, but the
  SEED benchmark vs DGCNN/RGNN is **pending/parity** (and the SEED GNN line uses DE features
  while we use raw — a confound to flag, not a SOTA claim).
- **TSMNet/SPDNet = baseline, NOT carrier:** SPD tangent features are the most leakage-prone
  (erm leakKL 2.0/1.7); applying `lpc_prior` either no-ops (2a) or **collapses to chance** (2b
  65.6→50.0). No λ removes leakage without collapse. Use only as a geometric DG baseline.
- **Route 2 (FMCA / chain-rule):** Y-erasure ablation confirmed (labelSep 59.1→39.5); dominated
  by `lpc_prior` everywhere → appendix only.
- **chsic (kernel cond-HSIC):** weaker leakage remover (2a 0.43 vs lpc 0.12).

---

# PART II — WHY ONE CMI ISN'T THE WHOLE STORY

## II.1 The encoder/decoder decomposition and the EXACT identity (A1)

The representation can leak domain info in **two** distinct ways: through the *features given
the label* (`I(Z;D|Y)`, encoder/covariate) and through the *label given the features*
(`I(Y;D|Z)`, decoder/concept). These are the two legs of one chain-rule identity — not a
heuristic pairing.

> **A1 (exact, any joint `p(Z,Y,D)`):**
> ```
> I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D).
> ```

**Proof (one line).** Expand the budget `I((Y,Z);D)` two ways:
`I(Y;D) + I(Z;D|Y) = I(Z;D) + I(Y;D|Z)`; rearrange. ∎

A1 is a **conservation law**: total domain info in the (label, representation) pair is
attributable two ways, and the two attributions must agree — which **couples** the encoder and
decoder leakages. Verified to **max abs error 2.22e-15** over 3000 random discrete joints
(`verify_tension.py [i]`).

## II.2 The TENSION THEOREM (A2/A3) — with verified numbers

> **A2 (the fight).** If the encoder is conditionally invariant, `I(Z;D|Y)=0`, then
> ```
> I(Y;D|Z) = I(Y;D) − I(Z;D).
> ```
> Under label shift `I(Y;D) > 0`, this is **strictly positive** unless `I(Z;D)=I(Y;D)`.
> Driving the encoder leakage to zero **forces the decoder leakage up** by exactly the
> label-shift gap.

> **A3 (the unreachable escape).** Both CMIs vanish jointly **iff** `I(Z;D)=I(Y;D)`. Under the
> Markov chain `D→Y→Z` that invariance imposes, DPI gives `I(Z;D) ≤ I(Y;D)`, with equality
> **iff** `p(z|y)` is label-invertible, i.e. `Y=f(Z)` deterministically — **zero Bayes error**.
> With irreducible clinical label noise (`H(Y|Z)>0`) the escape is unreachable, so the two
> terms **must fight**.

**Physical reading.** An invariant encoder pushes domain signal out of the feature *shapes*
`p(z|y)` and into the feature *prevalences*: `p(z|d) = Σ_y π_d(y) p(z|y)`. Because the mixing
weights `π_d(y)` differ by site (label shift) while `p(z|y)` is now shared, `p(z)` still carries
the label-prior shift. A decoder reads `p(y|z,d) ∝ π_d(y) p(z|y)`, so the *same* `z` implies a
*different* label posterior at different sites — `I(Y;D|Z) > 0`. You cannot zero both ends while
prevalences differ, unless you neutralize the prevalence difference itself (Part IV).

**Verified anchor** (`verify_tension.py [iv]`, fixed invariant symmetric encoder, per-domain
priors `0.5 ∓ 0.20`):

```
I(Z;D|Y) = 4.4e-16 ≈ 0      I(Y;D) = 0.0823
I(Z;D)   = 0.0340            I(Y;D|Z) = 0.0483 = I(Y;D) − I(Z;D)   ✓
```

**Monotone tradeoff** (fixed invariant encoder, shift strength `s: 0→0.95`): `I(Z;D|Y)` pinned
at ≤4.4e-16; forced `I(Y;D|Z) = I(Y;D)−I(Z;D)` grows monotonically
`0 → 0.0108 → 0.0444 → 0.1050 → 0.2039 → 0.3268`. The fight scales with label shift.

## II.3 The three-shift decomposition

A1 names every shift type and tells you exactly which knob handles it:

| shift type | quantity | handled by | what survives |
|---|---|---|---|
| **covariate** | `I(Z;D\|Y)` (encoder) | conditional-invariance regularizer (LPC-CMI) | → 0 |
| **label** | `I(Y;D)` via `π_d(y)` | GLS reweighting `w_d(y)=π*(y)/π_d(y)` | → 0 |
| **concept** | `I(Y;D\|Z)` (decoder, residual after correction) | explicit `I(Y;D\|Z)→0` penalty | **diagnostic readout** |

**The resolution (A4, GLS).** Fix a reference prior `π*`; reweight each domain by
`w_d(y) = π*(y)/π_d(y)`. Under the reweighted law `Ĩ(Y;D)=0`, and A1 becomes

```
Ĩ(Z;D|Y) = Ĩ(Y;D|Z) + Ĩ(Z;D).      (★)   — all three non-negative, all-positive signs
```

No forced trade-off survives: an invariant + sufficient `Z` makes all three vanish *together*,
**with no requirement of zero Bayes error**. Verified (`verify_resolution.py`, `ALL PILLAR-2
CLAIMS VERIFIED: True`): on the A2 fight joint, reweighting collapses `I(Y;D|Z)` from 0.0483 to
machine zero while **`I(Z;Y)` predictive content is unchanged** (0.2564 before = after).

**The CONTROL that proves the concept term is independent.** With a *concept-shifted*
(domain-dependent) channel `p(z|y,d)`, GLS reweighting drives `Ĩ(Y;D)=0` but leaves
**`Ĩ(Y;D|Z) = 0.1449 > 0`** — reweighting alone does *not* touch concept shift; only the
explicit decoder constraint does.

## II.4 What it predicts

**Invariance controls the gap, not the mean.** Conditional invariance does not improve mean
accuracy (Part I confirms parity); it certifies a *worst-case / cross-site / calibration*
property and removes a specific failure mode. The decomposition predicts the empirical pattern
we see: the encoder penalty cleans `I(Z;D|Y)` but, under label shift, **raises** `I(Y;D|Z)`
(A2) — which is precisely what the real-data ladder shows (Part III).

---

# PART III — DUAL-CMI: WHAT WE TRIED (brutally honest)

## III.1 The naive `dual` (sum)

`method = 'dual'` is the only method optimizing **both** CMIs:

```
L = CE + λ_enc · I(Z;D|Y) + λ_dec · I(Y;D|Z)              (NAIVE SUM)
```

Step A fits **both** `q_ψ(D|Z,Y)` (encoder posterior) **and** `h(Y|Z,D)` (decoder auxiliary
head) on detached `Z`. Step B (`trainer.py:206–211`):

```python
if is_dual:
    r_enc = post.reg("lpc_prior", z, yb)          # KL(q_dzy ‖ log_pi_y[y])  — UPPER bound on I(Z;D|Y)
    r_dec = ce_q - post.iib_ce_h(z, yb, db)        # CE_q(Y|Z) − CE_h(Y|Z,D) = H(Y|Z)−H(Y|Z,D) = Î(Y;D|Z)
    loss  = loss + lam_t * r_enc + gamma_t * r_dec
```

- `r_enc` = posterior-KL **upper bound** on `I(Z;D|Y)` (safe to minimize).
- `r_dec` = entropy-gap **consistent plug-in** for `I(Y;D|Z)` (exact at the joint optimum
  `q=p(y|z), h=p(y|z,d)`; over-estimate if `q` loose, under-estimate if `h` under-fit — so
  Step A must converge). Note `r_dec` is **two-sided**, not a clean single-sided bound.

`dual = lpc_prior`'s `r_enc` **+** `iib`'s `r_dec`, independent warmed weights `λ_enc ≡ lam_t`,
`λ_dec ≡ gamma_t`. Non-adversarial alternation, hence more stable than DANN/GRL.

## III.2 `label_correct` — partial and empirically inert

The GLS correction A4 is wired but **partial**: the `--balance` / `--label_correct` flag applies
the per-sample weight `w_i = π*(y_i)/π_{d_i}(y_i)` (or class-inverse-freq) **only to the task CE
`ce_q`**, *not* to the `h(Y|Z,D)` half of `r_dec` nor to the encoder KL or the Step-A fits. So
`I~(Y;D)` is not actually set to zero inside the CMI estimators. Empirically it is **inert**:

| MUMTAZ (subj bAcc / leakKL) | dual | dual + balance (`lc_*`) | dual + reweight (`rwdual`) |
|---|---|---|---|
| acc | 85.5±1.5 | 89.3 (single-seed) / 85.4 (lc) | 87.4 |
| leakKL | 0.025 | 0.027 | 0.025 |
| `I(Y;D\|Z)` | 0.029 | — | 0.029 |

| ADFTD (subj bAcc) | dual | dual + balance |
|---|---|---|
| acc | 59.8±5.6 (3-seed) / 60.2 (single) | 54.0 |

The `--balance` arm does **not** move leakage or `I(Y;D|Z)` and does not reliably help accuracy
(on ADFTD it *hurts*: 60.2→54.0 single-seed). This is the gap Route B is built to close — apply
the weights to **both** CMI estimators, not just `ce_q` (Part IV).

## III.3 The decoder leakage probe — a new held-out metric

`decoder_leakage_probe` in `cmi/eval/metrics.py` computes the **held-out** `I(Y;D|Z) =
H(Y|Z) − H(Y|Z,D)` by fitting fresh probes on frozen features (reported as `decoder_cmi` in
every summary). This is the diagnostic that separates shift conditions — and it is the most
important new measurement of the dual line.

## III.4 The multi-seed ladder + the `I(Y;D|Z)` diagnostic (`results/ladder_*.json`)

This is the key empirical table. `I(Y;D|Z)` = `decoder_cmi`, held-out:

| Dataset | method | subj bAcc | leakKL `I(Z;D\|Y)` | **`I(Y;D\|Z)` decoder** |
|---|---|---:|---:|---:|
| **ADFTD** (3-cls, ×3) | erm | 57.1 ± 3.8 | 1.334 | **0.200 ± 0.054** |
| | lpc_prior | 58.7 ± 6.6 | 0.169 | **0.300 ± 0.027** ↑ |
| | iib (decoder-only) | **60.9 ± 0.4** | 1.373 | 0.193 ± 0.035 |
| | dual | 59.8 ± 5.6 | 0.229 | 0.275 ± 0.026 ↑ |
| **MUMTAZ** (depr, ×3) | erm | 86.6 ± 0.8 | 1.528 | **0.005** |
| | lpc_prior | 85.5 ± 1.5 | 0.020 | 0.034 |
| | iib | 86.6 ± 0.8 | 1.541 | 0.005 |
| | dual | 85.5 ± 1.5 | 0.025 | 0.029 |
| **TUAB** (×2) | erm | 55.6 ± 1.9 | 1.476 | **0.101** |
| | lpc_prior | 57.5 ± 5.0 | 0.047 | 0.133 ↑ |
| | iib | 56.9 ± 0.6 | 1.441 | 0.102 |
| | dual | 58.8 ± 3.7 | 0.057 | 0.126 ↑ |
| **SCZ cross-site** | erm | 62.0 (worst .56) | 1.596 | **0.007** |
| (`dual_scps_SCZ`) | lpc_prior | 58.8 | 0.276 | 0.017 |
| | iib | 60.2 | 1.613 | 0.006 |
| | dual | 58.7 | 0.284 | 0.016 |

**What the diagnostic says (the headline of Part III):** the `I(Y;D|Z)` column **separates the
conditions**.

- **ADFTD (dementia): `I(Y;D|Z) ≈ 0.20` = REAL concept shift.** And there the **decoder-only
  `iib` is best AND most stable** (60.9 ± 0.4 vs ERM 57.1 ± 3.8) — the only place a CMI variant
  clearly wins, and it is the decoder term, on the dataset where the decoder term is large.
- **MUMTAZ ≈ 0.005 (none):** no concept shift; all methods parity; the encoder term cleans
  leakage 75× with no accuracy effect.
- **TUAB ≈ 0.10 (intermediate).**
- **SCZ cross-site ≈ 0.01 (none — but WEAK test):** only 2 task-based cohorts loaded; the
  resting-state FEP cohorts failed to load (see Part IV). This is *not* a real cross-site
  concept-shift test yet.

**TENSION CONFIRMED ON REAL DATA.** The encoder penalty **raises** `I(Y;D|Z)`, exactly as A2
predicts: ADFTD 0.20 → 0.30 (lpc) / 0.275 (dual); TUAB 0.10 → 0.13. The fight is not a synthetic
curiosity — it is measurable on clinical EEG.

## III.5 Synthetic v2 (`synthetic/dual_cmi_v2.py` → `results/synthetic_dual_v2.txt`)

Three DGPs {covariate-only / concept-only / all-three} × {erm, enc, dec, dual, dual+LC} × 5
seeds, with **held-out** CMI probes and an exact-MI tension sweep. Findings:

- **Exact-MI sweep reproduces A2/A3:** with `I(Z;D|Y)=0` and label shift, `I(Y;D|Z) =
  I(Y;D)−I(Z;D) > 0` for all `alpha<1`; both vanish only at `alpha=1` (zero Bayes error).
  Identity residual ≤ 2.2e-16.
- **Learned tension:** forcing `I(Z;D|Y)` down via `lam_enc: 0→4` raises held-out `I(Y;D|Z)`
  `0.162 → 0.165 → 0.168 → 0.172` (covariate+label DGP, no decoder term) — the A2 fight on a
  learned model.
- **dual helps under PURE covariate shift, HURTS under combined covariate+concept+label.**
  Covariate-only: dual 76.8 vs erm 74.8. All-three: dual 61.1 vs erm 59.1 (within noise, no
  compounding). Concept-only: enc/dual cannot fix the sign-flipped target.

## III.6 Brutally honest bottom line for Part III

- **`dual ≈ encoder-only` on accuracy.** No compounding benefit from adding the decoder term to
  the encoder term. The reweight/balance variants are inert with the current partial wiring.
- **The decoder term `I(Y;D|Z)` is a DIAGNOSTIC, not a lever.** Penalizing it does not reliably
  improve accuracy; *measuring* it cleanly separates "real concept shift" (ADFTD) from "none"
  (MUMTAZ). Where the decoder term is large (ADFTD), the **decoder-only `iib`** — not the dual —
  is what wins and stabilizes.
- **The one clean win is `iib` on ADFTD** (60.9 ± 0.4 vs 57.1 ± 3.8), and it lines up with the
  largest measured `I(Y;D|Z)`. That coherence (big concept term → decoder method helps) is the
  most promising empirical thread.

---

# PART IV — PLANNED ADJUSTMENTS (the 休整 / regroup)

## IV.1 Route B — reweighted-dual (full GLS reweighting of both estimators)

**Status: implemented, CPU-smoke-tested, validation jobs submitted; first real result landed.**
Code/notes: `notes/route_B_reweighted_dual.md`, flag `--reweight_dual` (gated on `method ==
"dual"`; off ⇒ bit-identical to naive dual, backward compatible).

**The fix.** Naive `dual`+`label_correct` weights only `ce_q` (the inert half). Route B applies
the GLS per-sample weight `w_i = π*(y_i)/π_{d_i}(y_i)` to **all three** estimators so the
reweighted measure genuinely has `I~(Y;D)=0` and the two CMIs **decouple** (★) *before*
co-minimization:

1. **Step A fits** `q(D|Z,Y)` and `h(Y|Z,D)` on the reweighted measure (so the Step-B KL is a
   valid bound on the *reweighted* CMI).
2. **Decoder term** `r_dec = ce_q_w − H_w(Y|Z,D)` — both halves now weighted (closing the inert
   gap).
3. **Encoder term** `r_enc = E_w[KL(q(D|Z,Y) ‖ π*)]` against the **uniform** reference, weighted
   batch mean.

**First landed result (`results/rwdual_MUMTAZ.json`, the sanity-null dataset):**

| MUMTAZ | subj bAcc | leakKL | `I(Y;D\|Z)` |
|---|---:|---:|---:|
| erm | 85.4 | 1.538 | 0.005 |
| rw-dual 0.5:0.5 | 87.4 | 0.025 | 0.029 |

As expected on a no-concept-shift dataset, rw-dual is **inert relative to naive dual** (matches
dual 85.5 within noise; leakage and `I(Y;D|Z)` unchanged) — the correct sanity null. The
decisive test is **ADFTD** (real `I(Y;D|Z)≈0.20`): the prediction is that decoupling should
**not raise** `I(Y;D|Z)` the way the naive encoder penalty does (0.20→0.30). That ADFTD rw-dual
run is the linchpin and is the next number to harvest.

## IV.2 Route A — GLS-VAE (structured latent, decoupling by construction)

**Status: synthetic-complete (4-seed matrix), honest mixed verdict.** Code:
`synthetic/gls_vae.py`; notes: `notes/route_A_gls_vae.md`; raw: `results/route_A_gls_vae*.txt`.

DIVA-style partitioned latent: shared class-conditional `p(z_y|y)`, per-domain `p(z_d|d)`,
per-domain free label prior `π_d(y)`, GLS reference-prior decode `p(y|z_y) ∝ π*(y) p(z_y|y)`,
optional encoder penalty, and a per-domain likelihood correction `δ_d` as a concept-shift test.

**Honest verdict — partly works, and the failure is the interesting part:**

- **WORKS — tension dissolved.** Pushing encoder pressure up does **not** raise `I(Y;D|Z)`
  (`0.136 → 0.125 → 0.127` as `lam_inv: 0→2→6`), unlike naive-enc (`0.149 → 0.175`). The GLS
  decode neutralizes the label term by construction, so the encoder pressure is **fight-free**.
  This is the cleanest single piece of Route-A evidence.
- **WORKS — lower simultaneous leakage at matched accuracy** on covariate shift
  (covariate+label: `glsvae+inv` `I(Z;D|Y) 0.371 / I(Y;D|Z) 0.106` vs erm `0.624 / 0.179`, dual
  `0.574 / 0.189`).
- **WORKS — concept test separates** ~2.5×: `δ_d` held-out ELBO gain `0.41–0.48` on
  concept-bearing DGPs vs `0.17–0.22` on covariate/label DGPs (caveat: needs **null-calibrated**
  threshold ≈0.30, not the hard 0.20).
- **DOES NOT WORK — "both CMIs vanish by construction" is FALSE** for an amortized encoder.
  `I(Z;D|Y)` floors at 0.37–0.47 and **still needs the explicit penalty**; structure alone gives
  only ~25–30% reduction. The defensible framing is "structured GLS decoupling makes the dual
  objective well-posed," **not** "free-lunch architecture."
- **COST — concept shift still breaks transfer accuracy** (concept-only 38–40 vs erm/dual 54–61).
  Correct (concept shift is unfixable by invariance, only detectable) but a real DG accuracy cost.

**Verdict:** a legitimate *reframing* + a working concept diagnostic, not a free-lunch
architecture; does not beat naive dual/erm on accuracy. Next: null-calibrate the `δ_d` test;
port to real ADFTD (does `δ_d` fire there and stay quiet on MUMTAZ?).

## IV.3 Fix the resting-state FEP SCZ loader — the real cross-site concept test

The SCZ cross-site test is currently **weak**: only 2 task-based cohorts loaded
(`I(Y;D|Z)≈0.01`, no concept shift); the **resting-state FEP cohorts failed to load**. Without
them there is no genuine same-disease, different-site, different-diagnostic-criteria test — which
is the *entire clinical motivation* for the concept term. Fixing this loader is the single
highest-leverage experimental task: it is the decisive test of whether `I(Y;D|Z) > 0` survives
on real psychiatric cross-site data after GLS reweighting + encoder invariance. (Plan & route:
`cmi-scps-crossdataset` memory; multi-level `D=(cohort,subject)` via OpenNeuro same-disease
cohorts.)

## IV.4 Honest venue assessment

- **AAAI-27 main track: VIABLE as positioned** — on the strength of (1) the **tension theorem**
  (A2/A3) stated as a formal proposition with the GLS resolution framed *as resolving a tension
  between two simultaneously-desired CMIs* (this framing is, to our reading, not in Zhao19 / GLS
  / entropyDG); (2) the **three-shift decomposition** with the **concept term `I(Y;D|Z)` as a
  measured nat-valued diagnostic** of differing diagnostic criteria across sites; (3) the
  **robustness/calibration** payoff (rock-solid leakage removal, ECE/NLL wins) — *not* a
  mean-accuracy-beats-ERM claim. AAAI rewards a correctly-positioned framework + a clinically
  real use case over a raw SOTA delta. **This is our lane.**
- **ICLR: borderline** without one of (a) a **new generalization bound** in the three CMIs
  (carrying the concept residual explicitly, tighter/more estimable than Zhao19's unestimable
  `|f_S−f_T|`), or (b) a **clear benchmark win** over IWDAN/IWCDAN/entropyDG on a shared
  label-+concept-shift protocol. The decomposition + tension alone will not clear ICLR's novelty
  bar (a skeptic's "just MI bookkeeping" objection about the *algebra* is correct — sell the
  *use*, not the identity).
- **The linchpin (hostage to experiments):** show `I(Y;D|Z) > 0` after GLS reweighting + encoder
  invariance on **real** psychiatric cross-site data, and that penalizing it improves worst-site
  accuracy / calibration. If that story is weak the paper collapses to "we re-derived GLS and
  added an entropy term"; if it holds, dual-CMI is a clean, honest, defensible AAAI paper. The
  ADFTD `iib`-best + `I(Y;D|Z)≈0.20` coherence is the current best evidence; the SCZ resting-FEP
  loader fix is what would make it decisive.

**Do NOT claim:** "we beat ERM on mean accuracy" (we do not — 4 validations + the multi-seed
ladder reproduce parity); "the chain-rule identity A1 is the contribution" (it is one line of
algebra).

---

# PART V — CODE MAP

| Piece | Location |
|---|---|
| **Encoder CMI estimator** `I(Z;D|Y)` (posterior-KL), `DomainPosteriors`, `kl_to_prior`, `reg("lpc_prior")` | `cmi/methods/regularizers.py` |
| **Decoder auxiliary head** `h(Y|Z,D)`, `iib_ce_h`, `posterior_loss` | `cmi/methods/regularizers.py` |
| **Two-step trainer**, warm-ups, method dispatch | `cmi/train/trainer.py` |
| — `is_dual` dispatch | `trainer.py:66` |
| — `balance` / `ce_weight` (BER/GLS label correction, partial) | `trainer.py:67–71`, consumed `:198` |
| — Step A (fit posteriors on detached Z) | `trainer.py:172–182` |
| — `iib` (decoder-only) Step B | `trainer.py:204–205` |
| — `dual` Step B (`r_enc + r_dec`) | `trainer.py:206–211` |
| — `lpc_prior` (encoder-only) Step B | `trainer.py:212–214` |
| — `λ_t`, `γ_t` warm-ups | `trainer.py:118–119` |
| — Route B reweighting (`reweight_dual`, weighted Step A + both CMI terms + uniform ref) | `trainer.py` (gated `rw_dual = reweight_dual and is_dual`); `regularizers.py` (`weight=` args, `reference="uniform"`) |
| **Decoder leakage probe** (held-out `I(Y;D|Z) = decoder_cmi`) | `cmi/eval/metrics.py` |
| **LOSO runner** (`--configs erm:0 lpc_prior:λ iib:λ dual:λ:γ`, `--balance`, `--reweight_dual`) | `cmi/run_loso.py` |
| **SCPS cross-dataset runner** (Protocol C, `dual:lam:gamma`) | `cmi/run_scps_crossdataset.py:51–52` |
| **Synthetic — dual v2** (3 DGPs × 5 methods × 5 seeds, held-out CMI, tension sweep) | `synthetic/dual_cmi_v2.py` → `results/synthetic_dual_v2.txt` |
| **Synthetic — Route A GLS-VAE** (DIVA-style, fight-free test, `δ_d` concept test) | `synthetic/gls_vae.py` → `results/route_A_gls_vae*.txt` |
| **Theory + machine verification** (A1–A4) | `notes/theory/01_tension.md`..`04_positioning.md`; `verify_tension.py`, `verify_resolution.py` |
| **Results — multi-seed ladder** (per-config acc, `leakage_kl`, `decoder_cmi`) | `results/ladder_{ADFTD,MUMTAZ,TUAB}_seed{1,2,3}.json` |
| **Results — dual / label-correct / reweight-dual / SCPS** | `results/dual_*.json`, `lc_*.json`, `rwdual_*.json`, `scps_*.json`, `dual_scps_*.json` |
| **Calibration** (ECE/NLL from `*.preds.npz`, no retrain) | `notes/calibration.md` |

---

*Document assembled 2026-06-11 from verified theory checks, the multi-seed results JSONs, and
the synthetic validators. Numbers are harvested, not asserted. Where empirics say parity, this
document says parity.*
