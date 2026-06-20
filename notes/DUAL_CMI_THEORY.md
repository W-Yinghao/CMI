# DUAL-CMI THEORY — the definitive answer

**Question.** *Do you agree the dual-CMI framework (encoder `I(Z;D|Y)` + decoder
`I(Y;D|Z)`, both variational, focused on psychiatric cross-site DG, aimed at
AAAI/ICLR) is right — and why do the two terms fight?*

This document is assembled from the four verified theory pillars
(`notes/theory/01_tension.md … 04_positioning.md`), their numerical checks
(`verify_tension.py`, `verify_resolution.py`, both re-run and **passing** in this
pass), the implemented `dual` method in `cmi/train/trainer.py`, and the live
experimental jobs. Every number below is reproduced, not asserted.

---

## (1) VERDICT — PARTIAL AGREEMENT

**Yes to the decomposition and the dual framing; no to "minimize both to zero."**

- The encoder/decoder split is **mathematically correct and exact**. `I(Z;D|Y)`
  (encoder leakage / covariate term) and `I(Y;D|Z)` (decoder leakage / concept
  term) are the two legs of one chain-rule identity (A1, below). This is not a
  heuristic pairing; it is forced by the chain rule of mutual information.

- The dual framing is **publishable** — as a *unification + tension* result for
  **psychiatric cross-site DG**. It correctly places Zhao19 (the marginal trap),
  GLS/Combes20 (the encoder term + label reweighting), and entropyDG (the
  decoder-flavoured term) as three faces of one identity, and adds the
  **concept term as a first-class, estimated quantity** that all three omit.

- **BUT** naively asking the optimizer to drive *both* CMIs to zero is
  **self-defeating under label shift**. The tension theorem (A2/A3) proves that,
  with a conditionally-invariant encoder (`I(Z;D|Y)=0`), the decoder leakage is
  *forced* to exactly `I(Y;D) − I(Z;D) > 0`. You cannot wish it away; the only
  noiseless escape (A3) is unreachable in clinical EEG. **The real contribution is
  not "minimize both" — it is the label-correction operator (A4) that *decouples*
  the two so that minimizing both becomes feasible.** Stated as "minimize both,"
  the framework is wrong; stated as "label-correct, then the two decouple and the
  residual `I(Y;D|Z)` is genuine concept shift," it is right and novel.

**One-line verdict:** *The dual decomposition is exactly right; the naive dual
objective is wrong under label shift; the GLS label correction is what makes the
dual objective well-posed, and the surviving `I(Y;D|Z)` is the paper's true
deliverable — a measurable concept-shift diagnostic for psychiatric cross-site
data.*

---

## (2) THE EXACT IDENTITY + THE TENSION PROOF + VERIFIED NUMBERS

### Identity A1 (exact, holds for **any** joint `p(Z,Y,D)`)

```
I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D).                     (A1)
```

**Proof (one line of chain rule).** Expand the joint information the pair
`(Y,Z)` carries about `D` in the two orders:
```
I((Y,Z);D) = I(Y;D) + I(Z;D|Y)        (condition on Y first)
           = I(Z;D) + I(Y;D|Z)        (condition on Z first).
```
Both equal the same budget `I((Y,Z);D)`; equate and rearrange → (A1). ∎
(`notes/theory/01_tension.md` §a; `verify_tension.py` block `[i]`.)

**Interpretation.** A1 is a *conservation law*: the total domain information in
the (label, representation) pair can be attributed two ways, and the two
attributions must agree — which couples the encoder and decoder leakages.

### Tension theorem A2 (the fight) + A3 (the unreachable escape)

> **A2.** If the encoder is conditionally invariant, `I(Z;D|Y)=0`, then from A1
> ```
> I(Y;D|Z) = I(Y;D) − I(Z;D).                                (A2)
> ```
> Under label shift `I(Y;D) > 0`, this is **strictly positive** unless
> `I(Z;D)=I(Y;D)`. Driving the encoder leakage to zero *forces* the decoder
> leakage *up* by exactly the label-shift gap.

> **A3 (escape).** Both CMIs vanish jointly **iff** `I(Z;D)=I(Y;D)`. Under a
> Markov chain `D→Y→Z` (which conditional invariance imposes) the DPI gives
> `I(Z;D) ≤ I(Y;D)`, with equality **iff** `p(z|y)` is label-invertible, i.e.
> `Y=f(Z)` deterministically — **zero Bayes error**. With irreducible clinical
> label noise (`H(Y|Z)>0`, overlapping class-conditionals, non-separable
> physiology) the escape is unreachable, so the two terms **must fight**.

### Verified numbers — anchor **A2** (`verify_tension.py [iv]`, re-run, exact)

A 2-domain, binary-`Y`, binary-`Z` construction with a *fixed invariant* symmetric
encoder `p(z=1|y=1)=p(z=0|y=0)=a=0.8241` and per-domain priors
`p(y=1|d)=0.5∓0.2000`:

```
I(Z;D|Y) = 4.4e-16  ≈ 0      (encoder invariant by construction)
I(Y;D)   = 0.0823            (label shift present)
I(Z;D)   = 0.0340
I(Y;D|Z) = 0.0483  = I(Y;D) − I(Z;D)   ✓  (the forced decoder leakage)
```

**Monotone tradeoff** (`verify_tension.py [ii]`, fixed invariant encoder, shift
strength `s`): `I(Z;D|Y)` pinned at ≤4.4e-16 for all `s`, while the forced
`I(Y;D|Z)=I(Y;D)−I(Z;D)` grows monotonically `0 → 0.0108 → 0.0444 → 0.1050 →
0.2039 → 0.3268` as `s: 0→0.95`. Identity A1 holds across **3000 random discrete
joints** to **max abs error 2.22e-15** (machine precision). The encoder objective
and the decoder objective genuinely fight, and the fight scales with label shift.

---

## (3) WHY THEY FIGHT — in physical terms

Strip the algebra. Under **label shift** the class proportions `π_d(y)=p(y|D=d)`
differ across sites (e.g., one clinic enrolls 60% depressed, another 30%).

1. **You make the encoder conditionally invariant.** You force `p(z|y)` to be the
   same in every domain (`I(Z;D|Y)=0`). The *shape* of each class cluster in
   feature space is now domain-identical — good, no covariate leakage.

2. **But the feature marginal `p(z)` is a label-prior-weighted mixture of those
   clusters:** `p(z|d) = Σ_y π_d(y) p(z|y)`. Because the *mixing weights* `π_d(y)`
   differ by site (label shift) while `p(z|y)` is now shared, **`p(z)` itself still
   carries the label-prior shift** — the high-prevalence class dominates the
   feature cloud at one site and not the other. The domain signal you evicted from
   the *conditional* `p(z|y)` re-enters through the *marginal* mixing weights.

3. **A decoder reads `p(y|z) ∝ π_d(y) p(z|y)`.** Since `π_d(y)` differs by site,
   the **same point `z` implies a different label posterior at different sites** —
   so `p(y|z,d) ≠ p(y|z)`, i.e. `I(Y;D|Z) > 0`. The decoder *cannot* be invariant.
   This is not a training artifact; it is exactly the `I(Y;D)−I(Z;D)` of A2: the
   part of the label-prior shift the invariant encoder could not absorb resurfaces
   in the decoder.

4. **The only escape (A3) is zero Bayes error** — `z` determines `y` outright, so
   the prior-reweighting of an *ambiguous* posterior never happens because there is
   no ambiguity. In **noisy clinical EEG** (overlapping AD/HC or depressed/healthy
   spectra, montage and impedance noise, irreducible `H(Y|Z)>0`) this is
   **unattainable**. Hence the fight is real on exactly the data we care about.

**Plain English:** an invariant encoder pushes the domain signal out of the feature
*shapes* and into the feature *prevalences*; the decoder, which sees prevalences,
then has to be domain-dependent. You cannot zero both ends while the prevalences
differ — unless you neutralize the prevalence difference itself. That is (4).

---

## (4) THE RESOLUTION = the real contribution (the three-shift spine)

The single cause of the fight is the label-shift gap `I(Y;D)` sitting on the RHS of
A1. **Remove it by importance reweighting** (Generalized Label Shift, Combes et
al., NeurIPS 2020):

> **A4 (GLS resolution).** Fix a reference prior `π*(y)`. Reweight each domain by
> `w_d(y) = π*(y)/π_d(y)`, giving `p̃(z,y,d) ∝ w_d(y) p(z,y,d)`. Then every domain
> shares prior `π*`, so `Ỹ ⟂ D` and **`Ĩ(Y;D)=0`**. Identity A1 under `p̃` becomes
> ```
> Ĩ(Z;D|Y) = Ĩ(Y;D|Z) + Ĩ(Z;D).                              (★)
> ```
> All three terms are non-negative MIs, **linked additively with all-positive
> signs**: there is no longer a forced trade-off where lowering one raises the
> other. An invariant + sufficient `Z` now makes all three vanish *together*,
> with **no requirement of zero Bayes error** — irreducible noise `H(Y|Z)>0` is
> fully compatible with `Ĩ(Y;D|Z)=0`, because the noise is now domain-independent.

**Verified** (`verify_resolution.py`, re-run, `ALL PILLAR-2 CLAIMS VERIFIED: True`).
On the exact A2 fight joint:

| quantity        | original (fight) | reweighted (π* = avg / uniform) |
|-----------------|-----------------:|--------------------------------:|
| `I(Z;D)`        | 0.034010         | 0.000000 |
| `I(Y;D)`        | **0.082302**     | **0.000000** (label shift removed) |
| `I(Z;D|Y)` enc  | 0.000000         | 0.000000 |
| `I(Y;D|Z)` dec  | **0.048292**     | **0.000000** (tension gone) |

`I(Z;Y)` (predictive content) is **unchanged** by reweighting — GLS buys
invariance of both leakages without sacrificing the encoder's ability to predict
`Y` (verified: `I(Z;Y)=0.2564` before = after, `verify_tension.py [iii]`).

### The three-shift decomposition (the paper's spine)

| shift type | quantity | handled by | what survives |
|---|---|---|---|
| **covariate** | `I(Z;D|Y)` (encoder) | conditional-invariance regularizer | → 0 |
| **label** | `I(Y;D)` via `π_d(y)` | GLS reweighting `w_d(y)=π*(y)/π_d(y)` | → 0 |
| **concept** | `I(Y;D|Z)` (decoder, **residual after correction**) | explicit `I(Y;D|Z)→0` penalty | **diagnostic readout** |

**The residual `I(Y;D|Z)` after label correction is genuine CONCEPT shift** —
differing label-generating mechanisms across sites (different diagnostic criteria,
rater conventions, inclusion thresholds: a clinic that calls borderline cases
"depressed" vs one that does not). This is *exactly* what GLS structurally omits
(it assumes the residual is 0 / absorbs it into BER). The **CONTROL experiment**
proves the term is independent: with a *concept-shifted* (domain-dependent)
channel `p(z|y,d)`, GLS reweighting drives `Ĩ(Y;D)=0` but leaves
**`Ĩ(Y;D|Z)=0.1449 > 0`** — reweighting alone does *not* touch it; only the
explicit decoder constraint does (`verify_resolution.py`, CONTROL block).

**Positioning in one line:** *GLS = invariance + label correction (kills the
encoder term and the label-shift gap); ours = GLS + explicit `I(Y;D|Z)` control
(kills/measures the concept-shift term GLS leaves in BER), in the multi-source DG
regime, for psychiatric cross-site EEG.*

### How the code realizes it (`cmi/train/trainer.py`)

The `dual` method (`is_dual = method == "dual"`, `trainer.py:66`) is the **only**
method optimizing both CMIs. Step B (`trainer.py:206–211`):
```python
if is_dual:                                  # DUAL: encoder I(Z;D|Y) + decoder I(Y;D|Z)
    r_enc = post.reg("lpc_prior", z, yb)     # KL(q_dzy ‖ log_pi_y[y]) — UPPER bound on I(Z;D|Y)
    r_dec = ce_q - post.iib_ce_h(z, yb, db)  # CE_q(Y|Z) − CE_h(Y|Z,D) = H(Y|Z)−H(Y|Z,D) = Î(Y;D|Z)
    loss = loss + lam_t * r_enc + gamma_t * r_dec
```
- `r_enc` = posterior-KL **upper bound** on `I(Z;D|Y)` (Barber–Agakov; safe to
  minimize — driving it to 0 forces the true CMI to 0). Verified: `E KL ≥ I(Z;D|Y)`
  for all perturbed `q`, equality at optimum `0.194167` (`03_estimators.md` §a).
- `r_dec` = entropy-gap **consistent plug-in** for `I(Y;D|Z)` (exact at the joint
  optimum `q=p(y|z), h=p(y|z,d)`, `=0.198942`; over-estimate if `q` loose,
  under-estimate if `h` under-fit — so Step-A must converge).
- **Label correction** = the `--balance` flag (`trainer.py:67–71`): per-class CE
  weight `N/(n_cls·count_y)` = inverse-frequency = BER/GLS reweighting to uniform
  `π*` — this is anchor A4 in the task loss. `dual` only escapes the A2/A3 tension
  **when paired with `--balance`** (or a matched `prior_mode`).
- Weights map to the objective directly: `λ_enc ≡ lam_t`, `λ_dec ≡ gamma_t`, both
  warmed. Two-step alternating (Step A fits posteriors on detached `z`; Step B
  moves only the encoder) — **not adversarial**, hence more stable than DANN/GRL.

---

## (5) EXPERIMENTAL STATUS — every submitted job + landed numbers

**Theory (CPU, complete, passing now):**
- `notes/theory/verify_tension.py` — A1 over 3000 joints (max err 2.22e-15), the
  monotone forced-leakage sweep, GLS collapse, and the A2 anchor. **PASS.**
- `notes/theory/verify_resolution.py` — anchors reproduced, label shift removed,
  reweighted identity (★), both CMIs jointly zeroed, π*-robustness, and the
  concept-shift CONTROL. **`ALL PILLAR-2 CLAIMS VERIFIED: True`.**

**Submitted SLURM jobs (live as of 2026-06-10 18:32):**

| job id | part. | name | what it runs | status / what it confirms |
|---|---|---|---|---|
| **846957** | CPU | `dual-syn` | `synthetic/dual_cmi_v2.py` — 3 DGPs {covariate / concept / all-three} × {erm,enc,dec,dual,dual+LC} × 5 seeds, **held-out** CMI probes + strong concept arm (target flips concept→label sign) + exact-MI tension sweep | **RUNNING** (~5 min in). Output `results/synthetic_dual_v2.txt` still **empty (0 B)**. Will confirm: dual+LC is the *only* arm that zeros `I(Y;D|Z)` on the concept-flip target and survives on target accuracy; ERM/enc-only leave a measurable `I(Y;D|Z)`. |
| **846929** | V100 | `cmi-loso` | MUMTAZ (depression) LOSO, EEGNet, `erm:0 lpc_prior:0.5 iib:0.5 dual:0.5:0.5` | **RUNNING** — first real-EEG run of `dual`. **Landed numbers below.** → `results/dual_MUMTAZ.json` |
| **846930** | A40 | `cmi-loso` | MUMTAZ LOSO `erm:0 dual:0.5:0.5 --balance` (= dual + GLS label correction A4) | **RUNNING** → `results/dual_MUMTAZ_bal.json`. Will confirm the *label-corrected* dual (the A4 arm) on real depression data. |
| **846931** | A40 | `cmi-loso` | ADFTD (3-class dementia) LOSO `erm:0 lpc_prior:0.3 iib:0.3 dual:0.3:0.3`, resample 128 | **RUNNING** → `results/dual_ADFTD.json`. The SCPS dataset with the one prior accuracy win. |
| **846932** | V100 | `cmi-loso` | ADFTD LOSO `erm:0 dual:0.3:0.3 --balance` | **RUNNING** → `results/dual_ADFTD_bal.json`. Label-corrected dual on dementia. |
| 845483 | A100 | `cmi-loso` | Lee2019_MI EA-strict, `lpc_prior:0.1` | running (alignment, not dual) |
| 845480/1/2 | V100/A40 | `cmi-loso` | SEED_IV GraphCMI/DGCNN/RGNN | running (GNN carriers; 845480 `graphcmi:0.3:0.3:0.3` is a 3-term cousin) |
| 846933/846934/846935/846936/846964 | A100/A40/V100 | `cmi-xdata` | `cmi.run_cross_dataset` (Protocol C) | PENDING (QOS cap) — logs empty |
| 846956 | A100 | `matched_v3` | **unrelated** (`eeg2025/OBB` project) | not part of CMI |

**Landed real-EEG numbers — MUMTAZ depression LOSO (job 846929, 10 targets; dual
complete for 9):**

```
method      mean bAcc   worst-target   leakKL (frozen-probe)
erm           78.3          1.7          1.56 – 1.68
lpc_prior     77.5          0.0          0.016 – 0.030   (~60–80× lower)
iib           80.9          1.7          1.51 – 1.55     (decoder-only: leakage NOT reduced)
dual          76.7*         0.0          0.019 – 0.028   (~60–80× lower)
```
(*dual mean over the 9 completed targets; matched-9 means: erm 76.7, lpc 75.9,
iib 80.9, dual 76.7.)

**What MUMTAZ already tells us (consistent with the 4 prior validations):**
- `dual` and `lpc_prior` cut **encoder leakage 60–80×** (1.6 → ~0.02 nats);
  `iib` (decoder-only) does **not** reduce encoder leakage (expected — it never
  touches `I(Z;D|Y)`). This cleanly separates the two terms on real data.
- On **mean accuracy**, `dual ≈ lpc ≈ erm` within target-to-target noise; `iib`
  edges ahead here but with full leakage. **This is parity, not a win** — exactly
  the empirical-findings position. The `--balance` (A4-corrected) arms (846930/932)
  are the ones that test the *resolution* claim and are still running.
- Note: MUMTAZ is small and several targets are degenerate (tgt5 ~0, tgt3 low for
  all methods) — per-target variance is high; this dataset will inform, not settle.

---

## (6) HONEST TOP-VENUE ASSESSMENT

**The defensible claim is a provable-gap / worst-case / calibration-control result
plus the theory, focused on psychiatric cross-site DG — NOT a mean-accuracy-beats-
ERM claim.** Four independent validations show invariance **ties** ERM on mean
balanced accuracy (the universal DomainBed result), and MUMTAZ above reproduces it.

**Do claim (each backed):**
1. **The tension theorem (A2/A3) as a formal proposition** + the GLS resolution
   (A4) framed *as resolving a tension between two simultaneously-desired CMIs*.
   This framing is, to our reading, **not stated in Zhao19 / GLS / entropyDG**
   (GLS has the reweighting but only one CMI). This is the one piece reviewers
   cannot attribute to prior art. (Pillar 4 §2–3.)
2. **The three-shift CMI decomposition as one identity** (covariate / label /
   concept), with the **concept term `I(Y;D|Z)` as a measured nat-valued
   diagnostic** of differing diagnostic criteria across clinical sites. This is the
   application-novel, AAAI-fit contribution. The CONTROL experiment proves the
   concept term is independent of GLS correction.
3. **Provable / worst-case control:** the leakage estimator is rock-solid
   (60–80× reduction here; proxy-validated `r=0.85` vs independent kNN `Î(Z;D|Y)`),
   and the natural headline is **worst-site robustness + calibration (ECE/NLL)**
   under label shift, where invariance + correction has a principled edge — not
   the mean-acc number.

**Do NOT claim:**
- "We beat ERM on mean accuracy." We do not, on balanced tasks (4 validations +
  MUMTAZ). Saying so would be refuted by our own runs.
- That the chain-rule identity A1 is *the* contribution. It is one line of
  algebra (a skeptic's "just MI bookkeeping" objection is *correct* about the
  algebra). Sell the **use** — separable, estimable, clinically meaningful shift
  diagnostics — and the **tension→resolution** dynamics, not the identity.

**Venue verdict (Pillar 4):**
- **AAAI-27 main track: viable as positioned**, conditional on the empirical
  concept-shift evidence (job 846957 + the `--balance` ADFTD/MUMTAZ arms showing
  residual `I(Y;D|Z)>0` that the explicit penalty reduces, with a
  robustness/calibration payoff). AAAI rewards a correctly-positioned framework +
  a clinically real use case over a raw SOTA delta. **This is our lane.**
- **ICLR: borderline** without one of (a) a *new generalization bound* in the
  three CMIs (carrying the concept residual explicitly, tighter/more estimable
  than Zhao19's unestimable `|f_S−f_T|`), or (b) a clear benchmark win over
  IWDAN/IWCDAN/entropyDG on a shared label-+concept-shift protocol. The
  decomposition + tension alone will not clear ICLR's novelty bar.

**The linchpin (hostage to experiments):** *show `I(Y;D|Z)` is non-zero after GLS
reweighting + encoder invariance on real psychiatric cross-site data, and that
penalizing it improves worst-site accuracy / calibration.* If that empirical
concept-shift story is weak, the paper collapses to "we re-derived GLS and added an
entropy term." If it holds, the dual-CMI framework is a clean, honest, defensible
AAAI paper.

---

### Artifacts cited
- Theory: `notes/theory/01_tension.md` (A1–A4), `02_resolution.md` (GLS decoupling
  + CONTROL), `03_estimators.md` (variational estimators, code map),
  `04_positioning.md` (Zhao19/GLS/entropyDG, novelty ledger, venue call).
- Verification (re-run, passing): `notes/theory/verify_tension.py`,
  `notes/theory/verify_resolution.py`.
- Code: `cmi/train/trainer.py:66, :67–71 (balance), :172–182 (Step A), :206–211
  (dual Step B)`; `cmi/run_loso.py`; `cmi/run_scps_crossdataset.py:51–52 (dual:lam:gamma)`.
- Synthetic validator (running): `synthetic/dual_cmi_v2.py` → `results/synthetic_dual_v2.txt`.
- Live results: `results/dual_MUMTAZ.json` (partial), plus pending
  `dual_MUMTAZ_bal.json`, `dual_ADFTD.json`, `dual_ADFTD_bal.json`.
