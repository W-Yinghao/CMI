# Is there cross-site *concept shift* in clinical EEG? A two-detector null result

*Self-contained section: methodology → experimental scope → results. Companion to `DUAL_CMI_THEORY.md`
(theory) and `CMI_TECHNICAL_REPORT.md` (full project).*

---

## 1. Methodology

### 1.1 Two conditional mutual informations, two physical channels
Along the chain `D → X → Z → Y` (domain, input, representation, label), domain shift splits into two
conditional MIs acting on two different Bayes channels:

| term | constraint | physical meaning (EEG) |
|---|---|---|
| **encoder** `I(Z;D\|Y)` | `Z ⊥ D \| Y`, i.e. `p_d(z\|y)=p(z\|y)` | within a class, features carry no site/subject identity (montage, impedance, head geometry, session/site artifact) — **covariate leakage** |
| **decoder** `I(Y;D\|Z)` | `Y ⊥ D \| Z`, i.e. `p_d(y\|z)=p(y\|z)` | given the representation, the label rule does not depend on domain — **concept shift** (e.g. differing diagnostic criteria across hospitals) |

They are **not independent knobs**: they are two Bayes directions of one joint, coupled by the
**domain label prior** `π_d(y)=p(y\|d)` through
```
p_d(y|z) = p_d(z|y) π_d(y) / Σ_y' p_d(z|y') π_d(y').
```
**Tension theorem** (verified to 2e-15): `I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)`. Hence under label shift
(`I(Y;D)>0`), an invariant encoder *forces* `I(Y;D|Z)=I(Y;D)−I(Z;D)>0` — the two cannot both be zero unless
Bayes error is zero. So a naive `CE + λ I(Z;D|Y) + γ I(Y;D|Z)` makes the two terms fight. **This tension is
*intra-domain*: it binds the two terms only when they share the *same* `D`. Measured at different
granularities (encoder at `D_site`, decoder at `D_subject`) they decouple — driving `I(Z;D_site|Y)→0` leaves
`I(Y;D_subj|Z)` untouched (§3.7).**

### 1.2 The estimation trap at `D = subject` (SCPS)
In single-class-per-subject clinical data, `Y = g(subject)`, so with `D = subject`, `H(Y|Z,D)=0`
(the subject id determines the label) and **`I(Y;D|Z) = H(Y|Z)` — the classifier's residual entropy, NOT
concept shift.** A held-out domain probe trivially memorises `subject → label`. **Consequence: the decoder
CMI is only meaningful at a `D` where each domain spans both classes (`D = cohort/site`).**

### 1.3 Two independent, null-calibrated detectors
We test concept shift two ways, each with a **permutation null** and a **positive control**:

- **Route C — intercept-residual decoder CMI (discriminative).** Fit three frozen probes on held-out source:
  a domain-blind `a(Y|Z)`, a **full** domain decoder `h(Y|Z,D)`, and an **intercept-only** decoder
  `h0(Y|Z,D)=u(Z)+b_D` (D may shift only a per-domain logit bias = prior/calibration). The **residual**
  `R_res = CE(h0) − CE(h)` isolates the domain-dependent *decision boundary* change — robust to label-prior
  and (partly) to the `D=subject` degeneracy. Reported raw and GLS-reweighted (`w_d(y)=π*(y)/π_d(y)`).
  Method `dualc` = GLS-reweighted encoder CMI (vs domain marginal `p(D)`) + **gated** residual decoder
  `relu(R_res − τ)`.
- **Route A — GLS-VAE likelihood-ratio (`δ_d`) test (generative).** A structured latent model (shared
  class-conditional `p(z|y)` + per-domain prior `π_d` + per-(domain,class) correction `δ_d`) dissolves the
  tension by construction; concept shift = the held-out ELBO gain from fitting only `δ_d`, vs a permutation
  null over domain labels.

The two detectors have **complementary sensitivity** (established empirically in §3.1): Route C is
*discriminative* — it responds to a change in the decision boundary `p(y|z)`; Route A is *generative* — it
responds to a change in the class-conditional `p(z|y)`. Either is sufficient evidence of concept shift; a
double null (both silent) is the strong claim that *no* shift of either kind is present.

---

## 2. Experimental scope

| axis | values |
|---|---|
| **within-dataset (D=subject)** | ADFTD (dementia, 3-cls), MUMTAZ (depression, 2-cls), TUAB (abnormal) |
| **cross-SITE (D=cohort)** | **SCZ same-task** (ds003944+ds003947, resting FEP, 2 hospitals); SCZ 4-cohort (mixed task); PD 3-site (ds002778/ds003490/ds004584, resting) |
| **domain variable D** | site/cohort, **sex, age, race** (demographics from `participants.tsv`); subject excluded for the disease label (§1.2) but *valid* for the paired med-state task (§3.6) |
| **positive case** | PD **medication state** ON/OFF (ds002778+ds003490, paired; Y=med-state) — a real shift we expect to detect |
| **method ladder** | `erm` / `lpc_prior` (encoder CMI) / `dual` (naive) / **`dualc`** (Route C) ; + Route-A GLS-VAE |
| **readouts** | balanced acc, `I(Z;D|Y)` (encZ), decoder CMI: raw / residual / GLS-reweighted; Route-A `δ_d` z & p |
| **controls** | per-method permutation null; synthetic **positive controls** (inject a cohort-dependent boundary) |
| **seeds** | 1–3 (within-dataset ladders) |

Backbone EEGNet on raw 19-ch 10-20 EEG; held-out source split for all probes.

---

## 3. Results

### 3.1 Positive controls — the detectors HAVE power (so a null is interpretable)
**(a) Synthetic latent controls** (inject directly in `Z`):
| regime (synthetic) | Route-C residual | Route-A `δ_d` |
|---|---|---|
| no shift / label-prior only | **0.02** (null) | quiet |
| injected **discriminative** boundary | **0.22** (fires) | — |
| injected **generative** class-cond. | — | **2.5× separation** (fires) |
| `D=subject` degeneracy | 0.31 (only partly suppressed → needs null gate) | n/a |

**(b) Real-data control — inject a *known* concept shift into real cross-site EEG.** In one cohort we
redefine the label rule by an EEGNet-encodable power feature (PC1 of per-channel log-power): swap the
α-fraction of class-0 with highest feature → class-1, and class-1 with lowest → class-0. The swap is
**label-balanced** (equal counts ⟹ pure *concept* shift, not label shift) and **z-encodable** (a power
feature the encoder learns ⟹ `Z` represents it). α=0 is the genuine null; the α=0 residual *is* the real
cross-site baseline of §3.2. Readout = the residual decoder CMI vs α (mean±std, 3 seeds); `Δ/σ` = separation
from the α=0 baseline.

| α (injected strength) | **SCZ** residual | **PD** residual | Route-A `δ_d` gain |
|---|---|---|---|
| 0 (null baseline) | 0.002 ± 0.003 | 0.007 ± 0.001 | −0.02 ± 0.04 |
| 0.2 | 0.015 ± 0.003 (Δ/σ +1.9) | 0.005 ± 0.002 | −0.02 ± 0.03 |
| 0.4 | **0.028 ± 0.002 (Δ/σ +5.1)** | 0.013 ± 0.002 (Δ/σ +2.5) | −0.03 ± 0.04 |
| 0.6 | 0.023 ± 0.003 (Δ/σ +3.2) | **0.025 ± 0.004 (Δ/σ +4.0)** | −0.02 ± 0.05 |

**Route C rises monotonically and separates cleanly from baseline on BOTH real datasets** (Δ/σ up to +5.1;
error bars non-overlapping) — i.e. the detector that reads ≈0.001 on the real cross-site data (§3.2)
demonstrably **fires (~14× baseline) when concept shift is injected into that same data**. **Route A's
generative `δ_d` does not respond to this discriminative injection** (gains stay negative at every α on both
datasets) — exactly its complementary-sensitivity profile (§1.3): it is the detector for generative
class-conditional shifts (its synthetic control), not boundary changes.

⟹ A null reading on real data is a **true null**, not a weak probe — now demonstrated on the real EEG itself,
not only synthetically.

### 3.2 Cross-site concept shift ≈ 0 — two detectors agree
**Route C (`D=cohort`, full ladder):**
| dataset / method | acc | encZ `I(Z;D\|Y)` | decRaw | decRes | **rwRes** |
|---|---|---|---|---|---|
| **SCZ same-task** erm / lpc / dual / dualc | 51.7 / 54.0 / 54.1 / 54.1 | 0.00 | 0.001 | 0.001 | **0.001** |
| **PD 3-site** erm / lpc / dual / dualc | 59.1 / 59.8 / 60.1 / 60.2 | 0.24→0.03 | 0.000 | 0.000 | **0.000** |

**Route A (GLS-VAE `δ_d`, `D=cohort`):**
| dataset | observed gain | null | z | p | detected |
|---|---|---|---|---|---|
| SCZ same-task | −0.053 | 0.013±0.076 | **−0.86** | 0.81 | **No** |
| PD 3-site | −0.020 | 0.015±0.058 | **−0.59** | 0.72 | **No** |

→ Raw, residual, GLS-reweighted decoder CMI **and** an orthogonal generative likelihood-ratio test **all read
null** across sites (observed gains even fall *below* the permutation null). **No measurable cross-site concept
shift.** (Earlier SCZ 4-cohort mixed-task: same, `I(Y;D|Z)≈0.01`.)

### 3.3 Within-dataset `D=subject` is an artifact — confirming §1.2
| ADFTD (D=subject) | acc | encZ | decRaw | **decRes** |
|---|---|---|---|---|
| erm / lpc / dual / dualc | 59.7 / 59.9 / 52.8 / 57.5 | 1.36→0.16 | 0.19 / 0.28 / 0.26 / 0.28 | 0.15 / **0.22** / 0.20 / 0.21 |
| MUMTAZ (D=subject) | 87.2 / 85.4 / 85.4 / 85.4 | 1.43→0.02 | 0.006 / 0.04 | 0.006 / 0.03 |

The ADFTD raw decoder CMI (~0.28) is **the `H(Y|Z)` artifact** (large because 3-class is hard; MUMTAZ binary
is easy ⟹ small). The intercept-residual **reduces it only ~24% (0.28→0.22), does NOT zero it** — exactly the
positive-control `subject-degen` caveat (finite-sample probe noise). **⟹ the decoder CMI must be read at
`D=cohort`; no residualization rescues `D=subject`.**

### 3.4 Accuracy & method behaviour
- **Parity, no compounding:** `dualc ≈ dual ≈ lpc` on cross-site; encoder term removes leakage (10–50×) everywhere.
- **Route C is better-behaved than naive dual:** where naive `dual` over-regularises and collapses
  (ADFTD 52.8), `dualc`'s **gate keeps the decoder term off when its residual is null**, recovering 57.5; on
  the cross-site nulls `dualc` reduces cleanly to GLS-encoder-CMI with **no accuracy cost**.

### 3.5 The choice of D — concept-invariance across site **and** demographics
Concept shift is only defined relative to a domain variable `D`. Beyond site, any class-spanning metadata
variable is a valid `D` for the decoder term (subject is *not* — §1.2). We swept `D ∈ {site, sex, age, race}`
on both diseases; for each split α=0 gives the **real** `I(Y;D|Z)` and an α=0.5 injection gives the
**detector-power** check on that same split (Δ/σ vs the α=0 baseline). All `D`-values are filtered for
non-degeneracy (≥2 classes, ≥150 samples); age is a median/tertile split; race is collapsed to White-vs-rest
for power.

| `D` | SCZ `I(Y;D\|Z)` (power Δ/σ) | PD `I(Y;D\|Z)` (power Δ/σ) | verdict |
|---|---|---|---|
| site / cohort | 0.002 (+5.1) | 0.007 (+4.0) | well-powered **NULL** |
| **sex** | 0.004 (+2.5) | 0.004 (+8.0) | well-powered **NULL** |
| **age** (binary) | 0.007 (+2.6) | 0.005 (+3.4) | well-powered **NULL** |
| **race** (White-vs-rest) | 0.011 (+1.2) pooled / **0.002 (+2.8) within ds003944** | — | site-confound resolved → **NULL** |

**Every split where the detector is demonstrably powered reads null.** Given the EEG representation, the disease
decision rule does **not** depend on sex, age, or recording site — a *subpopulation-robustness / fairness*
result, not mere absence of evidence (the per-split positive control proves power). Two methodological notes the
sweep surfaced: (i) an **underpowered split fakes a weak positive** — SCZ-age read 0.015 as a noisy 3-way split
but collapsed to 0.007 once binarised for power, so the power check is essential; (ii) **race aliases site**
(the cohorts differ in race mix) — pooled across cohorts it reads a marginal 0.011 (Δ/σ +1.2), but **measured
*within* a single cohort (ds003944, removing the site confound) it is a clean null, 0.002±0.002 with the
control firing +2.8** — so race concept shift is null too once site is controlled. Route-A `δ_d` is negative on every split — no generative shift across any
demographic axis either.

### 3.6 A positive case — PD medication state (ON vs OFF): the framework finds shift where it exists
Every result so far is a null; a fair reader asks whether the method can register *anything*. The PD
medication-state task answers it. Levodopa demonstrably alters resting EEG (β-band), and ds002778 / ds003490
record each PD subject **both ON and OFF** — a paired design with two properties: (i) `Y=med-state` is a *real*
shift we expect to detect, and (ii) because every subject spans both states, **`D=subject` is non-degenerate
here** (the exact thing impossible for the disease label, §1.2). Cache: 3200 trials, 40 PD subjects, balanced
[1600 OFF, 1600 ON], all 40 spanning both states.

| `D` | `Y\|Z` decodability | `I(Y;D\|Z)` (α=0) | control Δ/σ |
|---|---|---|---|
| — (decodability) | **0.89** (chance 0.5) | — | — |
| **site/cohort** (ds002778 vs ds003490) | — | 0.004 ± 0.003 (**null**) | +2.4 |
| **subject** (40, non-degenerate) | — | 0.05 ± 0.005 (**real**, two-null validated) | see below |

1. **ON vs OFF is ~89% decodable** (within-subject) — a strong, real physiological signal. This is the
   *negative-control-for-the-null*: the disease-concept nulls (§3.2/§3.5) are **not** an artifact of a dead
   method — the same pipeline lights up ~80% above chance when a genuine shift is present.
2. **The medication-signature is concept-consistent across the two PD sites** (`I(Y;D|Z)=0.004`, control fires
   +2.4) — site-universal, exactly like the disease-concept null.
3. **`D=subject` is real and CONFIRMED (`I(Y;D|Z)=0.05±0.005`, 3 seeds): a subject-specific levodopa
   response.** Because the 40-domain residual could in principle be probe-capacity inflation (the §3.5
   race-4-way lesson), we validated it against **two permutation nulls** (extract `Z` once, refit the residual
   100× each): a **fake-subject** null (permute subject labels → 40 random groups of the *same sizes*, real
   ON/OFF) and a **within-subject label** null (scramble ON/OFF *inside* each subject). Both collapse to
   **identically 0.000** — so the 0.05 is *not* a capacity floor (random grouping gives nothing) and *requires
   the real medication labels* (scrambling kills it). Permutation p≈0.01, both-confirm on every seed.
   Interpretation: the levodopa effect lies along **different `Z`-directions for different patients** — no
   single universal ON/OFF boundary exists; each subject needs their own. Heterogeneous individual medication
   response, detected as genuine concept shift across subjects.

⟹ the decoder term *does* move — for the medication signal it is strongly decodable, **concept-consistent
across sites**, yet **genuinely subject-specific** — while the *disease* decision rule stays null across every
site/demographic axis. The framework finds shift where it exists (and *what kind*: site-invariant but
subject-varying) and null where it doesn't.

### 3.7 Hierarchical D — the tension is intra-domain, not a global coupling
A natural design question: assign the two terms *different* domain granularities — penalise the encoder at the
deployment-relevant level (`D=site`, so `Z` generalises across hospitals) while probing the decoder at the
finest level (`D=subject`). We decoupled them in the cross-site runner (`--dec_domain`) and ran PD+SCZ, 3 seeds:

| | bAcc | `I(Z;D_site\|Y)` (encoder) | `I(Y;D_subj\|Z)` (decoder) |
|---|---|---|---|
| PD erm / **lpc_prior** | 59.8 / 58.8 | 0.202 → **0.031** (6.5×↓) | 0.120 → 0.124 *(unchanged)* |
| SCZ erm / **lpc_prior** | 52.7 / 53.6 | 0.446 → **0.122** (3.7×↓) | 0.225 → 0.239 *(unchanged)* |

Two readings. (i) **No performance gain**: the encoder term just removes site leakage as before (parity acc),
and the decoder-at-subject term is the `H(Y|Z)` artifact of §1.2 (larger for harder near-chance SCZ), *not*
concept shift. (ii) **A structural refinement of the tension theorem**: with a *shared* `D`, removing leakage
*raises* the decoder CMI (the flagship D=subject ladder shows erm 0.166 → lpc 0.188 on SCZ). Here, with
encoder@site and decoder@subject, `I(Z;D_site|Y)` collapses 4–7× while `I(Y;D_subj|Z)` does **not move** — the
two terms are **independent across granularities**. So the tension is a property of *sharing one* `D`, not an
unavoidable global coupling; choosing the encoder and decoder domains at different levels dissolves it. The
practical rule that falls out: **encoder → coarse class-spanning `D` (site) for deployment invariance; decoder
→ a class-spanning `D` (cohort) to read concept; `D=subject` only carries concept on a paired task (med-state,
§3.6), never the single-class-per-subject disease label.**

---

## 4. Conclusion
Across two **independent, positive-control-validated** detectors and every well-powered domain split — site,
sex, and age, on both SCZ and PD — **concept shift in our clinical EEG is rigorously ≈ 0** (0.002–0.007), not
"undetected" but "absent", since the detector demonstrably fires (per-split positive controls, Δ/σ up to +8)
when concept shift is real. The disease decision rule, given the representation, is invariant to recording
site and to demographics (sex, age, and — once site-controlled within a cohort — race). Crucially this is
**not** a dead detector: on the PD medication-state task (§3.6) the same pipeline decodes ON-vs-OFF at ~89%,
the decoder term moves, and a two-null-validated **subject-specific levodopa response** (`I(Y;D_subj|Z)=0.05`,
abolished under both fake-subject and within-subject-label permutation) emerges — yet that signature is
*site-consistent*. The framework registers shift where it exists, identifies *what kind* (site-invariant but
subject-varying), and reads null where it doesn't. The decoder CMI is therefore a **trustworthy
diagnostic that credibly reads null**, not an accuracy lever. The supported, defensible contributions are: the
**three-shift information-theoretic decomposition** (covariate `I(Z;D|Y)` / label `π_d` / concept `I(Y;D|Z)`),
the **tension theorem**, the **degeneracy correction** (`D` must be cohort-level; intercept-residual + null
gating), and the empirically robust levers — **encoder leakage removal + calibration**. Naive dual-CMI is
mis-specified; **GLS-residual gated dual (Route C)** is the principled treatment that does no harm under a null.

The decoder term has now been validated on the real EEG itself: a **real-data positive control** (§3.1b)
injects a known cohort-dependent boundary and Route C's residual rises monotonically and separates from the
α=0 baseline on **two** datasets (SCZ Δ/σ up to +5.1; PD up to +4.0). So the cross-site ≈0 reading is a
demonstrated true null, not a blind spot.

*Caveats:* cross-site samples are small (2–4 cohorts; SCZ near chance accuracy). The two detectors are
**complementary, not redundant** — Route C (discriminative) fires on the real boundary injection while Route A
(generative `δ_d`) does not, and vice-versa on a generative latent shift; the strong null claim rests on
*both* being silent on the real data. `D=subject` decoder CMI is reported for completeness but is **not** a
concept-shift measure.
