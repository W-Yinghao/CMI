## 6. Synthetic positive/negative controls & pipeline-validation gate

**Purpose.** Before C87E is permitted to read any real target-cohort model outcome, the *exact same* selection/estimation/inference code must be exercised on synthetic worlds whose ground truth is known by construction. Three controls are registered: **POS** (the pipeline must *detect* success — recover positive transport-consistency and a positive active gain when both truly exist), **NEG** (the pipeline must *not manufacture* success — no false transport claim, no spurious active gain when there is none), and **CALIB** (the active/LURE estimators are unbiased and the patient-cluster bootstrap attains nominal coverage on synthetic ground truth). **All three must pass; a failure is an engineering blocker, never a scientific result about ECG.** No control parameter depends on any real model outcome.

### 6.0 What is being validated and why it is leak-free

The real study answers, per untouched target cohort `e`, whether finite-budget CANDIDATE MEASUREMENT recovers the deployment-best model. The controls validate the *machinery* that produces those answers — `a*C_e`, `a*H_e`, `T_e`, `R_{e,pi,B}`, `G_{e,pi,B}`, and their patient-clustered CIs — on data where the answers are known a priori. The controls run on a **synthetic loss tensor** generated from frozen design constants; they never touch PhysioNet fields, never train a model, and never read a real target outcome. This is the standard "test the test" discipline: a real NULL result (measurement→control gap) is only credible if the pipeline provably *can* detect a positive when one is planted (POS/CALIB) and provably *does not* invent one from noise or leakage (NEG).

### 6.1 Shared synthetic generator (one simulator, three parameterizations)

The generator emits the *sufficient statistics the pipeline consumes* — a per-record loss for every candidate — so the controls exercise the real selector and estimator code, not a toy reimplementation. For cohort `e`, view `v ∈ {C (construction/acquisition), H (held/deployment)}`, candidate `a ∈ {1..A}`, patient `p`, and record `r` of patient `p`:

```text
GENERATIVE MODEL  (frozen skeleton; per-world settings in 6.2/6.3/6.4)
--------------------------------------------------------------------
  loss_{v,a,p,r} = clip( mu_{v,a}                       # candidate "skill" in view v
                         + s_p                          # patient random effect (shared across a patient's records)
                         + b_{p,a}                      # patient x candidate interaction
                         + D_{p,r} * delta_{v,a}        # discriminative-record signal
                         + eta_{v,a,p,r} ,  0, 1 )      # iid record noise

  s_p        ~ N(0, rho * sigma^2)            # induces intra-patient loss correlation = rho
  b_{p,a}    ~ N(0, kappa * sigma^2)          # keeps correlation present in loss-DIFFERENCES
  D_{p,r}    ~ Bernoulli(phi)                 # per-record "informative" indicator
  eta        ~ N(0, (1-rho-kappa) * sigma^2)  # residual, variances sum to sigma^2
  delta_{v,a}= (mu_{v,a} - mean_a mu_{v,a})   # candidate separation carried ONLY by D=1 records

  Ground truth (KNOWN, never revealed to the selector):
    L^v_e(a)  = E_{p,r}[ loss_{v,a,p,r} ]          # true per-view risk of candidate a
    a*C_e     = argmin_a L^C_e(a) ;  a*H_e = argmin_a L^H_e(a)
    T_e       = L^H_e(a*C_e) - L^H_e(a*H_e)        # full-view transport gap
  Pipeline sees ONLY loss_{H,a,p,r} for records it QUERIES (one label = one record), plus,
  where the design provides it, the free construction-view ranking a*C_e (label-free proxy).
```

Structural parameters that **must match the real C87E configuration** (resolved by the NO-OUTCOME rules in 6.5, not by any outcome): candidate cardinality `A`, number of target cohorts `E`, the budget grid `B_grid`, patients-per-cohort `N_p^{(e)}`, and the records-per-patient (cluster-size) distribution. The discriminative fraction `phi`, correlation `rho`, interaction share `kappa`, separation magnitudes, and transport magnitudes are **frozen design constants** chosen a priori (below); none are fit to data or to any model result.

Design intuition for the levers: `phi` creates genuine label-efficiency structure — only `D=1` records distinguish candidates, so a policy that concentrates its budget on high-disagreement/high-information records recovers `a*H_e` with fewer labels than uniform, yielding `G>0`; setting all `mu_{H,a}` equal removes that structure (null world); `rho, kappa` make patient-level clustering matter for the CIs.

### 6.2 POS — the pipeline must recover positive transport-consistency AND positive active gain

```text
POS WORLD  (planted: FULL transports, AND a label-adaptive policy genuinely beats P0)
-------------------------------------------------------------------------------------
  Transport:  mu_{C,a} and mu_{H,a} share the SAME argmin and the same top-k order for
              k>=3 (Spearman(mu_C, mu_H) >= 0.9).  => a*C_e = a*H_e, T_e = 0 (exactly, by
              construction) in ALL E synthetic cohorts.
  Separation: L^H_e(a*H_e) is SEP_POS below the runner-up (SEP_POS = 0.05 loss units),
              well above sampling noise at the real held n.
  Info:       phi in {0.15}  (only 15% of records are discriminative) with rho>0, kappa>0.
              => an information/disagreement-seeking adaptive policy exploits D=1 records
                 and finds a*H_e faster than uniform P0.
  Effect size SEP_POS and phi are set so the POS active gain clears the gate with
  >=80% Monte-Carlo power at the REAL held-view n and the frozen B_grid (see 6.6).
```

**POS pass criteria (all must hold):**
1. **T-consistency recovered:** in every one of the `E` synthetic cohorts, the patient-clustered 95% CI for `T̂_e` covers 0 **and** the estimated argmins agree (`â_C_e == â_H_e`). The pipeline reports "transports" where the truth transports.
2. **Active gain recovered:** there exists a budget `B ∈ B_grid` and at least one **label-adaptive** policy `pi ∈ {LURE/Active-Testing-guided, Hara-VMA, Model-Selector, CODA}` for which the patient-clustered 95% **lower** confidence bound `LCB[G_{e,pi,B}] > 0` in **all** `E` cohorts (same `pi`, same `B`, no pooling — exactly the real formal gate).
3. **Power floor:** over `K ≥ 1000` Monte-Carlo redraws of the POS world, criterion 2 is met in `≥ 80%` of redraws; the run also **reports the minimum detectable effect (MDE)** of the real gate as a design output.

If POS fails, the pipeline (or the gate) cannot detect a real positive of the planted size → any subsequent real NULL is uninterpretable → **blocker**.

### 6.3 NEG — the pipeline must not manufacture transport or active gain

Two sub-worlds isolate the two failure modes.

```text
NEG-A WORLD  (planted: FULL does NOT transport; but held-view signal EXISTS)
---------------------------------------------------------------------------
  Nontransport: mu_C and mu_H are ANTI-aligned (Spearman(mu_C,mu_H) <= 0), and a*C_e is
                placed among the WORST third on the held view, so
                T_e = L^H_e(a*C_e) - L^H_e(a*H_e) >= TAU_NEG (TAU_NEG = 0.05 loss units)
                in ALL E cohorts.  (This mirrors, but IMPOSES a priori, the C86D-style
                full nontransportability; it is a design constant, not a model reading.)
  Held signal:  a*H_e is well-separated (SEP_POS) so active methods CAN reduce held regret
                on the held view even though construction points to the wrong model.

NEG-B WORLD  (planted: NO information — no candidate is truly better on held)
---------------------------------------------------------------------------
  Null:  mu_{H,a} = mu_0 for ALL a  (identical expected held loss); phi irrelevant;
         a*H_e is defined by sampling noise only.  rho>0, kappa>0 retained.
  => No policy can systematically beat P0; any G>0 is spurious (optimism/leakage).
```

**NEG pass criteria (all must hold):**
1. **No false transport (NEG-A):** in every cohort the patient-clustered 95% `LCB[T̂_e] > 0` and `â_C_e ≠ â_H_e` are recovered. The pipeline correctly reports nontransport and does not collapse `T` to 0.
2. **No spurious active gain (NEG-B):** for **every** policy `pi` and **every** budget `B ∈ B_grid`, it is **NOT** the case that `LCB[G_{e,pi,B}] > 0` holds in all `E` cohorts. Equivalently, over `K ≥ 1000` NEG-B redraws, the family-wise rate of "same-`pi`, same-`B`, all-`E`-cohort `G>0`" is `≤ α = 0.05`.
3. **Sign discipline:** in NEG-B the pooled point estimate of `G` is statistically indistinguishable from 0 for the label-adaptive policies (`|mean_K Ĝ| ≤ 2·SE_MC`).

If NEG-B fails, the pipeline invents active gain from noise or from a hidden held-label leak → **blocker** (see the NEG-M3 leak mutation in 6.4).

### 6.4 CALIB — estimator unbiasedness and patient-cluster bootstrap coverage

The registry's active estimators (LURE / Active-Testing; the Hara variance-minimizing estimator; the estimators internal to Model-Selector / CODA / ASE) all estimate a **held risk / loss-difference** from *adaptively, without-replacement* queried labels. The verified LURE definition that the CALIB control checks the implementation against (Farquhar et al. 2021; Kossen et al. 2021):

```text
LURE  (Levelled Unbiased Risk Estimator) — VERIFIED reference implementation
----------------------------------------------------------------------------
  R_LURE = (1/M) * sum_{m=1..M} v_m * loss(i_m)                 # M queried of pool size N
  v_m    = 1 + ((N - M)/(N - m)) * ( 1/((N - m + 1) * q(i_m)) - 1 )
    - acquisition is WITHOUT replacement; q(i_m) = proposal prob. of the m-th queried
      record given the previous m-1 picks and the pool.
  Properties (used as exact assertions):
    E[v_m] = 1  for all m, M, N, and all proposals   => R_LURE is UNBIASED for the risk.
    Uniform proposal  => every v_m = 1  => R_LURE == naive mean of queried losses.
    M = N (full pool) => every v_m = 1.
  For active model selection (Hara et al. 2024) the same weights give an unbiased estimate
  of the loss DIFFERENCE L^H_e(a) - L^H_e(a'); the ideal q minimizes its variance.
```

**CALIB recipe.** Generate a synthetic cohort with **known** `L^H_e(a)` and known pairwise differences, with `rho>0, kappa>0` (so patients are genuine clusters). Run each registry estimator under (i) a **uniform** proposal and (ii) an **adaptive** (variance-minimizing / disagreement) proposal, over `K ≥ 1000` Monte-Carlo query draws. Estimand set = the per-candidate held risk `L^H_e(a)` for the contenders and the top pairwise loss-differences that drive selection.

**CALIB pass criteria (all must hold):**

```text
CALIB THRESHOLDS
----------------
  Exact identity:   uniform-proposal LURE weights are all == 1 (bit-exact) and
                    R_LURE == naive mean to ||.||_inf <= 1e-9.
  Unbiasedness:     for EACH estimand and EACH estimator, under BOTH proposals,
                    |mean_K(estimate) - truth| <= max( 2 * SE_MC , 0.003 loss-units ).
  Coverage:         patient-cluster (resample PATIENTS with replacement, carry all their
                    records) 95% bootstrap CIs attain empirical coverage in [93.5%, 96.5%]
                    for T_e, R_{e,P0,B}, and G_{e,pi,B}, over K >= 1000 synthetic cohorts.
                    Paired/common-random-number bootstrap across policies for G.
```

**Mutation tests (the validators must have power — each MUST trigger its failure signature, else the corresponding control is non-diagnostic → blocker):**

```text
CALIB-M1  unweighted (naive) estimator under the ADAPTIVE proposal
          => MUST show detectable bias ( |bias| > 2*SE_MC ).   [proves 6.4 catches a missing correction]
CALIB-M2  record-level (NON-clustered) bootstrap under rho>0
          => MUST under-cover ( empirical coverage < 90% at nominal 95% ). [proves clustering is load-bearing]
NEG-M3    oracle-leak variant (selector peeks at the FULL held loss field)
          => MUST produce spurious all-E-cohort G>0 in NEG-B.  [proves 6.3-crit-2 catches held-label leakage]
```

### 6.5 NO-OUTCOME resolution of not-yet-pinnable choices

Each choice below is fixed by resources / public metadata / the frozen C87E config — never by a model result:

```text
NO-OUTCOME RULES
----------------
  A_syn        := | candidate set the C87E selector ranges over |  (frozen config).
                  Real zoo is architecture-controlled: per context {1 ERM + 40 OACI + 40 SRC}
                  = 81 candidates; contexts = 2 panels x 2 seeds x 2 support levels.
  E_syn        := number of untouched target cohorts = 3  (Georgia, Chapman-Shaoxing, Ningbo)  [metadata].
  B_grid       := the C87E interface-section budget grid  (frozen config; controls reuse it verbatim).
  N_p^{(e)}    := held-view patient counts of the corresponding target cohort  [public metadata / headers].
  cluster-size := empirical records-per-patient distribution of that cohort  [metadata audit of public
                  headers]; if a cohort is ~1 record/patient, INJECT multi-record patients + rho>0 anyway
                  in CALIB so the clustered bootstrap is genuinely stressed (control, not realism).
  per-record loss ell := the ADDITIVE per-record loss pinned by the C87E metric section. If the deployment
                  metric is non-additive (e.g., the Challenge-2021 weighted score), the estimand is its
                  additive surrogate and the controls validate the estimators for THAT surrogate; the
                  non-additivity is routed to the metric section, out of scope here.
  policy set   := the frozen C87E method registry (P0 uniform; a prediction-driven active method; and
                  >=1 label-adaptive method — Hara-VMA / Model-Selector / CODA); controls run the SAME set.
  design constants (frozen a priori, world-specific): SEP_POS=0.05, TAU_NEG=0.05, phi=0.15,
                  rho, kappa (variance shares), K>=1000, alpha=0.05, coverage band [93.5%,96.5%].
```

### 6.6 The gating rule (registered)

```text
GATING RULE — C87E may read real target fields IFF all controls pass
--------------------------------------------------------------------
  PASS(controls) := POS(6.2 crit 1-3) AND NEG(6.3 crit 1-3) AND CALIB(6.4 all)
                    AND all mutation tests (CALIB-M1, CALIB-M2, NEG-M3) trigger their failure signatures.

  1. Controls run under the FROZEN commit, on the SAME modules C87E uses (selector, estimators,
     patient-cluster bootstrap, budget grid, A, E). A separate toy reimplementation is FORBIDDEN.
  2. Determinism/provenance: fixed seeds; bit-identical self-replay (sha256 of the control outputs
     equal on re-run); record code_sig + synthetic-gen param hash + config; emit a sha256 manifest.
  3. On PASS: emit a signed CONTROL_PASS token. The C87E real-field stage checks this token as a
     hard precondition and REFUSES to run without it.
  4. On FAIL or AMBIGUOUS (any statistic landing between pass/fail bands): this is an ENGINEERING
     BLOCKER. Fix the pipeline and re-run the controls. NEVER relax a threshold to pass. NEVER report
     a control outcome as an ECG scientific finding. Ambiguous == FAIL.
  5. Power/MC discipline: K >= 1000 everywhere (a 100->1000 change has flipped a conclusion before).
```

### 6.7 Scope boundaries (what the controls do NOT claim)

The controls certify that the pipeline can *detect* planted transport/gain, *refuses* to invent them from noise/leakage, and produces *unbiased, nominally-covered* estimates. They do **not** predict the real ECG outcome, do not tune any real threshold, and cannot substitute for the untouched-cohort gate. Passing controls is a **necessary engineering precondition**, not evidence about the Measurement→Control gap. The synthetic effect sizes are deliberately generous (to test detectability), so the reported MDE (6.2-3) is the *sensitivity floor* of the real gate, not a claim about real effect magnitude.