## 3. Acquisition/selection method registry (reference-fidelity)

**Scope and freeze status.** This section registers the candidate acquisition/selection methods, classifies each as *baseline / prediction-driven active / genuinely label-adaptive*, maps each to our estimand (choose the single deployment-best of the 81-candidate zoo **per target cohort** under a target-label budget, statistical unit = **patient**, one query = one record's SNOMED-CT label), and specifies the **reference-fidelity audit** that gates freezing the primary family. Every classification below is taken from the methods' **public description**; the primary/secondary split is resolved by a **synthetic toy-oracle audit + public-code numerical agreement + metadata**, never by any real ECG target-model outcome (see §3.5 leakage note). Where a public description is thin, it is **flagged, not invented**.

### 3.0 Terminology used in this section
- **Prediction-driven (active):** the query/acquisition rule uses only candidate-model *predictions* (or a surrogate's predicted loss/uncertainty) computed on unlabeled target records; acquiring true labels does **not** change which records the method prefers to query beyond removing them from the pool, and it does **not** maintain an updated preference/posterior over *which candidate is best*. (This is the class the EEG line C86D already tested: P0 uniform, A1 expected-loss, A2H pairwise-disagreement.)
- **Genuinely label-adaptive:** the method maintains an explicit belief/posterior over *which candidate model is best* and **updates that belief from queried ground-truth labels**, and its next query is chosen to disambiguate that belief (e.g., expected information gain about the identity of the best model). This is the strictly stronger class C87 adds relative to C86.
- **Estimator vs. selector:** an *estimator* (LURE/ASE/VMA) produces an unbiased/low-variance estimate of each candidate's held-view risk from actively-acquired labels; we then select `argmin`. A *selector* (MODEL SELECTOR/CODA) directly maintains a posterior over the best-model index.

### 3.1 Method-by-method registry

**(a) P0 — uniform without replacement (mandatory reference floor).**
Draw records from the target-cohort held pool uniformly at random **without replacement**, query one clinical label per draw, and after budget `B` select the candidate with lowest empirical held loss on the acquired labels (ties broken by a pre-registered deterministic rule). This is the non-adaptive control and the denominator of the active-gain quantity `G_{e,pi,B} = R_{e,P0,B} − R_{e,pi,B}`. **Class: baseline (neither prediction-driven nor label-adaptive).** Mapping: identical estimand for all 81 candidates share the same acquired label set; CIs computed with patient-clustered resampling.

**(b) Active Testing / LURE — Kossen, Farquhar, Gal, Rainforth, ICML 2021, "Active Testing: Sample-Efficient Model Evaluation."** Active Testing selects test points to label so as to estimate a model's risk with far fewer labels, and corrects the selection bias with the **LURE** (Levelled Unbiased Risk Estimator) weight, originally derived in Farquhar, Gal & Rainforth, "On Statistical Bias in Active Learning: How and When to Fix It" (ICLR 2021). For a pool of `N` records, drawing `M` labels without replacement under acquisition proposal `q`, with `L_m` the pointwise loss of the acquired record at step `m`, the estimator is

```text
R_LURE = (1/M) Σ_{m=1..M} v_m · L_m ,
v_m = 1 + ((N − M)/(N − m)) · ( 1/((N − m + 1)·q(i_m)) − 1 ).
```

The **acquisition proposal** `q` is proportional to a *surrogate/predicted* expected loss (from a proxy model or the candidate itself) — i.e. acquisition is **prediction-driven**; the estimator then consumes the *true* acquired labels to remain unbiased for the full-pool risk under any `q`. **Class: prediction-driven active + unbiased estimator; NOT genuinely label-adaptive** (the proposal does not update a "which-candidate-is-best" belief from acquired labels). Mapping / fidelity flags: (i) basic Active Testing targets evaluating **one** model; using it for *selection* across 81 candidates requires a registered multi-candidate proposal (e.g. shared proposal + per-candidate LURE, then `argmin`) — this is an explicit adaptation, not the paper's default, and is flagged. (ii) LURE assumes a **decomposable pointwise loss**; a non-decomposable deployment metric (macro-F1, Challenge score) breaks unbiasedness and must be replaced by the registered decomposable acquisition loss (§3.4 rule 3). Public code: `github.com/jlko/active-testing`.

**(c) ASE / XWED — Kossen, Farquhar, Gal, Rainforth, NeurIPS 2022, "Active Surrogate Estimators: An Active Learning Approach to Label-Efficient Model Evaluation."** ASE replaces the Monte-Carlo LURE estimator with a **surrogate** that imputes the loss of every unlabeled pool point; the surrogate is *actively learned*, and its acquisition function **XWED (eXpected Weighted Disagreement)** prefers points with high epistemic surrogate-uncertainty that also contribute strongly to the final risk estimate. **Class: prediction-driven active, surrogate estimator — *label-informed* (the surrogate retrains on acquired labels) but NOT a best-model posterior**; the object remains a per-candidate risk estimate, not a preference over the zoo. Mapping / flags: requires training and maintaining a surrogate over 12-lead ECG inputs (heavy); surrogate mis-specification biases the estimate. Assigned to the **secondary tier** on cost/complexity grounds, not performance. Public code: `github.com/jlko/active-testing` (ASE branch), arXiv 2202.06881.

**(d) Hara / Matsuura variance-minimizing active model selection — Matsuura & Hara, "Active model selection: A variance minimization approach," *Machine Learning* (Springer), 2024 (DOI 10.1007/s10994-024-06603-1; earlier NeurIPS 2023 RealML workshop).** The method estimates the **sign of the difference of test losses** between models using LURE, then derives the **query distribution that minimizes the variance of that LURE loss-difference estimate**, and labels adaptively so the best model is identified with the fewest labels. **Class: active, variance-minimizing estimator with an *adaptive query distribution*** (the query distribution is recomputed as loss estimates refine, so it is *label-refined*), but the preference object is an estimator ranking, **not a Bayesian best-model posterior** — it sits between prediction-driven and genuinely label-adaptive; classified here as **prediction-driven active with adaptive query distribution** and flagged as such. Mapping / flags: the paper is framed around **pairwise / multiple-model** loss differences; scaling to 81 candidates requires either O(K) or O(K²) pairwise comparisons or the paper's multiple-model variant — registered explicitly and flagged. Public description of the exact multi-model query rule is **moderately thin** (workshop → journal); the reference-fidelity audit (§3.3) must pin it against the authors' implementation or a faithful re-derivation.

**(e) MODEL SELECTOR — Okanovic, Kirsch, Kasper, Hoefler, Krause, Gürel, AISTATS 2025 (PMLR v258), "All models are wrong, some are useful: Model Selection with Limited Labels."** MODEL SELECTOR maintains an explicit **posterior over which candidate is best**, models each candidate by a single scalar error parameter `ε` (bootstrapped from ensemble/consensus pseudo-labels, no ground truth needed to initialize), and at each step queries the unlabeled record that **maximizes expected information gain about the best-model identity** (minimizes expected posterior entropy over both hypothetical label outcomes); the posterior is **updated by Bayes' rule from each true label** via simple multiplicative updates. It is model-agnostic, using only **hard predictions**. **Class: genuinely label-adaptive (preference-posterior).** Mapping / flags: (i) the single-`ε`-per-model assumption is a strong simplification for a **133-class, multi-label** SNOMED-CT ECG setting — flagged; a registered multi-label→pointwise reduction is required (§3.4 rule 3). (ii) Uses hard predictions → maps cleanly to the 81-candidate zoo. Public code available (Okanovic et al.). This is the primary genuinely-label-adaptive candidate.

**(f) CODA — Kay, Van Horn, Maji, Sheldon, Beery, ICCV 2025 (Highlight), "Consensus-Driven Active Model Selection."** CODA instantiates a **Dawid–Skene-style Bayesian model** over (classifiers × categories × data points): consensus of all candidate predictions seeds per-model, per-class **confusion-matrix Dirichlet priors**; it maintains a posterior `P_Best` over the best-model index, and at each step queries the record with maximal **expected information gain** `EIG = H(P_Best) − Σ_c π̂(c|x)·H(P_Best^c)`; on receiving a true label it performs Dirichlet–categorical updates (with a partial-update rate `η`, default 0.01). **Class: genuinely label-adaptive (preference-posterior).** Mapping / flags: (i) confusion-matrix parameterization assumes **single-label multiclass**; the multi-label ECG structure requires a registered adaptation (per-label binary decomposition or a defined multiclass reduction) — flagged as nontrivial. (ii) Very recent (2025); reference behavior must be pinned against public code `github.com/justinkay/coda`. In their own benchmarks CODA's baselines include Active Testing, VMA (Matsuura & Hara) and MODEL SELECTOR — i.e. our registry mirrors that method's comparison set. Second genuinely-label-adaptive candidate.

**Lineage (context only, not run):** classical variance-minimizing active model comparison — Sawade et al., "Active Comparison of Prediction Models," NeurIPS 2012; online/bandit active model selection — Karimi et al., "Online Active Model Selection for Pre-trained Classifiers," AISTATS 2021 (PMLR v130). Cited for provenance; not part of the registry.

### 3.2 Classification summary (registry table)

```text
Method            Venue                     Object        Acquisition uses      Updates best-model    Class
                                                          true labels to        posterior from
                                                          re-target queries?    true labels?
----------------- ------------------------- ------------- --------------------- --------------------- -----------------------------
P0 uniform        —                         empirical     no                    no                    BASELINE (reference floor)
Active Testing/   ICML 2021 (Kossen)        per-cand.     no (proposal from     no                    PREDICTION-DRIVEN active
  LURE                                      risk est.       predicted loss)                             + unbiased estimator
ASE / XWED        NeurIPS 2022 (Kossen)     per-cand.     partial (surrogate    no                    PREDICTION-DRIVEN active
                                            risk est.       retrains on labels)                         (label-informed surrogate)
VMA (Matsuura/    Springer ML 2024          loss-diff     yes (adaptive query   no (estimator rank,   PREDICTION-DRIVEN active
  Hara)                                     sign est.       dist. refines)        not posterior)        + adaptive query dist.
MODEL SELECTOR    AISTATS 2025 (Okanovic)   best-model    yes                   yes                   GENUINELY LABEL-ADAPTIVE
                                            posterior
CODA              ICCV 2025 (Kay)           best-model    yes                   yes                   GENUINELY LABEL-ADAPTIVE
                                            posterior
```

### 3.3 Primary family, secondary tier, and the freeze gate

**PRIMARY FAMILY (pre-registered; each runs at every budget in every untouched cohort; the formal gate — SAME method + SAME budget holding in ALL of Georgia, Chapman-Shaoxing, Ningbo with patient-clustered inference and NO pooled p-value — is evaluated on these):**

```text
PRIMARY (mandatory):
  1. P0 uniform-without-replacement            [baseline / denominator]         — MANDATORY
  2. Active Testing / LURE  (Kossen 2021)      [prediction-driven active]       — MANDATORY
  3. MODEL SELECTOR         (Okanovic 2025)    [genuinely label-adaptive #1]    — MANDATORY
PRIMARY (conditional):
  4. CODA                   (Kay 2025)         [genuinely label-adaptive #2]    — PRIMARY iff it PASSES
                                                                                  the §3.3 reference-fidelity
                                                                                  audit AND a multi-label +
                                                                                  patient-cluster mapping is
                                                                                  registered; else SECONDARY
```

This satisfies the requirement that the primary family contain **P0 + ≥1 prediction-driven active (LURE) + ≥1 genuinely label-adaptive (MODEL SELECTOR)**; CODA is a second, mechanistically-independent label-adaptive method (Dawid–Skene posterior vs. MODEL SELECTOR's single-`ε` multiplicative posterior) that strengthens falsification if it clears the audit.

**SECONDARY TIER (reported for completeness; NOT subject to the all-cohort formal gate):**

```text
SECONDARY:
  - Hara / Matsuura VMA        (Springer ML 2024)  [adaptive query dist.; pairwise→multi-model scaling flagged]
  - ASE / XWED                 (Kossen 2022)       [surrogate estimator; heavy; complexity-demoted]
  - CODA                       (Kay 2025)          [only if it fails the primary-conditional audit]
```

### 3.4 Reference-fidelity audit (toy oracle) — the freeze gate

The primary family is **frozen only after** every primary method passes the following audit. **The audit is fully synthetic and reads NO real ECG target-model outcome** — pass/fail is defined on synthetic ground-truth knowns and numerical agreement with public reference code.

```text
TOY-ORACLE CONSTRUCTION (synthetic, no real data, no real model):
  - N synthetic records grouped into synthetic PATIENTS (clustered), with KNOWN multi-hot labels.
  - K synthetic candidate "models" with KNOWN per-record losses, constructed so a UNIQUE best model
    exists with a KNOWN margin; include a decomposable pointwise loss + a non-decomposable metric
    variant to exercise the loss-mapping rule.
  - Because everything is known, there is a GROUND-TRUTH best index and GROUND-TRUTH full-pool risks.

AUDIT CHECKS (each pass/fail at a pre-pinned tolerance):
  A. Unbiasedness/consistency  — LURE / VMA / ASE risk estimates recover the true full-pool risk
     within tau_bias (bias CI covers 0) at each budget; active-variance <= uniform-variance on the
     constructed-advantageous toy.
  B. Reference-code agreement  — our re-implementation's per-step quantities (LURE weights v_m;
     MODEL SELECTOR P_Best multiplicative updates; CODA Dirichlet posteriors + EIG; VMA query dist.)
     match the authors' public code (jlko/active-testing; okanovic model-selector; justinkay/coda)
     within tau_num on the toy.
  C. Adaptive-method correctness — MODEL SELECTOR / CODA P_Best converges to the KNOWN best model
     as budget -> full; EIG acquisition is non-negative and selects higher-information points than P0.
  D. Clustered-sampling validity — one query = one RECORD label, but variance/CIs are PATIENT-clustered;
     verify each method's without-replacement weighting stays valid under patient-cluster sampling,
     or register the exact modification (cluster-level proposal / cluster-robust weighting).
  E. Multi-label mapping check  — verify the registered multi-label->pointwise-loss reduction preserves
     each method's assumptions (decomposable loss for LURE/VMA/ASE; well-defined confusion structure
     for MODEL SELECTOR/CODA). Flag any violation.

GATE RULE (freeze condition):
  FREEZE the primary family iff  P0 + Active Testing/LURE + MODEL SELECTOR  each PASS A–E, i.e. at least
  {baseline + 1 prediction-driven + 1 genuinely label-adaptive} clear the audit.
  A method failing any check is DEMOTED to secondary or EXCLUDED per this rule (CODA's primary/secondary
  status is decided exactly here). No pass/fail depends on any real ECG target result.
```

### 3.5 NO-OUTCOME rules (resolve every not-yet-pinnable choice as a rule, not a value)

1. **Primary label-adaptive membership.** `{MODEL SELECTOR, CODA}` are PRIMARY iff they pass the §3.4 audit (A–E) and admit a registered multi-label + patient-cluster mapping; else SECONDARY. Decided on synthetic-toy behavior + public-code numerical agreement only — **not** on any ECG target performance.
2. **Budget grid `B`.** Pinned by a **metadata audit** of the target cohorts (patient counts and per-class SNOMED-CT support from PhysioNet/CinC Challenge-2021 v1.0.3 public documentation), choosing `B` as pre-declared fractions of the median cohort patient count spanning the reference papers' regimes; pinned **before** any model result.
3. **Pointwise deployment loss for risk-estimator methods.** Fixed to a **decomposable per-record loss** so LURE/VMA/ASE unbiasedness holds; if the deployment metric is non-decomposable (macro-F1 / Challenge score), register the decomposable acquisition surrogate separately from the held-view metric used for `T_e`, `R_{e,pi,B}`, `G_{e,pi,B}`. Fixed by estimator theory, not by any target outcome.
4. **Reference-implementation pinning.** Public reference commits (jlko/active-testing, okanovic/model-selector, justinkay/coda) are **content-addressed and pinned before** the audit; the audit is a signal-integrity check against those exact versions.
5. **VMA multi-model rule + Active-Testing multi-candidate proposal.** The exact O(K)/O(K²) multi-model query rule (VMA) and the shared-proposal-plus-per-candidate-LURE selection rule (Active Testing) are pinned against reference code / faithful re-derivation in the audit **before** freeze; neither is tuned to a target result.

### 3.6 Honesty flags (thin or adapted descriptions)
- **VMA (Matsuura & Hara):** exact multi-model (beyond pairwise) query rule is moderately under-specified in public materials → pinned via audit-B.
- **MODEL SELECTOR:** single-`ε`-per-model assumption is a strong simplification for 133-class multi-label ECG → requires registered multi-label reduction (rule 3); flagged.
- **CODA:** single-label confusion-matrix parameterization; multi-label ECG adaptation is nontrivial and gates its primary status.
- **Active Testing / LURE:** paper's default is single-model *evaluation*; multi-candidate *selection* use is our registered adaptation, not the authors' default.
- **ASE / XWED:** requires training/maintaining an ECG surrogate; demoted to secondary on cost, not performance.