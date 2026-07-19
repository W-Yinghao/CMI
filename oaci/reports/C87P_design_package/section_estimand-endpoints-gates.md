## 4. Estimand, endpoints & statistical gates

All quantities below are **frozen as formulas/rules**. No value here is read from, or chosen in light of, any target-cohort model outcome. Standardizers, optima, and ceilings are *estimators* whose numeric values are computed only at analysis time from held/candidate losses that do not yet exist; the freeze pins the estimator, never the number.

### 4.1 Views, loss, and per-cohort estimands

Two disjoint **patient** partitions of each target cohort `e`:

- **View C (acquisition / construction)** — a fully-labeled reference view (source-panel validation and/or a within-target reference partition; pinned by §Interface). Its full labels define the surrogate prior `a*C_e` — the "measurement you already paid for." No query cost.
- **View H (held / deployment)** — the deployment distribution. Its labels are the deployment truth and are **never revealed** to any policy except through the `B` labels the policy pays to query. One query = one **record's** clinical label (SNOMED-CT scored set `S`).

The statistical unit is the **patient**: every record of a patient lies in exactly one view (see the §4.5 grouping audit); cost is counted per record, but splitting and inference are per patient.

```text
SPEC 4.A — LOSS AND PER-COHORT ESTIMANDS (frozen)
------------------------------------------------------------
Scored-label set S   : pinned by §Data metadata rule (intersection of
                       Challenge-2021 scored classes present in ALL included
                       cohorts, from dx_mapping_scored.csv). |S| <= 26.
Candidate set   A_e  : frozen zoo (§Zoo); a*_ ranges over A_e.
                       (81 per context; 8 contexts = 2 panels x 2 seeds x
                        2 support levels; per-context vs pooled pinned by §Zoo.)

Per-record loss  l(a,r):
  PRIMARY  = mean binary NLL over S:
             l(a,r) = -(1/|S|) sum_{k in S} [ y_k log p_{a,k}(r)
                                            + (1-y_k) log(1-p_{a,k}(r)) ]
  SECONDARY= multilabel Hamming error at fixed op-point 0.5 (robustness only).
  Selection & every gate decision use PRIMARY (NLL). No metric-shopping.

Patient loss (unit = patient j, record set R_j, all in one view):
  lbar(a,j) = (1/|R_j|) sum_{r in R_j} l(a,r)

View loss (V in {C,H}; P^V_e = patients of cohort e in view V):
  L^V_e(a)  = (1/|P^V_e|) sum_{j in P^V_e} lbar(a,j)

Optima:      a*C_e = argmin_{a in A_e} L^C_e(a)   (acquisition-view best)
             a*H_e = argmin_{a in A_e} L^H_e(a)   (deployment best)
Worst held:  a#_e  = argmax_{a in A_e} L^H_e(a)
Span:        D_e   = L^H_e(a#_e) - L^H_e(a*H_e)   ( > 0 ; standardizer )
Cand. spread s_e   = SD_{a in A_e} L^H_e(a)

Policy pick: â_{pi,B,e} = model chosen by policy pi after querying B held
             labels. CONVENTION: at B=0 (no held label) the policy deploys the
             acquisition-view pick a*C_e.

Transport gap (FULL acquisition-view ceiling):
  T_e        = L^H_e(a*C_e) - L^H_e(a*H_e)          ( = R at B=0 )
Finite-budget held regret:
  R_{e,pi,B} = L^H_e(â_{pi,B,e}) - L^H_e(a*H_e)      ( >= 0 )
Active gain (vs uniform P0, matched budget):
  G_{e,pi,B} = R_{e,P0,B} - R_{e,pi,B}
Standardized forms (comparable across cohorts & metrics):
  R~ = R/D_e in [0,1];  T~ = T_e/D_e in [0,1];  G~ = G/D_e in [-1,1]

DEGENERACY GUARD (pre-registered, decided at analysis, outcome of the
selection problem not of any single model choice): if D_e is not bounded away
from 0 relative to the patient-cluster bootstrap SE of L^H (i.e. all candidates
are within held-loss noise), cohort e's selection task is VACUOUS and is
reported as "no selection signal" — neither a gate pass nor a gate fail.
```

The active question is exactly: does querying **held-view** labels close the regret from the ceiling `T_e` (B=0) down toward 0 **faster** for a label-adaptive method than for uniform, **robustly in every untouched cohort**? This makes `T_e` the ladder's zero-budget anchor and reproduces the C86D object (large `T_e` = FULL nontransportability) while adding the finite-budget-adaptive axis.

### 4.2 Endpoint set

```text
SPEC 4.B — ENDPOINT REGISTRY (frozen; PRIMARY = E1 & E5a)
------------------------------------------------------------
E1  Cohort-level standardized selection regret
      R~_{e,pi,B} = R_{e,pi,B} / D_e        (curve over the budget ladder)
      PRIMARY descriptive object; drives transport & convergence pictures.

E2  Patient-level held loss of the pick
      L^H_e(â_{pi,B,e})                      (raw deployment NLL of chosen model)

E3  Patient-tail CVaR of the pick
      CVaR_{0.10}[ lbar(â_{pi,B,e}, j) : j in P^H_e ]
      = mean patient loss over the worst-decile held patients (safety tail).

E4  Probability of ε-near-optimal selection
      P_near(e,pi,B) = Pr_{seeds,splits}[ L^H_e(â) - L^H_e(a*H_e) <= eps_e ]
      eps_e = 0.25 * s_e   (near-optimal = within a quarter of the candidate
      held-loss SD of the deployment optimum; scale-free, metric-agnostic).

E5  Active-minus-passive label efficiency
   (a) G~_{e,pi,B} = G_{e,pi,B}/D_e          (regret reduction at matched B) PRIMARY
   (b) Budget-equivalence factor
      rho_{e,pi,B} = min{ B'/B : R_{e,P0,B'} <= R_{e,pi,B} }
      (how many x more UNIFORM labels to match pi at budget B; the
       LURE/active-testing horizontal-efficiency reading).

E6  FULL acquisition-view ceiling (transport diagnostic)
      T_e and T~_e = T_e/D_e  ( = R~ at B=0 )
      Reproduces C86D: how wrong deployment is when the acquisition-view-best
      model is fielded with ZERO held labels.
```

### 4.3 Budget ladder (pre-registered from label-cost realism)

```text
SPEC 4.C — BUDGET LADDER (frozen; justified by cost realism, not by any result)
------------------------------------------------------------
B in { 0, 8, 16, 32, 64, 128, 256, 512 }   held-view record-labels per cohort.
  B=0    : ceiling anchor (deploy a*C_e; R = T_e).
  8–32   : opportunistic / single reading session.
  64–128 : modest funded pilot.
  256–512: funded annotation batch.
Geometric doubling. Cost basis: a board-certified reader labels the full scored
set of one 12-lead ECG in ~1–3 min, so 512 labels ~ 1–2 reader-days — a
realistic small-budget campaign. Every budget is << cohort size:
  512 <= ~5.0% of Chapman-Shaoxing (10,247 rec) and Georgia (10,344 rec),
  512 <= ~1.5% of Ningbo (34,905 rec)  -> genuinely finite-label.
No budget, and no ladder endpoint, is set from any model outcome.
```

### 4.4 Pre-registered thresholds and inference config

```text
SPEC 4.D — THRESHOLDS & INFERENCE (frozen)
------------------------------------------------------------
tau_G   = 0.10   PRIMARY active-control threshold, in standardized-gain units:
                 active must close >= 10% of the full candidate held-loss span
                 beyond uniform. Rationale (design, not result): a reduction
                 below 10% of span does not change which model a clinician would
                 field. Gate = LCB_95(G~_{e,pi,B}) >= tau_G.
eps_e   = 0.25 * s_e                    near-optimal margin (E4).
tau_near= 0.80   secondary: LCB_95(P_near) >= 0.80 at the gate budget.
alpha_CVaR=0.10  tail level (E3); tail-no-harm secondary:
                 LCB_95( CVaR^{P0} - CVaR^{pi} ) >= 0  (active must not worsen
                 worst-decile patient loss vs uniform).
tau_T   = 0.05   transport classification (E6):
                 TRANSPORTS      iff UCB_95(T~_e) <= tau_T
                 NON-TRANSPORT   iff LCB_95(T~_e) >  tau_T
                 AMBIGUOUS       otherwise.
alpha   = 0.05 one-sided (gains); two-sided where a CI is reported.
Bootstrap: B_boot = 10,000 patient-CLUSTER resamples; BCa intervals.
Runs     : K = 10 policy seeds x M = 5 acquisition/held splits (frozen);
           endpoint = mean over the K*M runs; the split is an added resample
           layer inside the cluster bootstrap.
```

### 4.5 Patient-clustered inference (per cohort; never pooled)

Inference is run **separately in each cohort**. There is **no pooled cross-dataset p-value** and no meta-analytic borrowing of strength; the only cross-cohort combination is the all-must-pass conjunction of §4.6.

For each cohort `e` and endpoint, the confidence interval is a **BCa cluster bootstrap over held-view patients** (the cluster is the patient): each resample draws patients with replacement, recomputes `L^H` (and `a*H_e`, `a#_e`, `D_e`, `s_e`) on the resample, re-evaluates the `K×M` policy runs, and recomputes the endpoint; the lower/upper confidence bounds (`LCB_95`/`UCB_95`) used by the gates are read from this per-cohort bootstrap distribution.

Because the statistical unit is the patient but the challenge distribution does not ship patient IDs for every cohort, the **cluster unit is fixed by a NO-OUTCOME metadata audit**, not asserted:

```text
SPEC 4.E — PATIENT-GROUPING NO-OUTCOME AUDIT (frozen rule)
------------------------------------------------------------
Scope: resource / signal-integrity + metadata audit. Reads ONLY .hea header
fields, native source metadata, and waveform first-moment summaries. Reads NO
queried label and NO model output. Run per cohort BEFORE any model result.

G-a  Documented patient-ID linkage recoverable (native metadata maps
     records->patients): adopt it. All of a patient's records in one view;
     cluster unit = patient.
G-b  Source documented one-record-per-patient (e.g. Chapman-Shaoxing per its
     data descriptor): record == patient; record-level clustering IS
     patient-level. (Chapman-Shaoxing 10,247 rec = one ECG/patient.)
G-c  Linkage NEITHER in the challenge distribution NOR recoverable from native
     metadata, AND source not documented-singleton (the Ningbo case as
     distributed: 34,905 rec, multiple records/patient, NO patient ID in the
     .hea headers):
     run a metadata-only near-duplicate audit (cluster records by exact/near
     header demographics + low-dim waveform summary stats; NO labels, NO model
     outputs).
       (i)  No material within-cohort multi-record grouping above chance:
            treat records as patient-proxy clusters and PROCEED, carrying a
            documented CONSERVATISM CAVEAT (record-clustered CIs may be
            anticonservative under latent within-patient correlation).  [DEFAULT]
       (ii) Material unresolved grouping found: cohort is REPORTED-BUT-EXCLUDED
            from the formal gate. The all-cohorts gate is evaluated over the
            INCLUDED untouched cohorts; a cohort dropped for a
            metadata/integrity reason (decided with zero model information) does
            not bias the gate.
This rule fixes the clustering UNIT by audit, never by a value or an outcome.
```

### 4.6 The formal gate

```text
SPEC 4.F — FORMAL GATE (frozen)
------------------------------------------------------------
Registry: P0 uniform (reference) + >=1 prediction-driven method
(A1 expected-loss / A2H pairwise-disagreement, C86D lineage) + >=1
label-ADAPTIVE method (Hara variance-minimizing active model selection;
Active-Testing/LURE; Model-Selector; CODA) — pinned by §Method Registry.

Per (pi in registry\{P0}, B in ladder\{0}):
  Per cohort e (included set): one-sided patient-cluster bootstrap test
     H0: G~_{e,pi,B} <= tau_G   vs   H1: G~_{e,pi,B} > tau_G
     -> p_{e,pi,B}  (equivalently: PASS_e iff LCB_95(G~_{e,pi,B}) >= tau_G).

  CONJUNCTION over cohorts (Intersection-Union Test = "same method + same
  budget must clear the threshold in EVERY untouched cohort"):
     (pi,B) passes iff PASS_e for ALL included cohorts.
     Conjunction p-value: p^conj_{pi,B} = max_e p_{e,pi,B}.
     IUT is valid at level alpha for the conjunction with NO correction and is
     conservative; it is exactly the all-cohorts requirement (no pooling).

  FAMILY over the grid: Holm-Bonferroni across the |registry\{P0}| x
  |ladder\{0}| conjunction p-values {p^conj_{pi,B}} to control FWER at
  alpha = 0.05.

PROGRAM VERDICT:
  "ACTIVE CONTROL DEMONSTRATED"  iff >= 1 (pi,B) survives Holm AND, for that
     (pi,B), the secondaries hold in EVERY included cohort:
        E4 LCB_95(P_near) >= tau_near   AND
        E3 LCB_95(CVaR^{P0}-CVaR^{pi}) >= 0 (tail-no-harm).
  "ACTIVE CONTROL WITH CAVEAT"   iff primary Holm passes but a secondary fails.
  "NO ACTIVE ADVANTAGE"          iff no (pi,B) survives Holm.

Transport gate (diagnostic; feeds §4.7): per cohort classify FULL via E6/T~_e
using tau_T {TRANSPORTS | NON-TRANSPORT | AMBIGUOUS}.
```

### 4.7 Four-way failure decomposition (mapped to endpoints)

Axis A = FULL transport class (E6 / `T~_e`, `tau_T`). Axis B = active-control gate (E5a / `G~`, `tau_G`), with a homogeneity read on the sign/pass-agreement of `G~` across cohorts.

```text
TABLE 4.G — 4-WAY DECOMPOSITION (frozen mapping to endpoints & verdicts)
------------------------------------------------------------------------------
Cell 1  FULL TRANSPORTS + ACTIVE WINS
  Trigger : UCB(T~_e)<=tau_T all cohorts  AND  Holm-surviving (pi,B) passes.
  Reading : Acquisition view already informative; label-adaptive querying still
            closes residual finite-budget regret. POSITIVE actionability.
  Endpoints: E6 low; E1 low even at small B; E5a LCB(G~)>=tau_G all cohorts;
             E4 high; E3 tail-no-harm.

Cell 2  FULL TRANSPORTS + ACTIVE FAILS
  Trigger : UCB(T~_e)<=tau_T all cohorts  AND  no (pi,B) survives Holm
            (UCB(G~)<tau_G in >=1 cohort).
  Reading : Transport holds; passive uniform already near-saturates, adaptivity
            superfluous. Extreme-action control achievable PASSIVELY.
  Endpoints: E6 low; E1(P0) already low at small B; E5a G~ ~ 0.

Cell 3  FULL DOES NOT TRANSPORT
  Trigger : LCB(T~_e)>tau_T in >=1 cohort (C86D reproduces on ECG).
  Reading : Acquisition-view labels insufficient. Sub-question via E5a:
              3a  label-adaptive HELD querying still closes regret
                  (Holm passes)              -> active RESCUES nontransport
                                                (novel positive beyond C86D);
              3b  it does not (no Holm pass)  -> Measurement->Control gap
                                                PERSISTS across modality AND
                                                under adaptivity (strong
                                                negative reproduction).
  Endpoints: E6 high; E1 curve over B; E5a decides 3a vs 3b; E3/E4 report cost.

Cell 4  HETEROGENEOUS ACROSS COHORTS  (Conditional Nontransportability)
  Trigger : transport class OR sign(G~) differs across cohorts, OR the
            conjunction fails only because ONE cohort fails.
  Reading : Cohort-conditional actionability — the all-cohorts gate cannot
            pass; this IS the conditional-nontransportability object.
  Endpoints: report per-cohort E1/E5/E6, name the discordant cohort(s).
            NO pooled p-value; heterogeneity is descriptive and drives the cell.
```

### 4.8 Open NO-OUTCOME decisions carried by this section

1. **Patient-grouping unit per cohort** → resolved by SPEC 4.E metadata audit (G-a/G-b/G-c); Ningbo is the live G-c case.
2. **Cohort inclusion in the all-cohorts gate** → INCLUDED = untouched cohorts passing the SPEC 4.E audit; decided metadata-only, zero model information.
3. **Standardizers `D_e`, `s_e` and ceiling `T_e`** → formulas frozen (SPEC 4.A); numeric values computed at analysis from candidate held losses that do not yet exist.
4. **Scored-label set `S`** → §Data metadata rule (intersection of Challenge-2021 scored classes across included cohorts, from `dx_mapping_scored.csv`).
5. **Selection pool `A_e` (per-context 81 vs pooled 648)** → §Zoo; endpoint/gate algebra is invariant to the choice.
6. **Degenerate-span handling** → SPEC 4.A guard: a cohort with `D_e` inside held-loss noise is "no selection signal," neither pass nor fail.