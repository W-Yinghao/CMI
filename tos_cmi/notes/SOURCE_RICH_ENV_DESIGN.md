# Source-rich environment discovery (Fork 2) --- DESIGN LOCK

Branch `science-source-rich-v1` (off method-deepen-v1). A FINITE, decidable test of the LAST constructive
possibility for the strict source-only line, before we (if needed) move to the target-information frontier
(Fork 1, next paper). Design-lock only; NO experiments until PM go. Gate thresholds FROZEN (0.02/0.01);
target used ONLY in the post-hoc audit; environment selection uses SOURCE-ONLY scores (no target labels).

## 1. Scientific question
V2 proved: source-INVISIBLE deployment-shift benefit cannot be certified by a source-only gate (Prop 1). The
remaining question (Prop 2, constructive):
> Can a BETTER source-**environment** construction --- beyond leave-one-subject-out --- make the
> deployment-relevant shift VISIBLE inside the source, so that a source-LOEO benefit lower bound becomes a
> conservative proxy for target benefit (and the gate can safely ACCEPT), all WITHOUT target labels?
Goal is NOT to prove source-only always accepts; it is to test whether some benefit can be made source-visible.

## 2. source-visible vs source-invisible (definitions)
```
source-visible benefit   : the target-relevant shift IS represented among source environments;
                           source-LOEO benefit can certify target benefit up to eps_coverage (Prop 2).
source-invisible benefit : the target-relevant shift is ABSENT from source environments;
                           source-only gate must reject/abstain (Prop 1; V2 World A witness).
```
Fork 2 tests whether a chosen environment definition moves cells from source-invisible toward source-visible.

## 3. Environment definitions (pre-registered E0-E6; leave-one-ENVIRONMENT-out replaces leave-one-subject-out)
* **E0 subject** (baseline = current Track B): leave-one-subject-out.
* **E1 session / subject x session**: leave-one-session-out / leave-one-(subject,session)-out. ONLY if the dump
  carries session metadata (2a/2b do; Lee/Cho/HGD checked at runtime -> mark `unavailable`, never fabricated).
* **E2 covariance-geometry clusters**: cluster SOURCE trials/subjects by covariance/tangent features (KMeans,
  k in {4,8,16}); leave-one-cluster-out. Source-only.
* **E3 spectral clusters**: cluster by bandpower / spectral centroid (delta..gamma). NOTE: frozen Z is
  post-encoder, not raw spectra -> requires raw-signal access (EEG_DATALAKE_RAW) OR a Z-proxy; flagged
  data-dependent, may be deferred.
* **E4 margin / error-profile clusters**: cluster by SOURCE-trained-model margin / entropy / error pattern.
  Uses SOURCE labels (legal) but the clustering must be CROSS-FIT / fixed on a training split (no overfit).
* **E5 augmentation-defined environments**: construct source shifts (channel dropout, band-stop, covariance
  scaling, sensor-noise, temporal jitter). No target labels, but introduces the ASSUMPTION that augmentations
  approximate deployment shift. Data-dependent (raw or Z-level perturbation).
* **E6 cross-dataset-as-source** (OPTIONAL, not first round): source environments from multiple datasets,
  held-out target dataset scored only at final report. Complex; deferred.

## 4. Discovery / confirmation split (fixed BEFORE looking at target)
```
development : Lee2019_MI
confirmation: Cho2017
secondary   : Schirrmeister2017 (HGD)  # small subject count -> secondary
```
Environment definitions and hyperparameters are chosen on DEVELOPMENT only; the CONFIRMATION dataset is scored
once. No re-selection after seeing target on either.

## 5. Source-only environment-selection rule
An environment definition may be selected using ONLY source-only scores:
* source-LOEO benefit-estimate STABILITY (cluster-bootstrap width),
* source-only calibration of the benefit estimator,
* environment COVERAGE score (how much source-LOEO variation the environments span).
FORBIDDEN as selection inputs: actual target delta, target labels, target performance. (Target enters ONLY the
post-hoc audit.)

## 6. Metrics (per environment definition)
```
source-LOEO predicted ΔbAcc   (the gate's benefit signal under this environment)
actual target ΔbAcc            (audit only)
corr(source-LOEO, target)      (Pearson + Spearman, cluster-bootstrap CI)
sign-agreement rate
false accepts / true accepts / abstain rate
harm prevented
accepted target gain
```
Headline table: | Env | source-target corr | false accepts | true accepts | abstain | harm prevented | notes |.

## 7. Pass / fail criteria (three cases)
* **Case A (source-rich finds true accepts):** strong positive source-only result. MANDATORY audit: target
  leakage? random-k reproduces? source task drop? consistent across datasets? -> only then a source-only positive.
* **Case B (avoids all harms, still no accepts):** real EEG has no source-VISIBLE beneficial erasure; Fork 2
  supports the ceiling, no positive route -> move to Fork 1.
* **Case C (source-rich false-accepts):** environment discovery overfits; source-only positive route unreliable
  -> move to Fork 1, keep source-rich as a cautionary appendix.

## 8. p-hacking controls (mandatory)
1. **Discovery/confirmation split** (S4) fixed a priori; no post-hoc dataset picking.
2. **Environment selection uses SOURCE-ONLY scores** (S5); never target delta/labels/performance.
3. **Report the FULL candidate set**: all E0-E6 (available) + random partitions matched by cluster size +
   an ORACLE target-informed environment selector labeled DIAGNOSTIC_ONLY (upper bound, never a method result).

## 9. Semi-synthetic source-rich SMOKE (Phase 2; the Prop-2 experimental witness)
Real EEG may have no positive target gain, so we FIRST build a semi-synthetic **source-rich World A** to test
whether the gate CAN accept when the shift IS source-visible (contrast with V2's source-invisible World A):
```
V2 World A          : target-beneficial shift NOT represented in source -> gate abstains.
source-rich World A : source environments include BOTH aligned and reversed/noisy shortcut regimes;
                      target regime is REPRESENTED among source environments -> source-LOEO sees benefit -> ACCEPT.
```
Construction: inject nuisance on real Z as in V2, but make source environments explicitly span multiple shortcut
regimes (some aligned, some reversed/noisy); target = a regime present in source. Estimate source-LOEO benefit
under each environment definition. **Pass:** (1) source-rich gate ACCEPTs some interventions; (2) accepted cells
have actual target ΔbAcc LCB > +0.01; (3) false accept ~ 0; (4) subject-baseline (E0) still abstains/rejects or
is weaker; (5) domain-gain-only false-accepts useless/harmful cells. **Forbidden:** tune thresholds; select
environment by target; report only the best cluster k; must report the full grid.

## 10. Real EEG source-rich VALIDATION (Phase 3; after the smoke passes)
```
datasets   : Lee2019_MI (dev), Cho2017 (confirm) [HGD secondary]
backbones  : EEGNet, TSMNet
interventions: identity, LEACE, RLACE, TOS_VD, tp_leace, alpha_leace, random_k
environments : E0-E5 (E3/E5 as data allows; E6 deferred)
```
Report the headline table (S6) per environment; classify by Case A/B/C (S7). A/B/C decides: source-only route
remains alive (Case A) vs close source-only accept and move to Fork 1 (Case B/C).

## Phase order + what is NOT done now
Phase 0 (cheap mechanistic meta-analysis, no new training) -> Phase 1 (this design-lock) -> Phase 2 (semi-synth
smoke) -> Phase 3 (real EEG) -> decide. NOT now: target-information budget experiment, few-shot/active target
labels, Track E, paper rewrite, dataset expansion, World-A redesign. Fork 1 is postponed, not cancelled: Fork 2
is its pre-flight diagnostic (source-rich coverage vs target labels).
