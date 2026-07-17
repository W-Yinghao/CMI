# C86R Final Report Red Team

## Verdict Matrix

| Check | Verdict | Evidence |
|---|---|---|
| Historical C86P stack preserved | PASS | six historical/additive identities replayed; no in-place rewrite |
| One effective downstream manifest | PASS | V2 manifest required; stale parent rejected by code and test |
| Yang adult criterion | PASS_FAIL_CLOSED | public minimum age 17; excluded from adult primary set |
| Kumar adult criterion | PASS_FAIL_CLOSED | no valid public minimum-age proof; mean age not imputed |
| Brandl adult criterion | PASS | explicit public 22-30 range |
| At least two adult cohorts | BLOCKER | one eligible primary cohort remains |
| Hara general-K fidelity | PASS | exact sum-over-unordered-pairs replay; distinct from max-pair |
| A2M attribution | PASS | renamed project heuristic; development-only; not labelled Hara |
| ASE/MODEL SELECTOR/CODA disposition | PASS | explicit availability or incompatibility for every method |
| Linear/plugin claim boundary | PASS | no nonlinear composite unbiasedness claim |
| C86L development/outcome separation | PASS | construction acquisition pool; C85U held outcome; zero overlap |
| Unsupported budgets | PASS | Physionet B32 unavailable; no substitution/replacement/deletion |
| Common prospective interface | PASS_METADATA_ONLY | 11-channel common interface; new training required |
| Target-cluster/inference arithmetic | PASS_METADATA_ONLY | 16/18/51 targets; exact 1/65537 plus-one resolution |
| License/resource scope | PASS_WITH_FUTURE_GUARD | planning only; terms/canary required before execution |
| Synthetic V2 chronology | PASS | 11 scenarios bound; zero registered draws |
| Protected access boundary | PASS | zero EEG, label, active, training, forward, GPU accesses |
| Downstream authorization boundary | PASS | no C86 lock or authorization created |
| Regressions | PASS | focused, C65, C23, full; accepted stderr empty |

## Adversarial Findings

1. Treating Yang's reported cohort range as adult-only would violate the locked
   rule because the public minimum is 17. The registry fails closed.
2. Inferring Kumar's minimum age from a mean and standard deviation would be an
   unsupported imputation. The public participant age field is not valid
   evidence, so Kumar also fails closed.
3. Retaining the historical max-pair score under the Hara label would be a
   fidelity error. The exact sum and heuristic are now separate methods.
4. Calling the historical composite a per-trial full loss vector would be
   mathematically false. Only linear moments are per-trial additive; the final
   utility and action remain plugins.
5. Reusing C84 evaluation labels as active development queries would contaminate
   the C85U-held outcome. The V2 development contract forbids that path.
6. A common montage alone does not clear confirmation: it requires new model
   training, license replay, a canary, and at least two eligible adult cohorts.

## Final Disposition

The repair is internally coherent, but its locked multi-cohort eligibility
premise is not satisfied. Reporting the nominal success gate would be false.

```text
C86_UNTOUCHED_COHORT_AGE_ACCESS_OR_INTERFACE_ELIGIBILITY_RECONCILIATION_REQUIRED
```
