# C17 — Source-signal identifiability audit

> Diagnostic study: are the target-accuracy-good OACI checkpoints (C16) identifiable from SOURCE-ONLY checkpoint observables? Target labels are used POST HOC only (diagnostic_only_non_deployable); no deployable selector is produced.

- **CASE: `case_III_multivariate_weak_identifiability`** — no scalar source signal works, but source-only combinations weakly beat permutation -> competence information exists but is not captured by simple selectors
- next science: a low-freedom source-only competence probe MAY be pre-registered; not deployable yet

## Univariate identifiability (within-fold-level Spearman; permutation p)

- verdict `weak_accuracy_needs_multivariate`; strong accuracy signals 0, weak 3, NLL-identifying 5
- C10's oracle signal (source_audit_worst_bacc) within-fold ρ(target bAcc) = +0.120; best |ρ| any signal = +0.236; accuracy signal families = ['accuracy', 'objective', 'risk']

| signal | axis | ρ(tgt bAcc) | ρ(tgt NLL) | perm p | strong | weak |
|---|---|---:|---:|---:|:--:|:--:|
| source_guard_worst_bacc | accuracy | +0.024 | +0.045 | +0.571 | False | False |
| source_guard_worst_nll | calibration | -0.106 | +0.107 | +0.003 | False | False |
| source_guard_worst_ece | calibration | +0.108 | -0.219 | +0.003 | False | False |
| source_audit_worst_bacc | accuracy | +0.120 | -0.073 | +0.003 | False | False |
| source_audit_worst_nll | calibration | -0.071 | +0.222 | +0.010 | False | False |
| source_audit_worst_ece | calibration | -0.013 | +0.160 | +0.638 | False | False |
| selection_leakage_point | leakage | +0.075 | +0.003 | +0.010 | False | False |
| audit_leakage_point | leakage | +0.030 | -0.008 | +0.282 | False | False |
| R_src | risk | -0.236 | +0.084 | +0.003 | False | True |
| balanced_err | accuracy | -0.222 | +0.140 | +0.003 | False | True |
| train_surrogate | objective | +0.206 | -0.201 | +0.003 | False | True |
| epoch | meta | +0.038 | -0.154 | +0.193 | False | False |

## Multivariate competence probe (DIAGNOSTIC-ONLY, leave-one-target-out)

- LOTO AUC **+0.602** vs permutation mean +0.537 (p +0.008); LOSO AUC +0.542; base rate +0.556
- **beats permutation: True** · non_deployable = True

## Calibration-vs-accuracy axis decomposition

- calibration-axis→target-NLL visibility +0.177 vs accuracy-axis→target-bAcc visibility +0.122 → **source signals calibration-biased: True**

## Class-boundary rotation identifiability (selected checkpoints)

- source↔target per-class recall-delta correlation +0.547 → class-boundary source-identifiable: True (SELECTED checkpoints only (per-candidate class recall is not committed))

> no scalar source signal works, but source-only combinations weakly beat permutation -> competence information exists but is not captured by simple selectors