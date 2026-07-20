# C87 synthetic control gate — CONTROL_PASS

signature `b31cd7314b1a4c59`  ·  elapsed 4613.6s

```text
[C87 CONTROL GATE] config={"A": 648, "n_pat": 400, "pos_n_pat": 1000, "E": 3, "K_seeds": 10, "B_boot": 2000, "K_MC": 1000, "fast_reps": 100, "budgets": [16, 32, 64], "power_budget": 32, "alpha": 0.05, "coverage_band": [0.935, 0.965], "tau_G": 0.25}
  note: control-scoped subset of C87E (A_real=648 matched). COVERAGE controls (CALIB/POS_DENSE/M2) use N_p=400 as a HARDER-than-real coverage stress (real held ~5000). POWER/specificity controls (POS/NEG_B) use N_p=pos_n_pat=1000 (still << real ~5000) because power is a property at deployment n; small n unfairly understates it. B_boot main=2000; K_MC=1000; fast-loop bootstrap=100 reps; power = P(MODEL-SELECTOR LCB[G]>0 in ALL cohorts at B=power_budget).
  POS: transport=True gain=('MODEL-SELECTOR', 16) power=0.806
  POS_DENSE: insample_min=0.2656 xfit_ref=0.2686 optimism_gap=+0.0030 (corrected=True); naiveT=+0.0001 xfitT=-0.0028 (naive_inflates=True)
  NEG_A: no-false-transport=True
  NEG_B: real-gate FP rate=0.0000 (<= 0.05: True); LCB>0 rate=0.0010; null-vacuous frac=1.00; mean null G/s_e (diag)=+0.185
  CALIB: gate-statistic (G-like) cluster-bootstrap coverage=0.960 band=[0.935, 0.965] ok=True (cross-fit T finite-mean diagnostic=-0.0039)
  MUTATIONS: M1=True M2=True M3=True
[C87 CONTROL GATE] VERDICT = CONTROL_PASS  (4613.6s)  sig=b31cd7314b1a4c59
```

## Per-control verdicts

- **POS**: PASS
- **POS_DENSE**: PASS
- **NEG_A**: PASS
- **NEG_B**: PASS
- **CALIB**: PASS
- **MUTATIONS**: PASS