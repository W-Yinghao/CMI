# S2P_22 - Phase B1-Core Results

**Status:** B1-Core complete and independently verified. **Project:** OPEN. **B2:** HOLD.

This is a technical experiment record, not a manuscript draft. It reports the variance forensic disposition and
the pre-registered subject/task/geometry core without adding metrics, changing ranks, or selecting results.

## Authority and execution

- Ten representation contracts: deterministic random, released CBraMod, and H={200,500,1000,2000} x two seeds.
- Immutable checkpoint authority: `20ffcf9c91814956a404ca186ad7ee80eb2007cf`.
- Frozen scientific replay source: `960104e77dd897c7e695924da2e3bff45968cb10`, source SHA256
  `6574301ca95efe98ee75748a3481a9f55dcd0747af758f4836ae2f39b8d4962f`.
- Successful exact replay/core job: `893169`.
- Successful verifier-only job: `893274`.
- Ten checkpoint hashes, ten closure checksum features, clip-group folds, and the target-label firewall passed.
- No pretraining, fine-tuning, H4000, CodeBrain, new downstream dataset, or B2 analysis was run.

## Variance forensic disposition

The exact replay reproduced the original failed checkpoint set: `released` and `H1000_s0`. In both objects,
the subject-fraction and residual-fraction source-subject bootstrap 95% interval widths exceeded the frozen 0.20
threshold. Fold-deviation and negative-residual gates passed.

| Object | Component | Bootstrap CI width | Threshold | Other gates |
|---|---|---:|---:|---|
| released | subject fraction | 0.26166 | 0.20 | pass |
| released | residual fraction | 0.26048 | 0.20 | pass |
| H1000_s0 | subject fraction | 0.22230 | 0.20 | pass |
| H1000_s0 | residual fraction | 0.22346 | 0.20 | pass |

The variance family is therefore `FAILED_STABILITY_NOT_INTERPRETABLE` for all checkpoints. No surviving
checkpoint is reported selectively, and no variance or interaction result enters primary inference, the mechanism
verdict, or a B2 decision.

## Primary family

The three tests use 5,000 biological-subject cluster bootstrap replicates and one Holm family.

| Test | Directional estimate | 95% CI | Holm p | Decision |
|---|---:|---:|---:|---|
| P1 random minus H200 subject NLL | 3.59028 | [3.37455, 3.78857] | 0.00060 | pass |
| P2 H200 minus pooled-higher target NLL | -0.01949 | [-0.05465, 0.01608] | 0.86323 | no pass |
| P3 pooled-higher minus H200 rank-8 overlap | -0.01678 | [-0.02464, -0.00446] | 0.01200 | pass, decrease |

The primary subject NLL did not trigger its saturation rule, so no fallback metric was needed.

## Descriptive endpoints

Budget rows average the two training seeds. These values do not define a monotonic scaling claim.

| Budget | Subject NLL | Subject accuracy (diagnostic) | Target NLL | Target Kappa | Target bAcc | Rank-8 overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 200 h | 0.52905 | 0.86124 | 2.25791 | 0.04576 | 0.15170 | 0.02570 |
| 500 h | 0.11167 | 0.98482 | 2.26309 | 0.07263 | 0.17492 | 0.00939 |
| 1000 h | 0.09069 | 0.98936 | 2.30057 | 0.06821 | 0.17116 | 0.00670 |
| 2000 h | 0.11161 | 0.98534 | 2.26853 | 0.06733 | 0.17016 | 0.01067 |

Continuous subject separation continues after H200: H200 minus pooled-higher subject NLL is `0.42439`, with
95% CI `[0.35946, 0.49129]`. Thus raw subject accuracy approaches saturation while subject confidence continues
to strengthen.

P2 does not confirm later task structure under the frozen primary target-NLL estimand. Kappa and balanced-accuracy
sensitivities are positive (`+0.02363` and `+0.02037`, respectively, with subject-cluster intervals excluding
zero), but they cannot replace the failed primary task test.

P3 indicates lower measured linear overlap at higher budgets. The sign remains negative when any one high budget
is omitted and when either training seed is retained. All held-out subspace-capture gates pass. This supports only
a statement about the measured rank-8 linear effect subspaces; it is not evidence for subject invariance,
functional reliance, or a subject-by-class interaction.

## Core verdict

```text
mechanism_verdict: D
mechanism_label: CORE_GEOMETRY_UNRESOLVED
operational reason: P2 primary target NLL did not establish later task structure
geometry status: stable, with a reproducible decrease in measured overlap
variance status: failed stability and excluded
interaction claim: forbidden
```

The label `D` does not mean every geometry measurement failed. Geometry was stable and P3 was independently
reproduced. It means the complete A/B/C mechanism chain cannot be certified because its pre-registered primary
task premise, P2, did not pass.

## Independent verification

The verifier independently reproduced checkpoint hashes, closure features, clip grouping, subject/task sample
aggregates, all 5,000 bootstrap replicates, P1-P3 signs and intervals, Holm decisions, and geometry gates.

Independent float32-derived SVD recomputation differed from the generating pass by at most `5.04e-9` in overlap
and `5.73e-9` in subject capture. A geometry-only absolute tolerance of `1e-7` was fixed before the final verifier
run. P3 sign, both CI endpoints' side relative to zero, Holm decision, and all 0.05 capture gates matched exactly.

## Claim boundary

Allowed:

- subject-identifiable structure is acquired by H200 and continuous subject confidence strengthens thereafter;
- pooled higher budgets do not improve the pre-registered target-NLL primary endpoint relative to H200;
- rank-8 subject-task overlap decreases at pooled higher budgets under the measured linear geometry;
- variance partition is unstable and non-interpretable in this B1 analysis.

Not allowed:

- task structure is confirmatorily established to emerge after H200 by B1-Core;
- subject-by-class interaction increases or decreases;
- reduced overlap proves subject invariance or lack of functional subject reliance;
- monotonic scaling, an optimal budget, or a variance result from the surviving checkpoints;
- automatic Phase B2 launch.
