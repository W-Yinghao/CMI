# C86D — Corrected Development Result (last execution, under fresh 授权 C86D)

**Gate:** `C86D_DEVELOPMENT_ACTIVE_POLICY_FIELD_FROZEN_C86H_REVIEW_REQUIRED`

The corrected D1→D2 pipeline (all four reconciliation rounds) executed under a fresh
direct `授权 C86D` as the LAST development execution. SLURM job 902594. Development
only; no confirmatory claim.

## Preflight / integrity (all passed, fail-closed)

```text
D1: 2,832 / 2,832 freezes (3 methods × 118 targets × 8 chains); path_blind_worker=True; c85u_accessed=False
    (D1 manifest sha 447a4488725d41bf7bca5a26)
freeze verification (before C85U): registry==accepted C86L / methods=={P0,A1,A2H} / chains=={0..7}
    / Cartesian 2832 / seed==target-bound / lengths / nested-prefix / composite==first-argmax
FULL acquisition invariance (BLOCKING): within-group True, across-chains True
C86L acceptance replay: acceptance_ok + gate + inventory 1891
    (D2 manifest sha cc0d81fa03552dd529a9335e)
```

## Development disposition

```text
label   = ACQUISITION_VIEW_NONTRANSPORTABLE  (FULL ceiling fails per cohort)
budget  = FULL
```

Even with all construction labels, the construction-view composite selection does not
reach the held-evaluation optimum: **near-optimal probability = 0.000 in every cohort**
(FULL best mean/tail/near-opt — Cho2017 0.209/0.323/0.000; Lee2019_MI 0.286/0.475/0.000;
PhysionetMI 0.378/0.535/0.000). Primary risk = held standardized regret; near-opt uses
the raw-gap ε geometry (indicator-first).

## Active vs passive (no registered active gain)

```text
max active−P0 cohort mean gain over cross-cohort-eligible budgets {4,8,16,FULL} = 0.006  (< TAU 0.02)
budget 4 (all warm-start): P0=A1=A2H identical -> paired chain-level MC SE = 0.0 (paired-CRN signature)
B32 INPUT_UNAVAILABLE: PhysionetMI (76 cells) -> budget 32 excluded from cross-cohort gate
C86H method registry: [P0, A1, A2H]  (unchanged; not pruned by development performance)
```

## Honest reading (development, not confirmation)

Corrected result: with the locked estimator/methods/isolation, adaptive acquisition
(A1 mixture-CE, A2H sum-over-pairs) gives no advantage over passive uniform, and the
construction acquisition view does not transport to held-evaluation actionability at any
budget — a stronger, more precise negative than the rejected attempt. This is
development only, not a confirmatory/transport claim, and did not touch untouched
Brandl/ds007221. The real scientific test remains a future untouched C86H (NOT
authorized). Outputs: /projects/…/oaci-c86d-dev-v2{,_d1}/.
