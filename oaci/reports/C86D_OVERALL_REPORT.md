# C86D — Overall Report (executed under 授权 C86D)

**Gate reached**

```text
C86D_DEVELOPMENT_ACTIVE_POLICY_FIELD_FROZEN_C86H_REVIEW_REQUIRED
```

Development only. P0/A1/A2H were run on the accepted C86L field through the
process-isolated sealed server and evaluated against the real C85U held utility.
**No confirmatory claim**; trials/queries/replicates are not scientific N. C86H /
C87 / manuscript remain NOT authorized; C86H does not auto-start C87.

## 1. Execution

```text
authorization : 授权 C86D (direct)
runner        : oaci/active_testing/c86d/run.py (via c86d.execute)
SLURM job     : 902203 (cpu-high), run 300.6 s
methods       : P0 uniform / A1 LURE / A2H Hara       budgets: 4 / 8 / 16 / 32 / FULL
targets       : 118        replicates/cell: 8         cohorts: dataset (Cho2017, Lee2019_MI, PhysionetMI)
finalize      : atomic (.staging → os.replace); output C86D_RESULT_MANIFEST.json (sha 00696d54…)
```

## 2. Verified real inputs (fail-closed)

```text
C86L field   : /projects/…/oaci-c86l-development-field-v1 (content-addressed, accepted)
C85U utility : …/stage_u1_candidate_utility_v2/candidate_utility_index.csv
               sha256 83bddf56…; 76,464 rows / 944 contexts;
               candidate_index == C86L canonical order in ALL 944 contexts (0 misaligned)
C85U identity: acceptance manifest opened + hashed (dfcf8456…, 944/76,464) — verified, not hardcoded
```

## 3. Result — NO registered active gain (the development finding)

Endpoint: **target regret** = equal-weight mean over a target's 8 contexts of
(max composite utility − selected composite utility); pooled over targets per cohort.

```text
pooled mean regret (P0 / A1 / A2H):
  budget  4   : 0.3621 / 0.3542 / 0.3605
  budget  8   : 0.3338 / 0.3322 / 0.3379
  budget 16   : 0.3115 / 0.3171 / 0.3149
  budget 32   : 0.2972 / 0.2987 / 0.2978
  budget FULL : 0.2867 / 0.2867 / 0.2867   (identical — acquisition-order-invariant at FULL: a correctness check)

max active-minus-P0 gain over ALL (method × budget × cohort) cells = 0.0143  (< TAU 0.02)
target_near_opt_prob (regret ≤ 0.05) = 0.000 everywhere
```

**C86H method freeze (deterministic, pre-registered rule):**

```text
method = P0 ;  reason = no_registered_active_gain
```

No budget shows an active method (A1/A2H) beating P0 by ≥ TAU in any cohort; in most
cells A1/A2H are marginally worse. The registry stays {P0, A1, A2H}; the frozen
selection is P0.

## 4. Honest reading (development, not confirmation)

* Within this development, **adaptive acquisition (A1 LURE, A2H Hara) does not beat
  passive uniform** at any budget or cohort — the active-testing regime provides no
  advantage for identifying the held-evaluation-best candidate here.
* Even the FULL-budget ceiling regret is ~0.29 (near-opt 0): the construction-view
  composite selection has **limited held-evaluation actionability** — construction-
  estimated candidate rankings do not closely match the held-evaluation ranking.
* This is **not** a transport/confirmatory claim and does not touch untouched
  Brandl/ds007221. The near-optimal threshold (ε = 0.05) is tight relative to the
  composite-utility regret scale, but the load-bearing finding — active ≈ passive,
  max gain 0.0143 < 0.02 — is scale-independent.
* The real scientific test remains a future **untouched C86H** (not authorized).

## 5. Boundary

`c86d.execute` refuses without `授权 C86D`. Development only; no confirmatory claim.
C86H / C87 / manuscript NOT authorized; C86H does not auto-start C87. Output field
identity: `/projects/…/oaci-c86d-development-field-v1/C86D_RESULT_MANIFEST.json`.
