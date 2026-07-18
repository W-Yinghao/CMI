# C86H — Integrated Terminal Runner: readiness object (compact)

**Status**

```text
C86H_INTEGRATED_TERMINAL_RUNNER_IMPLEMENTED (prep) ; SYNTHETIC E2E + RESOURCE BENCHMARK PASS
C86H_REAL_EXECUTION_NOT_AUTHORIZED  (separate 授权 C86H required)
C87_NOT_AUTHORIZED ; MANUSCRIPT_WORK_NOT_AUTHORIZED
```

The five requested items are complete; no real Brandl2020/ds007221 EEG or label was touched.

## 1. Integrated runner — `oaci/active_testing/c86h/runner.py`

One gated entrypoint, F0→H4, physical barriers not extra authorization rounds:

```text
F0 preflight   : verify_bindings (V3 manifest + field/training manifest + registry tables +
                 frozen C86D dispatcher blobs incl. server.py + selection_worker.py) — outcome-free
F1 zoo train   : fresh 11-ch 648-model zoo (real: gated, authorized modules)   [not built in prep]
F2 predict+split: target-unlabeled predictions + locked label-blind split       [not built in prep]
H1 selection   : reuse frozen C86D SPAWN server + PATH-BLIND worker (2,048 chains)
H2 held-eval   : SEPARATE spawned process (no server/oracle capability); verifies EVERY freeze
                 BEFORE opening the held split; endpoints
H3 inference   : within-cohort max-T (C86_MAXT_V1 seed; Brandl exhaustive 2^16, ds007221 MC 65536;
                 plus-one adjusted p) + tail CVaR + LOTO; formal C86-A..E + L1..L4 + Level-2 descriptor
H4 result      : one immutable result manifest; stop (no auto-C87)
```

`execute()` and `run_confirmation()` BOTH refuse the real path without `授权 C86H`; a synthetic
run must present a synthetic field and may never target the real field root.

## 2. Synthetic production-equivalent end-to-end test

`runner.run_synthetic` drives F0→H4 over a synthetic field in the exact C86 on-disk format
(pool probs, sealed oracle labels.csv, sealed contribution store, sealed held split). Tests
(`oaci/tests/test_c86h_runner.py`): full e2e (two-level output, held-after-verify, Cartesian
completeness), freeze tamper guard, gated refusals, benchmark. NO real EEG/label.

## 3. Outcome-free resource benchmark (synthetic arrays; opened no real data)

```text
selection (frozen dispatcher inner loop) : 0.174 s / selection  (A2H O(NM^2) dominates)
full campaign 53 targets x 2,048 chains x 3 methods : ~15.7 CPU-core-hours (single-core);
   embarrassingly parallel per target -> ~18 min wall on 53 cores / ~1 h on 16 cores
selection-freeze storage                 : 325,632 freeze files, ~0.68 GiB (rough)
max-T per cohort                         : Brandl 1.15 s (exhaustive 2^16) ; ds007221 1.47 s (MC 65536)
peak RAM (per target)                    : ~0.04 GiB   (envelope 128 GiB)
DECISION                                 : FEASIBLE. If ever infeasible vs envelope ->
                                           STOP_BEFORE_DATA_ACCESS (never reduce the chain count).
```

## 4. Exact field/training manifest — `oaci/reports/C86H_FIELD_TRAINING_MANIFEST.json`

Content-addressed (sha256 pinned in `contract.py`). Binds: legacy sources Lee2019_MI /
Cho2017 / PhysionetMI; panels A/B (12 train + 4 held-out source-audit each); seeds {5,6};
levels {0,1}; 81 = 1 ERM + 40 OACI + 40 SRC; Adam wd0 frozen EngineConfig; 40-checkpoint
cadence range(4,200,5); canonical candidate index 0..80; same 648-model zoo x 2 cohorts =
1296 artifacts; **fresh 11-channel training (C84's 20-channel outcomes NOT retained)**;
integer canonical target ids (Brandl 1..16, ds007221 37..73); ds007221 deterministic
inclusion (sub-37..73, hybrid, left/right); resource envelope.

## 5. Three fixes (from the prior review) + adversarial red-team fixes

The three PM-requested fixes: (a) `verify_bindings` opens+hashes the V3 manifest (==c6b7e490…)
and replays schema/status/gate/cohort identities; (b) Level-2 descriptor computed from real
secondary objects (FULL ceiling + C86D CROSSED margins), decoupled from the gate, POLICY_LIMITED
fixed to NOT_IDENTIFIABLE; (c) max-T seed = `SHA256(C86_MAXT_V1|dataset|registered_family)`,
Brandl exhaustive / ds007221 MC, plus-one adjusted p.

An adversarial red-team (6 independent skeptics) then found and I fixed **2 blockers + 6 majors +
nits** — disclosed in full:

```text
BLOCKER  freeze verification was a dead 'or True' no-op with no completeness check
         -> expected-target registry + Cartesian completeness + duplicate rejection
BLOCKER  verify_bindings did not hash server.py / selection_worker.py (the H1 isolation code)
         -> both now content-addressed
MAJOR    run_confirmation (the path that opens real data) was ungated; token only on execute()
         -> token + synthetic gating enforced on run_confirmation
MAJOR    LOTO was folded into mean-qualification (would demote a true C86-B to C86-C)
         -> LOTO scoped to a separate C86-A-only stability check
MAJOR    FULL-ceiling near-opt used per-context-then-average, not C86D indicator-first geometry
         -> per-replicate mean-8-context gap then threshold (matches run_d2)
MAJOR    descriptor 'robust' equalled the gate's mean predicate (table-lookup) and lacked tail
         -> C86D CROSSED margins (mean+tail+near-opt, DESC_TAU=0.02/DESC_NEAROPT=0.05), decoupled
MAJOR    non-AVAILABLE budgets escaped validation; FULL not required AVAILABLE; vacuous FULL invariance
         -> status whitelist + FULL-AVAILABLE + INPUT_UNAVAILABLE validity + per-(tgt,ctx,chain)
            all-methods FULL invariance
MINOR    H2 shared a process with the H1 launcher (sealed-oracle reach)
         -> H2 runs in a separate spawned process with no server/oracle capability
NIT      max-T zero-variance asymmetry + family np.stack misalignment + residual schema checks
         -> shared _t_vec convention; common aligned target set; q_seq/lure/receipt checks
```

A second adversarial pass (verify-the-fix) confirmed those closed and found residual issues,
all now fixed and re-tested:

```text
MEDIUM   FULL invariance was within-(target,ctx,chain) only, not across chains
         -> FULL candidate now required single across all methods AND chains + pool-length consistency
MED-LOW  removing LOTO from mean made label_frontier ignore stability heterogeneity
         -> frontier now returns C86-L3 on stability heterogeneity; best label taken across methods
LOW-MED  target registry was caller-supplied, not bound to the registered population
         -> real path binds cohort set == registered COHORTS and target count == 53
LOW      descriptor 'any_material' was per-cohort, not the C86D pooled-mean TAU
         -> pooled cross-cohort mean >= DESC_TAU
LOW      component_sha context-set / realpath root guard -> both added
```

Residual known item (documented, not a correctness bug): the Brandl exhaustive branch retains the
plus-one estimator (floor 2/65537, conservative) rather than a pure #/2^n exact p — never
anti-conservative, far below alpha 0.05.

## 6. Pre-execution review (§13) and stop rule

Before any authorized data access, the five §13 checks apply (bindings; three-stage isolation;
two-level output; 65,536-draw max-T + LOTO + registered thresholds; resource feasibility — met).
C86H remains terminal: one field, one confirmation, one audit, stop; no auto-C87. The only valid
trigger for real execution is a separate direct `授权 C86H`.
