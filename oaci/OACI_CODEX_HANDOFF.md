# OACI (Direction 1) — Codex Handoff

**Purpose.** Everything a fresh agent (Codex) needs to continue the **OACI mechanism-audit line** with no loss of
context. Two things come FIRST, by the PM's instruction: **(§1) what C23–C31 actually established**, and **(§2) how
this PM works with you**. Then the repo/env/how-to-continue mechanics. Written 2026-07-08. Read §1–§2 before anything.

> **One-sentence orientation.** OACI is a **strict-DG (source-only, no target data) EEG mechanism study**. The method
> line is **CLOSED / NEGATIVE** (C8 stop, C14 falsified, C21 estimand boundary locked). What survives is a
> **read-only, DIAGNOSTIC-ONLY mechanism chain (C22→C31)** that dissects *why* source-only competence selection fails
> to transport. **It is not a deployable method, has no selector, and never uses an oracle as a feature.** Nothing here
> imports `cmi/` or `h2cmi/`.

---

## 0. Current continuation state (2026-07-13)

The authoritative milestone is **C81R2 repair readiness complete; C81E held-evaluation execution is not reauthorized**:

```text
C81 base protocol commit:         16a0d2eba4715a1cec78da6a79a182fd416a6629
C81 base protocol SHA-256:        cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
C81 source-schema repair:         6371b2220979b61cabfb105521036bb02f47aaea
C81R2 descriptor repair:          5062f5ade0f45d6fd34f80556fb77470c2c6d717
C81R2 repaired implementation:    225df1c2066b50abedec4bacf043f6359c715190
C81R2 execution lock:             f82ffa4b147c0b1329a98649b898691cf1fdc983
C81R2 lock SHA-256:               13414dde0a88eb8a1a0810b3b36f25c718669d4cfe3178b871239eff6e292705
C81R2 readiness result:           6118a13df264bc64e79fe6789fc215b1a5e96b55
C81R2 red team:                   52 / 52 PASS
C81R2 gate:                       C81_SELECTION_DESCRIPTOR_REPAIR_LOCKED_READY_FOR_PI_REAUTHORIZATION
held-evaluation statistics:       0
same-label oracle accesses:       0
```

C81P locked the 34-method same-field comparison, including fixed zero-label
representatives ATC, NuclearNorm, MaNo, COTT, SND, and ALine. Selection job
`894915` completed under direct PI authorization and froze all 32 contexts and
19 feasible selectors without evaluation-label, oracle, target-4, training,
forward, or GPU access. Its manifest self-hash is
`4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519`;
the external payload hash is
`1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257`.

The independent freeze audit then stopped before evaluation because the generic
C74 verifier assumed one common array length, while the registered payload has
32 context rows and 19 method identifiers. C81R2 fixes only that descriptor
verification with an exact per-array shape map, binds the existing payload, and
forbids selection recomputation. The shared C74 verifier and all scientific
objects remain unchanged.

Regression on lock `f82ffa4` is green: focused `47`; C65-C81R2 `416 + 1
conditional skip + 3 historical deselections`; C23-C81R2 `827 + 1 + 3`; full
OACI `1,751 + 1 + 3`. All stderr files are empty.

The direct authorization bound at `b2f9fca` was consumed by job `894915` and is
not reusable for the new lock. Under policy `3d9dd76`, a new direct PI statement
such as `授权 C81R2 修复后的 C81E 继续执行` is sufficient; no token or repeated
hash recital is required. Until then, held-evaluation scoring remains stopped.

The latest completed scientific milestone remains **C80E**:

```text
C80E authorization policy:      3d9dd76
C80E repaired preflight:        7937740
C80E selection repair protocol: c19ef34
C80E replacement lock:          0797599
C80E machine result freeze:     a43aa27
C80E scientific red team:       be3e5c7 (32/32 PASS)
C80E report/final red team:      212d864 (30/30 PASS)
C80E lifecycle regression:      ebc6afe
C80E regression/memory delivery: 42a9f72
C80E primary gate:              C80-A_stable_low_regret_label_budget_frontier_across_training_seeds
project-control gate:           C80E_COMPLETE_C81_PROTOCOL_REVIEW_REQUIRED
```

The PI's direct C80E authorization was accepted without a token or repeated
hashes and automatically bound to the unique operative repaired protocol and
lock. All five paths ran unconditionally on the frozen seed-3/seed-4 primary
fields. Target 4 remained excluded and the same-label oracle remained closed.

```text
B*_seed3 / B*_seed4:          1 / 1 label per class
ordinal grid distance:        0
cross-seed stability:         PASS under the registered rule
B=1 regret seed3 / seed4:     0.353383 / 0.373705
B=1 source-relative gain:     0.426093 / 0.423742
B=1 max-T p, each seed:       0.042802
B=1 top-1 seed3 / seed4:      0.037842 / 0.038391
leave-one-target B* changes:  16 / 16 analyses
```

The exact C80-A taxonomy is retained, but the scientific claim is narrower: a
stable **source-relative** regret-reduction frontier under the fixed Q0 policy
in these existing fields. The source baseline is weaker than random, absolute
B=1 regret remains substantial, top-1 recovery is about 3.8%, and the one-label
boundary is small-target sensitive. This is not universal one-label
sufficiency, deployment, independent confirmation, or external generality.

Final regression on exact clean commit `ebc6afe` is green: focused `54`;
C65-C80E `369 passed + 1 conditional skip + 3 historical deselections`;
C23-C80E `776 + 1 + 3`; full OACI `1,704 + 1 + 3`. Final stderr is empty.
Superseded wrong-worktree and obsolete-lifecycle attempts remain disclosed in
`reports/c80e_tables/regression_attempt_ledger.csv`.

### Superseded pre-execution state

The following C80R record is retained for chronology. Its instruction to stop
for C80E authorization is no longer current:

```text
C80E safe-stop evidence:       6c18fd4
C80R repair protocol:          e88a244
C80R protocol SHA-256:         2d72eb5119056a6520fd33fc0ac14ee6270bfd573b59c36b74be6aa3dc25fe39
C80R final adapter:            e5cb41a
C80R replacement lock:         f19acd8
C80R replacement lock SHA-256: e18f2b5f1d79b6fcd96207339c5842e30b7aecb5bc22b8939a475487068b1b82
C80R readiness result:         101146b
C80R readiness gate:           C80_REPAIRED_PROTOCOL_AND_REAL_ADAPTER_LOCKED_READY_FOR_PI_REAUTHORIZATION
real-data budget statistics:   0
evaluation-label value reads:  0
same-label oracle accesses:    0
new C80E authorization:        absent at this historical readiness point
```

The accepted preflight found three pre-outcome defects in the historical C80P
execution objects: missing C80-A--E precedence/near-FULL semantics, a nested
authorization-guard schema mismatch, and no real-data adapter bound by
`972f47c`. C80R preserves those historical objects and the blocked
authorization record, then repairs them additively. The machine-locked
precedence is C80-E blocker, C80-D absent B*, C80-B unstable paired frontier,
C80-C stable with both B* in `{32,FULL}`, then C80-A. `FULL` remains
cell-specific and is never rewritten as 61 labels/class.

The fail-closed adapter now binds construction-only nested-Q0 selection,
content-addressed selection freeze, evaluation-after-freeze, source-relative
top-k/regret, target-cluster simultaneous inference, all-budget paired
cross-seed heterogeneity, fixed descriptive S3 LOTO, all five unconditional
paths, and machine-result freezing. The first repaired lock revision
`9617760` is transparently superseded by `f19acd8` after a preauthorization
completeness red team; registry entries and scientific thresholds did not
change and outcome access remained zero.

At this historical point, the old C80E authorization was not reusable and C80E
remained stopped pending direct PI authorization of the repaired objects. That
authorization was subsequently received and is recorded in the authoritative
C80E section above.

C80R final regression is green on exact clean commit `93d2099`: focused
`53 passed`; C65-C80R `368 passed, 1 conditional skip, 3 deselected`;
C23-C80R `775 passed, 1 conditional skip, 3 deselected`; full OACI
`1,703 passed, 1 conditional skip, 3 deselected`. All failures and stderr byte
counts are zero. The skip is the finalized C78F guard, and the three
deselections are historical C79P preauthorization-state tests; no C80R path
was skipped or deselected.

The completed **C80P cross-seed label-budget frontier protocol/readiness
milestone** remains the accepted base:

```text
C80P protocol commit:        f5d83b3
C80P protocol SHA-256:       c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85
C80P synthetic implementation: c98e084
C80P analysis lock:          972f47c
C80P analysis-lock SHA-256:  05a99e7ccc357b90b6675756caa680fd16541358bb697fdded89351f1e7ae4a8
C80P readiness result:       1b02454
C80P final gate:             C80_LABEL_BUDGET_FRONTIER_PROTOCOL_LOCKED_READY_FOR_PI_AUTHORIZATION
scientific registry:         5 paths x 16 categories = 80 / 80
pre-execution red team:      36 / 36
real-data budget statistics: 0
C80E authorization at C80P readiness: false
```

C80P is explicitly designed after C79E and is prospective only to the new C80
budget computations. It is a retrospective existing-field design study, not
independent confirmation, new-subject replication, target-population
confirmation, or external validation. It performed no EEG load, training,
forward pass, re-inference, GPU work, real-data label-budget computation, or
same-label-oracle access.

The requested finite budget grid was `[1,2,4,8,16,32,64]` labels per class
plus `FULL`. The permitted availability-only audit found a minimum construction
class count of 61. Budget 64 was therefore removed before protocol hashing as
infeasible; no candidate score or evaluation outcome was computed. The locked
grid is `[1,2,4,8,16,32,FULL]`, using nested class-stratified uniform sampling
without replacement and 2,048 Monte Carlo chains.

The future C80E primary object is expected held-evaluation standardized regret
under the exact C79 P1 construction-label selector. The frontier uses an exact
target-level max-T family procedure, the locked 0.05 regret-reduction margin,
at least 6/8 positive targets, a no-catastrophic-target rule, and all-larger-
budget closure. Seed 3 and seed 4 remain paired training factors over shared
targets/trials. Reliability is secondary and is not an actionability
precondition. The exact top-gap/effective-multiplicity path is descriptive
moderation only and cannot rescue H2/H2R.

Synthetic-only calibration passed all nine registered scenarios, all 18 B*
seed-scenario cells, family-wise error `0.044922`, target-cluster coverage
`0.953125`, and the pseudoreplication trap. Regression is green:
`focused=29`, `C65-C80P=344 pass + 1 conditional skip`,
`C23-C80P=751 pass + 1 conditional skip`, and
`full_OACI=1679 pass + 1 conditional skip`.

The accepted scientific base remains the completed **C79E post-seed-3
prospective seed-4 replication**:

```text
C79P replacement protocol:    ec4834c
C79P protocol SHA-256:        e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587
C79P field / analysis locks:  35d0c65 / 7cebf2e
C79E authorization record:    b67ba6c
C79E full-field freeze:       50232df
C79E label provisioning:      6c3dc91
C79E primary-output freeze:   cfd57cc
C79E scientific red team:     439c8c5
C79E result report:           7dee4be
C79E audit namespace repair:  e48edda
C79E final validation:        a12dc8b
C79E final gate:              C79-E_seed4_does_not_replicate_either_core_pattern
scientific red team:          17 / 17
final-report red team:        27 / 27
```

C79E remains explicitly outcome-informed after C78S and prospective only to
seed-4 checkpoint outcomes. It tests training-seed robustness over the same
targets and raw trials; it is not pre-C78S confirmation, new-subject
replication, target-population confirmation, or external validation. The
historical timing-invalid C79 artifact remains preserved and superseded rather
than relabeled.

The complete seed-4 engineering field contains 1,458 units: 18 ERM anchors,
720 OACI checkpoints, and 720 SRC checkpoints. The primary field contains
1,296 units over targets `[1,2,3,5,6,7,8,9]`; target 4 contributes 162
engineering-only units and enters no primary estimand, null, family, count, or
success rule. Strict-source and target-unlabeled caches contain 6,718,464 and
839,808 rows. Construction/evaluation views contain 2,235/2,373 disjoint rows.
All state, optimizer, genealogy, cadence, and numerical instrumentation gates
pass; target-label training reads and outcome-driven retention/retry are zero;
the same-label oracle remains closed.

Seed-4 decisions are mixed at the component level but neither co-primary
compound object replicates:

```text
P1 reliability:             0.756456; raw p=0.011673; Holm p=0.070039 (inactive)
P1 construction top-1/5/10: 0.1250 / 0.5000 / 0.6875 (material actionability)
P1 construction regret:     0.110667; reduction 0.686781
P1 compound transition:     false

H2R deviance reduction:    -8.717406; p=0.862 (does not qualify)
P2 local association:       0.210137; 32/32 positive cells
P2 worst-control / Holm p:  0.092 / 0.368 (local gate inactive)
P2 LOTO / LORO R2:         -0.098497 / -0.032944 (both unqualified)
P2 compound local/nontransport: false

H4R strict-source F2 R2:   -0.096288 (does not qualify)
H5R target-unlabeled F4 R2: 0.010450 (does not qualify)
H6R effect:                 0.415635; raw p=0.011673; Holm p=0.070039 (inactive)
```

All registered aggregate effects retain their seed-3 direction, but P2-L and
P2-overall change gate status. The P2 local seed4-minus-seed3 difference is
`-0.032519` with paired-target 95% CI `[-0.061609, 0.004581]`: the interval
includes zero, while the fixed gate differs. Report this as gate-level
training-seed heterogeneity, not a significant effect-size difference. No
cross-seed p-values were combined.

The accepted C78S scientific base remains:

```text
C78S protocol SHA:          df85699090a65d1e1766d754bcebd9eb5648cc13e4441d8074a3f4884487c7f8
C78S implementation:        e561a15
C78S execution lock:        ce1fb14
C78S result commit:         43a046c
C78S handoff commit:        48be5b7
C78S provenance correction: dcd4c28
C78S lock SHA:              aee520820cb7b2b94ab43f4e2bf8a30278a36f479296e145c24eaada99df36ad
C78S analysis job:          893151 (cpu-high, 48 CPUs)
C78S final gate:            SEED3_MIXED_RESULTS_C79_PROTOCOL_REVIEW_REQUIRED
C78S active primary:        H3 + H4 + H5
```

C78S consumed the 1,296-unit primary seed-3 field over targets `[1,2,3,5,6,7,8,9]`, levels 0/1, and exact historical ERM/OACI/SRC paths. Target 4 was mechanically excluded from every primary estimand, null pool, and multiplicity family. Construction/evaluation IDs were disjoint and covered all 576 trials per target. The primary route contained no same-label-oracle descriptor; trial ID and row order were join/split/cluster keys only.

```text
split-label trajectory reliability:  0.770863
target-cluster 95% CI:               [0.699990, 0.841695]
construction top-1 / top-5 / top-10: 0.1250 / 0.6875 / 0.7500
random top-1 / top-5 / top-10:       0.0123 / 0.0617 / 0.1235
construction standardized regret:    0.0828
random expected standardized regret: 0.4820
strict-source F2 incremental R2:     -0.073086
target-unlabeled F4 incremental R2:   0.005176
target-unlabeled local association:   0.242656 (worst max-stat p=0.002)
target-unlabeled nonlinear LOTO R2:  -0.212875
target-unlabeled nonlinear LORO R2:  -0.085796
```

The key counter-result is H1: prior measurement-control separation did **not** replicate for the split-label target-information class. The measurement was highly reliable and passed the pre-locked top-k/regret materiality gate, so H1 is inactive; it also narrowly misses H1-H6 Holm correction (`p=0.0584`). This does not rescue source-only selection: construction labels are target-label-derived and diagnostic-only.

H2 is inactive: the registered effective-multiplicity/top-gap model worsened held-target top-1-miss deviance (`-9.5059`, p=0.896). H3 is active: target-unlabeled geometry retains a local nonlinear association in all 32 eligible trajectory cells and passes six blocked controls, but fixed-kernel held-target/held-regime prediction worsens and actionability does not qualify. H4/H5 mean only that the exact registered strict-source F2 and target-unlabeled F4 candidates fail the full gate; they are not universal impossibility claims. H6 is descriptively strong (`incremental R2=0.4043`, raw p=0.0195) but inactive after Holm (`p=0.0778`).

Pre-execution red team: `45/45`. Independent scientific result red team: `60/60`. Final report red team: `20/20`. Regression: `focused=43`, `C65-C78S=256 pass + 1 conditional skip`, `C23-C78S=663 pass + 1 conditional skip`, `full_OACI=1591 pass + 1 conditional skip`; all stderr logs are empty.

Authorization governance: direct, explicit PM authorization is sufficient; no magic token is required. Real execution must still bind that approval to a committed, scope-specific protocol/execution lock before restricted data access or job submission.

Historical C80R readiness artifacts at that point:

```text
oaci/reports/C80R_ADDITIVE_REPAIR_PROTOCOL.json
oaci/reports/C80R_ADDITIVE_REPAIR_PROTOCOL.sha256
oaci/reports/C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.json
oaci/reports/C80R_REPAIRED_ANALYSIS_EXECUTION_LOCK.sha256
oaci/reports/C80R_IMPLEMENTATION_REPLAY.md
oaci/reports/C80R_PRE_EXECUTION_RED_TEAM.md
oaci/reports/C80R_PROTOCOL_READINESS.md
oaci/reports/C80E_PI_AUTHORIZATION_RECORD.json
oaci/reports/C80E_AUTHORIZATION_AND_PREFLIGHT.md
oaci/reports/C80E_AUTHORIZATION_AND_PREFLIGHT.json
oaci/reports/c80e_tables/authorization_lock_replay.csv
oaci/reports/c80e_tables/failure_reason_ledger.csv
oaci/reports/OACI_EEG_DG_PROJECT_MEMORY_THROUGH_C80E.md
oaci/reports/C80_LABEL_BUDGET_FRONTIER_PROTOCOL.json
oaci/reports/C80_LABEL_BUDGET_FRONTIER_PROTOCOL.sha256
oaci/reports/C80_PROTOCOL_TIMING_AUDIT.md
oaci/reports/C80_LABEL_BUDGET_CLAIM_CONTRACT.md
oaci/reports/C80P_ANALYSIS_EXECUTION_LOCK.json
oaci/reports/C80P_PROTOCOL_READINESS.md
oaci/reports/C80P_PROTOCOL_READINESS.json
oaci/reports/C80P_PRE_EXECUTION_RED_TEAM.md
oaci/reports/C80P_REGRESSION_VERIFICATION.md
```

Historical stopping instruction: C80R stopped for PI reauthorization with no
real-data frontier computed. It is retained to document the chronology and is
superseded by the completed C80E state at the top of this handoff.

---


## 1. What C23–C31 established (the mechanism chain)

**Setup.** Earlier rungs found: target-good checkpoints EXIST, a pre-registered source-only competence probe (**C19**,
frozen config hash `664007686afb520f`) is weakly positive **in-regime** but does **NOT transport** to new targets
(**C20**), and the estimand boundary was locked (**C21**). C22→C31 are **read-only audits over two frozen artifacts**
(the C22 score sidecar + the C10 replay) that trace the transport failure to its root. Each rung is pre-registered
with a **lettered case taxonomy**, read **gate-first** (numbers before narrative), and (C30/C31) **adversarially
red-teamed** by a multi-agent workflow.

| Rung | Commit | What it established (gate-first verdict) |
|---|---|---|
| **C22** | (pre-chain) | Transport failure = a **per-target score OFFSET** the source-only estimand doesn't calibrate (T1). The within-regime signal is real; the cross-target intercept is not. |
| **C23** | `c7b0234` | That offset is **source-UNOBSERVABLE** — a target-free source "gauge" fails leave-one-target-out (G5). You cannot calibrate the offset from source alone. |
| **C24** | `7dc5987` | **Target-UNLABELED marginal geometry PARTIALLY recovers** the offset (I2, permutation-robust p≈0.024); **source HURTS**, grouping/oracle fully recover ⇒ a **0-label transductive** problem. Done via a **P0-gated, no-retraining re-inference** (byte-identical replay 27/27). |
| **C25** | `40f5aa6` | The carrier is the **predicted-class MIX** (U2, Shapley 0.983); source = small-N high-dim noise (U6); grouping = a separate 0-label problem class (U7). |
| **C26** | `da62772` | predmix = a **split-STABLE target DECISION-OCCUPANCY pattern** reflecting the model's per-class error geometry (recall-corr 0.881); that SAME pattern **IS the target-identity FINGERPRINT** (P4, NN 1.0); offset recovery is ONLY the confidence-mix **synergy interaction** (P5), not standalone. Labels **quarantined** (never in the feature path). P0-gated re-persistence, byte-identical 27/27. |
| **C27** | `1f84e17` | Dissects P5 in **logit space**: the interaction is **class-conditioned confidence**, a **SINGLE sufficient factor** (conf-gap +0.524 survives permutation) — **REVISES** C26's "irreducible synergy". Still identity-entangled + only partially error-coupled. |
| **C28** | `c9a67ab` | **Source–target homology**: a source class-cond-conf factor EXISTS but does **NOT predict the target offset** (source gauge HURTS −0.378 vs target +0.524). **Source-unobservability CONFIRMED at the logit-factor level** — it's not a registry omission. |
| **C29** | `1bb1ba0` | **Origin**: the carrier is a **representation-projection-induced effective logit bias mean(W·z)**, NOT the parameter head-bias `b` (R2); a **target-specific** projection shift (R3); source representation can't substitute (R6). |
| **C30** | `ab41366` | **Rank–Gauge separation** (1st ultracode red-team): a **within-target RANK axis** (source-visible, weak) is orthogonal to a **cross-target GAUGE axis** (target-specific, source-unobservable). Red-team CONFIRMED the separation (G1) but **caught 2 overstatements**: "tracks source error" is tautological (G5), and the score-beats-family gap is within bootstrap noise (G7). The transferable part is the **multivariate probe** (sign-consistent 9/9), NOT R_src. **C19's positive = the within-target rank, NOT a target-free detector.** |
| **C31** | `611988f` | **Endpoint-axis / accuracy–calibration geometry** (2nd ultracode red-team): cases **E3 + E7**; E1/E2/E5/E6/E8 falsified; **E4 DOWNGRADED**. Accuracy, calibration, and the joint Pareto point are **largely the SAME within-target-rankable object** (they co-improve, +0.60, 9/9 targets; source rank orders them indistinguishably — E4 by-construction) but the pooled/cross-target transport is **gauge-broken and non-deployable**. This **reconciles** the earlier "accuracy↔calibration trade-off" (C16) as a **source-observability/gauge failure, NOT a checkpoint-space trade-off.** |

**The arc, in one breath.** Target-good checkpoints exist → the source probe sees only a **weak, distributed,
non-transferable within-target RANK** → a **target-specific GAUGE** (per-target score offset = a
representation-projection-induced, class-conditioned-confidence effective logit bias = the target-identity fingerprint)
**breaks cross-target transport and is source-unobservable** → **target-unlabeled** confidence geometry partially
recovers it (0-label transductive), but **source never can**. Every rung is **DIAGNOSTIC-ONLY**; no selector was built;
the oracle is never a feature; target labels are quarantined out of every feature path.

**Authoritative status:** `oaci/reports/C21_FINAL_CLAIM_LEDGER.md` (claim ledger) + the per-rung reports
`oaci/reports/C2*_*.md/.json` + `C30_RED_TEAM_VERIFICATION.md`. Regression: **99 tests green (C23–C31)**.

---

## 2. How this PM works with you (communication & collaboration style)

**The PM directs each rung with a detailed pre-registration.** A rung spec = a **question**, a **lettered case
taxonomy** decided in advance (C24 I-cases, C25 U-cases, C26 P1–P7, C27 L1–L7, C28 H1–H7, C29 R1–R8, C30 G1–G7,
C31 E1–E9), **hard gates**, a **pre-committed interpretation grid**, and the PM's **prior** — always closing with
*"but let the data decide."* You implement exactly that taxonomy; you do not invent new endpoints mid-rung.

**Non-negotiable working discipline:**
- **Gate-first reading:** compute the numbers and read the verdict from them BEFORE writing any narrative. Numbers win.
- **Disclose surprising cells BEFORE concluding; UNDER-claim.** Hold "PASS" until verified; report the worst cells
  first. (Over-confidence has been caught repeatedly — err low.)
- **Pre-register + red-team BEFORE compute; adversarially verify AFTER.** When **ultracode** is on, run a **multi-agent
  Workflow** that tries to REFUTE each gate-first verdict on the real artifacts. This is not ceremony — it caught real
  overstatements in **C30** (G5 tautology, G7 within-noise) and **C31** (E4 "accuracy-specific" was by-construction;
  E3 "≈chance" was too strong). Amend the verdict/report to match what survives.
- **Identity-gate-first for any re-inference:** a **P0 replay-identity gate** (byte-identical re-inference, e.g. 27/27)
  must pass before trusting a no-retraining re-inference. Prefer **real re-inference over a proxy** when GPU is idle;
  a proxy is a secondary appendix, never the central rung.
- **Reason-code every dropped/skipped unit; fail loud.** A silent `except → None` manufactures fake results.
- **Two-commit, unsquashed, auto-push.** Result-first, interpretation-second — never mixed. Terminal/frozen artifacts
  are read-only (never re-tag; build on top). **Push to origin after every commit** (the PM reviews via GitHub). End
  commit messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- **Precise claims only.** Distinguish observation from mechanism; call oracle quantities "oracle diagnostic, not
  deployable"; never "X converges to Y" without the supporting metric. Verify against real git (`git show <sha>:<file>`),
  not stale raw/CDN URLs (they mislead).

**PM communication:** be concise and decisive — give a **recommendation, not a survey**. Present decisions as
**free-text prose options + trade-offs + WAIT** — the PM **rejects the multiple-choice/AskUserQuestion picker**. **Own
mistakes plainly.** Report the gate-checklist, then WAIT for explicit "go" before consuming compute.

**Recurring catches (the discipline earns its keep):** gate-first + adversarial controls caught a C22 epoch-definition
bug, a C24 within-target block confound, the C25→C26 Shapley-synergy-vs-standalone distinction, a C28 raw-cosine
mean-structure artifact, a C29 linear-vs-nonlinear-carrier mis-fire, C30 sign-flip masking, and C31's E4/E3
overstatements. Expect to be wrong on the first read; the controls exist to catch it.

---

## 3. GitHub — repo / branch / commits

- **Repo:** `git@github.com:W-Yinghao/CMI.git`  ·  **Branch:** `oaci` (fully pushed).
- **Worktree on the lab machine:** `/home/infres/yinwang/CMI_AAAI_oaci`.
- **C80P readiness result = `1b02454`**; protocol = `f5d83b3`, protocol SHA = `c6292597f5610cb96e8eaf0313eaa741f8fa9b11dd89ff9e4d84db1fa33add85`, synthetic implementation = `c98e084`, analysis lock = `972f47c`, analysis-lock SHA = `05a99e7ccc357b90b6675756caa680fd16541358bb697fdded89351f1e7ae4a8`.
- **C79E final HEAD = `dadd166`**; final validation = `a12dc8b`, protocol = `ec4834c`, protocol SHA = `e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587`, field/analysis locks = `35d0c65` / `7cebf2e`, authorization record = `b67ba6c`, scientific red team = `439c8c5`.
- **C78S implementation = `e561a15`**, scope-specific lock = `ce1fb14`, result = `43a046c`, handoff = `48be5b7`, regression-provenance correction = `dcd4c28`; all are unsquashed.
- **C78F result = `51022f4`**; protocol/lock/repair anchors are `1d210fd`, `a902966`, and `f0d49c2`.
- C31 remains `611988f`; the complete C23–C78S trail is retained unsquashed on `origin/oaci`.

---

## 4. The `oaci/` package (which subpackage = which rung)

Read-only mechanism rungs each live in their own subpackage (module set: `schema.py` = frozen constants + taxonomy,
`artifact_loader.py`, the analysis modules, `taxonomy.py` = deterministic case decision, `report.py` = render + tables
+ forbidden-claim guard, `__init__.py`, plus `oaci/tests/test_c<NN>_*.py`).

| Rung | Subpackage | Report prefix |
|---|---|---|
| C23 | `oaci/score_gauge/` | `C23_*` |
| C24 | `oaci/information_ladder/` | `C24_*` |
| C25 | `oaci/unlabeled_gauge/` | `C25_*` |
| C26 | `oaci/predmix_mechanism/` | `C26_*` |
| C27 | `oaci/logit_geometry/` | `C27_*` |
| C28 | `oaci/source_target_homology/` | `C28_*` |
| C29 | `oaci/rep_head_geometry/` | `C29_*` |
| C30 | `oaci/rank_gauge/` | `C30_*` |
| C31 | `oaci/endpoint_geometry/` | `C31_*` |

Shared: `oaci/identifiability/` (multivariate_probe `_auc`, signal_atlas), `oaci/competence_probe/` (C19 probe),
`oaci/reports/` (all `.md`/`.json` + per-rung `c<NN>_tables/`). Method-line scaffolding (CLOSED) lives in
`oaci/{methods,train,runner,eval,leakage,support_stress,...}` — do not resurrect it as a method.

---

## 5. Frozen artifacts, config lock, environment

- **Frozen C19 config hash: `664007686afb520f`** — every rung's `schema.frozen_config_hash()` must equal this; the
  report `_lock_config()` **refuses to run on drift**. Do not change it.
- **Frozen artifacts (read-only inputs, verified present):**
  - C22 score sidecar: `/projects/EEG-foundation-model/yinghao/oaci-c22-scores.json` (per-candidate frozen source-only
    LOTO probe score, competence label, R_src, source robust-core features; modes `in_regime` / `cross_regime`).
  - C10 replay: `/projects/EEG-foundation-model/yinghao/oaci-c10-replay/seed-*-target-*.json` (per-candidate target
    bAcc/NLL/ECE + per-(seed,target,level) ERM reference; used by C31).
- **Environment:** CPU-only, `/home/infres/yinwang/anaconda3/bin/python` (base conda) — sklearn 1.0.2, numpy 1.24.4,
  scipy. **No GPU needed** for the C-series (read-only over frozen artifacts). Re-inference rungs (C24/C26) used SLURM
  GPU with a P0 replay gate; those are already done and byte-identical.
- **Regression:** `python -m pytest oaci/tests/test_c2[3-9]_*.py oaci/tests/test_c3[01]_*.py -q` → **99 passed**
  (~2 min; C24/C26 re-inference tests are the slow ones). Tree is clean (`git status` = 0) — keep it clean
  (`require_clean_git`-style guards trip on stray files).

---

## 6. How a rung is built (the reusable pattern) & how to run it

Each rung: (1) `schema.py` imports the prior rung's schema, re-locks the config hash, defines the **frozen case
taxonomy** (letters) + `FORBIDDEN_CLAIM_SUBSTRINGS`; (2) `artifact_loader.py` loads/joins the frozen artifacts (no
refit/tune); (3) analysis modules compute the endpoints **gate-first**; (4) `taxonomy.py` decides cases
deterministically, **reporting imbalance/base-rates BEFORE the verdict**; (5) `report.py` renders `.md`/`.json` +
`c<NN>_tables/` CSVs with a **negation-aware forbidden-claim guard**; (6) `test_c<NN>_*.py` with synthetic fixtures of
known geometry + a real-artifact smoke; (7) when ultracode is on, an **adversarial-verify Workflow** red-teams the
verdicts and a `C<NN>_RED_TEAM_VERIFICATION.md` records what survived.

Run a rung's report: `python -m oaci.<subpackage>.report --out-dir oaci/reports`. Run its tests:
`python -m pytest oaci/tests/test_c<NN>_*.py -q`.

---

## 7. How to continue (C80E complete; C81 review required)

- Treat `C80-A_stable_low_regret_label_budget_frontier_across_training_seeds` as the locked C80 taxonomy and preserve the narrower source-relative interpretation in [C80_LABEL_BUDGET_FRONTIER.md](reports/C80_LABEL_BUDGET_FRONTIER.md).
- Preserve the failed pre-evaluation job `894641`, additive descriptor repair, successful job `894646`, result hashes, both red teams, and the complete regression-attempt ledger.
- Do not reinterpret `B*=1` as universal one-label sufficiency. Report the weak source baseline, non-low absolute regret, low top-1 rate, 16/16 leave-one-target B* changes, and existing-field retrospective status.
- Keep target 4 excluded, the same-label oracle closed, construction/evaluation views disjoint, and trial ID/row order restricted to join/split/dependence keys.
- Stop for PM review. C80E does not authorize C81, seed 5, BNCI2014_004, another target, active acquisition, same-label-oracle analysis, checkpoint recommendations, or manuscript drafting.

---

## 8. Standing constraints

DIAGNOSTIC-ONLY. No selector / no target-free detector / no OACI rescue / no joint-deployable-improvement / no
external-validation-success claim. **No target labels in source or target-unlabeled feature construction**; label
views stay physically quarantined. Frozen C19 config hash **`664007686afb520f`** remains unchanged. No probe tuning,
feature selection, checkpoint recommendation, BNCI2014_004 access, additional
training-seed execution, or manuscript prose. Existing seed-3/seed-4 views may
be read only under a separately authorized, scope-locked analysis such as
C80E.

Historical “no training / no seed 3” defaults were explicitly superseded for
the completed C78/C78R/C78F field generation and C79E seed-4 replication under
their scope-specific direct authorizations. C78S, C79P, and C80P were
analysis/protocol-only. C80E was explicitly authorized and completed as a
CPU-only read-only existing-field analysis after additive pre-outcome repair
and relock. Authorization did not waive lock completeness; physical-view
isolation, red-team, report, commit, and push remained mandatory.

**FORBIDDEN wording** (the report guards enforce this): "deployable selector", "target-free detector", "OACI rescue",
"joint deployable improvement", "endpoint/pareto selector", "target oracle as method", "X converges to Y" (without the
metric). **Oracle quantities are "oracle diagnostic, not deployable."**

---

## 9. Embedded memory (so Codex needs nothing from this machine's `~/.claude`)

- **OACI is Direction 1** (the PM's taxonomy: Dir 1 = OACI, Dir 2 = ACAR, Dir 3 = TOS-CMI, Dir 4 = CSC). This handoff
  is the OACI/`oaci`-branch line — distinct from Project S2P (subject-scaling pretraining, branch
  `project/s2p-subject-scaling`) which is a separate direction.
- Discipline pointers (all reflected above): gate-first + disclose-all-cells + under-claim; pre-reg + red-team before,
  adversarial multi-agent verify after; P0 replay-identity gate before re-inference; prefer real re-inference over
  proxy when GPU idle; reason-code every feature/unit loss and fail loud; two-commit + auto-push + `Co-Authored-By`;
  PM decisions as free-text prose (never a picker); verify against real git, not stale CDN URLs.
- Updated one-line ledger: *registered source-only and target-unlabeled
  candidates remain unqualified across two training seeds; construction-label
  actionability is directionally robust while compound reliability is
  seed-heterogeneous. C80E finds a stable source-relative Q0 frontier at one
  label per class in the existing fields, but absolute regret, top-1 recovery,
  leave-target sensitivity, and retrospective reuse prevent a universal,
  deployable, or externally general claim.*
```
