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

## 0. Current continuation state (2026-07-11)

The authoritative tip is C78S, the completed full seed-3 multi-regime scientific analysis:

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

Authoritative artifacts:

```text
oaci/reports/C78S_SEED3_SCIENTIFIC_ANALYSIS.md
oaci/reports/C78S_SEED3_SCIENTIFIC_ANALYSIS.json
oaci/reports/C78S_AUTHORIZED_RED_TEAM_VERIFICATION.md
oaci/reports/C78S_FINAL_REPORT_RED_TEAM.md
oaci/reports/C78S_ARTIFACT_MANIFEST.json
oaci/reports/C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.json
```

Wait for PM review. C79 execution, seed 4, BNCI2014_004, and manuscript work remain unauthorized and untouched.

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

## 7. How to continue (C78S is DONE; C79 is protocol-only)

- Wait for PM review of the mixed C78S result. Do not silently start C79.
- C79's protocol is locked, but its execution requires a later direct authorization. Seed 4 has not been accessed.
- If C79 is approved, analyze seed 4 alone before any seed-3 pooling and preserve the fixed C78S feature, null, multiplicity, materiality, view-isolation, and ERM-anchor contracts.
- Do not reinterpret C78S split-label actionability as source-only selection, OACI rescue, or deployment. Do not add features, kernels, regimes, endpoints, selectors, checkpoint recommendations, BNCI2014_004 access, or manuscript prose outside the locked C79 protocol.

---

## 8. Standing constraints

DIAGNOSTIC-ONLY. No selector / no target-free detector / no OACI rescue / no joint-deployable-improvement / no
external-validation-success claim. **No target labels in source or target-unlabeled feature construction**; label
views stay physically quarantined. Frozen C19 config hash **`664007686afb520f`** remains unchanged. No probe tuning,
feature selection, checkpoint recommendation, BNCI2014_004 access, seed-4 access, or manuscript prose.

Historical “no training / no seed 3” defaults were explicitly superseded only for the completed C78/C78R/C78F
prospective field-generation scopes. C78S was analysis-only. None of these authorize further training or C79. Direct PM authorization no
longer needs a token, but scope/protocol timing, physical-view isolation, red-team, report, commit, and push remain
mandatory.

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
- Updated one-line ledger: *target-good checkpoints exist; registered source-only and target-unlabeled representation
  paths still fail transport/control, while split-label target information can be reliable and materially actionable
  inside the seed-3 diagnostic field. That counter-result does not make the signal source-only or deployable; seed-4
  confirmation remains required.*
```
