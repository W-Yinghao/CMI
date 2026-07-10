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

## 0. Current continuation state (2026-07-10)

The detailed C23-C31 history below remains useful background, but the authoritative tip is now C76:

```text
C75 result commit: fb8a412
C76 protocol SHA:  1a1b4255601d6178ffbe8a8245625845fdb4057c445ca8db25e84b4ddcd8528f
C76 final gate:    LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE
C76 primary:       C76-D_local_nonlinear_measurement_nontransportable_nonactionable
```

C76 is a no-forward/no-training T2 robustness audit of C75's nonlinear association. It preserves the untouched
1,052-unit T3-HO z/Wz new-variable holdout. It ran 29 function-preserving orbit variants over all 216 T2 units,
24 registered kernel/statistic paths, six 499-replicate blocked null families, nested leave-target-out prediction,
actionability gates, and seven 500-replicate synthetic known cases.

Gate-first result:

```text
strict-source architecture:
  best registered effect 0.234144
  worst six-null max-stat p 0.054
  strict control pass false
  incremental R2 -0.042483; prediction max-stat p 0.998; actionability false

target-unlabeled architecture geometry (F4[0:20], Wz tail excluded):
  Laplacian-HSIC effect 0.237725; target-bootstrap CI [0.195773, 0.277323]
  worst six-null max-stat p 0.030; all six nulls pass
  incremental R2 -0.011041; prediction max-stat p 0.820
  positive prediction targets 4/9; actionability false
```

The surviving target association is local and diagnostic. It is robust over the registered orbit but does not
transport as held-out prediction or improve checkpoint actionability. It does not identify W-versus-z origin,
representation causality, target gauge, source-only rescue, or a deployable method. No C77 protocol was created;
the T3-HO representation campaign is not justified and the representation branch is saturated under the current
architecture/frozen universe.

Independent red-team job `892694` passed 26/26 blocking checks after rehashing 1,080/1,080 C74 descriptors. It
forced three repairs before reporting: isolate C75 F4's 20-d geometry from its 15-d Wz/logit-redundant tail; require
all six nulls before calling association-prediction separation; and repair synthetic S5 so a random extreme winner
preserves association without top1 control. Final verification is 18 focused C76,
115 C65-C76 (job 892715), 522 C23-C76 (job 892716),
and 1450 full-suite tests (job 892717), all with empty stderr.

Authoritative artifacts:

```text
oaci/reports/C76_REPRESENTATION_ASSOCIATION_ORBIT.md
oaci/reports/C76_REPRESENTATION_ASSOCIATION_ORBIT.json
oaci/reports/C76_RED_TEAM_VERIFICATION.md
oaci/reports/c76_tables/artifact_manifest.csv
oaci/reports/c76_tables/execution_attempt_ledger.csv
```

Do not launch C77, T3-HO representation re-inference, new training, GPU work, BNCI2014_004, seeds [3,4], a selector,
checkpoint recommendations, or manuscript drafting without a new explicit PM instruction.

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
- **C31 = `611988f`** on `origin/oaci` — the whole C23–C31 chain is on GitHub. This handoff adds one **doc-only**
  commit on top of `611988f` (so `origin/oaci` tip = that handoff commit; `611988f` is its immediate parent). Verify
  the chain: `git -C /home/infres/yinwang/CMI_AAAI_oaci log --oneline origin/oaci | grep -m1 C31` → `611988f`.
- Recent trail (each rung is its own result→interpretation commit sequence; see §1 table for SHAs).

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

## 7. How to continue (C31 is DONE; no pending rung)

C31 is complete, committed, pushed. There is **no pre-specified C32**. Depending on the PM's next instruction:

- **Reproduce / verify:** `git -C /home/infres/yinwang/CMI_AAAI_oaci fetch && git checkout oaci`; run the 99-test
  regression; `python -m oaci.endpoint_geometry.report` (and any C2x) to regenerate reports; confirm the config hash
  and both artifact paths. This re-establishes the chain end-to-end.
- **New rung (C32):** wait for the PM's pre-registered spec (question + lettered taxonomy + hard gates + prior). Build
  it with the §6 pattern in a new `oaci/<name>/` subpackage over the SAME frozen artifacts + config hash; gate-first;
  then ultracode adversarial-verify; two-commit; push.
- **Do NOT** turn any diagnostic into a selector, use the oracle as a feature, put target labels in a feature path,
  change the config hash, run EEG training, execute BNCI2014_004, use seeds [3,4], or write manuscript prose (the
  method line is CLOSED and the estimand boundary is locked).

---

## 8. Standing constraints (VERBATIM — apply to every rung)

DIAGNOSTIC-ONLY. No selector / no target-free detector / no OACI rescue / no joint-deployable-improvement / no
external-validation-success claim. **No target labels in feature construction** (post-hoc, quarantined only). Frozen
C19 config hash **`664007686afb520f`** unchanged. No probe tuning. No feature selection. **No re-inference unless
artifact integrity fails** (and then only P0-replay-gated). No EEG training. No BNCI2014_004 execution. No seeds
`[3,4]`. No manuscript prose. Auto-push after every commit; `Co-Authored-By` trailer.

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
- The mechanism chain's one-line ledger entry: *target-good checkpoints exist but the source-only signal is a weak
  within-target rank whose cross-target gauge (class-conditioned-confidence effective logit bias = target-identity
  fingerprint) is source-unobservable and only 0-label-transductively recoverable — diagnostic-only, never a selector.*
```
