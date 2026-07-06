# FSR_00 — Project Specification

**Project FSR: Functional Shortcut Reliance in EEG Representations**

Status: FROZEN SPEC (definitions only — contains no experiment results).
Branch: `project/functional-shortcut-reliance` (cut from `project/cita-target-unlabeled-cmi` @ `c889730`).
Companion documents:
- `docs/FSR_01_EVIDENCE_LEDGER.md` — current cross-branch evidence table.
- `results/fsr_artifact_index/artifact_index.csv` — machine-readable artifact map.

This document defines the project. It does not report findings, propose a new loss, or make empirical claims. All evidence lives in `FSR_01`.

---

## 1. Project thesis

> **Measurable subject leakage is not necessarily functional shortcut reliance. In EEG representations, a harmful shortcut requires task-coupled domain information, not merely domain-decodable information.**

中文：EEG 表征里能解码出 subject/domain，不等于模型把它当作有害捷径。真正需要研究的是：哪些 domain 信息与任务头耦合、被模型功能性依赖、并且干预后会改变目标域风险。

The core scientific object of this project is the *gap* between what is **measurable** (domain-decodable information in `Z`) and what is **relied upon** (information the task head functionally uses such that intervening on it changes the task or the target-domain risk).

Operational definition used throughout the project:

```text
harmful shortcut ≠ leakage magnitude
harmful shortcut = leakage × task coupling × functional reliance × harmful target effect
```

A representation direction is a **harmful functional shortcut** only if it is simultaneously:
1. measurable (domain-decodable from `Z`, even conditioning on `Y`),
2. localizable (attributable to a specific subspace / branch),
3. interventionable (removable by a linear or non-linear eraser),
4. functionally relied upon (removing it changes the task head's output), and
5. harmful in consequence (removing it improves — or at least does not degrade — target-domain risk in an interpretable way).

Domain-decodable information that fails any of (4)–(5) is **not** a harmful shortcut for the purposes of this project; it is at most an audit/privacy signal.

---

## 2. What this project breaks: the default assumption

The default assumption this project is designed to falsify:

```text
I(Z;D|Y) high
 → subject shortcut strong
 → reduce I(Z;D|Y)
 → shortcut reliance drops
 → target generalization improves
```

The repository evidence (frozen in `FSR_01`) already shows this chain breaks in the middle: measured leakage can be reduced by a proxy without reducing functional reliance, and stronger erasure does not stably produce target benefit. The replacement working hypothesis is that reliance is governed by *task coupling* and *branch load*, not by leakage magnitude:

```text
R (reliance)  ~  L (leakage) × A (task coupling) × B (branch load) × harmful target effect
```

This project's job is to *test* that hypothesis on frozen artifacts, not to assume it.

---

## 3. Non-goals

This project explicitly does **not**:

```text
- run new CMI λ/β/η/k sweeps
- attempt to rescue static-DGCNN CIGL
- develop a new CITA-CMI training objective
- write any CMI-control result up as a positive method
- treat erasure / leakage removal as automatically equal to DG improvement
- make FBCSP spatial-CMI training the main line
- use target labels for fit, model selection, or early stopping
```

The premise that **source-only CMI reliance-control is closed** (CIGL_70) and that the target-unlabeled **active-λ=1.0 CMI term does not beat matched TTA-Control** (CITA_03) are treated as **frozen premises**, not pending hypotheses. See `FSR_01`.

The success of this project is **not** a target-bAcc improvement. See §7.

---

## 4. The Functional Shortcut Reliance (FSR) Ladder

Every representation / branch / method is placed on the same six-level ladder. A project claim may only be stated in terms of the *relationship between levels* — never by asserting a high level from a low one (e.g. never infer L5/L6 reliance from an L1 detection).

| Level | Question | Indicators |
|---|---|---|
| **L1 Detectability** | Can `D` be decoded from `Z`, even conditioning on `Y`? | posterior-KL proxy, domain probe accuracy, permutation-null ratio |
| **L2 Reducibility** | Can a method reduce *measured* leakage? | ΔKL, Δprobe-acc, Δdomain-advantage |
| **L3 Erasability** | Can a linear/non-linear eraser remove the subject signal? | LEACE / INLP / RLACE / random-k leakage drop |
| **L4 Task coupling** | Is the erased/residual subject subspace aligned with the task head? | `align_k`, task-head row-space overlap, branch gate, branch ablation |
| **L5 Functional reliance** | Does deleting that subspace change the model's output / task? | R3 head-replay `task_drop`, logit SymKL, CE/NLL delta |
| **L6 Target consequence** | Is that reliance harmful, benign, or task-useful? | target bAcc/NLL/ECE delta, worst-subject delta, refusal/accept gate |

**Rule:** a claim must be anchored to a relationship among L1–L6. L1/L2 (measurement) may not be reported as if it were L5/L6 (reliance/consequence). This is the boundary that separates Project FSR from the closed CMI-control story, which optimized L1/L2.

---

## 5. Core research questions

- **RQ1 — Does measured leakage predict functional reliance?**
  Compare `graph_kl / node_kl / spatial_kl` (L1/L2) vs `R3 task_drop / logit SymKL / CE delta` (L5).
- **RQ2 — Does erasure strength predict target benefit?**
  Compare LEACE/INLP/RLACE/TOS-deletion leakage drop (L3) vs target bAcc/NLL/ECE delta (L6).
- **RQ3 — Does task-head alignment explain reliance better than leakage magnitude?**
  Compare `align_k` (L4) vs KL proxy (L1) vs domain-probe accuracy (L1) as predictors of R (L5).
- **RQ4 — Does branch-locality determine the meaning of leakage?**
  Leakage in a non-load-bearing branch (graph/node) may be audit/privacy signal; leakage in a load-bearing branch (spatial) may be task-coupled. Audit branch-locally; do **not** train spatial-CMI in this project.

---

## 6. Deliverables

- **A — FSR evidence ledger** (`FSR_01`): cross-branch table unifying all existing results into one schema (route, regime, dataset, backbone, representation/branch, leakage metric, intervention, reliance metric, target metric, collapse guard, result, allowed claim, forbidden claim, artifact path, SHA). **First deliverable — this Step 1.** The evidence base spans the whole repository, in tiers: **primary** — CMI-control (CIGL / FCIGL / dCIGL / MetaCMI on `project/cigl-*` + `project/metacmi-eegnet-conformer`; CITA on `project/cita-target-unlabeled-cmi`), TOS erasure (`tos`), FBCSP branch-locality (`project/fbcsp-lgg-spatial-cmi-fusion`); **supporting** — FBCSP bottleneck/blueprint (`project/fblgg-2a-bottleneck-analysis`, `project/fbcsp-lgg-dualcmi-scaffold`), OACI source-side observability failure (`oaci`), ACAR action-conditional deployment successor (`acar`), CSC information-contract boundary (`csc`), LPC-CMI legacy boundary (`exp/lpc-cmi`); **background only** — prior-decoupled TTA / H²-CMI (`exp/h2cmi-*`), which supports the measurement→control framing but is NOT FSR leakage/reliance evidence. A companion `results/fsr_artifact_index/branch_inventory.md` records the SHA/worktree/clean-dirty state of every cited branch.
- **B — Unified functional-reliance benchmark schema**: reuse frozen artifacts; scores L, E, A, R, T, B; statistical explanation only (no deployment policy yet).
- **C — CIGL × TOS mechanism synthesis**: proxy leakage ↓ but reliance not (CIGL); subject signal removable but target benefit not / task-safety only conditional (TOS/LEACE/RLACE/INLP) ⇒ leakage reduction and erasure are insufficient; harmful shortcut requires task-coupled reliance evidence.
- **D — Branch-local extension**: use FBCSP-LGG frozen branch diagnostics (spatial = load-bearing) to ask whether leakage in a load-bearing branch behaves differently from leakage in a non-load-bearing one. Mark missing branch-leakage metrics rather than running GPU.
- **E — Paper skeleton**: working title *Measurable Is Not Reliance: Functional Shortcut Auditing in EEG Representations* (alt: *Subject Leakage Is Not Necessarily a Shortcut: Functional Reliance Audits for EEG Representations*).

---

## 7. Success criteria

Success is **not** a target-accuracy gain.

**Strong success**
```text
leakage magnitude L does NOT predict reliance R;
task-head alignment A significantly predicts R;
branch load B modulates the L→R relationship;
erasure strength E does NOT stably predict target benefit T;
⇒ propose the functional-shortcut-reliance audit ladder.
```

**Moderate success**
```text
L, A, B each only weakly predict R;
but we systematically show leakage/erasure/CMI-control are insufficient to certify shortcut reliance;
⇒ a high-quality negative / boundary paper.
```

**Failure**
```text
artifacts incomplete; metrics cannot be aligned across branches;
R3 / erasure / branch metrics cannot be reproduced;
only narrative remains, no statistical closure.
```

---

## 8. Allowed claims

Claims the project MAY make (each must cite an artifact path + SHA in `FSR_01`):

```text
- "Subject/domain leakage is measurable in Z even conditioning on Y." (L1)
- "A given method reduces the measured leakage proxy." (L2, method-scoped)
- "Subject signal is linearly/non-linearly erasable from Z." (L3, operator-scoped)
- "The residual/erased subspace is more/less task-head-aligned." (L4, with align_k evidence)
- "Removing subspace S changes the task head's output by <amount>." (L5, R3 evidence)
- "Reducing measured leakage did NOT reduce functional reliance." (L1/L2 vs L5 relationship)
- "Erasure did NOT stably improve — or degraded — the target endpoint." (L3 vs L6 relationship)
- "Task-head alignment predicts reliance better than leakage magnitude." (L4 vs L1, IF the statistics support it)
- "Source-only CMI reliance-control is closed as a positive method." (frozen premise, CIGL_70)
- "Target-unlabeled active-λ CMI does not beat matched TTA-Control." (frozen premise, CITA_03)
- "The spatial branch is load-bearing; graph/node branches are not (on the tested datasets)." (L4/branch, branch-ablation evidence)
- Boundary/negative claims of the form "current observables do not certify shortcut reliance."
```

## 9. Forbidden claims

Claims the project may NOT make:

```text
- "Reducing I(Z;D|Y) improves target generalization." (the falsified chain)
- "High measured leakage ⇒ the model relies on it as a harmful shortcut." (L1 → L5/L6 leap)
- "Erasing subject signal makes the model domain-general / improves DG." (L3 → L6 leap)
- "CIGL / FCIGL / dCIGL / MetaCMI / CITA-CMI is a positive DG/reliance-control method." (closed)
- "align_k is a validated/perfect estimator of reliance." (only that it is closer than leakage, if shown)
- Any claim that uses target labels for fit / selection / early stopping. (target y is EVAL/AUDIT-only)
- Any target-accuracy SOTA claim.
- Any claim from an artifact whose path/SHA is not recorded in FSR_01.
```

Every target-label use MUST be tagged `NO` / `YES_FORBIDDEN` / `AUDIT_ONLY` / `UNKNOWN` in the artifact index; any `UNKNOWN` must be explained.

---

## 10. Phase gates

- **Phase 0 — Evidence freeze & artifact map** (this Step 1). Gate: locate CIGL R3 / gap diagnostics, CIGL source-only closure summary, CITA λ=1.0 readout, TOS erasure/LEACE/refusal evidence, FBCSP-LGG branch ablation / gate summaries; mark missing metrics rather than inventing them.
- **Phase 1 — Unified metric schema.** Gate (revised 2026-07-06, PM patch — replaces the earlier "≥3 of {L1,L2,L5,L6}", which was over-strict and penalised honest diagnostic rows):
  ```text
  Phase-1 quantitative inclusion gate:
    A route may enter a cross-level quantitative test only if it has
    at least one predictor-side level among {L1, L2, L3, L4}
    and at least one endpoint-side level among {L5, L6}.

  Support-only / boundary / protocol rows are allowed, but must be tagged:
    SUPPORT_ONLY
    BOUNDARY_ONLY
    PROTOCOL_ONLY
    BACKGROUND_ONLY

  They may support interpretation but may not be used in RQ1/RQ2/RQ3 regression
  unless the required predictor and endpoint levels are present.
  ```
  This is not a lowered bar — it is an executable, auditable one. Example: FBCSP's spatial branch is strong **L4** evidence (`zero_spatial` −7.4pp on 2a, −8.8pp on 2015 = load-bearing), but with **no per-branch leakage probe** it has no L1/L5 pairing, so it is `SUPPORT_ONLY` (branch-locality background) and cannot enter the RQ4 leakage×branch quantitative test. Every claim still needs an artifact path; every target-label use still tagged.
- **Phase 2 — Frozen CPU re-analysis.** No new models. Compute `corr(L,R)`, `corr(E,T)`, `corr(A,R)`; mixed-effects `R ~ L + A + branch + dataset + backbone`; hierarchical bootstrap CIs; random-subspace control; per-dataset leave-one-out. Gate: if A explains R better than L ⇒ mechanism-positive main line; else ⇒ stronger boundary line.
- **Phase 3 — Branch-local audit.** Bring in FBCSP-LGG spatial branch, no CMI training. Gate: frozen branch ablation + gate summary suffice for the "load-bearing branch" background; if `spatial_z` frozen embeddings are missing, record as missing metric; only after Phase 2 discuss whether to approve a small frozen-probe run.
- **Phase 4 — Minimal confirmatory GPU run (optional).** Approved only if Phase 2 shows a specific gap. Allowed: frozen representation extraction, frozen `spatial_z/graph_z/node_z` probe, R3-style branch/subspace-removal replay. Forbidden: `fbdualpc` training, CMI regularizer sweeps, new graph-architecture search, ConformerFull rescue.

---

## 11. Provenance discipline

Every row in `FSR_01` and `artifact_index.csv` must carry: source branch, source SHA, artifact path, and the metric's source document/section. Numbers are quoted from frozen artifacts; anything not found on disk is labelled `NOT FOUND` / `missing metric`, never inferred. GitHub raw branch-ref URLs are CDN-cached and can mislead — provenance is taken from local `git show <sha>:<path>`, not raw URLs. Priority order for any judgement: **provenance > metrics > interpretation > new idea.**

### Red-flag discipline (a claim carrying any of these must be flagged, not passed)

1. Only a README claim, with no result artifact.
2. A result artifact with no branch SHA / manifest.
3. An artifact rewritten without a manifest note explaining the change.
4. Target labels used for fit / selection / early-stopping (must tag `YES_FORBIDDEN`).
5. A tiny-pilot result written as a method-level conclusion.
6. A seed-0 positive without seeds {0,1,2} confirmation.
7. CMI-control written as a positive method (closed by CIGL_70 / CMI_SYNTHESIS_01).
8. An erasure / leakage-drop auto-read as a target benefit.
9. An ACAR protocol/gate written as a scientific-efficacy result (gates are DEFINED, not passed; Stage-2B ended DEV_STOP).
10. A CSC simulator pass written as a real-EEG certificate.
11. An FBCSP P6 spatial-CMI scaffold written as a real-EEG CMI result.

Default rule: if the artifact is unclear, do not write a strong claim; if a target-label use is unclear, tag it `UNKNOWN` and explain; if a route is CMI-control, treat it by the closure documents unless a newer, fully-provenanced update exists.
