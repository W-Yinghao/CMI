# FSR_17 — RQ4 Branch-Local Verification Audit (Phase 4B)

**Project FSR — Phase 4B.** The branch-local L1–L6 evidence chain from the ERM refit (FSR_16), and the verification verdict. This is the first *direct, real-EEG, branch-local* test of whether a subject/domain signal is a **verified harmful shortcut** or merely measurable leakage. Seed 0, 21 LOSO folds (2a 9 + 2015 12). Target labels score L5/L6 only; probes/subspaces source-fit. CPU-only analysis on frozen dumps.

## Method (per branch `b ∈ {graph_z, temporal_z, spatial_z, fused_z}`)
- **L1** source-only subject-decode probe on `z_b` (5-fold, permutation null).
- **L4** ablation drop (`zero_b`) + fusion-gate weight (from the run).
- **L5** erase the source-fit top-2 subject subspace from the branch's **target** latent, recompose `head3(_fuse3(...))`, and measure `task_drop = bAcc_orig − bAcc_erased` and logit SymKL, each vs a random-subspace control.
- **L6** target bAcc/NLL.
Sign convention: `task_drop < 0` (erasing *helps* target) ⇒ harmful shortcut; `task_drop > 0` (erasing *hurts* target) ⇒ target-useful/benign reliance.

## Results (mean, bootstrap CI; `*` = CI excludes 0)

| dataset | branch | L1 subj bAcc (chance) | L4 ablation drop | L5 task_drop (vs random) | L5 SymKL specificity | verdict |
|---|---|---|---|---|---|---|
| 2a | graph_z | 0.433 (.125) | −0.027 | −0.014 (spec −0.015\*) | 0.278\* | measurable_only |
| 2a | temporal_z | 0.584 (.125) | −0.018 | −0.007 (ns) | 0.088\* | measurable_only |
| 2a | **spatial_z** | **0.922** (.125) | **+0.086** | **+0.015** (spec +0.017) | **0.590\*** | **measurable+coupled, NOT harmful** |
| 2015 | graph_z | 0.307 (.091) | −0.012 | −0.010\* (spec −0.010\*) | 0.187\* | measurable_only |
| 2015 | temporal_z | 0.411 (.091) | −0.017 | −0.008 (ns) | 0.043\* | measurable_only |
| 2015 | **spatial_z** | **0.821** (.091) | **+0.085** | **+0.050\*** (spec +0.051\*) | **0.316\*** | **measurable+coupled, NOT harmful** |

L6 target bAcc: 2a 0.358, 2015 0.599.

## Verdict: `NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT` on natural EEG

The spatial branch is the shortcut *candidate* on both datasets — it satisfies four conditions: **measurable** (subject-decode 0.92/0.82, far above chance and every other branch), **task-coupled / load-bearing** (largest ablation drop +0.086, highest gate), and **functionally relied-upon** (erasing its subject subspace shifts logits far more than a random subspace, SymKL specificity 0.59/0.32, CI excludes 0). But it **fails the target-harmful condition**: erasing the subject subspace *raises* task_drop (+0.015 on 2a, **+0.050 significant on 2015**), i.e. removing it **hurts** the held-out target. The subject-predictive directions in the load-bearing spatial branch are **entangled with task-useful signal** — mechanistically expected, since CSP-style spatial filters encode both *who* and *what*.

So the most subject-leaky, load-bearing, functionally-coupled branch is a **task-useful (benign) reliance, not a harmful shortcut** — and **blind erasure of it would harm the target**. This is the branch-local, mechanistic explanation of why global erasure failed (TOS, `benefit_claimable=0/40`): removal hits task-entangled signal.

The graph branch shows a small *significant* target-useful-removal signal (task_drop −0.010\* on 2015, spec-vs-random −0.010\*), i.e. a weak hint of a mildly-harmful direction — but its leakage is low (0.31) and it is **not** load-bearing (ablation drop negative), so it does not meet the harmful-shortcut bar. We report it as a weak, non-load-bearing candidate, not a finding.

## What this licenses (and what it does not)
- **Licensed (verified):** on these benchmarks, real-EEG branch-local subject leakage is **not automatically a harmful shortcut**; the strongest candidate is task-entangled, so verification correctly *refuses* blind repair.
- **Not licensed:** "spatial leakage is harmful", "erase the spatial subject subspace to improve DG" (it would harm), or any repair claim — none is verified.
- **Implication for the framework:** because natural EEG here yields no verified harmful shortcut, the **positive control (Phase 4C, PC1 subject-token injection / PC2 prevalence stress)** is required to demonstrate the verification protocol *can* detect and localize a known harmful shortcut, and that a task-protected repair recovers it. Phase 4D repair candidates are gated on a verified target-harmful branch (injected or, if found, natural).

## Firewall
`rq4_target_label_firewall.json`: all 21 folds clean — target y used only for final L5/L6 scoring; probes and subject subspaces fit on source only; target subject held out (unseen domain, never a probe class).
