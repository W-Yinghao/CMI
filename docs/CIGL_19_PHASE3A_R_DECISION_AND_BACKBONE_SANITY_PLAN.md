# CIGL Phase 3A-R — Decision and Backbone-Sanity Plan

Decision record for the Phase 3A-R baseline-repair pilot
(`docs/CIGL_18_PHASE3A_R_RESULTS_BNCI2014_001.md`, job 876328). Records the Gate-3A-R verdict and the
next authorized step. Project record, not a results table.

## Gate-3A-R decision

- **Baseline repair: FAIL.** None of the six GraphCMINet-ERM candidates reached a non-degenerate
  source-only baseline (all `source_probe` bAcc ≈ **0.33**, below the **0.45** adequacy floor; 4-class
  chance = 0.25). `baseline_gate_pass = false`; Part B (gentle re-pilot) was **skipped**.
- **Controllability: still PASS** (from Phase 3A): graph/node/edge CMI regularization reduces leakage to
  ~0. Leakage remains strong in ERM here (graph KL 0.53–0.73, edge ~0.85–0.89).
- **Task-preserving CIGL method: still FAIL / not evaluable** — there is no credible task baseline on
  which to measure a task-vs-leakage tradeoff.

## Why

- All 6 candidates (`current_default`, `source_channel_zscore`, `stronger_graphcmi_backbone`,
  `lower_lr_longer` @150 ep, `no_classbal_sampler`, `ce_balance_check`) give `source_probe` bAcc ≈ 0.33.
- The sanity **controls pass**: `overfit_small_source` train bAcc = 0.531 (the architecture *can* fit a
  tiny subset), `label_shuffle_control` src bAcc = 0.256 (≈ chance — the probe is not cheating).
- Train bAcc is only ~0.39 with a tiny train−source gap (~0.05) ⇒ **GraphCMINet underfits** BNCI2014_001
  4-class MI; this is underfitting, not overfitting and not target/protocol leakage.
- `source_channel_zscore` was a **no-op** (the loader already trial-z-scores), so that "repair" couldn't
  have helped.

## NOT authorized

- A **full CIGL** method claim (needs a credible baseline).
- An **Edge-CMI** narrow method claim (also needs a baseline that can learn the task).
- **Full LOSO** (all folds), **SEED / DEAP**, a **large λ-grid**, or any **SOTA** table.
- Phase 3B or any method-paper framing.

## Next authorized direction — Phase 3A-S (backbone/protocol sanity)

> **The only authorized next work** is a **known-good MI decoder sanity check** on the **same** strict
> source-only BNCI2014_001 fold-0 protocol (target subject excluded from training and `source_probe`;
> target labels evaluation-only; success judged on `source_probe`). Branch:
> `project/cigl-phase3a-backbone-sanity`. It has its own dry-run + reviewer checkpoint before any GPU run.

Candidates (small, no new dependencies; minimal internal wrappers if the repo's braindecode-backed ones
are unavailable in the run env): GraphCMINet current reference, **EEGNet**, **ShallowConvNet**,
optionally DeepConvNet and an existing DGCNN/RGNN baseline. ERM only — **no CMI regularization**.

## Decision rule (after the Phase 3A-S real run)

- **Known-good decoder reaches `source_probe` bAcc ≥ 0.45** → the **protocol is usable** and
  **GraphCMINet is the bottleneck**: redesign/replace the graph task backbone (e.g. around a known-good
  temporal stem) **before** any CIGL regularizer claim.
- **All known-good decoders also stay near 0.33** → the **protocol / preprocessing / data split is
  suspect**: diagnose MOABB loading, time window, scaling, labels, and fold construction.
- **A known-good decoder succeeds but no graph-compatible model does** → the CIGL **method** path is
  paused; keep CIGL as a **diagnostic / audit** framework (Gate-2 leakage evidence) until a
  graph-compatible task backbone exists.

Only once a credible source-only task baseline exists may CIGL regularization be revisited. No
full-LOSO / SEED / λ-grid / Phase 3B until then.
