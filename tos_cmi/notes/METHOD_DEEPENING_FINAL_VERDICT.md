# Method-deepening line --- FINAL VERDICT (frozen: tag tos-cmi-method-deepen-v2-final @ 0ef7be3)

The method-deepening line is COMPLETE and frozen. No further experiments; target-informed branch parked;
Track E not supported by current evidence; TSMNet World A not redesigned. This doc consolidates the three
sub-results and the honest caveats. Per-sub-result detail: notes/REAL_EEG_VALIDATION.md (Track B),
Phase-2 dry-run (commit f8bd0ef), notes/V2_*.md (V2 Stage 1/1B/2).

## Headline
Conditional domain leakage / subject erasure is measurable and sometimes removable, but removal is NOT a
reliable DG control lever. A strict source-only selective-invariance controller should be **refusal-first**: it
can reliably REJECT unsafe/useless interventions, but it CANNOT certify deployment-shift benefit when that
benefit is not visible in the source domains. The source-only gate has **safe refusal power** and exposes an
**acceptance ceiling** --- NOT acceptance power.

## 1. Track B --- source-OOD benefit gate on real EEG
Three frozen source-only layers (safety task-drop UCB<=0.02 REJECT; source leave-one-source-subject-out benefit
LCB>+0.01 ACCEPT; domain-gain diagnostic-only). Target used ONLY in post-hoc audit.
* Lee2019/Cho2017 sampled gate (15 of 52/54 folds; labeled pilot, not full LOSO).
* **0 false accepts; 8/8 harmful erasures prevented; 20/20 correct decisions.**
* Naive controllers on the SAME source signals FALSE-ACCEPT: domain-gain-only accepts 16 (8 harmful);
  always-erase-if-domain-gain accepts 20 (8 harmful); safety-only accepts 4. Our benefit+safety gate accepts 0.
* **Acceptance power UNTESTED on real EEG** (no real positive exists there).

## 2. Phase 2 --- task-preserving / conditional erasure (Lee/Cho EEGNet)
* tp-LEACE (LEACE on the task-orthogonal complement) FIXES plain-LEACE's task->chance collapse: it PRESERVES
  the deployed task decision (source task-drop UCB +0.000; argmax literally unchanged; MLP-robust) and still
  ERASES source subject (subject decode 0.31->0.03).
* Yet TARGET transfer stays FLAT (target dbAcc +0.000). The gate ABSTAINS (safe, but no source-OOD benefit).
* Disclosed: cc-predicted's exact-zero is a STRUCTURAL TAUTOLOGY (the probe re-learns the router); tp-LEACE is
  the clean result; V2 uses a probe-independent fair_conditional eraser.

## 3. V2 --- source-only acceptance CEILING (semi-synthetic: real latents + injected ground-truth nuisance)
Worlds A (target-beneficial -> expect REJECT/ABSTAIN, the ceiling), B (task-entangled -> REJECT), C
(removable-useless -> ABSTAIN). Nuisance dim normalized to latent capacity m=max(4,round(0.20*z_dim)).
Stage 2 scoped (5-shard, 72,000 tasks, 0 fail, 0 degenerate; config b8e24e34fc84):
* **EEGNet World A = CLEAN ceiling, robust** across source_subject_counts {8,16,32,all} x seeds {0,1,2}: clean
  target-beneficial-but-source-uncertifiable cells at EVERY n_source (51/67/65/53), principled source-only gate
  0 ACCEPT, oracle-supported, random-k never reproduces (LCB max +0.0057).
* **World B/C = robust refusal on BOTH backbones**: 0 principled ACCEPT (unsafe erasers REJECTed;
  high-domain-gain-useless cells abundant, 530/546).
* **0 false accepts across all 72,000 cells.** Naive source-only controllers false-accept 1807-3368; the ORACLE
  target-informed selector picks 284 true beneficial / 0 false --- proving the ceiling is a SOURCE-ONLY limit
  (benefit exists; source evidence can't certify it), not an absence of benefit.
* The gate's ACCEPT branch is LIVE (proven) but never fires because source-LOSO benefit is never >+0.01.

## 4. Honest caveats (do NOT overclaim)
* **TSMNet World A is NOT cleanly demonstrable** under the appended-nuisance construction: the injected-nuisance
  oracle is not target-beneficial at any nuisance_fraction {0.15..0.30} (best +0.009, LCB<+0.01, flat in m); the
  RLACE-carried positives are not injected-nuisance-specific. A high-dimensional-latent limitation, kept honest.
* The source-only gate does NOT have acceptance power; it has safe refusal power + an acceptance ceiling.
* Target-informed acceptance ("how much target information crosses the source-only ceiling?") is FUTURE WORK,
  parked, not implemented.
* Track E (end-to-end selective training) is not supported by current evidence (no source-visible positive).

## Three-layer final claim
| layer | final state |
|---|---|
| Real EEG deployment | source-fitted erasers (incl LEACE) yield no practically meaningful target-bAcc gain; some harm the task |
| Source-only controller | benefit+safety gate: 0 false-accept; reliably rejects real EEG's harmful/useless erasures |
| Semi-synthetic ceiling | EEGNet source-only acceptance ceiling robust; B/C refusal robust both backbones; TSMNet World A not clean (caveat) |
