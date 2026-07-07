# FSR_47 — CodeBrain + CBraMod 8B Encoder-Audit Results (Phase 8B)

**Project FSR — Phase 8B.** Frozen-encoder FSR audit (protocol FSR_46) of two EEG foundation encoders on SHU-MI
(25 subj, 5 sessions, 2-class MI). After the PM revision, the audit uses **F1 = spatially-preserving per-channel
feature** (mean over patches, keep channels → C×200, PCA-95%/cap-128 on source-train) as the **main feature for
L4/L5/L6**, with **F0 (pooled) kept as L1-only**. Frozen CodeBrain (GPU) + CBraMod (control); target labels final
scoring only. Design-red-teamed (FSR_46) + adversarially self-checked.

## Headline — subject is strongly ENCODED but not RELIED-UPON, on a genuinely decodable task, in BOTH architectures
The spatial feature **recovers the MI task** (F0 pooled washed out C3/C4 lateralization); on this task-decodable
representation the finding is solid and architecture-general:

| SHU-MI (25 subj), F1 spatial — chance ⟨⟩ | CodeBrain | CBraMod |
|---|---|---|
| **task** source-val / target (F0 pooled was) | 0.609 / **0.605** (0.53) | 0.597 / **0.598** (0.548) |
| task **gate pass** (≥0.60/0.58 or +0.04 over F0) | **PASS** (+0.075) | **PASS** (+0.050) |
| **L1 subject decodability** ⟨0.04⟩, session-held-out, perm p | **0.628** (eff +0.59, p=0) | **0.562** (eff +0.52, p=0) |
| L1 **class-conditional** (beyond the task label) | 0.627 | 0.562 |
| **L4** task-head↔subject-subspace alignment (k=2) | 0.036 | 0.036 |
| **L5** reliance drop vs **variance-matched** null; beats it? | −0.0002 vs +0.0012, **No** | +0.0001 vs −0.0001, **No** |
| **L6** target consequence Δ (erase subject subspace) | +0.0012 (CI∋0) | +0.0014 (CI∋0) |

**Finding:** even on a **genuinely decodable** MI task (both gate-pass ≈0.60, near CodeBrain's fine-tuned 0.64),
both frozen foundation encoders **strongly and class-conditionally encode subject identity** (L1 0.56–0.63 vs 0.04
chance, perm p=0) **yet do not functionally rely on it**: erasing the subject subspace does **not** beat a
variance-matched erase (L5), and does **not** change the held-out-target task (L6≈0). It **replicates across two
different architectures** (EEGSSM/SGConv vs criss-cross transformer). This is the FSR thesis — **subject encoding ≠
harmful subject reliance** — established on frozen EEG foundation representations, on a real task.

## Resolution of the two prior open issues
- **Open issue 1 (weak task, STOP-5) — RESOLVED.** F0 pooled decoded MI at only 0.53/0.55; **F1 spatial recovers it
  to 0.60/0.60** (gate-pass both), confirming the weakness was a **channel-mean-pooling artifact**, not a frozen-
  encoder limit. L4/L5/L6 are therefore promoted from preliminary to **main conclusions** on a decodable task; the
  L1 finding (already solid on F0) is *stronger* on F1 (0.628/0.562).
- **Open issue 2 (CodeBrain determinism, STOP-1) — DOCUMENTED + CLAIM-INVARIANT.** CodeBrain `F0/F1` differ across
  batch sizes by only a tiny SGConv FFT numerical path: **F0 max-abs 1.8e-3** (median-trial-L2 6.7e-4, relative
  3.7e-5), **F1 max-abs 4.2e-3** (relative 2.0e-4); repeat-determinism exact (0.0); CBraMod fully deterministic
  (≤3e-7). The bs=64 vs bs=32 **claim-level invariance** is confirmed on **both** the F0 audit (L1 0.5907 vs 0.5914,
  Δ=0.0007; L6 base 0.5302 vs 0.5304) and the **F1 audit** (`codebrain_batch_invariance.csv`, `invariance_pass=True`):
  task Δ=0.0018, **L1 Δ=0.0025**, L4 Δ=0.0009, L5 Δ=0.0001, L6 Δ=0.0009 — all ≤ tolerance, and the **task-gate +
  L5-beats-variance-sign decisions are unchanged**. CBraMod additionally provides a fully-deterministic control, so
  no 8B conclusion rests on the CodeBrain numerical path. **STOP-1 disposition: documented + claim-invariant (PASS).**

## Firewall / discipline (clean)
`target_label_firewall.json`: L6 is the only target-label read (final scoring). PCA (95%/cap-128) fit on
source-train, task head selected on source-val, subject subspaces fit on source; z-score per-trial within-window;
no target label in feature extraction / PCA / head / probe / subspace / selection. Per-dataset (2-class only here);
L5 nulls = variance-matched + oracle-task with removed-variance reporting; L1 perm nulls (1000×, marginal +
class-conditional).

## 8B closure addendum (PM-required before 8C execution) — all clear
- **F1 batch-invariance (STOP-1) — PASS** (§ above): every claim-level metric batch-size-invariant.
- **Temporal-token side-check (one-time, no tuning; STOP-6) — CONFIRMED COLLAPSED.** The pretrained temporal
  codebook emits **1 unique code across 1000 real SHU trials (~305k tokens)** AND on the canonical 30-patch BNCI
  window, while the **frequency** codebook uses **4009/4096** codes. `temporal_token_status=collapsed` → temporal
  tokens **excluded from the main analysis**; CodeBrain's temporal-token interpretability claims are **not**
  transported (C23). Frequency-token diagnostic usable (secondary).
- **BNCI2014_001 alignment sanity — done, recorded as a limitation.** **L1 subject-encoding replicates
  cross-dataset** (strong: CodeBrain 0.586 / CBraMod 0.697 vs 0.111 chance). But the **frozen 4-class 2a task is
  weak** (target 0.267 / 0.320 vs 0.25) so the **task gate FAILS** → L4/L5/L6 are *not* concludable on BNCI (frozen
  4-class MI needs fine-tuning; the CodeBrain/CBraMod papers full-fine-tune for 2a). Per PM, this is a small-subject
  / weak-frozen-task **sanity limitation**; the F1 pipeline + channel mapping are reproducible on BNCI, and it does
  **not** rewrite the SHU-MI conclusion. **No stop rule hit.**

## Gate status + 8C readiness
`measurability = PASS`; **F1 task gate PASSES on BOTH encoders**; firewall clean; determinism invariant (CodeBrain)
+ CBraMod deterministic. **8B closure complete (invariance PASS + temporal collapsed + BNCI sanity; no stop rule).**
Per FSR_46 / PM, all 8C conditions are **met** and PM authorized 8C execution after clean closure →
**`proceed_to_8c = True`.** 8C = `N_source ∈ {2,4,8,16,all}` subject-scaling on PhysioNetMI (109), frozen
encoder + F1 spatial + source-val head, fixed-vs-growing-trials, per FSR_48 (design-red-teamed). Next = 8C-0
(PhysioNetMI manifest + source-subset plan); specialist baselines deferred to 8C-2 (return for PM review first).
PC2 paused; Paper 1 unaffected; Paper 2 frozen (Phase 8 is a separate axis).

## Deliverables (`results/fsr_codebrain_cbramod_8b/`)
F0/F1 `.npz` (both encoders, +bs32 invariance), `feature_dump_manifest_*.json`, `f0`/`f1` audit summaries,
`f1_task_gate.csv`, `f1_l1_subject_probe.csv`, `f1_l4_task_alignment.csv`, `f1_l5_replay.csv`,
`f1_l6_target_consequence.csv`, `codebrain_batch_invariance.csv`, `codebrain_cbramod_8b_verdict.json`,
`target_label_firewall.json`.
