# FSR_52 — Trained-Representation Subject-Scaling Results (Phase 8D-0) + Phase 8 close-out

**Project FSR — Phase 8D-0.** Result of the **metric-power gate** (protocol FSR_51 v2; the red-team-mandated first
stage). CBraMod, PhysioNetMI, F1 spatial → fixed PCA-128, N_source=8, subset_seeds=3, train_seeds=5. Arms **A0
frozen** vs **A1-SE** (subject-erase positive control: bottleneck adapter `z'=LayerNorm(z + W2·relu(W1·z))` + linear
task head + **adversarial subject classifier via gradient reversal** on the bottleneck code `h=relu(W1·z)`). Primary
metric = **mean pairwise subject separability on `h`** for source subjects **held out** from adapter training. No
target labels (the 15-subject PhysioNetMI target panel is excluded entirely).

## Result — the metric moves, but too little: gate FAILS → STOP Phase 8
| A1-SE subject-erase vs A0 frozen (held-out source, bottleneck `h`) | value |
|---|---|
| A0 frozen held-out L1 (ceiling) | **0.948** |
| A1-SE held-out L1 | 0.872 |
| **drop (held-out)** | **+0.076 [0.054, 0.097]** — significant (CI excludes 0) |
| **relative drop** | **~8%** |
| pre-registered gate | ≥0.10 abs **OR** ≥20% rel |
| in-sample L1 (A0→A1-SE) | 0.94 → **0.98** (rises — adapter memorizes training subjects) |
| source-val task | 0.60 (not collapsed) |
| **`metric_power_gate_pass`** | **False** |

**Reading:** an **explicit adversarial subject-erasure** — the strongest push this design allows — reduces held-out
bottleneck-code subject separability only **~8%** (0.948→0.872), a **small, statistically significant, but
sub-threshold** move; and it **raises** in-sample separability (the adapter memorizes the training subjects). The
~0.95 subject fingerprint is so dominant, and a light adapter on a **fixed** PCA so limited, that the **bottleneck-
code L1 metric has insufficient power to resolve a representation-level subject-diversity effect** against this
ceiling. Per the pre-registered gate (FSR_51 v2, PM), **8D-1 does not run and Phase 8 STOPS** — this is a
**method-limit** result (`l1_untestable_metric_saturated`), **not** a foundation-model null, and per STOP-1 we do
**not** search for another adapter architecture.

## Phase 8 — final close-out (what it does and does not establish)
- **8B (encoder audit, decodable SHU-MI task, two architectures):** frozen CodeBrain and CBraMod **strongly and
  class-conditionally encode subject identity** (L1 0.63/0.56 vs 0.04), yet the subject subspace is **not a task
  lever** — erasing it does not beat a variance-matched null and does not change the target. *Subject encoded ≠
  harmful reliance*, architecture-general. (Frozen determinism claim-invariant; temporal-token collapse disclosed.)
- **8C (frozen scaling boundary, PhysioNetMI, weak task):** across source-subject count, the **subject subspace is
  never a task lever** (extends 8B); transfer gains are a **sample-size** effect; the frozen full-pool-PCA design
  **cannot** test whether diversity changes the representation.
- **8D-0 (trained-representation probe):** a subject-erasure positive control moves the bottleneck-code L1 only ~8%
  (sub-threshold) — the **light-adapter design lacks the metric power** to test representation-level diversity.

**Frozen 8C-1 wording still holds (do not exceed):** *"In frozen EEG foundation embeddings, subject structure is
highly separable but the measured subject subspace is not a functional task lever across source-count scaling; the
gain is a source-sample-size effect, and this design cannot test whether subject diversity changes the
representation."* Now extended: *"and a light trained adapter does not give the bottleneck-code separability metric
enough power to test it either."*

**Forbidden (permanent):** "subject diversity does/does not reduce subject leakage / separability"; "foundation
encoders become subject-invariant"; any diversity-erasure claim; SOTA/full-FT/leaderboard framing.

## Status
Phase 8 **stops** here (8D-0 gate fail; no 8D-1, no A2, no full FT, no specialist baselines, no new datasets). The
Phase-8 result is a coherent **FSR extension** (frozen + lightly-trained EEG foundation encoders encode subject
identity strongly but the measured subject subspace is not a task lever; representation-level diversity is not
testable under these frozen/light-adapter protocols) — likely an **appendix/extension**, not a standalone main
paper, pending a *trained-encoder* study (out of scope). PC2 paused; Paper 1 (Prior-Decoupled TTA) unaffected;
**Paper 2 frozen.**

## Deliverables (`results/fsr_trained_rep_scaling/`)
`trained_rep_metric_power_gate.csv` (15 cells: A0/A1-SE held-out + in-sample L1, source-val), `adapter_architecture_manifest.json`,
`trained_rep_metric_power_verdict.json` (`metric_power_gate_pass=false`, `adapter_signal=l1_untestable_metric_saturated`,
`stop_phase8=true`), `trained_rep_target_label_firewall.json`.
