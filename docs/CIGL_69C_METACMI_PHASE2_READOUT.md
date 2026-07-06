# CIGL_69C — MetaCMI Phase-2 seed0 readout (MetaCE + MetaCMI-Direct β0.1)

```
84 runs: {EEGNetMini, EEGConformerMini} × {MetaCE, MetaCMI-Direct β0.1} × {BNCI2014_001 (9), BNCI2015_001 (12)},
seed0, full LOSO. ERM reused from frozen CIGL_69A comparators (metacmi_gate/); MetaCE+MetaCMI in isolated
metacmi_phase2/. Jobs 885957/885958 COMPLETED. Fixed-epoch (80), NO early stopping, all methods same train_model
path. Attribution order (PM): 1) MetaCE vs ERM, 2) MetaCMI vs MetaCE (the CMI-specific test), 3) MetaCMI vs ERM.
seed0 = scientific SCREENING (backbone/method-level judgment needs seeds 0/1/2).
```

## 0. Integrity — all 8 stop-conditions CLEAN
finite (all); feature_z(graph_z)+model_logits present; **exact classifier-level head-replay** replay_ok=True on
all 126 records (max|Δ|≤9e-6, removal_mode=head_replay); **random_subspace control ≈0** (−0.003…+0.002) and
stable; projector firewall recorded+true on all 84 meta records (fit meta_train subjects only; excludes
meta_heldout/pool/target; detached); same backbone path across methods (z_dim eegnet 192 / conformer 1696,
single-valued); same fixed-epoch(80)/no-early-stop protocol for MetaCE+MetaCMI (ERM ran the identical train_model
path). No stop-and-report trigger fired.

## 1. Target generalization (Δ vs comparators)
| dataset·backbone | ERM | MetaCE | MetaCMI | MetaCE−ERM | **MetaCMI−MetaCE** | MetaCMI−ERM |
|---|---|---|---|---|---|---|
| 2a·eegnet | 0.423 | 0.404 | 0.399 | −0.019 | **−0.006** | −0.025 |
| 2a·conformer | 0.411 | 0.405 | 0.405 | −0.006 | **−0.001** | −0.007 |
| 2015·eegnet | 0.637 | 0.635 | 0.632 | −0.002 | **−0.003** | −0.005 |
| 2015·conformer | 0.640 | 0.646 | 0.645 | +0.006 | **−0.001** | +0.005 |

- **MetaCE vs ERM:** helps target in only **1/4** cells (2015·conformer +0.006); flat/negative in 3/4. Source-
  episodic CE is not robustly beneficial.
- **MetaCMI vs MetaCE (the CMI-specific test): ≤ 0 in ALL 4 cells** (−0.001…−0.006). The CMI term adds nothing
  to accuracy over episodic CE — it marginally hurts.
- **MetaCMI vs ERM:** positive only 1/4 cells, and there MetaCE (+0.006) ≥ MetaCMI (+0.005).

## 2. Leakage / CMI audit (feature_z; NOT graph leakage — non-graph backbones)
featKL Δ vs ERM: MetaCE −0.027/−0.014/−0.005/−0.018; **MetaCMI −0.044/−0.039/−0.036/−0.031**. The CMI term does
reduce measured label-conditional subject leakage **more than MetaCE** (as designed) — but the reduction is small
(~2–4%) and leakage stays **significant (perm_p<0.05 on all folds)**: reduced, not controlled. No rebound. So the
β·SymKL term measurably acts on the leakage proxy, but the effect is tiny and does not propagate downstream.

## 3. R3 reliance (classifier-level head_replay, firewall=True)
| dataset·backbone | ERM k2 | MetaCE k2 | MetaCMI k2 | **MetaCMI−MetaCE** | MetaCMI−ERM | random_ctrl | ERM k8→MetaCMI k8 |
|---|---|---|---|---|---|---|---|
| 2a·eegnet | +0.032 | +0.034 | +0.031 | **−0.003** | −0.001 | ~0.002 | +0.113→+0.092 |
| 2a·conformer | +0.047 | +0.043 | +0.042 | **−0.001** | −0.005 | ~0.000 | +0.097→+0.094 |
| 2015·eegnet | +0.027 | +0.028 | +0.024 | **−0.004** | −0.003 | ~0.000 | +0.074→+0.064 |
| 2015·conformer | +0.057 | +0.064 | +0.063 | **−0.002** | +0.005 | ~0.000 | +0.075→+0.085 |

- ERM has a small-but-real, SPECIFIC classifier-level reliance (k2 +0.03…+0.06; random control ≈0; k8 larger).
- **MetaCMI − MetaCE (k2): −0.001…−0.004 in all 4 cells** — consistent DIRECTION but **10–20× within the fold-sd
  (0.05–0.07)** → statistically negligible.
- **MetaCMI − ERM (k2): mixed** (−0.001/−0.005/−0.003/**+0.005**) — increases on 2015·conformer, so not a
  consistent vs-ERM reduction. k8 likewise mixed.

## 4. Attribution decision — **FAIL** (clean negative, no CMI-specific seed0 signal)
Against the pre-registered tiers, per backbone and pooled (the pattern is uniform across both datasets, so pooling
hides nothing):
- **NOT Strong CMI pass** — MetaCMI ≤ MetaCE on target in every cell; > ERM only 1/4.
- **NOT Functional pass** — the R3 reduction vs MetaCE is within noise, and vs ERM it is *not* consistent (2015·
  conformer increases); leakage is only marginally reduced (still significant), not controlled.
- **NOT even Meta-only pass** — MetaCE itself improves target in only 1/4 cells, so episodic CE is not robustly
  positive either.
- ⇒ **FAIL.** MetaCMI does not improve target or reliance beyond noise; the only consistent effect is a tiny
  extra reduction of the measured leakage proxy that buys no accuracy and no functional reliance change. The
  audit is fully clean (classifier-level R3, random control valid, firewall enforced) — this is a **well-measured
  negative, not an artifact.**

### Recommendation (per the PM's fail rule)
- **Stop the source-episodic MetaCMI line.** No β=0.5 sweep, no more k / meta schedules / Conformer arms.
- **Do NOT request seeds 1/2** — none of the expansion triggers hold (MetaCMI never beats MetaCE on target; the
  R3 vs-MetaCE reduction is within noise, not "clear"; no backbone gives a clean interpretable CMI signal).
- **Scientific conclusion:** source-episodic MetaCMI on stable, auditable, leakage-rich, audit-compatible
  backbones (EEGNetMini + EEGConformerMini) shows **no seed0 CMI-specific signal** — the CMI term adds nothing
  beyond episodic CE. This is consistent with the static-DGCNN three-level negative (CIGL/FCIGL/dCIGL): controlling
  a measured-CMI proxy/objective does not yield functional benefit on real EEG. The measurement→control gap holds
  across both the graph and the source-episodic formulations.
- **Next setting (per the PM's tree), only if authorized:** move off source-only reliance control toward a
  target-unlabeled regime (CITA / TTA), where target-distribution information is available at adaptation time.
```
