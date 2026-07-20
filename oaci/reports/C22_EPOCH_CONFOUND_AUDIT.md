# C22 — Epoch / trajectory-position confound audit

> Reported BEFORE any normalization-rescue interpretation (hard gate). T2 fires on TRAJECTORY (epoch/order) ONLY; training-log baselines are a separate source-observable overlap.

T2 (trajectory confound) fires ONLY if the probe fails to beat the epoch/order baselines OR the residual controlling epoch vanishes. Training-log baselines (R_src/train_surrogate) are a SEPARATE source-observable overlap: source_risk_overlap means a single source-risk scalar matches the probe (a low-dimensional / risk-family finding, echoing C17), NOT a trajectory proxy. Reported BEFORE any normalization-rescue interpretation.

- probe within-target strength: +0.659
- TRAJECTORY baseline strengths (epoch/order): {'epoch': 0.5545847694569417, 'order': 0.5545127963791404}
- probe beats trajectory: True
- partial Spearman(score, label | epoch): +0.225 (n_targets 9); residual present: True
- **epoch_confounded (trajectory): False**

## Separate source-observable overlap (NOT a trajectory downgrade)
- TRAINING-LOG baseline strengths (R_src/train_surrogate): {'R_src': 0.6690819429027496, 'train_surrogate': 0.6002453552973308}
- source_risk_overlap: True; partial Spearman(score, label | R_src): +0.163; probe adds over R_src: True
