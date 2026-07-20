# C24 — Target-unlabeled gauge audit (R3/R4): REQUIRES RE-INFERENCE

> Status **computed**. R3/R4 are NOT computed read-only and are NOT proxied from method-final checkpoints (wrong population), and C24 is NOT finalized without them.

## Why re-inference is required
- The per-target offset is defined over the ~60 feasible-OACI CANDIDATE checkpoints per seed×target. The committed artifacts cache target logits only for method-final checkpoints (216 `target_audit.npz`; example `seed-0/target-001/artifacts/6a946cb12ac4c03ca21e90ab8ad2f9e3294d5039df712acac923590019e05466/levels/level-000/methods/ERM/target_audit.npz`). cached target logits are METHOD-FINAL checkpoints (~4 per seed x target x level), NOT the ~60 per-seed x target feasible-OACI CANDIDATE checkpoints the offset is defined over -- using them as R3/R4 would swap the population; REFUSED as science.
- The C18 extract holds source logits only; the C10 replay holds a `target_pred_hash` + label-dependent target scalars only. No per-candidate target-unlabeled confidence geometry exists.

## Planned R3 target-unlabeled features (label-free)
target_entropy_mean, target_entropy_std, target_confidence_mean, target_confidence_std, target_margin_mean, target_margin_std, target_logit_norm_mean, target_logit_norm_std, target_pred_prop_c0, target_pred_prop_c1, target_pred_prop_c2, target_pred_prop_c3

## C24-R3R4-P0 replay-identity smoke gate (must pass before full re-inference)
- G1 overlapping-checkpoint target logits match cached method-final within declared tolerance
- G2 sample IDs / order match `target_audit.npz`
- G3 checkpoint hashes match the manifest
- G4 repeated forward is deterministic
- G5 no target labels read by the R3/R4 feature builder
- G6 no target-derived endpoint metric computed in R3/R4
- G7 no selected-checkpoint artifact
- G8 target labels joined only later for diagnostic validation / offset evaluation

## Then
- full no-retraining target-audit re-inference over the feasible-OACI candidate population → per-candidate target logits → R3 (target-unlabeled gauge) + R4 (source + target-unlabeled) → final C24 taxonomy.
- R3/R4 are target-unlabeled TRANSDUCTIVE DIAGNOSTIC rungs: not source-only, not deployable, not DG success.
