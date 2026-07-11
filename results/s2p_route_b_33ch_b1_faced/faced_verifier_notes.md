# FACED Frozen-Probe Verifier Notes

- Phase: immutable H2000 re-audit.
- Dataset: FACED native32 LMDB, 10s, 200Hz, 9 classes.
- Split: train subjects 1-80, validation subjects 81-100, test subjects 101-123.
- Encoder is frozen; no fine-tuning or pretraining is launched by this script.
- PCA, classifier, subject subspace, and rank are source train/val only.
- FACED test labels are used only for final scoring.
- H2000_s0 and H2000_s1 use SHA-pinned, read-only immutable checkpoints.
- Random and released rows are reused from the D2-1 FACED audit.
- H4000, CodeBrain, fine-tuning, and any extra dataset are excluded.
- H500/H1000/H2000 comparisons are descriptive budget-floor calibration, not a monotonic scaling-law claim.
- Floor crossed by 1000h: True.
- Floor crossed by 2000h: True.
- Best descriptive budget: {'budget_h': 500, 'mean_target_kappa': 0.07235083341954623, 'mean_target_bacc': 0.1747181964573269, 'descriptive_only_not_optimality_claim': True}.
