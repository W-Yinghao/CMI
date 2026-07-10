# FACED Frozen-Probe Verifier Notes

- Phase: D2-2 H2000 update.
- Dataset: FACED native32 LMDB, 10s, 200Hz, 9 classes.
- Split: train subjects 1-80, validation subjects 81-100, test subjects 101-123.
- Encoder is frozen; no fine-tuning or pretraining is launched by this script.
- PCA, classifier, subject subspace, and rank are source train/val only.
- FACED test labels are used only for final scoring.
- H2000_s0 and H2000_s1 were added to the existing through-1000h audit tables.
- Random and released rows are reused from the D2-1 FACED audit.
- H4000, CodeBrain, fine-tuning, and any extra dataset are excluded.
- H500/H1000/H2000 comparisons are descriptive budget-floor calibration, not a monotonic scaling-law claim.
- Floor crossed by 1000h: True.
- Floor crossed by 2000h: True.
- Best descriptive budget: {'budget_h': 2000, 'mean_target_kappa': 0.0789308315851821, 'mean_target_bacc': 0.18008588298443373, 'descriptive_only_not_optimality_claim': True}.
