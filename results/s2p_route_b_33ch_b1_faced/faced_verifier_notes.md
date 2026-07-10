# FACED Frozen-Probe Verifier Notes

- Phase: D2-fleet
- Dataset: FACED native32 LMDB, 10s, 200Hz, 9 classes.
- Split: train subjects 1-80, validation subjects 81-100, test subjects 101-123.
- Encoder is frozen; no fine-tuning or pretraining is launched by this script.
- PCA, classifier, subject subspace, and rank are source train/val only.
- FACED test labels are used only for final scoring.
- H2000, CodeBrain, and any extra dataset are excluded.
- Channel-name metadata was not present in the selected local LMDB tree; the native array order is pinned by hash.
- Random target kappa: 0.028537; target bAcc: 0.137144.
- Released target kappa: 0.075472; target bAcc: 0.176329.
