# FACED Frozen-Probe Verifier Notes

- Current load-bearing phase: final confirmatory verification through 1000 h.
- Dataset: FACED native32 LMDB, 10s, 200Hz, 9 classes.
- Split: train subjects 1-80, validation subjects 81-100, test subjects 101-123.
- Encoder is frozen; no fine-tuning or pretraining is launched by this script.
- PCA, classifier, subject subspace, and rank are source train/val only.
- FACED test labels are used only for final scoring.
- Random, released, H200, H500, and H1000 committed metrics reproduce exactly.
- Final uncertainty uses 5000 paired cluster-bootstrap draws over 23 FACED test subjects.
- All 6 pretrained cells pass the frozen source-val task gate.
- No cell's subject-subspace intervention exceeds the source-val-energy-matched random null after Holm correction.
- Historical rank-50 null outputs also reproduce exactly.
- D2-2 H2000 rows are not load-bearing: they were computed while H2000 training was still overwriting `best.pth`.
- H2000 requires completed, SHA-pinned checkpoints and a fresh audit before interpretation.
- H4000, CodeBrain, fine-tuning, and any extra dataset are excluded.
- H500 and H1000 clear the random baseline under paired target-subject uncertainty; H200 does not.
- The valid budget means are non-monotone, so no scaling-law or optimal-budget claim is licensed.
