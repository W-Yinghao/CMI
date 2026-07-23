# CMI-Trace Task 7 — cross-backbone AMOUNT vs USE — PRE-REGISTRATION (frozen before the fleet)

**Question.** Does the four-object separation — subject identity is *encoded* (**Amount**, λ) in the frozen
representation but is *not functionally load-bearing* for the task classifier (**Use**, τ) — replicate across
**both** paper backbones (EEGNet d_z=16, TSMNet d_z=210), i.e. is it not an EEGNet artifact?

**Method (ported E1).** For each frozen LOSO dump we run the tested whitened subject-spectrum
`cmi.eval.subject_spectrum.subject_spectrum` (SOURCE-only fit; pooled within-class Σ_W^{-1/2} whitening;
between-subject-within-label scatter S_B → top-K directions). Per direction j:
- **Amount λ_j** = null-calibrated conditional-subject info of the 1-D whitened projection (`flat_conditional_cmi`,
  retrained within-label permutation null): `lambda_excess_over_null`, `lambda_perm_p`.
- **Use τ_j** = CE reliance of removing the raw direction u_j via a **replay-exact linear head**, minus a
  same-rank-1 **random-direction control** τ_random (n_random directions).

**Use head (disclosed asymmetry).**
- PRIMARY, both backbones: a SOURCE-fit multinomial-logistic **probe head** → representation reliance
  (linear ⇒ replay-exact by construction ⇒ comparable across backbones).
- VALIDATION, TSMNet only: the **exact deployed head** recovered from stored logits (lstsq; replay max|Δ|~1e-6).
  EEGNet's stored 16-d penultimate→logits map is NON-linear (replay ~2 nats, QC-measured) so its deployed head
  is not exact-recoverable → EEGNet Use is representation-level **only**. This is a disclosed limitation, not a
  bug; the probe head makes the cross-backbone comparison apples-to-apples.

**Disclosed probe (single-dump, n_perm=10).** Ran ONE EEGNet + ONE TSMNet Lee2019 dump before freezing:
firewall passed, replay verified, probe-τ ≈ exact-τ on TSMNet (top 0.019/0.013 vs 0.015/0.011; both USE>random
2/8), λ excess>null positive (0.09–0.23), τ tiny (|τ|≈0.002/0.006) and ≤ τ_random. n_perm=10 cannot reach
p<0.05 (min p≈0.09) → the fleet uses **n_perm=100**. Freezing this analysis before viewing any aggregate.

## Scope (pre-committed)
- Datasets: **Lee2019_MI, Cho2017, BNCI2015_001** (the paper's Tier-1 / deployment sets).
- Backbones: **EEGNet, TSMNet**. Method: frozen **ERM** (`erm_lam0_seed0`).
- Folds: **first-12 LOSO folds per (dataset, backbone)** (balances the 3 datasets; BNCI2015 = full 12).
  Disclosed cap; 72 dumps total. Reason-code any dropped dump; refuse to aggregate under-coverage.
- Spectrum: `k_spec=16` top directions, `n_perm=100`, `n_random=50`, `shrink="lw"`, `seed=0`.

## Primary endpoints (per backbone; pooled over folds×datasets, fold = cluster bootstrap unit, n_boot=10000)
- **E7.1 Amount present** — fraction of scored directions with `lambda_perm_p < 0.05`. Prediction: **> 0**
  (subject info is encoded), and > a rank-matched Amount-null.
- **E7.2 Use ≈ random / absent** — paired `τ − τ_random` over directions: point + 95% CI, plus mean `|τ|` in
  nats. Prediction: **CI includes 0 or is negative** (Use not above a random rank-1 removal) and `|τ|` tiny.
- **E7.3 Decoupling** — `corr(λ_j, τ_j)` across directions. Prediction: **CI includes 0** (λ-ordering carries
  no reliance information).
- **E7.4 Cross-backbone consistency** — E7.1–E7.3 hold for **both** EEGNet and TSMNet.
- **E7.5 Probe validity (TSMNet)** — `corr(τ_probe, τ_exact)` and mean|τ_probe − τ_exact|. Prediction: high
  corr / small gap (probe reliance tracks deployed reliance).

## Pre-committed interpretation grid
- **λ significant + τ ≤ random on BOTH backbones** → four-object separation **replicates cross-backbone**
  (Amount ≠ Use is not an EEGNet artifact). Expected/supporting outcome.
- **Some backbone: τ ≫ random for high-λ directions** → Use tracks Amount there → separation is
  **backbone-dependent** (falsifies "not an artifact"); report which backbone and do not generalize.
- **λ not significant on a backbone** → Amount **underpowered / weak encoding** on that backbone (measurement
  caveat), NOT evidence against the separation; report as inconclusive-Amount.
- **E7.5 fails (probe ≠ exact on TSMNet)** → the probe-head Use proxy is invalid → down-weight EEGNet Use;
  report TSMNet exact-head result as primary.

## QC sentinels (halt/flag)
- `firewall_passed` must be True on every dump (SOURCE-only fit; target rows eval-only, distinct domain tag).
- `head_replay_verified` True on every dump (probe head replay-exact; TSMNet exact head max|Δ|≤1e-5).
- Determinism: same dump + seed → bit-identical spectrum.
- Coverage: 72/72 dumps or reason-code the gap; proper unique count, not string-sort.
- No silent `except→None`: every dropped direction/dump reason-coded.

## Claims discipline
- "encoded but not used" only when λ significant AND (τ ≤ τ_random). Do NOT say "subject info is unused" if λ
  is merely non-significant (that is underpowered Amount, not absent Use).
- EEGNet Use is **representation-level** (probe head), TSMNet Use is **deployed-classifier-level** (exact head);
  never conflate. Cross-backbone claim rests on the common probe-head measure.
- This is a cross-backbone REPLICATION of the separation, not a new mechanism.
