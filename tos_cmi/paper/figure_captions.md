# Figure & table captions (draft) — allowed wording; claim tags in [Cx]

## Figure 2 — Synthetic certification: geometry alone is not safety.
Certification on synthetic Gaussian-mixture controls with a known Bayes oracle (R=30, β=0.2).
**(A)** Certification frontier: the gate fires on real leakage (oracle detection ≈1 at the boundary)
while the plug-in estimator is conservative, and neither fires under the no-leakage null [C3].
**(B)** A weak nested critic *unsafe-accepts* a conditionally-unsafe deletion — its probe task-risk UCB
sits far below the true Bayes task gap (the accepted point, in red) [C2, C3]. **(C)** The estimator gap
(Bayes − probe UCB) shrinks with calibration sample size; the unsafe-accept occurs only at the smallest n.
**(D)** With the cross-fitted plug-in estimator and a power floor, the certified gate produces **zero**
unsafe-accepts but is conservative (it abstains or rejects), i.e. it defaults to refusal rather than
weak invariance [C3]. *(The score-Fisher-vs-mean-scatter detection on covariance-only leakage is
established qualitatively in the text; not plotted here.)*

## Figure 3 — Global LPC removes subject leakage only by collapsing the representation (TSMNet/2a).
**Raw global LPC removes subject decodability only by collapsing the latent representation to the origin;
when this collapse path is blocked or avoided, task performance recovers but subject leakage returns** [C5, C6].
Per-epoch curves (median over folds×seeds): **(A)** task cross-entropy rises to ln 4 (chance) for λ≥1;
**(B)** the feature norm ‖Z‖ goes to 0 (collapse to the origin); **(C)** the λ·LPC penalty goes to 0
(trivially satisfied by Z→0). This is an **objective-scaling / training-basin pathology, not a gradient
explosion** — the diagnostic encoder gradient is about an order of magnitude *smaller* than in healthy
low-λ training and is never non-finite, and the apparently stable effective rank is a scale-invariant-metric
artifact. **(D)** Outcome bars: raw LPC at λ=1 drives both source task and subject decode to chance (Z→0
collapse); a warm-up schedule or scale-normalized penalty prevents the collapse, restoring task accuracy —
but subject leakage then returns to the ERM level (leakage is *not* removed without collapse) [C6].

## Figure 4 — TSMNet subject leakage is high-dimensional and redundant.
ERM TSMNet latent (z_dim=210; n=27 folds×seeds, 3 seeds). **(A)** Subject decode for the full latent Z,
the complement RZ after deleting the score-Fisher subspace V_D, the deleted subspace V_D itself, and a
same-k random removal (linear and MLP probes; chance dashed). Deleting V_D only **dents** subject decode
and essentially matches random removal [C4]. **(B)** Task decode is preserved after deletion. **(C)** The
subject decode *removed* by the informed V_D versus same-k random deletion is small and comparable — the
deletion is **not selective** here. **(D)** The candidate rank nDcand (≈3 of 210) is compact, but
low-rank deletion is nonetheless insufficient because subject identity is redundantly re-encoded across the
high-dimensional latent [C4].

## Figure 5 — EEGNet contrast: low-rank deletion removes leakage, but removal yields no DG gain.
ERM EEGNet latent (z_dim=16) and its global-LPC sweep. **(A–C)** Same axes as Fig 4: deleting V_D
**selectively removes** subject decode far below same-k random removal (linear 0.82→0.35, MLP 0.88→0.54)
while preserving the task — an order-of-magnitude larger informed-vs-random selectivity than TSMNet [C7];
a nonlinear residual remains (RZ MLP 0.54 ≫ chance), so the removal is partial [C8]. **(D)** Global LPC
reduces subject decode (0.89→0.18) **without** collapse (the feature norm never goes to 0), yet **mean**
LOSO target accuracy is flat-to-worse across λ (0.43→0.39, paired-t worse, p≤0.001) — removing the leakage
**does not improve target generalization** [C9]. (Curves show the mean over folds×seeds, the standard DG
metric.)

## Table 1 — One-glance summary across representations.
Columns: backbone; latent dim; ERM subject decode; low-rank deletion effect (subject; task); **raw LPC
outcome**; **collapse-free LPC outcome**; **DG gain under task-preserving leakage control**; certified
decision. TSMNet: deletion dents only; raw LPC reduces leakage via collapse (task→chance); collapse-free
LPC restores task but leakage returns to ERM; no task-preserving leakage reduction exists → **abstain**.
EEGNet: deletion removes leakage (partial, selective) at no task cost; raw LPC is collapse-free; target
accuracy flat → **diagnostic deletion that removes leakage but is not claimed to improve DG** [C4–C10].
(The "n/a" in EEGNet's collapse-free column means raw LPC is already collapse-free, not a missing result.)
