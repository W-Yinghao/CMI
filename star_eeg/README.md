# STAR-EEG

**Source-Task Anchored Representation Reorganization for EEG**

> Can a source-only task anchor move an H200 foundation representation from
> subject-identifiable but task-floor to task-transferable, without requiring
> subject erasure?

This directory is an independent project scaffold based on the frozen S2P
dependency commit `a9134eb5eb7f8486a5e1ee41831823dab39381ed`.

STAR_00A froze project design, artifact inventory, deterministic schedules, and
synthetic red-team checks. STAR_00B adds SHA-named immutable H200 starts, the
complete FACED source_train-only manifest/stream, the exact H200 Route-B SSL
streams, and a bounded B/C/D ten-step real CBraMod CUDA smoke. STAR_00B does not
authorize any 3,750-step cell, FACED source_val/test access, target scoring, or
STAR_01.

STAR_00C adds launch-only hardening: per-step persistent telemetry, hashed run
manifests/summaries, atomic no-overwrite checkpoints, immutable attempt
directories, a commit/artifact-bound approval lock, and an executable six-cell
array-to-afterok-closure chain. It does not change any scientific
hyperparameter, variant, or gate threshold. Target scoring remains blocked.

Future STAR_01 is restricted to H200_s0/H200_s1 and the four frozen variants
H200_BASE, H200_SSL_CONT, H200_STAR_TRUE, and H200_STAR_SHUFFLED. All other S2P
budgets and the released checkpoint are descriptive frozen references only.

Run the low-cost checks from the repository root:

```bash
python -m compileall -q star_eeg
PYTHONPATH=. pytest -q star_eeg/tests
```

The STAR_00A artifact-aware preflight additionally needs the environment that
can load frozen CBraMod checkpoints and remains toy-only. STAR_00B uses the
dedicated source-only loader and bounded GPU runner; its real-path smoke is
explicitly separated from scientific training and downstream evaluation.

The STAR_00B artifacts and gate state are summarized in
`reports/STAR_00B_PREFLIGHT_READOUT.md`; STAR_00C is summarized in
`reports/STAR_00C_PREFLIGHT_READOUT.md`. Formal training refuses execution
unless a read-only SHA-named approval manifest binds the clean execution commit,
exact six-cell universe, and all required artifact/source hashes.
