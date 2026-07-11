# STAR-EEG

**Source-Task Anchored Representation Reorganization for EEG**

> Can a source-only task anchor move an H200 foundation representation from
> subject-identifiable but task-floor to task-transferable, without requiring
> subject erasure?

This directory is an independent project scaffold based on the frozen S2P
dependency commit `a9134eb5eb7f8486a5e1ee41831823dab39381ed`.

The current scope is **STAR_00A only**: project design, artifact inventory,
deterministic schedule and manifest implementation, synthetic smoke tests, and
red-team preflight. It does not authorize or run real STAR EEG training, FACED
target scoring, SLURM jobs, or STAR_01.

Future STAR_01 is restricted to H200_s0/H200_s1 and the four frozen variants
H200_BASE, H200_SSL_CONT, H200_STAR_TRUE, and H200_STAR_SHUFFLED. All other S2P
budgets and the released checkpoint are descriptive frozen references only.

Run the low-cost checks from the repository root:

```bash
python -m compileall -q star_eeg
PYTHONPATH=. pytest -q star_eeg/tests
```

The artifact-aware preflight additionally needs the environment that can load
the frozen CBraMod checkpoints. It performs read-only hashing/loading and a toy
smoke only; it does not read real EEG arrays.
