# C43 Red-Team Verification

Scope: C43 Source-Objective Scalarization Frontier / Escape-Hatch Closure Audit.

## Checks

- Artifact scope: C43 reads committed C30/C41/C42 artifacts only. No training, GPU, score tuning, feature selection, or reinference path is used.
- Registry freeze: source-objective registry and the 103-row scalarization grid are fixed before analysis.
- Hindsight boundary: the best grid row is labeled as a hindsight diagnostic ceiling. It is not a method artifact.
- Multiplicity: top1 joint-good gains are tested against trajectory-conditioned random baselines with Poisson-binomial p values and Holm/BH correction.
- Stability: per-target AUC sign consistency is required before any positive scalarization claim.
- Base-rate check: every top-k result carries a trajectory-conditioned random baseline.
- Pareto-front caveat: source Pareto front is very broad, so F1 means target-good candidates are source-front feasible, not localized by the frontier.
- Method boundary: C43 emits no candidate ids, checkpoint hashes, model hashes, or selected-checkpoint artifact.

## Verification

- C43 focused Slurm job `890129`: `9 passed in 0.20s`.
- C23-C43 regression Slurm job `890130`: `192 passed in 32.00s`.

## Verdict

C43 is internally consistent with its gates. The accepted candidate taxonomy for remote review is conservative: `F1 + F3 + F4 + F5 + F7 + F8`; `F2`, `F6`, `F9`, and `F10` remain inactive.
