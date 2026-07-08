# SPDIM Clean-run Policy

Confirmatory SPDIM runs require a clean worktree.

Dirty-worktree launches are exploratory-only.

Full-run escalation is blocked unless the clean provenance gate passes.

The runner records launch commit, `git status --porcelain` output,
clean-worktree status, runner checksum, config checksum, external SPDIM commit,
environment name, command line, and Slurm job id. Future official SPDIM probe or
expansion runs must not use `--allow-dirty`.
