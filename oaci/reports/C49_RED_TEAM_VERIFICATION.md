# C49 - Red-Team Verification

C49 red-team checks were run after artifact generation and before commit.

- self_neighbor_and_label_path_quarantine: pass - Local purity excludes self labels; target labels are diagnostic-only outputs.
- coverage_accounting: pass - Every reported coverage curve row carries coverage, empty fraction, and neighbor count.
- same_group_random_baseline: pass - Coverage rows compare local Bayes hit against covered candidates inside the same group.
- broad_vs_sparse_gate: pass - Reliability is evaluated under predeclared coverage thresholds 0.25/0.50/0.75.
- underuse_not_selector_repair: pass - Existing scores are audited against diagnostic islands without emitting selected artifacts.

Verdict: C49 is a diagnostic coverage audit; it does not emit a selected-checkpoint artifact.
