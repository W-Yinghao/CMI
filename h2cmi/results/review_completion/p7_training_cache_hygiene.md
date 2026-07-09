# P7 Training Cache Hygiene

- cache_path: `results/h2cmi/p7_w1_repaired_bundles/`
- cache_inside_git_root: `True`
- file_count: `549`
- total_bytes: `57224235`
- moved_or_preserved: `moved_outside_git_root_preserved`
- new_cache_path_if_moved: `/home/infres/yinwang/.cache/h2cmi_training_caches/p7_w1_repaired_bundles_bc61ee1`
- committed_to_git: `False`
- worktree_clean_after_hygiene: `True`

A sha256 manifest was recorded before moving the cache. The cache was moved outside the git repository root and was not committed. No broad `git clean -fd` command was used.

## Red Team Review

- The P7 training cache is preserved outside the repo, not deleted.
- The official P8 GPU launch must still happen from a clean post-P8A commit worktree.
- This hygiene step does not approve SPDIM seeds 1/2 or full SPDIM.
