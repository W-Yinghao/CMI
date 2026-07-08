# Branch Reconciliation

Checked after review-completion audit commit `29a2195` and hygiene/digest commit `483ff8c`.

## Current Worktree

- local worktree: `/home/infres/yinwang/CMI_AAAI_qxu`
- local branch: `exp/h2cmi-wave0-mechanism`
- remote tracking branch: `origin/exp/h2cmi-wave0-mechanism`
- artifact bundle commit: `29a219596a976c1300443dfcd4e890d6010db7e9`
- hygiene/digest head: `483ff8c01c95d3083864f0f0c8b8243d271bdb75`
- current remote head: `483ff8c01c95d3083864f0f0c8b8243d271bdb75`
- analysis base commit: `283832710c93d56cca45b27682286e64e37a4034`

## Responsibility Branch Check

- expected project branch named by PM: `exp/h2cmi-responsibility-qxu`
- remote responsibility branch HEAD: `09e92499ecec9d245e12d92f2c3b355e8e1b93d1`
- `29a219596a976c1300443dfcd4e890d6010db7e9` is **not** an ancestor of `origin/exp/h2cmi-responsibility-qxu`.
- `483ff8c01c95d3083864f0f0c8b8243d271bdb75` is **not** an ancestor of `origin/exp/h2cmi-responsibility-qxu`.
- Conclusion: the review-completion package and hygiene digest exist on `exp/h2cmi-wave0-mechanism`, not on `exp/h2cmi-responsibility-qxu`.

## Exact Migration Commands If Approved

Do not run these without explicit approval.

Cherry-pick only the review-completion audit commit:

```bash
git fetch origin exp/h2cmi-wave0-mechanism exp/h2cmi-responsibility-qxu
git switch exp/h2cmi-responsibility-qxu
git cherry-pick 29a219596a976c1300443dfcd4e890d6010db7e9
```

Cherry-pick the review-completion audit plus hygiene/digest commits:

```bash
git fetch origin exp/h2cmi-wave0-mechanism exp/h2cmi-responsibility-qxu
git switch exp/h2cmi-responsibility-qxu
git cherry-pick 29a219596a976c1300443dfcd4e890d6010db7e9
git cherry-pick 483ff8c01c95d3083864f0f0c8b8243d271bdb75
```

Merge the source branch instead:

```bash
git fetch origin exp/h2cmi-wave0-mechanism exp/h2cmi-responsibility-qxu
git switch exp/h2cmi-responsibility-qxu
git merge --no-ff origin/exp/h2cmi-wave0-mechanism
```

Recommended PM choice: cherry-pick `29a2195` plus `483ff8c` if only the review-completion package and hygiene digest should move. Use a merge only if the full `exp/h2cmi-wave0-mechanism` history is desired on the responsibility branch.
