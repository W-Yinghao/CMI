#!/bin/bash
# csc commit-separation guard (CSC-P1.4).
#
# A .gitignore does NOT stop ALREADY-TRACKED audit files from being re-staged by `git add csc/`,
# nor does a code commit's TREE stop inheriting a tracked audit from its parent. So:
#   * a CODE commit keeps its tree pure by UNTRACKING csc/results/ (git rm --cached) -> these
#     appear as DELETIONS (D) and are ALLOWED alongside code;
#   * adding/modifying audit files (A/M under csc/results/) mixed with any code change is
#     REJECTED (that was the stale-audit-in-code-commit bug);
#   * an AUDIT commit touches ONLY csc/results/ (force-added) -> allowed.
# Install: ln -sf ../../csc/tools/check_commit_separation.sh .git/hooks/pre-commit  (or CI).
set -e
ns=$(git diff --cached --name-status)
res_addmod=$(echo "$ns" | grep -E '^[AM][[:space:]]+csc/results/' || true)
other=$(echo "$ns" | grep -vE '[[:space:]]csc/results/' | grep -v '^$' || true)
if [ -n "$res_addmod" ] && [ -n "$other" ]; then
    echo "ERROR: commit ADDS/MODIFIES audit artifacts (csc/results/**) mixed with code." >&2
    echo "  audit add/mod:" >&2; echo "$res_addmod" | sed 's/^/    /' >&2
    echo "  other files:"   >&2; echo "$other"       | sed 's/^/    /' >&2
    echo "Commit code and audit artifacts SEPARATELY (CSC discipline)." >&2
    exit 1
fi
exit 0
