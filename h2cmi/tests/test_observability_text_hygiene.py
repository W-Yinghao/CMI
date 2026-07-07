"""Project A — text-hygiene tests: source/doc files use LF and are not single-giant-lines.

These pass on the current tree (the files already use LF); they exist to PREVENT regression
(e.g. a CRLF/CR-only commit, or a tool collapsing a file into one huge line) that would make the
files unreviewable in a diff / on GitHub. Run:

    python -m h2cmi.tests.test_observability_text_hygiene
"""
from __future__ import annotations

import glob
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

_PATTERNS = [
    "h2cmi/observability/*.py",
    "h2cmi/run_real_audited.py",
    "h2cmi/run_real_audited_grid.py",
    "h2cmi/tests/test_observability_*.py",
    "h2cmi/tests/test_eval_bridge_harness_contract.py",
    "notes/project_A_observability/*.md",
    "notes/project_A_observability/results_summaries/*.md",
    "scripts/project_A_*.slurm",
]


def _paths():
    out = []
    for pat in _PATTERNS:
        out += [Path(p) for p in glob.glob(str(_REPO / pat))]
    return sorted(set(out))


def test_project_a_files_use_lf_not_cr_only():
    for p in _paths():
        b = p.read_bytes()
        if not b:
            continue
        assert b.count(b"\n") >= 1, f"{p} has content but no LF newline"
        # no lone CR (CR that is not part of CRLF) -> old-Mac endings that render as one line
        assert b.replace(b"\r\n", b"").count(b"\r") == 0, f"{p} has CR-only line endings"


def test_project_a_review_files_are_not_single_giant_lines():
    for p in _paths():
        if p.suffix not in (".py", ".md"):
            continue
        text = p.read_text()
        if len(text) < 400:
            continue                                  # trivially small file; nothing to check
        lines = text.split("\n")
        nonblank = [ln for ln in lines if ln.strip()]
        assert len(nonblank) >= 5, f"{p} renders as <5 non-blank lines (giant-line file?)"
        longest = max((len(ln) for ln in lines), default=0)
        assert longest <= 2000, f"{p} has a {longest}-char line (>2000) — unreviewable"


ALL_TESTS = [test_project_a_files_use_lf_not_cr_only,
             test_project_a_review_files_are_not_single_giant_lines]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} TEXT-HYGIENE TESTS PASSED ({len(_paths())} files checked)")


if __name__ == "__main__":
    run()
