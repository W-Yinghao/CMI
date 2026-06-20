"""The shift-grid driver must be resumable: a second run adds no duplicate rows and skips
completed (seed,site,scenario,method,cmi) units."""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import pytest
    pytestmark = pytest.mark.integration
except ImportError:
    pass

from h2cmi import run_shift_grid


def _run(out_path):
    argv = ["run_shift_grid", "--scenarios", "no_shift,cov", "--seeds", "0",
            "--target-sites", "all", "--sites", "2", "--subjects", "1", "--sessions", "1",
            "--trials", "8", "--epochs", "1", "--fast", "--out", out_path]
    old = sys.argv
    sys.argv = argv
    try:
        run_shift_grid.main()
    finally:
        sys.argv = old


def test_resume_no_duplicates():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "grid.jsonl")
        _run(out)
        n1 = sum(1 for _ in open(out))
        assert n1 > 0
        _run(out)                                   # resume
        n2 = sum(1 for _ in open(out))
        assert n2 == n1, f"resume duplicated rows: {n1} -> {n2}"
        # uniqueness of keys
        import json
        keys = set()
        for line in open(out):
            r = json.loads(line)
            k = (r["data_seed"], r["target_site"], r["scenario"], r["method"], r["cmi"])
            assert k not in keys, f"duplicate unit {k}"
            keys.add(k)


if __name__ == "__main__":
    test_resume_no_duplicates()
    print("test_grid_resume PASSED")
