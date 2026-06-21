"""The action grid must be genuinely resumable: a second run adds no duplicate rows (this
was broken because load_done_keys read 'method' while action rows use 'action')."""
from __future__ import annotations

import json
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

from h2cmi import run_action_grid


def _run(out_path):
    argv = ["run_action_grid", "--scenarios", "population_null,cov", "--grid-seeds", "0",
            "--grid-target-sites", "all", "--sites", "2", "--subjects", "1", "--sessions", "1",
            "--trials", "8", "--epochs", "1", "--fast", "--allow-dirty", "--out", out_path]
    old = sys.argv
    sys.argv = argv
    try:
        run_action_grid.main()
    finally:
        sys.argv = old


def test_action_grid_resume_no_duplicates():
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "ag.jsonl")
        _run(out)
        n1 = sum(1 for _ in open(out))
        assert n1 > 0
        _run(out)                                          # resume must add nothing
        n2 = sum(1 for _ in open(out))
        assert n2 == n1, f"resume duplicated rows: {n1} -> {n2}"
        keys = set()
        for line in open(out):
            r = json.loads(line)
            k = (r["data_seed"], r["target_site"], r["scenario"], r["action"], r["cmi"])
            assert k not in keys, f"duplicate {k}"
            keys.add(k)


if __name__ == "__main__":
    test_action_grid_resume_no_duplicates()
    print("test_action_grid_resume PASSED")
