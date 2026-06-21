"""The canonical-estimator freeze decision rule (parsimony tie-break) must be pre-registered and
deterministic."""
from __future__ import annotations

from h2cmi.run_canonical_freeze import freeze_decision, SUBSTANTIVE_THRESHOLD


def test_tie_break_prefers_oneshot():
    sel, tie, d = freeze_decision({0: 0.002, 1: -0.003, 2: 0.001}, SUBSTANTIVE_THRESHOLD)
    assert tie and sel == "gen_oneshot_diag" and abs(d) < SUBSTANTIVE_THRESHOLD


def test_picks_better_above_threshold():
    sel, tie, d = freeze_decision({0: 0.05, 1: 0.04, 2: 0.03}, SUBSTANTIVE_THRESHOLD)
    assert not tie and sel == "gen_oneshot_diag" and d > 0
    sel2, tie2, d2 = freeze_decision({0: -0.05, 1: -0.04, 2: -0.03}, SUBSTANTIVE_THRESHOLD)
    assert not tie2 and sel2 == "gen_iterative_diag" and d2 < 0


def test_threshold_is_preregistered():
    assert SUBSTANTIVE_THRESHOLD == 0.01


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"  {n} PASSED")
    print("test_canonical_freeze PASSED")
