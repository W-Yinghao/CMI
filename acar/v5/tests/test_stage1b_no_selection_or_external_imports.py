"""Guard (Stage-1B2): importing the Stage-1B build code pulls NO heavy/real-data/selection/external deps, and its source performs
no selection / robustness / external calls. Synthetic only."""
from __future__ import annotations
import ast
import inspect
import sys
from acar.v5.tests._util import ok

HEAVY = ("torch", "mne", "braindecode", "moabb", "cmi", "acar.v3", "numpy", "scipy", "sklearn")


def _imported_names(mod):
    """All module/from targets imported anywhere in a module's source (AST — docstring-safe)."""
    names = set()
    for n in ast.walk(ast.parse(inspect.getsource(mod))):
        if isinstance(n, ast.Import):
            for a in n.names:
                names.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module)
    return names


def test_import_pulls_no_heavy_modules():
    before = set(sys.modules)
    import acar.v5.substrate.stage1b_build  # noqa: F401
    import acar.v5.substrate.dev_reader_contract  # noqa: F401
    import acar.v5.substrate.train_contract  # noqa: F401
    import acar.v5.substrate.stage1b_artifacts  # noqa: F401
    new = set(sys.modules) - before
    leaked = [m for m in new for h in HEAVY if m == h or m.startswith(h + ".")]
    assert not leaked, f"importing Stage-1B build eagerly pulled heavy modules: {sorted(set(leaked))}"
    ok("importing stage1b_build (+ contracts/artifacts) pulls NO torch/mne/cmi/acar.v3/numpy/... eagerly")


def test_source_imports_no_heavy_modules():
    from acar.v5.substrate import stage1b_build as B
    from acar.v5.substrate import dev_reader_contract, train_contract, stage1b_artifacts
    heavy_roots = {"torch", "mne", "braindecode", "moabb", "cmi", "numpy", "scipy", "sklearn"}
    for mod in (B, dev_reader_contract, train_contract, stage1b_artifacts):
        imported = _imported_names(mod)
        roots = {i.split(".")[0] for i in imported}
        assert not (heavy_roots & roots), (mod.__name__, "heavy import", sorted(heavy_roots & roots))
        assert not any(i == "acar.v3" or i.startswith("acar.v3.") for i in imported), (mod.__name__, "acar.v3 import")
    ok("stage1b_build (+ contracts/artifacts) import NO torch/mne/cmi/acar.v3/numpy/scipy/sklearn (AST; docstring-safe)")


def test_source_has_no_selection_external_calls():
    from acar.v5.substrate import stage1b_build as B
    from acar.v5.substrate import dev_reader_contract, train_contract, stage1b_artifacts
    src = "".join(inspect.getsource(m) for m in (B, dev_reader_contract, train_contract, stage1b_artifacts))
    # call/loader tokens that must never appear (chosen so they do not occur in the modules' docstrings)
    for tok in ("candidate_selection", "run_dev_select", "load_crossdataset", "load_cohort(", "external_read_gate("):
        assert tok not in src, f"Stage-1B build source must not reference {tok!r}"
    ok("Stage-1B build source performs no selection / real-loader / external-gate calls")


def main():
    print("ACAR v5 Stage-1B2 guard: no selection/external imports")
    test_import_pulls_no_heavy_modules()
    test_source_imports_no_heavy_modules()
    test_source_has_no_selection_external_calls()
    print("ALL V5 STAGE1B-NO-SELECTION-EXTERNAL-IMPORTS GUARDS PASS")


if __name__ == "__main__":
    main()
