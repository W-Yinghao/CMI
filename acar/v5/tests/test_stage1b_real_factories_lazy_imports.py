"""Guard (Stage-1B4): the real reader/trainer modules import NO heavy deps at module level — mne/torch appear ONLY as lazy imports
inside the read/train methods. Synthetic only."""
from __future__ import annotations
import ast
import importlib
import inspect
import sys
from acar.v5.tests._util import ok

HEAVY = ("torch", "mne", "braindecode", "moabb", "cmi", "numpy", "scipy", "sklearn")
MODULES = ("real_dev_reader", "real_trainer", "stage1b_file_artifact_writer")


def _toplevel_roots(mod):
    roots = set()
    for node in ast.parse(inspect.getsource(mod)).body:
        if isinstance(node, ast.Import):
            for a in node.names:
                roots.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_toplevel_heavy_imports():
    for name in MODULES:
        mod = importlib.import_module(f"acar.v5.substrate.{name}")
        assert not (set(HEAVY) & _toplevel_roots(mod)), (name, sorted(set(HEAVY) & _toplevel_roots(mod)))
    ok("real_dev_reader / real_trainer / file-artifact-writer have NO module-level heavy imports")


def test_import_pulls_no_heavy_modules():
    before = set(sys.modules)
    for name in MODULES:
        importlib.import_module(f"acar.v5.substrate.{name}")
    new = set(sys.modules) - before
    leaked = [m for m in new for h in HEAVY if m == h or m.startswith(h + ".")]
    assert not leaked, f"eagerly pulled heavy modules: {sorted(set(leaked))}"
    ok("importing the real modules pulls NO heavy module (mne/torch stay lazy)")


def test_lazy_imports_are_inside_methods():
    from acar.v5.substrate import real_dev_reader as RDR
    from acar.v5.substrate import real_trainer as RT
    assert "import mne" in inspect.getsource(RDR.RealBidsDevReader.read_subject_windows)
    assert "import torch" in inspect.getsource(RT.RealSubstrateTrainer.train_fold)
    assert "torch" not in _toplevel_roots(RT) and "mne" not in _toplevel_roots(RDR)
    ok("mne is lazy inside read_subject_windows; torch is lazy inside train_fold")


def main():
    print("ACAR v5 Stage-1B4 guard: real factories lazy imports")
    test_no_toplevel_heavy_imports()
    test_import_pulls_no_heavy_modules()
    test_lazy_imports_are_inside_methods()
    print("ALL V5 STAGE1B-REAL-FACTORIES-LAZY GUARDS PASS")


if __name__ == "__main__":
    main()
