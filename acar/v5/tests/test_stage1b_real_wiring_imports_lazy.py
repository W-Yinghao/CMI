"""Guard (Stage-1B3): the real-wiring modules import NO heavy deps at module level (torch etc. only appear as LAZY imports inside
functions, e.g. the runtime-capture probe). Synthetic only."""
from __future__ import annotations
import ast
import importlib
import inspect
import sys
from acar.v5.tests._util import ok

HEAVY = ("torch", "mne", "braindecode", "moabb", "cmi", "numpy", "scipy", "sklearn")
MODULES = ("subject_index", "fit_dataset_view", "stage1b_artifact_writer", "stage1b_registry_populate",
           "stage1b_build", "dev_reader_contract", "train_contract", "stage1b_runtime_capture")


def _toplevel_import_roots(mod):
    roots = set()
    for node in ast.parse(inspect.getsource(mod)).body:       # MODULE-LEVEL statements only (lazy in-function imports allowed)
        if isinstance(node, ast.Import):
            for a in node.names:
                roots.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_toplevel_heavy_imports():
    for name in MODULES:
        mod = importlib.import_module(f"acar.v5.substrate.{name}")
        roots = _toplevel_import_roots(mod)
        assert not (set(HEAVY) & roots), (name, "module-level heavy import", sorted(set(HEAVY) & roots))
    ok("all real-wiring modules have NO module-level heavy imports (torch etc. are lazy-only)")


def test_import_pulls_no_heavy_modules():
    before = set(sys.modules)
    for name in MODULES:
        importlib.import_module(f"acar.v5.substrate.{name}")
    new = set(sys.modules) - before
    leaked = [m for m in new for h in HEAVY if m == h or m.startswith(h + ".")]
    assert not leaked, f"importing the real-wiring modules eagerly pulled: {sorted(set(leaked))}"
    ok("importing the real-wiring modules pulls NO heavy module (lazy torch in capture is never triggered at import)")


def test_runtime_capture_has_lazy_torch_only():
    from acar.v5.substrate import stage1b_runtime_capture as CAP
    assert "torch" not in _toplevel_import_roots(CAP), "torch must NOT be a module-level import in the capture tool"
    assert "import torch" in inspect.getsource(CAP.capture_runtime_lock), "capture_runtime_lock should import torch lazily"
    ok("stage1b_runtime_capture imports torch ONLY lazily inside capture_runtime_lock")


def main():
    print("ACAR v5 Stage-1B3 guard: real wiring imports lazy")
    test_no_toplevel_heavy_imports()
    test_import_pulls_no_heavy_modules()
    test_runtime_capture_has_lazy_torch_only()
    print("ALL V5 STAGE1B-REAL-WIRING-LAZY GUARDS PASS")


if __name__ == "__main__":
    main()
