"""B2b-ii CPU tests: the GPU SLURM script sets the determinism env before any Python, uses the SLURM
GPU allocation (not a manual override), keeps outputs outside the repo, and propagates failures.

Standalone (``python -m oaci.tests.test_bnci_gpu_script``) and pytest-compatible.
"""
from __future__ import annotations

import os

import oaci

_SCRIPT = os.path.join(os.path.dirname(oaci.__file__), "slurm_gpu_smoke.sh")


def _text():
    with open(_SCRIPT) as f:
        return f.read()


def _first_index(text, needle):
    return text.find(needle)


def test_gpu_script_sets_env_before_any_python_command():
    t = _text()
    py = t.find("$PY ")                                       # first python invocation
    for var in ("CUBLAS_WORKSPACE_CONFIG=:4096:8", "PYTHONHASHSEED=0", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                "NVIDIA_TF32_OVERRIDE=0", "OMP_NUM_THREADS=1", "MKL_DYNAMIC=FALSE", "KMP_DETERMINISTIC_REDUCTION=true"):
        i = t.find(f"export {var.split('=')[0]}=")
        assert 0 <= i < py, f"{var} must be exported before the first python command"


def test_gpu_script_uses_slurm_allocation_for_one_gpu():
    t = _text()
    assert "export CUDA_VISIBLE_DEVICES=" not in t            # never override the SLURM allocation
    assert "torch.cuda.device_count()==1" in t                # python verifies exactly one visible GPU
    assert "--gres=gpu:1" in t


def test_gpu_script_keeps_outputs_outside_repo():
    t = _text()
    assert "artifact root must be OUTSIDE the repo" in t      # explicit guard
    assert "status --porcelain -- oaci" in t                  # clean scientific tree required
    assert "MNE-bnci-data" in t                               # datalake presence check


def test_gpu_script_is_strict_and_folds_all_return_codes():
    t = _text()
    assert "set -euo pipefail" in t
    for rc in ("rt_rc", "rn_rc", "demo_rc", "val_rc"):        # runtime / runner / demo / validator
        assert rc in t
    assert 'exit "$fail"' in t


def test_gpu_demo_separates_canonical_json_from_logs():
    t = _text()
    # the demo's stdout (the canonical JSON) goes to the report file; stderr (MNE/training) goes elsewhere
    assert 'bnci_gpu_demo' in t and '>"$OACI_ARTIFACT_ROOT/gpu-smoke-report.json"' in t
    assert '2>"$OACI_ARTIFACT_ROOT/gpu-smoke.err"' in t


def test_gpu_script_independently_deep_verifies_both_artifacts():
    t = _text()
    assert t.count("oaci.artifacts.verify") >= 2              # canonical + reversed verified independently


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.runner.bnci_gpu_demo  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} bnci-gpu-script tests")


if __name__ == "__main__":
    _run_all()
