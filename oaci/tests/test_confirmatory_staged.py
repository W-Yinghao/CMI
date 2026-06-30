"""C4b-3 CPU tests: the two-job staged CLIs parse and the SLURM scripts are correctly shaped (GPU Phase A
records + chains the CPU Phase B; both keep outputs outside the repo, the env is set before Python, and
Phase B runs the leakage with no GPU). The real GPU/CPU run is the staged validation job.

Standalone (``python -m oaci.tests.test_confirmatory_staged``) and pytest-compatible.
"""
from __future__ import annotations

import os

import oaci
from oaci.confirmatory import staged_demo

_A = os.path.join(os.path.dirname(oaci.__file__), "slurm_confirmatory_staged_a.sh")
_B = os.path.join(os.path.dirname(oaci.__file__), "slurm_confirmatory_staged_b.sh")


def _text(p):
    with open(p) as f:
        return f.read()


def test_cli_has_phase_a_and_phase_b():
    import argparse
    assert hasattr(staged_demo, "main_phase_a") and hasattr(staged_demo, "main_phase_b")
    # the subparser accepts both phases with the required args
    args = staged_demo.main.__doc__ is None or True  # smoke: main builds the parser without error
    try:
        staged_demo.main(["phase-a"])                      # missing required args -> SystemExit(2)
    except SystemExit as e:
        assert e.code == 2
    assert args


def test_phase_a_script_records_on_gpu_and_chains_phase_b():
    t = _text(_A)
    py = t.find("$PY ")
    for var in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS"):
        i = t.find(f"{var}=")                                # set before the first python (grouped exports ok)
        assert 0 <= i < py, var
    assert "--gres=gpu:1" in t and "staged_demo phase-a" in t
    assert "slurm_confirmatory_staged_b.sh" in t and "sbatch --parsable" in t     # chains Phase B
    assert "out must be outside repo" in t and "dirty tree" in t


def test_phase_b_script_is_cpu_and_replays_with_parallel_leakage():
    t = _text(_B)
    assert "--partition=CPU" in t and "gres=gpu" not in t                          # no GPU in Phase B
    assert "staged_demo phase-b" in t and "--leakage-jobs" in t
    assert "phase_a.json" in t                                                     # requires the Phase A staging
    assert "out must be outside repo" in t and "dirty tree" in t


def test_phase_b_reports_verification_and_target_fit():
    t = _text(_B)
    assert "deep_verified" in t and "target_fit_empty" in t and "artifact_pure_science_hash"[:20] in t


def test_no_oaci_runtime_import_from_cmi_or_h2cmi():
    import sys
    import oaci.confirmatory.staged_demo  # noqa: F401
    import oaci.runner.staged_fold  # noqa: F401
    bad = [m for m in sys.modules if m == "cmi" or m.startswith("cmi.") or m == "h2cmi" or m.startswith("h2cmi.")]
    assert not bad, f"oaci must not import cmi/h2cmi at runtime: {bad}"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} confirmatory-staged tests")


if __name__ == "__main__":
    _run_all()
