from pathlib import Path

from star_eeg.config import DEPENDENCY_COMMIT
from star_eeg.data.checkpoint_registry import inspect_spec, registry_specs


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_registry_has_exact_required_universe_and_start_flags():
    specs = registry_specs()
    assert len(specs) == 10
    assert {spec.tag for spec in specs} == {
        "H200_s0", "H200_s1", "H500_s0", "H500_s1",
        "H1000_s0", "H1000_s1", "H2000_s0", "H2000_s1",
        "released", "random",
    }
    assert {spec.tag for spec in specs if spec.usable_as_star_start} == {"H200_s0", "H200_s1"}
    assert {spec.tag for spec in specs if spec.usable_as_reference_only} == {
        "H500_s0", "H500_s1", "H1000_s0", "H1000_s1",
        "H2000_s0", "H2000_s1", "released", "random",
    }


def test_random_reference_config_is_hashable_and_strictly_instantiable_via_loader_stub():
    spec = next(spec for spec in registry_specs() if spec.tag == "random")

    def loader(_spec):
        return {
            "strict_reload_pass": True,
            "source_git_commit": DEPENDENCY_COMMIT,
            "route_manifest": None,
            "parameter_count": 4924000,
        }

    row = inspect_spec(spec, REPO_ROOT, loader=loader)
    assert row["exists"] is True
    assert len(row["sha256"]) == 64
    assert row["strict_reload_pass"] is True
    assert row["usable_as_star_start"] is False
