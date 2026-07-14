"""Leading-numeric pytest suite discovery for suffixed milestone accounting."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "oaci/tests"
MILESTONE_PATTERN = re.compile(
    r"^test_c(?P<number>[0-9]+)(?P<suffix>[a-z][a-z0-9]*)?_.*\.py$"
)


def milestone_number(path: str | Path) -> int | None:
    match = MILESTONE_PATTERN.match(Path(path).name)
    return None if match is None else int(match.group("number"))


def discover_milestone_files(start: int, end: int = 84, test_dir: Path = TEST_DIR) -> list[Path]:
    files = []
    for path in sorted(test_dir.glob("test_c*.py")):
        number = milestone_number(path)
        if number is not None and int(start) <= number <= int(end):
            files.append(path)
    return files


def suite_files(suite: str, test_dir: Path = TEST_DIR) -> list[Path]:
    if suite == "focused":
        names = (
            "test_c84_multidataset_external_validity.py",
            "test_c84r_montage_repair.py",
            "test_c84c_canary_contract.py",
            "test_c84r2_canary_runtime_repair.py",
            "test_c84c_complete_engineering_replay.py",
            "test_c84r3_float32_linear_replay.py",
            "test_c84r3_execution_lock.py",
            "test_c84c_engineering_result.py",
            "test_c84fl_full_field_lock.py",
            "test_c84f_full_field_contract.py",
            "test_c84l1_intervention.py",
            "test_c84l1_protocol_lock.py",
            "test_c84l1_canary_contract.py",
            "test_c84l1r1_float32_linear_replay.py",
            "test_c84l1c_engineering_result.py",
            "test_c84fl2_full_field_lock.py",
            "test_c84f_dual_level_training_contract.py",
            "test_c84f_target_instrumentation_contract.py",
        )
        return [test_dir / name for name in names]
    if suite == "c65":
        return discover_milestone_files(65, test_dir=test_dir)
    if suite == "c23":
        return discover_milestone_files(23, test_dir=test_dir)
    if suite == "full":
        return [test_dir]
    raise ValueError(f"unknown C84R suite {suite!r}")


def relative_paths(paths: Iterable[Path]) -> list[str]:
    return [str(path.relative_to(REPO_ROOT)) for path in paths]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("focused", "c65", "c23", "full"))
    args = parser.parse_args(argv)
    for path in relative_paths(suite_files(args.suite)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
