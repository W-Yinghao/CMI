from pathlib import Path

from star_eeg.data.faced_split_contract import canonical_hash
from star_eeg.runners.run_star00a_preflight import assemble_static_preflight


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_static_preflight_is_byte_content_deterministic():
    first = assemble_static_preflight(REPO_ROOT)
    second = assemble_static_preflight(REPO_ROOT)
    assert first == second
    assert canonical_hash(first) == canonical_hash(second)
