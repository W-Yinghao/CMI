"""C20 — frozen-config lock. Re-derives the C19 pre-registration hash and asserts it equals the value C20 is
locked to (664007686afb520f). If C19's config drifts, C20 refuses to run — this is what makes C20 an external
VALIDATION of the frozen probe rather than a re-parameterised C19b."""
from __future__ import annotations

import hashlib
import json

from ..competence_probe import report as c19_report
from ..competence_probe import schema as c19
from . import schema


def c19_config_hash() -> str:
    return hashlib.sha256(json.dumps(c19.frozen_config(), sort_keys=True).encode()).hexdigest()[:16]


def assert_locked() -> str:
    """Assert the executed C19 config hash matches the locked value; return it. Raises on drift."""
    got = c19_config_hash()
    # cross-check the two derivations agree (C19's own hasher and ours)
    if got != c19_report._config_hash():
        raise ValueError("C19 config hashers disagree — config drift")
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C20 is locked to C19 config {schema.LOCKED_C19_CONFIG_HASH} but C19 now hashes {got} "
                         "-> the probe changed; C20 would be C19b, not a validation. Refusing.")
    return got


def frozen_config() -> dict:
    assert_locked()
    return c19.frozen_config()
