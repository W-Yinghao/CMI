"""C17 — diagnostic-only target labels. The C16 target-oracle labels (target_bacc_good / target_joint_good /
target_oracle_rank) are attached to the atlas by signal_atlas.build_atlas; this module only enforces and
asserts that they are DIAGNOSTIC-ONLY and NON-DEPLOYABLE — they are never permitted to drive a selector."""
from __future__ import annotations

from .schema import DIAGNOSTIC_ONLY

TARGET_LABEL_COLUMNS = ("tgt__target_bacc_delta", "tgt__target_nll_delta", "tgt__target_bacc_good",
                        "tgt__target_joint_good", "tgt__target_oracle_rank")


def assert_diagnostic_only(rows) -> bool:
    """Every atlas row must carry the diagnostic_only_non_deployable flag; target columns exist only as
    labels. Raises if a target column is missing the flag (a guard against turning this into a selector)."""
    for r in rows:
        if not r.get(DIAGNOSTIC_ONLY, False):
            raise ValueError("atlas row missing diagnostic_only_non_deployable flag — target labels must not "
                             "be usable as a deployable selector")
    return True


def is_source_column(col) -> bool:
    return col.startswith("src__")


def is_target_column(col) -> bool:
    return col.startswith("tgt__")
