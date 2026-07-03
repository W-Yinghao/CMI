"""Guard (Stage-1B14): the ordinal-placeholder rename is pinned to ds003944/ds003947 — a type-prefixed ordinal header + valid
channels.tsv in any OTHER cohort gets NO rename plan (never a global fallback)."""
from __future__ import annotations
import tempfile
from acar.v5.substrate import brainvision_read_repair as BR
from acar.v5.tests._util import ok, make_brainvision_triplet, modern_channel_names

_REAL = modern_channel_names() + ["VEOG", "ECG"]


def test_other_cohorts_get_no_ordinal_rename():
    raw_dir = tempfile.mkdtemp()
    vhdr = make_brainvision_triplet(raw_dir, "sub-x_task-Rest_eeg", _REAL, with_marker=False,
                                    generic_header=True, write_channels_tsv=True, ordinal_prefix_overrides={20: "EOG", 21: "ECG"})
    for disease, cohort in (("PD", "ds002778"), ("PD", "ds003490"), ("PD", "ds004584"),
                            ("SCZ", "ds004000"), ("SCZ", "ds004367")):
        plan = BR.plan_repair(disease, cohort, "sub-x", vhdr)
        assert plan is None or plan.mode != BR.MODE_CHANNEL_NAMES_FROM_TSV, f"{cohort} must NOT rename"
    ok("a type-prefixed ordinal header + valid channels.tsv outside ds003944/ds003947 gets NO rename (never a global fallback)")


def main():
    print("ACAR v5 Stage-1B14 guard: type-prefixed ordinal rename forbidden outside ds003944/ds003947")
    test_other_cohorts_get_no_ordinal_rename()
    print("ALL V5 STAGE1B14-TYPE-PREFIXED-ORDINAL-FORBIDDEN-COHORT GUARDS PASS")


if __name__ == "__main__":
    main()
