"""Guard (Stage-1B10): if two participants.tsv columns collapse to the same casefolded name as the pinned column, resolution FAILS
closed (ambiguous which column is the label). Synthetic only."""
from __future__ import annotations
import os
import tempfile
from acar.v5.substrate import cohort_label_spec as CLS
from acar.v5.tests._util import expect_raises, ok


def test_duplicate_casefold_column_fails():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "participants.tsv")
        with open(p, "w") as f:                               # 'Group' and 'group' both present → collapse to the pinned 'Group'
            f.write("participant_id\tGroup\tgroup\n")
            f.write("sub-001\tControl\tPD\n")
        expect_raises(CLS.CohortLabelSpecError, lambda: CLS.resolve_label("PD", "ds004584", "sub-001", p))
    ok("two columns collapsing to the pinned label column (Group + group) → CohortLabelSpecError (ambiguous)")


def main():
    print("ACAR v5 Stage-1B10 guard: cohort label duplicate casefold column fails")
    test_duplicate_casefold_column_fails()
    print("ALL V5 STAGE1B-COHORT-LABEL-DUP-COLUMN GUARDS PASS")


if __name__ == "__main__":
    main()
