"""Guard (Stage-2A, PROP 8): no label is visible to routing/scalarization code, and a feature dump carrying a label-like field is
rejected at intake. Synthetic only (writes a tiny label-free npz, then poisons it)."""
from __future__ import annotations
import inspect
import os
import tempfile
from acar.v5 import deploy as DEPLOY
from acar.v5 import stage2_package_intake as INTAKE
from acar.v5 import stage2_selection_runner as RUN2
from acar.v5.tests._util import expect_raises, ok, write_synthetic_feat_dump


def test_routing_signatures_label_free():
    assert RUN2.assert_label_free_routing() is True
    # route takes ONLY (candidate, batch, thresholds) — no label/y/target parameter exists
    assert set(inspect.signature(DEPLOY.route).parameters) == {"candidate", "batch", "thresholds"}
    ok("route/decide/proposed_action/fit_quantiles expose no label-like parameter; route=(candidate,batch,thresholds) (PROP 8)")


def test_label_free_feat_dump_header_ok():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "feat_dump.npz")
        write_synthetic_feat_dump(p)
        header = INTAKE.read_feature_dump_header(p)                 # header-only read (no embedding materialized)
        assert str(header["schema_version"]).startswith("ACAR_V5_STAGE1B_FEAT_DUMP_V5")
        assert str(header["ref"]) == "PD/fold0/seed20260711"
    ok("a label-free feat_dump header reads cleanly (schema V5); embedding never indexed (PROP 8)")


def _poison(dirpath, **arrays):
    """Write a valid dump then splice in extra arrays (bypassing the self-validating writer); return the npz path."""
    import numpy as np
    p = os.path.join(dirpath, "feat_dump.npz")
    write_synthetic_feat_dump(p)
    with np.load(p, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data.update(arrays)
    with open(p, "wb") as f:
        np.savez(f, **data)
    return p


def test_labeled_feat_dump_rejected():
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        p = _poison(d, label=np.asarray([1, 0]))                   # exact denylist name
        expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.read_feature_dump_header(p))
    ok("a feat_dump carrying a 'label' field → Stage2IntakeError at header read (PROP 8)")


def test_out_of_schema_field_rejected():
    import numpy as np
    # a case-variant / synonym label name outside the 9-name denylist is caught by the closed-schema ALLOWLIST
    for name in ("Diagnosis", "dx", "phenotype", "Label"):
        with tempfile.TemporaryDirectory() as d:
            p = _poison(d, **{name: np.asarray([1, 0])})
            expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.read_feature_dump_header(p))
    ok("an out-of-schema field (Diagnosis/dx/phenotype/Label) evading the 9-name denylist → rejected by the allowlist (PROP 8)")


def test_nested_label_in_json_map_rejected():
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        # a nested label key inside a JSON-string provenance map (embedding never materialized)
        poisoned = np.asarray('{"sub-01":{"diagnosis":1,"interpolated":false}}')
        p = _poison(d, montage_completion_by_subject=poisoned)
        expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.read_feature_dump_header(p))
    ok("a nested label-like key ('diagnosis') inside a JSON provenance map → rejected (PROP 8)")


def main():
    print("ACAR v5 Stage-2A guard: no label in routing / label-free feature dumps (PROP 8)")
    test_routing_signatures_label_free()
    test_label_free_feat_dump_header_ok()
    test_labeled_feat_dump_rejected()
    test_out_of_schema_field_rejected()
    test_nested_label_in_json_map_rejected()
    print("ALL V5 STAGE2A-NO-LABEL-IN-ROUTING GUARDS PASS")


if __name__ == "__main__":
    main()
