"""Guard (Stage-2B2): stable_matched_coral_v1 is label-free — it consumes only source_state + z, its signature has no label
parameter, and its source references no label read. Torch-free. Synthetic."""
from __future__ import annotations
import inspect
from acar.v5 import stage2_stable_coral as SC
from acar.v5 import stage2_real_action_provider as RAP
from acar.v5.tests._util import ok

_LABELS = {"label", "labels", "y", "y_te", "y_true", "y_pred", "diagnosis", "target", "case_control",
           "group", "participant_group", "outcome"}


def test_label_free_signature_and_source():
    for fn in (SC.stable_matched_coral_v1, SC.transport_operator, RAP.real_action_provider):
        assert not (set(inspect.signature(fn).parameters) & _LABELS), f"{fn.__name__} exposes a label parameter"
    src = inspect.getsource(SC)
    for tok in ("read_label", "resolve_label", "participants", "label_view", "stage2_label_loader"):
        assert tok not in src, f"stage2_stable_coral references label code: {tok!r}"
    # signature is exactly (source_lda, Z)
    assert list(inspect.signature(SC.stable_matched_coral_v1).parameters) == ["source_lda", "Z"]
    ok("stable_matched_coral_v1 is label-free (consumes only source_state + z; no label parameter / no label read)")


def main():
    print("ACAR v5 Stage-2B2 guard: stable CORAL label-free")
    test_label_free_signature_and_source()
    print("ALL V5 STAGE2B2-STABLE-CORAL-LABEL-FREE GUARDS PASS")


if __name__ == "__main__":
    main()
