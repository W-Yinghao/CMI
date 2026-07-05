"""Guard (Stage-2B0): labels are confined to evaluation. Routing / scalarization / threshold-fitting entry points expose no
label parameter, and the evaluation label view is single-subject + authorized-only (no bulk dump). Synthetic only."""
from __future__ import annotations
import inspect
from acar.v5 import deploy as DEPLOY
from acar.v5 import scalarization as SCAL
from acar.v5 import stage2_thresholds as TH
from acar.v5 import stage2_label_loader as LL
from acar.v5.tests._util import expect_raises, ok

_LABELS = {"label", "labels", "y", "y_te", "y_true", "y_pred", "diagnosis", "target", "case_control",
           "group", "participant_group", "outcome"}


def test_routing_and_threshold_signatures_label_free():
    for fn in (DEPLOY.route, SCAL.decide, SCAL.proposed_action, SCAL.fit_quantiles, TH.fit_thresholds):
        leak = set(inspect.signature(fn).parameters) & _LABELS
        assert not leak, f"{fn.__module__}.{fn.__name__} exposes label param(s) {leak}"
    ok("route/decide/proposed_action/fit_quantiles/fit_thresholds expose no label parameter")


def test_label_view_evaluation_only():
    lv = LL.make_evaluation_label_view({"ds002778": None}, ["PD/ds002778/sub-hc001", "PD/ds002778/sub-pd002"])
    assert hasattr(lv, "resolve_label")
    # no bulk-dump / participants-path / label-list accessor
    for attr in ("labels", "_labels", "participants", "paths", "_paths", "dump", "all_labels", "values"):
        assert not hasattr(lv, attr), f"label view leaks bulk accessor {attr!r}"
    # resolves authorized subjects (ds002778 = id-prefix, no tsv), rejects an unauthorized one
    assert lv.resolve_label("PD/ds002778/sub-hc001") == 0
    assert lv.resolve_label("PD/ds002778/sub-pd002") == 1
    expect_raises(LL.Stage2LabelError, lambda: lv.resolve_label("PD/ds002778/sub-hc999"))
    ok("evaluation label view: resolve_label only, authorized subjects only, no bulk label/path accessor")


def main():
    print("ACAR v5 Stage-2B0 guard: labels do not enter routing/scalarization/thresholds")
    test_routing_and_threshold_signatures_label_free()
    test_label_view_evaluation_only()
    print("ALL V5 STAGE2B0-LABEL-FIREWALL GUARDS PASS")


if __name__ == "__main__":
    main()
