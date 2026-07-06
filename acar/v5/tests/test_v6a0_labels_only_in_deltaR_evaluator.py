"""Guard (V6-A0a): the LABEL FIREWALL — labels enter only ΔR (batch_action_delta_r). Flipping every label leaves the label-free
features byte-identical; only ΔR (and the derived beneficial target) change. Synthetic, torch-free."""
from __future__ import annotations
import inspect
from acar.v5 import stage2_action_records as AR
from acar.v5 import v6_a0_action_viability as AV
from acar.v5 import v6_a0_sign_predictability as SP
from acar.v5.tests._util import ok, v6a0_eval_fold, DictLabelView


def _labels_of(by_subject):
    return {sk: (0 if i % 2 == 0 else 1) for i, sk in enumerate(sorted(by_subject))}


def test_flip_labels_changes_only_deltaR():
    fold = v6a0_eval_fold([("PD/ds002778/sub-e0", "eval", 64, 0), ("PD/ds002778/sub-e1", "eval", 96, 1)])
    labels = {sk: fold["label_view"].resolve_label(sk) for sk in fold["by_subject"]}
    flip = {sk: 1 - v for sk, v in labels.items()}
    r0, _ = AV.collect_eval_records([fold], AR.synthetic_action_provider)
    fold_flip = dict(fold, label_view=DictLabelView(flip))
    r1, _ = AV.collect_eval_records([fold_flip], AR.synthetic_action_provider)
    assert len(r0) == len(r1) and r0
    for a, b in zip(r0, r1):
        assert a["subject_key"] == b["subject_key"] and a["batch_id"] == b["batch_id"]
        assert a["features"] == b["features"], "label-free features must be byte-identical under label flip"
        # ΔR is NLL(f_a,y)-NLL(f_0,y): a symmetric flip of a binary label changes it (features fixed)
        assert a["delta_r"] != b["delta_r"], "ΔR must change when labels flip (labels DO enter ΔR)"
    # structural: only batch_action_delta_r reads a label; the feature builder never does
    assert "resolve_label" not in inspect.getsource(AV.batch_label_free_features)
    assert "resolve_label" in inspect.getsource(AV.collect_eval_records)
    ok("labels enter ONLY ΔR; flipping labels leaves features byte-identical, changes only ΔR (structural label firewall)")


def main():
    print("ACAR v5 V6-A0a guard: labels only in the ΔR evaluator")
    test_flip_labels_changes_only_deltaR()
    print("ALL V6A0-LABEL-FIREWALL GUARDS PASS")


if __name__ == "__main__":
    main()
