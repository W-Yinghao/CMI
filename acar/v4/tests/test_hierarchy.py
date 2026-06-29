"""Guards for acar/v4/hierarchy.py (Direction-B deployed-risk calibration objects). SYNTHETIC FIXTURES ONLY; NO real
DEV cohort, NO v3 loader, NO V4 DEV result, NO coverage theorem. The decisive guards prove the V4-B vs v3 distinction:
B0 (all-action joint max) RESPONDS to a change in an UNEXECUTED action's risk, while B1/B2 (policy-only) IGNORE it —
i.e. V4-B calibrates the EXECUTED policy's deployed risk, not the all-action simultaneous risk. Also: B1 responds to
the executed action's risk; fallback identity contributes 0 but stays in the subject denominator; canonical /
permutation-independent subject order; exact hand-calculations; full fail-closed contract.
Run: python -m acar.v4.tests.test_hierarchy
"""
import numpy as np

from acar.v4 import policies as PO
from acar.v4 import hierarchy as H

ID = PO.IDENTITY


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except Exception as e:                       # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, no exception raised")


# ----------------------------------------------------------------------------- THE distinction

def test_unexecuted_action_risk_moves_B0_not_B1_B2():
    subj = np.array(["A", "A", "B"])
    choices = np.array([0, 0, 0])                                  # always execute action 0
    dr_base = np.array([[-1.0, 0.0], [2.0, 0.0], [-0.5, 0.0]])
    dr_alt = dr_base.copy(); dr_alt[:, 1] = 9.0                    # raise the UNEXECUTED action's risk
    # B1 / B2 depend only on the executed column ⇒ unchanged
    for fn in (lambda d: H.policy_subject_risk(choices, d, subj, loss="mean"),
               lambda d: H.hierarchical_policy_risk(choices, d, subj, loss="mean", batch_summary="mean")):
        assert np.allclose(fn(dr_base).values, fn(dr_alt).values), "B1/B2 must ignore unexecuted-action risk"
    # B0 uses every action ⇒ changes
    b0_base = H.all_action_joint_max(dr_base, subj).values
    b0_alt = H.all_action_joint_max(dr_alt, subj).values
    assert not np.allclose(b0_base, b0_alt), "B0 must respond to unexecuted-action risk"


def test_B1_responds_to_executed_action_risk():
    subj = np.array(["A", "A", "B"]); choices = np.array([0, 0, 0])
    dr1 = np.array([[-1.0, 0.0], [2.0, 0.0], [-0.5, 0.0]])
    dr2 = dr1.copy(); dr2[0, 0] = -3.0                            # change the EXECUTED action's risk
    assert not np.allclose(H.policy_subject_risk(choices, dr1, subj, loss="mean").values,
                           H.policy_subject_risk(choices, dr2, subj, loss="mean").values)


# ----------------------------------------------------------------------------- exact hand calculations

def test_hand_calculations_and_fallback_denominator():
    subj = np.array(["A", "A", "B"])
    dr = np.array([[-1.0, 0.3], [2.0, 0.1], [-0.5, 4.0]])
    choices = np.array([0, 0, ID])                                # B's batch is fallback identity
    b1m = H.policy_subject_risk(choices, dr, subj, loss="mean")
    assert b1m.subject_ids == ("A", "B") and b1m.n_batches_by_subject == (2, 1)
    assert np.allclose(b1m.values, [0.5, 0.0])                    # A=(-1+2)/2 ; B=0/1 (fallback in denominator)
    assert np.allclose(H.policy_subject_risk(choices, dr, subj, loss="positive").values, [1.0, 0.0])
    assert np.allclose(H.policy_subject_risk(choices, dr, subj, loss="harm_indicator").values, [0.5, 0.0])
    # B0 = max over batches×actions per subject: A=max(0.3,2)=2 ; B=max(-0.5,4)=4
    assert np.allclose(H.all_action_joint_max(dr, subj).values, [2.0, 4.0])
    # B2(mean, mean) == B1(mean)
    b2 = H.hierarchical_policy_risk(choices, dr, subj, loss="mean", batch_summary="mean")
    assert np.allclose(b2.values, b1m.values)


def test_immutability_and_one_value_per_subject():
    subj = np.array(["A"] + ["B"] * 100)
    rng = np.random.default_rng(0)
    dr = rng.normal(size=(101, 2))
    choices = rng.integers(-1, 2, size=101)
    sr = H.policy_subject_risk(choices, dr, subj, loss="mean")
    assert sr.values.shape == (2,) and sr.n_batches_by_subject == (1, 100)
    _expect(ValueError, lambda: sr.values.__setitem__(0, 123.0))   # values are read-only


def test_canonical_and_permutation_independent():
    subj = np.array(["A", "A", "B"])
    dr = np.array([[-1.0, 0.3], [2.0, 0.1], [-0.5, 4.0]])
    choices = np.array([0, 0, 0])
    base = H.policy_subject_risk(choices, dr, subj, loss="mean")
    perm = np.array([2, 0, 1])
    shuf = H.policy_subject_risk(choices[perm], dr[perm], subj[perm], loss="mean")
    assert base.subject_ids == shuf.subject_ids == ("A", "B")
    assert np.allclose(base.values, shuf.values)                   # canonical order ⇒ permutation-independent


# ----------------------------------------------------------------------------- fail-closed

def test_fail_closed_validation():
    subj = np.array(["A", "B"]); dr = np.array([[-1.0, 0.0], [0.5, 0.0]]); ch = np.array([0, 0])
    _expect(ValueError, lambda: H.policy_subject_risk(np.array([-2, 0]), dr, subj, loss="mean"))     # choice=-2
    _expect(ValueError, lambda: H.policy_subject_risk(np.array([0, 5]), dr, subj, loss="mean"))      # choice>=A
    _expect(ValueError, lambda: H.policy_subject_risk(ch, np.array([[np.nan, 0.0], [0.0, 0.0]]), subj, loss="mean"))
    _expect(ValueError, lambda: H.all_action_joint_max(np.zeros((2, 0)), subj))                       # zero actions
    _expect(ValueError, lambda: H.all_action_joint_max(np.zeros((0, 2)), np.array([], dtype="<U1")))  # zero batches
    _expect(ValueError, lambda: H.policy_subject_risk(np.array([0]), dr, subj, loss="mean"))          # shape mismatch
    _expect(ValueError, lambda: H.policy_subject_risk(ch, dr, np.array(["A"]), loss="mean"))          # ids len mismatch
    _expect(ValueError, lambda: H.policy_subject_risk(ch, dr, np.array([1, 2]), loss="mean"))         # non-string ids
    _expect(ValueError, lambda: H.policy_subject_risk(ch, dr, np.array(["A", ""]), loss="mean"))      # empty id
    _expect(ValueError, lambda: H.policy_subject_risk(ch, dr, subj, loss="zzz"))                      # unknown loss
    _expect(ValueError, lambda: H.hierarchical_policy_risk(ch, dr, subj, loss="mean", batch_summary="zzz"))  # summary


def main():
    print("ACAR v4 hierarchy (Direction-B) calibration-object guards (synthetic fixtures only):")
    for t in (test_unexecuted_action_risk_moves_B0_not_B1_B2, test_B1_responds_to_executed_action_risk,
              test_hand_calculations_and_fallback_denominator, test_immutability_and_one_value_per_subject,
              test_canonical_and_permutation_independent, test_fail_closed_validation):
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL V4 HIERARCHY GUARDS PASS")


if __name__ == "__main__":
    main()
