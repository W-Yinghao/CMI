"""Guard (Stage-2B3): the v2-replay comparator uses the SAME forced-tail contract as V5 — it never calls the action provider for a
forced (sub-MIN_BATCH) tail (FIT/CAL skip it; EVAL routes it to identity, ΔR=0). No more permissive action universe than V5.
Synthetic, torch-free (constant regressor, no sklearn)."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_v2_replay as VR
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject


class _ConstReg:
    def fit(self, X, y):
        self.m = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def test_v2_replay_never_calls_provider_on_forced_tail():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=6))
    by_subject, lv = stage2b_by_subject([
        ("PD/ds002778/sub-fit0", "train", 64, 0),
        ("PD/ds002778/sub-fit1", "val", 64, 1),
        ("PD/ds002778/sub-cal0", "cal", 33, 0),                # [32, 1 forced]
        ("PD/ds002778/sub-eval0", "eval", 33, 1),              # [32, 1 forced] -> forced routed to identity ΔR=0
    ])
    folds = [{"by_subject": by_subject, "source_lda": lda, "label_view": lv}]
    prov, calls = stage2b_spy_provider()
    red = VR.v2_replay_red_by_disease("PD", folds, action_provider=prov, regressor_factory=lambda: _ConstReg())
    assert calls and all(c["n"] >= PE.STAGE2_MIN_BATCH for c in calls), \
        f"v2 replay must not call the action provider on a forced tail; offending={[c for c in calls if c['n'] < PE.STAGE2_MIN_BATCH]}"
    assert isinstance(red, float) and np.isfinite(red)
    ok("v2 replay excludes forced tails from all non-identity action evaluation (same contract as V5)")


def main():
    print("ACAR v5 Stage-2B3 guard: v2 replay excludes forced tails")
    test_v2_replay_never_calls_provider_on_forced_tail()
    print("ALL V5 STAGE2B3-V2REPLAY-EXCLUDE-FORCED GUARDS PASS")


if __name__ == "__main__":
    main()
