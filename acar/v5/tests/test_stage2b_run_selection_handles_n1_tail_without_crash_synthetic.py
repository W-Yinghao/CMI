"""Guard (Stage-2B3): the end-to-end regression for the Stage-2B FAIL — run_selection completes on subjects that have a 1-window
tail, and NO non-identity action is ever routed on a forced (sub-MIN_BATCH) tail. A strict provider raises if that contract is
violated, so an unfixed engine would crash. Synthetic, torch-free."""
from __future__ import annotations
import numpy as np
from acar.v5 import protocol as P
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_selection_report as RPT
from acar.v5.tests._util import ok, stage2b_auth, stage2b_disease_inputs


def _strict_provider(name, source_lda, Z):
    """Synthetic provider that FAILS if a non-identity action is ever routed on a forced (sub-MIN_BATCH) tail — mimics the real
    stable_matched_coral fail-closed contract, so an engine that routed a forced tail would crash here."""
    n = int(np.asarray(Z, float).shape[0])
    if name != "identity" and n < PE.STAGE2_MIN_BATCH:
        raise AssertionError(f"non-identity action {name!r} routed on a forced tail n={n} — engine must skip forced tails")
    return AR.synthetic_action_provider(name, source_lda, Z)


def test_run_selection_survives_n1_tails():
    di = stage2b_disease_inputs()                                          # default n_windows=16 (all eligible)
    r = np.random.RandomState(0)
    injected = 0
    for d in ("PD", "SCZ"):                                                # inject a 1-window tail into a CAL and an EVAL subject
        for role in ("cal", "eval"):
            for sk, rec in di[d]["folds"][0]["by_subject"].items():
                if rec["split_role"] == role:
                    rec["embedding"] = r.randn(33, 8) * 0.3                # 33 -> window_batches [32 eligible, 1 FORCED]
                    injected += 1
                    break
    assert injected == 4
    a = stage2b_auth()
    rep = ENG.run_selection(a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256=a["stage1b_registry_sha256"],
                            disease_inputs=di, action_provider=_strict_provider, v2_replay_provider=lambda d, ctx: -0.5)
    RPT.validate_selection_report(rep)                                     # completed without crash -> valid report
    assert rep["outcome"] in (RPT.OUTCOME_SELECTED, RPT.OUTCOME_DEV_STOP)
    assert set(rep["per_candidate"]) == set(P.CANDIDATE_IDS)
    ok("run_selection completes on 1-window-tail subjects; no non-identity action routed on any forced tail")


def main():
    print("ACAR v5 Stage-2B3 guard: run_selection handles n=1 tails without crash")
    test_run_selection_survives_n1_tails()
    print("ALL V5 STAGE2B3-RUN-SELECTION-N1-TAIL GUARDS PASS")


if __name__ == "__main__":
    main()
