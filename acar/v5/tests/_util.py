"""Shared test helpers (stdlib only)."""
from __future__ import annotations


def expect_raises(exc, fn, msg=""):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # noqa
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e} ({msg})")
    raise AssertionError(f"expected {exc.__name__}, no error raised ({msg})")


def ok(name):
    print(f"  [ok] {name}")


def stage1b_auth(run_id="run-syn-0001", **over):
    """A fully-valid SYNTHETIC Stage-1B authorization contract (override any field via kwargs)."""
    from acar.v5 import protocol as P
    from acar.v5.substrate import stage1b_authorization as SA
    from acar.v5.substrate import plan as PLAN
    a = {"stage": "Stage-1B", "protocol_tag": SA.PROTOCOL_TAG, "protocol_tag_target_sha": "4278435",
         "implementation_base_sha": "0" * 40, "allowed_ref_type": "fold_contained_only",
         "allowed_refs": sorted(r["ref"] for r in PLAN.fold_refs()), "allowed_seeds": list(P.S1_SEEDS),
         "selection_seed": P.SELECTION_SEED, "forbid_final_external_refs": True, "forbid_external_sites": True,
         "forbid_candidate_selection": True, "forbid_external_read": True,
         "run_id": run_id, "statement": SA.REQUIRED_STAGE1B_STATEMENT}
    a.update(over)
    return a


def stage1b_lock(run_id="run-syn-0001", device_kind="cpu", **over):
    """A fully-valid SYNTHETIC Stage-1B runtime lock (override any field via kwargs)."""
    from acar.v5.substrate import stage1b_authorization as SA
    lk = {"stage": "Stage-1B", "protocol_tag": SA.PROTOCOL_TAG, "protocol_tag_target_sha": "4278435",
          "run_id": run_id, "device_kind": device_kind, "status": "CAPTURED_AND_VERIFIED"}
    lk.update(over)
    return lk


def batch(batch_id, **per_action):
    """Build a synthetic action-indexed batch: batch(id, matched_coral={d_margin:..,flip_rate:..,JS:..,d_entropy:..,post_sep:..},
    spdim={...}, t3a={...}). Missing features default to neutral 0.0."""
    from acar.v5 import protocol as P
    feats = {}
    for a in P.ACTIONS:
        d = dict(per_action.get(a, {}))
        for f in P.FEATURES:
            d.setdefault(f, 0.0)
        feats[a] = d
    return {"batch_id": batch_id, "features": feats}
