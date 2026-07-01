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
          "implementation_base_sha": "0" * 40, "run_id": run_id, "device_kind": device_kind,
          "status": "CAPTURED_AND_VERIFIED"}
    lk.update(over)
    return lk


def stage1b_full_plan():
    """A SYNTHETIC full-build plan: every one of the 30 fold refs carries source_paths_by_cohort with per-cohort synthetic DEV
    paths (strings only; nothing is opened)."""
    from acar.v5 import protocol as P
    from acar.v5.substrate import plan as PLAN
    pl = PLAN.build_substrate_plan()
    for e in pl["fold_contained_refs"]:
        e["source_paths_by_cohort"] = {c: f"/projects/datalake/raw/bids/{c}/sub-XXX" for c in P.DEV_COHORTS[e["disease"]]}
    return pl


def stage1b_fake_subjects(n_per_cohort=20):
    """{(disease, cohort): [namespaced subject ids]} — synthetic, deterministic; enough per cohort for K=5 splits."""
    from acar.v5 import protocol as P
    out = {}
    for d, cs in P.DEV_COHORTS.items():
        for c in cs:
            out[(d, c)] = [f"{c}/sub-{i:03d}" for i in range(n_per_cohort)]
    return out


class FakeDevReader:
    """Synthetic DEV reader (no filesystem). Records list/read calls so tests can prove the gate runs before any read and that
    CAL/EVAL subjects are never read."""

    def __init__(self, subjects_by=None):
        self._subs = subjects_by if subjects_by is not None else stage1b_fake_subjects()
        self.listed = []
        self.read_calls = []

    def list_subjects(self, disease, cohort, path):
        self.listed.append((disease, cohort, path))
        return list(self._subs.get((disease, cohort), []))

    def read_subject_windows(self, disease, cohort, subject, path):
        self.read_calls.append((disease, cohort, subject, path))
        return {"marker": f"{disease}/{cohort}/{subject}"}


class FakeTrainer:
    """Synthetic trainer (no torch). Records exactly which (train, val) subjects the build handed it, and emits an artifact
    manifest with a complete dummy registry hash set."""

    def __init__(self):
        self.received = {}

    def train_fold(self, disease, fold, seed, train_subjects, val_subjects, cohort_paths):
        import hashlib
        from acar.v5 import protocol as P
        ref = f"{disease}/fold{fold}/seed{seed}"
        self.received[ref] = {"train": set(train_subjects), "val": set(val_subjects)}
        art = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
        for h in P.REGISTRY_HASH_FIELDS:
            art[h] = hashlib.sha256(f"{ref}:{h}".encode()).hexdigest()
        return art


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
