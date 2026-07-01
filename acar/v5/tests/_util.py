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
    """{(disease, cohort): [RAW subject ids]} — synthetic, deterministic; RAW (not namespaced) so the subject index must add
    disease/cohort (raw ids intentionally repeat across cohorts to exercise no-collapse)."""
    from acar.v5 import protocol as P
    out = {}
    for d, cs in P.DEV_COHORTS.items():
        for c in cs:
            out[(d, c)] = [f"sub-{i:03d}" for i in range(n_per_cohort)]
    return out


def stage1b_subject_index(subs_by, disease):
    """Build the same canonical SubjectIndex the orchestrator builds, from a stage1b_fake_subjects() mapping (for test asserts)."""
    from acar.v5 import protocol as P
    from acar.v5.substrate import subject_index as SI
    return SI.build_subject_index(disease, {c: subs_by[(disease, c)] for c in P.DEV_COHORTS[disease]})


class FakeDevReader:
    """Synthetic DEV reader (no filesystem). Records list/read calls so tests can prove the gate runs before any read and that
    CAL/EVAL subjects are never read."""

    def __init__(self, subjects_by=None):
        self._subs = subjects_by if subjects_by is not None else stage1b_fake_subjects()
        self.listed = []
        self.read_calls = []
        self.label_calls = []

    def list_subjects(self, disease, cohort, path):
        self.listed.append((disease, cohort, path))
        return list(self._subs.get((disease, cohort), []))

    def read_subject_windows(self, disease, cohort, subject, path):
        self.read_calls.append((disease, cohort, subject, path))
        return {"marker": f"{disease}/{cohort}/{subject}"}

    def read_subject_label(self, disease, cohort, subject, path):   # FIT-only (reachable only via the FIT training view)
        self.label_calls.append((disease, cohort, subject, path))
        return 0


class FakeTrainer:
    """Synthetic trainer (no torch). New (Stage-1B3) signature: receives FIT subject KEYS + an AuthorizedFitDatasetView, reads
    only via the view (proving CAL/EVAL are unreachable), and returns a RAW build output with bytes payloads (the artifact writer
    computes the hashes)."""

    def __init__(self):
        self.received = {}          # ref -> {"train": set, "val": set}
        self.reads = {}             # ref -> [subject_keys read via the view]

    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        from acar.v5.substrate import stage1b_artifact_writer as AW
        ref = f"{disease}/fold{fold}/seed{seed}"
        self.received[ref] = {"train": set(train_subject_keys), "val": set(val_subject_keys)}
        rd = []
        for k in list(train_subject_keys) + list(val_subject_keys):
            dataset_view.read_windows(k)                       # only FIT keys → all allowed; a CAL/EVAL key would raise
            rd.append(k)
        self.reads[ref] = rd
        raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
        for bytes_key in sorted(set(AW.HASH_SOURCE.values())):
            raw[bytes_key] = f"{ref}:{bytes_key}".encode()     # deterministic synthetic bytes; writer hashes them
        return raw


class FakeFileTrainer:
    """Synthetic FILE-emitting trainer (no torch). Reads only via the view; writes tiny temp files per ref and returns their
    paths (exercises the file-backed artifact writer). Never scans roots."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.received = {}

    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        import os
        from acar.v5.substrate import stage1b_file_artifact_writer as FW
        ref = f"{disease}/fold{fold}/seed{seed}"
        self.received[ref] = {"train": set(train_subject_keys), "val": set(val_subject_keys)}
        for k in list(train_subject_keys) + list(val_subject_keys):
            dataset_view.read_windows(k)
        d = os.path.join(self.out_dir, ref.replace("/", "_"))
        os.makedirs(d, exist_ok=True)
        raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
        for path_key in sorted(set(FW.FILE_SOURCE.values())):
            p = os.path.join(d, path_key + ".bin")
            with open(p, "wb") as f:
                f.write(f"{ref}:{path_key}".encode())
            raw[path_key] = p
        return raw


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
