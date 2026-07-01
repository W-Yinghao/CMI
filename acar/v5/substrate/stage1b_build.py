"""ACAR V5 Stage-1B build ORCHESTRATOR. Importing this module reads NOTHING and imports NO heavy/real-data deps (torch/mne/cmi/
acar.v3 are never imported here — the real reader/trainer import them lazily, and only after the gate). The orchestrator:

  1. calls require_stage1b_full_build_ready(...) FIRST — before any filesystem read, dataset open, model init, or training call;
  2. defaults to DRY-RUN (execute=False) — validates the gate and reports what WOULD build, reading nothing;
  3. on execute=True, requires an authorized DEV reader + trainer (the real ones are a later patch; the CLI default is unwired,
     so `--execute` fails closed) and builds EXACTLY the 30 fold-contained substrates, handing the trainer ONLY FIT (train/val)
     subjects (CAL/EVAL are never passed → no split contamination), and validating each artifact's registry hash set.
"""
from __future__ import annotations
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_artifacts as ART
from acar.v5.substrate import dev_reader_contract as DR
from acar.v5.substrate import train_contract as TR


class Stage1bBuildError(RuntimeError):
    """Raised when a Stage-1B build produces an incomplete / non-conforming set of substrates."""


def run_stage1b_build(plan, authorization, runtime_lock, *, execute=False, dev_reader=None, trainer=None):
    """Gate-first Stage-1B build. Returns a report. With execute=False (default) it reads/ trains NOTHING."""
    ready = RL.require_stage1b_full_build_ready(plan, authorization, runtime_lock)   # GATE BEFORE ANYTHING
    if not execute:
        return {"status": "STAGE1B_BUILD_DRYRUN", "n_would_build": ready["built_fold_substrates"],
                "would_build_refs": sorted(SA.CANONICAL_FOLD_REFS), "reads": 0, "trained": 0,
                "note": "dry-run; gate validated; NO read/train (needs execute=True + an authorized reader/trainer)"}

    DR.require_reader(dev_reader)                                 # execute path: real reader/trainer required (unwired by default)
    TR.require_trainer(trainer)
    subjects_by_disease, artifacts = {}, {}
    for e in plan["fold_contained_refs"]:
        disease, fold, seed = e["disease"], int(e["fold"]), int(e["seed"])
        cohort_paths = e["source_paths_by_cohort"]
        if disease not in subjects_by_disease:                    # metadata listing (post-gate), per disease across its cohorts
            subs = sorted({s for c, p in cohort_paths.items() for s in dev_reader.list_subjects(disease, c, p)})
            subjects_by_disease[disease] = subs
        split = SPL.make_fold(subjects_by_disease[disease], fold)  # subject-disjoint FIT/CAL/EVAL, TRAIN/VAL
        # hand the trainer ONLY FIT (train/val) subjects — CAL/EVAL never leave this function
        art = trainer.train_fold(disease, fold, seed, list(split["train"]), list(split["val"]), dict(cohort_paths))
        ART.validate_artifact_manifest(art, expected_ref=e["ref"], disease=disease, fold=fold, seed=seed)
        if e["ref"] in artifacts:
            raise Stage1bBuildError(f"duplicate built ref {e['ref']}")
        artifacts[e["ref"]] = art
    if set(artifacts) != set(SA.CANONICAL_FOLD_REFS):
        raise Stage1bBuildError(f"build produced {len(artifacts)} substrates != the 30 canonical fold refs")
    return {"status": "STAGE1B_BUILT", "n_artifacts": len(artifacts), "artifacts": artifacts,
            "run_id": ready["run_id"], "device_kind": ready["device_kind"]}


def main(argv=None):  # pragma: no cover — CLI; default dry-run, real execute is unwired in Stage-1B2
    import argparse
    import json
    ap = argparse.ArgumentParser(description="ACAR v5 Stage-1B build (default DRY-RUN; real execute unwired in Stage-1B2)")
    ap.add_argument("--execute", action="store_true", help="attempt a real build (requires an authorized reader/trainer — unwired here)")
    ap.add_argument("--auth-json")
    ap.add_argument("--runtime-lock-json")
    ap.add_argument("--full-build-manifest-json")
    args = ap.parse_args(argv)

    def _load(p):
        with open(p) as f:
            return json.load(f)

    if not args.execute:
        # dry-run needs the three specs too (to validate the gate); if absent, just report the intended contract shape
        if not (args.auth_json and args.runtime_lock_json and args.full_build_manifest_json):
            print(json.dumps({"status": "STAGE1B_BUILD_DRYRUN_NOSPEC",
                              "note": "supply --auth-json --runtime-lock-json --full-build-manifest-json to validate the gate"}, indent=2))
            return
        rep = run_stage1b_build(_load(args.full_build_manifest_json), _load(args.auth_json), _load(args.runtime_lock_json), execute=False)
        print(json.dumps({k: v for k, v in rep.items() if k != "artifacts"}, indent=2, sort_keys=True))
        return
    # --execute: real reader/trainer are unwired in Stage-1B2 → fails closed (proves the CLI cannot read real data yet)
    run_stage1b_build(_load(args.full_build_manifest_json), _load(args.auth_json), _load(args.runtime_lock_json),
                      execute=True, dev_reader=DR.UnwiredDevReader(), trainer=TR.UnwiredTrainer())


if __name__ == "__main__":
    main()
