"""ACAR V5 Stage-1B build ORCHESTRATOR. Importing this module reads NOTHING and imports NO heavy/real-data deps (torch/mne/cmi/
acar.v3 are never imported here — the real reader/trainer import them lazily inside their FACTORIES, only after the gate passes).

Guarantees (all synthetic-tested):
  1. `require_stage1b_full_build_ready(...)` runs FIRST — before any read/list/instantiation/import;
  2. FACTORY path: real reader/trainer are constructed ONLY after the gate (so no model init / GPU probe / BIDS scan pre-gate);
  3. default is DRY-RUN (execute=False) — reads nothing;
  4. builds EXACTLY the 30 fold substrates; per fold, the trainer gets ONLY the FIT (train∪val) canonical subject keys + an
     AuthorizedFitDatasetView (CAL/EVAL cannot be read); canonical SubjectKeys prevent cross-cohort id collapse;
  5. artifact hashes are COMPUTED from the trainer's output bytes (not trusted); the registry is populated exactly once per ref.
"""
from __future__ import annotations
import hashlib
import json
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_full_build_manifest as FBM
from acar.v5.substrate import subject_index as SI
from acar.v5.substrate import fit_dataset_view as FV
from acar.v5.substrate import stage1b_artifact_writer as AW
from acar.v5.substrate import stage1b_file_artifact_writer as FW
from acar.v5.substrate import stage1b_registry_populate as RP
from acar.v5.substrate import dev_reader_contract as DR
from acar.v5.substrate import train_contract as TR
from acar.v5.substrate.registry import SubstrateRegistry


class Stage1bBuildError(RuntimeError):
    """Raised when a Stage-1B build produces an incomplete / non-conforming set of substrates."""


def _disease_cohort_paths(plan, disease):
    for e in plan["fold_contained_refs"]:
        if e["disease"] == disease:
            return dict(e["source_paths_by_cohort"])          # consistent across the disease's refs (validated by the manifest)
    raise Stage1bBuildError(f"no fold ref for disease {disease}")


def run_stage1b_build(plan, authorization, runtime_lock, *, execute=False,
                      dev_reader=None, trainer=None, dev_reader_factory=None, trainer_factory=None, artifact_writer=None):
    """Gate-first Stage-1B build. execute=False (default) reads/trains NOTHING. On execute=True, either pass ready-made
    dev_reader+trainer (synthetic test path) OR dev_reader_factory+trainer_factory (real path — instantiated ONLY after the gate).
    `artifact_writer(raw, *, expected_ref, disease, fold, seed)` defaults to the bytes writer; the real build passes the
    file-backed writer. Returns a report incl. the populated SubstrateRegistry. (Production real runs use run_stage1b_real_build,
    which forbids preconstructed objects.)"""
    if artifact_writer is None:
        artifact_writer = AW.write_artifact
    ready = RL.require_stage1b_full_build_ready(plan, authorization, runtime_lock)   # GATE BEFORE ANYTHING
    if not execute:
        return {"status": "STAGE1B_BUILD_DRYRUN", "n_would_build": ready["built_fold_substrates"],
                "would_build_refs": sorted(SA.CANONICAL_FOLD_REFS), "reads": 0, "trained": 0,
                "note": "dry-run; gate validated; NO read/train/instantiate"}

    # gate passed → NOW resolve the reader/trainer. Factories are instantiated here (post-gate), never before.
    if dev_reader_factory is not None or trainer_factory is not None:
        if dev_reader_factory is None or trainer_factory is None:
            raise Stage1bBuildError("both dev_reader_factory and trainer_factory are required (or neither)")
        if dev_reader is not None or trainer is not None:
            raise Stage1bBuildError("pass factories OR objects, not both")
        dev_reader = dev_reader_factory()                     # <-- real import/model-init/GPU-probe happens here, post-gate
        trainer = trainer_factory()
    DR.require_reader(dev_reader)
    TR.require_trainer(trainer)

    index_by_disease, artifacts = {}, {}
    for e in plan["fold_contained_refs"]:
        disease, fold, seed = e["disease"], int(e["fold"]), int(e["seed"])
        cohort_paths = _disease_cohort_paths(plan, disease)
        if disease not in index_by_disease:                   # list raw subjects once per disease; canonical index (no collapse)
            per_cohort_raw = {c: list(dev_reader.list_subjects(disease, c, p)) for c, p in cohort_paths.items()}
            index_by_disease[disease] = SI.build_subject_index(disease, per_cohort_raw)
        idx = index_by_disease[disease]
        split = SPL.make_fold(idx.subject_keys, fold)         # split on canonical SubjectKeys
        allowed = set(split["train"]) | set(split["val"])     # FIT only
        view = FV.AuthorizedFitDatasetView(idx, allowed, dev_reader, cohort_paths)   # trainer gets NO raw roots
        raw = trainer.train_fold(disease, fold, seed, list(split["train"]), list(split["val"]), view)
        art = artifact_writer(raw, expected_ref=e["ref"], disease=disease, fold=fold, seed=seed)   # hashes computed, not trusted
        if e["ref"] in artifacts:
            raise Stage1bBuildError(f"duplicate built ref {e['ref']}")
        artifacts[e["ref"]] = art
    if set(artifacts) != set(SA.CANONICAL_FOLD_REFS):
        raise Stage1bBuildError(f"build produced {len(artifacts)} substrates != the 30 canonical fold refs")

    registry = SubstrateRegistry()                            # populate exactly once per canonical ref
    env_lock_sha256 = hashlib.sha256(json.dumps(runtime_lock, sort_keys=True).encode()).hexdigest()
    n = RP.populate_registry(registry, artifacts, git_commit=authorization["implementation_base_sha"],
                             env_lock_sha256=env_lock_sha256, channel_montage="10-20-19",
                             sampling_rate=128, windowing_config="4s/512")
    return {"status": "STAGE1B_BUILT", "n_artifacts": len(artifacts), "n_registered": n, "artifacts": artifacts,
            "registry": registry, "run_id": ready["run_id"], "device_kind": ready["device_kind"]}


def run_stage1b_real_build(plan, authorization, runtime_lock, *, dev_reader_factory, trainer_factory, artifact_writer=None):
    """PRODUCTION real-run entry. Accepts ONLY factories (no preconstructed objects) so the real reader/trainer can never be
    instantiated before the gate. Defaults to the FILE-backed artifact writer (real trainers emit files, not in-memory bytes).
    Both factories are required and must be callables. (Executing on real DEV data still requires an authorized run: the real
    reader/trainer emit their signal read/training only at the Stage-1B run.)"""
    if not callable(dev_reader_factory) or not callable(trainer_factory):
        raise Stage1bBuildError("run_stage1b_real_build requires callable dev_reader_factory and trainer_factory")
    if artifact_writer is None:
        artifact_writer = FW.write_artifact_from_files
    return run_stage1b_build(plan, authorization, runtime_lock, execute=True,
                             dev_reader_factory=dev_reader_factory, trainer_factory=trainer_factory,
                             artifact_writer=artifact_writer)


def main(argv=None):  # pragma: no cover — CLI; default dry-run, real execute is unwired here
    import argparse
    ap = argparse.ArgumentParser(description="ACAR v5 Stage-1B build (default DRY-RUN; real execute unwired without a factory)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--auth-json")
    ap.add_argument("--runtime-lock-json")
    ap.add_argument("--full-build-manifest-json")
    args = ap.parse_args(argv)

    def _load(p):
        with open(p) as f:
            return json.load(f)

    if not (args.auth_json and args.runtime_lock_json and args.full_build_manifest_json):
        print(json.dumps({"status": "STAGE1B_BUILD_DRYRUN_NOSPEC",
                          "note": "supply --auth-json --runtime-lock-json --full-build-manifest-json"}, indent=2))
        return
    plan, auth, lock = _load(args.full_build_manifest_json), _load(args.auth_json), _load(args.runtime_lock_json)
    if not args.execute:
        rep = run_stage1b_build(plan, auth, lock, execute=False)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("artifacts", "registry")}, indent=2, sort_keys=True))
        return
    # --execute here uses the UNWIRED reader/trainer → fails closed (real reader/trainer come via a later-authorized factory)
    run_stage1b_build(plan, auth, lock, execute=True, dev_reader=DR.UnwiredDevReader(), trainer=TR.UnwiredTrainer())


if __name__ == "__main__":
    main()
