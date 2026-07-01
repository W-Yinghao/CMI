"""ACAR V5 Stage-1B build ORCHESTRATOR. Importing this module reads NOTHING and imports NO heavy/real-data deps (torch/mne/cmi/
acar.v3 are never imported here — the real reader/trainer/dumper import them lazily inside their FACTORIES, only after the gate).

Guarantees (all synthetic-tested):
  1. `require_stage1b_full_build_ready(...)` runs FIRST — before any read/list/instantiation/import;
  2. FACTORY path: real reader/trainer/dumper are constructed ONLY after the gate, bound to the gate-issued execution context;
  3. default is DRY-RUN (execute=False) — reads nothing;
  4. builds EXACTLY the 30 fold substrates. Per fold the build is TWO-PHASE (stage1b_embedding_orchestrator):
       A. FIT-only training — the trainer gets ONLY the FIT (train∪val) canonical keys + an AuthorizedFitDatasetView (labels
          readable, CAL/EVAL unreadable) and emits the encoder/source-state artifacts but NOT feat_dump;
       B. label-free feature dump — an AuthorizedEmbeddingDatasetView over ALL fold subjects drives a separate dumper (no
          read_label) that emits feat_dump.
  5. artifact hashes are COMPUTED from the trainer/dumper output (not trusted); the registry is populated exactly once per ref via
     the finalize BARRIER (all-or-none count + global path uniqueness + canonical config sidecars, then a FINALIZED marker).
"""
from __future__ import annotations
import functools
import hashlib
import json
from acar.v5 import splits as SPL
from acar.v5.substrate import stage1_runtime_lock as RL
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import subject_index as SI
from acar.v5.substrate import stage1b_artifact_writer as AW
from acar.v5.substrate import stage1b_file_artifact_writer as FW
from acar.v5.substrate import stage1b_execution_context as EC
from acar.v5.substrate import stage1b_embedding_orchestrator as ORC
from acar.v5.substrate import stage1b_finalize as FIN
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
                      dev_reader=None, trainer=None, dumper=None,
                      dev_reader_factory=None, trainer_factory=None, dumper_factory=None,
                      artifact_writer=None, output_root=None):
    """Gate-first Stage-1B build. execute=False (default) reads/trains NOTHING. On execute=True, either pass ready-made
    dev_reader+trainer+dumper (synthetic test path) OR dev_reader_factory+trainer_factory+dumper_factory (real path —
    instantiated ONLY after the gate, each called with the gate-issued Stage1BExecutionContext). `artifact_writer` defaults to the
    bytes writer; the real build passes the file-backed writer. Returns a report incl. the populated SubstrateRegistry."""
    if artifact_writer is None:
        artifact_writer = AW.write_artifact
    ready = RL.require_stage1b_full_build_ready(plan, authorization, runtime_lock)   # GATE BEFORE ANYTHING
    if not execute:
        return {"status": "STAGE1B_BUILD_DRYRUN", "n_would_build": ready["built_fold_substrates"],
                "would_build_refs": sorted(SA.CANONICAL_FOLD_REFS), "reads": 0, "trained": 0,
                "note": "dry-run; gate validated; NO read/train/instantiate"}

    # gate passed → NOW resolve reader/trainer/dumper. Factories are instantiated here (post-gate), bound to the context.
    if dev_reader_factory is not None or trainer_factory is not None or dumper_factory is not None:
        if not (dev_reader_factory is not None and trainer_factory is not None and dumper_factory is not None):
            raise Stage1bBuildError("dev_reader_factory, trainer_factory and dumper_factory are required together (or none)")
        if dev_reader is not None or trainer is not None or dumper is not None:
            raise Stage1bBuildError("pass factories OR objects, not both")
        if not output_root:
            raise Stage1bBuildError("factory path requires output_root (for the execution context)")
        ctx = EC.build_execution_context(authorization, runtime_lock, plan, output_root=output_root)   # AFTER the gate
        dev_reader = dev_reader_factory(ctx)                  # <-- real import/model-init/GPU-probe happens here, post-gate
        trainer = trainer_factory(ctx)
        dumper = dumper_factory(ctx)
    DR.require_reader(dev_reader)
    TR.require_trainer(trainer)
    ORC.require_dumper(dumper)

    index_by_disease, artifacts, paths_by_ref, sidecars_by_ref = {}, {}, {}, {}
    for e in plan["fold_contained_refs"]:
        disease, fold, seed, ref = e["disease"], int(e["fold"]), int(e["seed"]), e["ref"]
        cohort_paths = _disease_cohort_paths(plan, disease)
        if disease not in index_by_disease:                   # list raw subjects once per disease; canonical index (no collapse)
            per_cohort_raw = {c: list(dev_reader.list_subjects(disease, c, p)) for c, p in cohort_paths.items()}
            index_by_disease[disease] = SI.build_subject_index(disease, per_cohort_raw)
        idx = index_by_disease[disease]
        split = SPL.make_fold(idx.subject_keys, fold)         # split on canonical SubjectKeys
        raw, sidecars = ORC.build_fold_raw(disease, fold, seed, ref, idx, split, dev_reader, trainer, dumper, cohort_paths)
        art = artifact_writer(raw, expected_ref=ref, disease=disease, fold=fold, seed=seed)   # hashes computed, not trusted
        if ref in artifacts:
            raise Stage1bBuildError(f"duplicate built ref {ref}")
        file_paths = art.pop("_paths", None)                  # ref-local resolved paths (file-backed writer only)
        if file_paths:
            paths_by_ref[ref] = file_paths
        sidecars_by_ref[ref] = sidecars
        artifacts[ref] = art
    if set(artifacts) != set(SA.CANONICAL_FOLD_REFS):
        raise Stage1bBuildError(f"build produced {len(artifacts)} substrates != the 30 canonical fold refs")

    # ALL-OR-NONE finalize BARRIER: count + global path uniqueness + canonical config sidecars validated IN FULL before any
    # registration; a failure leaves the registry EMPTY and writes no FINALIZED marker.
    registry = SubstrateRegistry()
    env_lock_sha256 = hashlib.sha256(json.dumps(runtime_lock, sort_keys=True).encode()).hexdigest()
    n = FIN.finalize_and_populate(
        registry, artifacts, git_commit=authorization["implementation_base_sha"], env_lock_sha256=env_lock_sha256,
        channel_montage="10-20-19", sampling_rate=128, windowing_config="4s/512",
        paths_by_ref=(paths_by_ref or None), sidecars_by_ref=sidecars_by_ref,
        output_root=output_root, run_id=ready["run_id"])
    return {"status": "STAGE1B_BUILT", "n_artifacts": len(artifacts), "n_registered": n, "artifacts": artifacts,
            "registry": registry, "run_id": ready["run_id"], "device_kind": ready["device_kind"]}


def run_stage1b_real_build(plan, authorization, runtime_lock, *, output_root, dev_reader_factory, trainer_factory,
                           dumper_factory, artifact_writer=None):
    """PRODUCTION real-run entry. Accepts ONLY factories (no preconstructed objects) so the real reader/trainer/dumper can never be
    instantiated before the gate; factories are called with the gate-issued execution context. Defaults to the FILE-backed artifact
    writer (real trainers/dumpers emit files) with PER-REF output containment enforced (output_root/run_id/safe_ref_slug). All
    three factories are required + callable."""
    if not (callable(dev_reader_factory) and callable(trainer_factory) and callable(dumper_factory)):
        raise Stage1bBuildError("run_stage1b_real_build requires callable dev_reader_factory, trainer_factory and dumper_factory")
    if not output_root:
        raise Stage1bBuildError("run_stage1b_real_build requires output_root")
    if artifact_writer is None:                               # per-ref containment via run_id
        artifact_writer = functools.partial(FW.write_artifact_from_files, output_root=output_root,
                                            run_id=authorization["run_id"])
    return run_stage1b_build(plan, authorization, runtime_lock, execute=True, output_root=output_root,
                             dev_reader_factory=dev_reader_factory, trainer_factory=trainer_factory,
                             dumper_factory=dumper_factory, artifact_writer=artifact_writer)


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
    # --execute here uses UNWIRED reader/trainer/dumper → fails closed (real ones come via a later-authorized factory)
    run_stage1b_build(plan, auth, lock, execute=True, dev_reader=DR.UnwiredDevReader(), trainer=TR.UnwiredTrainer(),
                      dumper=ORC.UnwiredEmbeddingDumper())


if __name__ == "__main__":
    main()
