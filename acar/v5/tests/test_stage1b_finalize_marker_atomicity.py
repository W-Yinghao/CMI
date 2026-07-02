"""Guard (Stage-1B7): the FINALIZED marker is written ATOMICALLY (tmp → os.replace); the marker exists IFF the registry is fully
populated. On a marker-write failure AFTER population, the registry is rolled back to empty (no registry-without-marker state).
Synthetic temp files only."""
from __future__ import annotations
import json
import os
import tempfile
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_output_layout as LO
from acar.v5.substrate import stage1b_file_artifact_writer as FW
from acar.v5.substrate import stage1b_finalize as FIN
from acar.v5.substrate import preprocessing_config as PC
from acar.v5.substrate import training_config as TC
from acar.v5.substrate import stage1b_feature_dump_writer as FDW
from acar.v5.substrate.registry import SubstrateRegistry
from acar.v5.tests._util import expect_raises, ok

RUN = "run-syn-0001"
_META = dict(git_commit="0" * 40, env_lock_sha256="a" * 64, channel_montage="10-20-19", sampling_rate=128,
             windowing_config="4s/512")


def _materialize(root, run_id):
    """Create 30 valid per-ref file artifacts + config sidecars on disk; return (artifacts, paths_by_ref, sidecars_by_ref)."""
    artifacts, paths_by_ref, sidecars_by_ref = {}, {}, {}
    for ref in SA.CANONICAL_FOLD_REFS:
        disease = ref.split("/")[0]
        fold = int(ref.split("fold")[1].split("/")[0])
        seed = int(ref.split("seed")[1])
        d = LO.ref_output_dir(root, run_id, ref)
        os.makedirs(d, exist_ok=True)
        raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
        for pk in sorted(set(FW.FILE_SOURCE.values())):
            if pk == "preprocessing_config_path":
                p = os.path.join(d, "preprocessing_config.json")
                with open(p, "w") as f:
                    f.write(PC.canonical_json())
            elif pk == "feat_dump_path":
                p = os.path.join(d, "feat_dump.npz")          # a schema-valid feature dump (finalize parses it)
                FDW.write_feature_dump(p, ref=ref, disease=disease, fold=fold, seed=seed,
                                       preprocessing_config_sha256="0" * 64, training_config_sha256="0" * 64,
                                       encoder_checkpoint_file_sha256="0" * 64, source_state_file_sha256="0" * 64,
                                       records=[(f"{disease}/dsX/sub-1", "train", 0, [0.0, 1.0])])
            else:
                p = os.path.join(d, pk + ".bin")
                with open(p, "wb") as f:
                    f.write((ref + pk).encode())
            raw[pk] = p
        tcp = os.path.join(d, "training_config.json")
        with open(tcp, "w") as f:
            f.write(TC.canonical_json())
        art = FW.write_artifact_from_files(raw, expected_ref=ref, disease=disease, fold=fold, seed=seed,
                                           output_root=root, run_id=run_id)
        paths_by_ref[ref] = art.pop("_paths")
        sidecars_by_ref[ref] = {"training_config_path": tcp}
        artifacts[ref] = art
    return artifacts, paths_by_ref, sidecars_by_ref


def test_marker_writer_atomic_success_and_failure():
    with tempfile.TemporaryDirectory() as root:
        FIN.write_finalized_marker(root, RUN, {"status": "FINALIZED", "n_registered": 30})
        mp = FIN.marker_path(root, RUN)
        assert os.path.isfile(mp) and not os.path.exists(mp + ".tmp")
        assert json.load(open(mp))["n_registered"] == 30
    with tempfile.TemporaryDirectory() as root:
        # make the run root a FILE so the marker directory cannot be created → fail closed, no marker
        with open(os.path.join(root, RUN), "w") as f:
            f.write("x")
        expect_raises(Exception, lambda: FIN.write_finalized_marker(root, RUN, {"status": "FINALIZED"}))
    ok("write_finalized_marker: success leaves FINALIZED.json (no .tmp); an unwritable target fails closed with no marker")


def test_finalize_success_marker_and_registry_together():
    with tempfile.TemporaryDirectory() as root:
        reg = SubstrateRegistry()
        arts, paths, sides = _materialize(root, RUN)
        n = FIN.finalize_and_populate(reg, arts, paths_by_ref=paths, sidecars_by_ref=sides, output_root=root, run_id=RUN, **_META)
        mp = FIN.marker_path(root, RUN)
        assert n == 30 and len(reg._entries) == 30 and os.path.isfile(mp) and not os.path.exists(mp + ".tmp")
        assert json.load(open(mp))["n_registered"] == 30
    ok("file-backed finalize success → 30 registered AND an atomic FINALIZED marker (present together)")


def test_marker_failure_rolls_back_registry():
    with tempfile.TemporaryDirectory() as root:
        reg = SubstrateRegistry()
        arts, paths, sides = _materialize(root, RUN)
        os.makedirs(FIN.marker_path(root, RUN))               # marker path pre-occupied by a DIRECTORY → os.replace fails
        expect_raises(FIN.Stage1bFinalizeError,
                      lambda: FIN.finalize_and_populate(reg, arts, paths_by_ref=paths, sidecars_by_ref=sides,
                                                        output_root=root, run_id=RUN, **_META))
        assert len(reg._entries) == 0                          # rolled back → no registry-without-marker state
    ok("a marker-write failure after population → registry rolled back to empty (marker exists IFF registry populated)")


def test_finalize_rejects_malformed_feat_dump():
    with tempfile.TemporaryDirectory() as root:
        reg = SubstrateRegistry()
        arts, paths, sides = _materialize(root, RUN)
        victim = sorted(paths)[0]
        with open(paths[victim]["feat_dump_path"], "wb") as f:   # clobber one ref's feat_dump with non-schema bytes
            f.write(b"not-a-valid-npz")
        expect_raises(FIN.Stage1bFinalizeError,
                      lambda: FIN.finalize_and_populate(reg, arts, paths_by_ref=paths, sidecars_by_ref=sides,
                                                        output_root=root, run_id=RUN, **_META))
        assert len(reg._entries) == 0 and not os.path.exists(FIN.marker_path(root, RUN))
    ok("a malformed feat_dump (not the pinned schema) → finalize barrier fails; registry empty; no marker (dumper-agnostic)")


def main():
    print("ACAR v5 Stage-1B7 guard: finalize marker atomicity")
    test_marker_writer_atomic_success_and_failure()
    test_finalize_success_marker_and_registry_together()
    test_marker_failure_rolls_back_registry()
    test_finalize_rejects_malformed_feat_dump()
    print("ALL V5 STAGE1B-FINALIZE-MARKER-ATOMICITY GUARDS PASS")


if __name__ == "__main__":
    main()
