"""Shared test helpers (stdlib only)."""
from __future__ import annotations


def _exc_name(exc):
    return getattr(exc, "__name__", None) or "/".join(getattr(x, "__name__", str(x)) for x in exc)


def expect_raises(exc, fn, msg=""):
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # noqa
        raise AssertionError(f"expected {_exc_name(exc)}, got {type(e).__name__}: {e} ({msg})")
    raise AssertionError(f"expected {_exc_name(exc)}, no error raised ({msg})")


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


def stage1b_repair_staging_root():
    """A fresh, non-existent, absolute repair-staging root for Stage-1B real-build tests (parent dir exists + empty, child absent) —
    passes stage1b_repair_staging validation (not a symlink, not under output_root/raw paths, absent-or-empty at launch)."""
    import os
    import tempfile
    return os.path.join(tempfile.mkdtemp(prefix="acar_v5_repair_staging_"), "staging")


def stage1b_reader_ctx(disease, cohort, path, repair_staging_root, output_root="/run/out"):
    """A gate-issued-style execution context whose approved source path for (disease, cohort) == `path`, carrying a Stage-1B15
    repair_staging_root — for exercising the production real reader without a full build."""
    from acar.v5.substrate import stage1b_execution_context as EC
    from acar.v5.substrate import stage1b_authorization as SA
    plan = {"fold_contained_refs": [{"disease": disease, "source_paths_by_cohort": {cohort: path}}]}
    return EC.build_execution_context(stage1b_auth(protocol_tag_target_sha=SA.PROTOCOL_TAG_TARGET_SHA_FULL), {}, plan,
                                      output_root=output_root, repair_staging_root=repair_staging_root)


def capture_preprocess_staging(fn):
    """Monkeypatch real_mne_reader.preprocess_subject to CAPTURE the staging_dir of each call (no real read), run fn(), restore.
    Returns (list_of_captured_staging_dirs, fn_result)."""
    from acar.v5.substrate import real_mne_reader as RMR
    cap, orig = [], RMR.preprocess_subject

    def _fake(disease, cohort, subject, subject_dir, *, mne=None, staging_dir=None):
        cap.append(staging_dir)
        return f"WINDOWS:{disease}/{cohort}/{subject}"

    RMR.preprocess_subject = _fake
    try:
        return cap, fn()
    finally:
        RMR.preprocess_subject = orig


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

    def subject_label_resolvable(self, disease, cohort, subject, path):
        return True                                            # synthetic subjects are all eligible

    def windows_only(self):
        """A label-INCAPABLE facade for the embedding view — shares the read_calls audit list but has NO read_subject_label and no
        reference back to this label-capable reader."""
        return _WindowsOnlyFakeReader(self._subs, self.read_calls)


class _WindowsOnlyFakeReader:
    """Fake windows-only reader (no read_subject_label). Holds only the subjects table + a shared read_calls list."""

    def __init__(self, subs, read_calls):
        self._subs = subs
        self.read_calls = read_calls

    def read_subject_windows(self, disease, cohort, subject, path):
        self.read_calls.append((disease, cohort, subject, path))
        return {"marker": f"{disease}/{cohort}/{subject}"}


def make_subject_windows(subject_key, n_windows=1):
    """A tiny VALID SubjectWindows for tests that exercise the real trainer/dumper (which validate the payload)."""
    import numpy as np
    from acar.v5.substrate import subject_windows as SW
    from acar.v5.substrate import preprocessing_config as PC
    disease, cohort, raw = subject_key.split("/")
    return SW.SubjectWindows(subject_key=subject_key, disease=disease, cohort=cohort, raw_subject_id=raw,
                             n_windows=n_windows, n_channels=19, n_samples=512, sfreq=128, channels=PC.CHANNELS_19,
                             preprocessing_config_sha256=PC.config_sha256(),
                             windows=np.zeros((n_windows, 19, 512), dtype=np.float32), provenance="fake")


class FakeWindowsDevReader:
    """Like FakeDevReader but read_subject_windows returns a VALIDATED SubjectWindows (for real trainer/dumper tests). read_label
    returns int 0/1. windows_only() returns a label-incapable facade that also yields SubjectWindows."""

    def __init__(self, subjects_by=None):
        self._subs = subjects_by if subjects_by is not None else stage1b_fake_subjects()
        self.read_calls = []
        self.label_calls = []

    def list_subjects(self, disease, cohort, path):
        return list(self._subs.get((disease, cohort), []))

    def read_subject_windows(self, disease, cohort, subject, path):
        self.read_calls.append((disease, cohort, subject, path))
        return make_subject_windows(f"{disease}/{cohort}/{subject}")

    def read_subject_label(self, disease, cohort, subject, path):
        self.label_calls.append((disease, cohort, subject, path))
        return 0

    def subject_label_resolvable(self, disease, cohort, subject, path):
        return True

    def windows_only(self):
        return _WindowsOnlyWindowsFake(self.read_calls)


class _WindowsOnlyWindowsFake:
    """Label-incapable facade returning SubjectWindows (no read_subject_label, no back-reference to a label-capable reader)."""

    def __init__(self, read_calls):
        self.read_calls = read_calls

    def read_subject_windows(self, disease, cohort, subject, path):
        self.read_calls.append((disease, cohort, subject, path))
        return make_subject_windows(f"{disease}/{cohort}/{subject}")


class FakeTrainer:
    """Synthetic FIT-only trainer (no torch). Receives FIT subject KEYS + an AuthorizedFitDatasetView, reads signal+labels only via
    the view (proving CAL/EVAL are unreachable), and returns a RAW build output with the 5 NON-feat bytes payloads — it never emits
    feat_dump (that is the dumper's job). The artifact writer computes the hashes."""

    def __init__(self):
        self.received = {}          # ref -> {"train": set, "val": set}
        self.reads = {}             # ref -> [subject_keys whose windows were read via the view]
        self.label_reads = {}       # ref -> [subject_keys whose labels were read via the view]

    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        from acar.v5.substrate import stage1b_artifact_writer as AW
        ref = f"{disease}/fold{fold}/seed{seed}"
        self.received[ref] = {"train": set(train_subject_keys), "val": set(val_subject_keys)}
        rd, lb = [], []
        for k in list(train_subject_keys) + list(val_subject_keys):
            dataset_view.read_windows(k)                       # only FIT keys → all allowed; a CAL/EVAL key would raise
            dataset_view.read_label(k)                         # FIT-only labels via the view
            rd.append(k)
            lb.append(k)
        self.reads[ref] = rd
        self.label_reads[ref] = lb
        raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
        for bytes_key in sorted(set(AW.HASH_SOURCE.values())):
            if bytes_key == "feat_dump_bytes":                 # NOT the trainer's job (the label-free dumper produces it)
                continue
            raw[bytes_key] = f"{ref}:{bytes_key}".encode()     # deterministic synthetic bytes; writer hashes them
        return raw


class FakeDumper:
    """Synthetic label-free embedding dumper (no torch). Reads ALL fold subjects via the label-free embedding view and emits ONLY
    feat_dump bytes. The view has no read_label, so this fake physically cannot read labels."""

    def __init__(self):
        self.reads = {}             # ref -> [ALL fold subject keys read (train∪val∪cal∪eval)]

    def dump_embeddings(self, disease, fold, seed, embedding_view, all_fold_subject_keys, train_result, role_by_subject=None):
        ref = f"{disease}/fold{fold}/seed{seed}"
        rd = []
        for k in all_fold_subject_keys:
            embedding_view.read_windows(k)                     # label-free
            rd.append(k)
        self.reads[ref] = rd
        return {"ref": ref, "disease": disease, "fold": fold, "seed": seed, "feat_dump_bytes": f"{ref}:feat_dump".encode()}


class FakeFileTrainer:
    """Synthetic FILE-emitting FIT-only trainer (no torch). Reads only via the view; writes the 5 NON-feat model/config files +
    training_config sidecar into the PER-REF output dir (output_root/run_id/safe_ref_slug) and returns their paths. Never scans
    roots, never emits feat_dump."""

    def __init__(self, out_dir, run_id):
        self.out_dir = out_dir
        self.run_id = run_id
        self.received = {}

    def train_fold(self, disease, fold, seed, train_subject_keys, val_subject_keys, dataset_view):
        import os
        from acar.v5.substrate import stage1b_file_artifact_writer as FW
        from acar.v5.substrate import stage1b_output_layout as LO
        from acar.v5.substrate import preprocessing_config as PC
        from acar.v5.substrate import training_config as TC
        ref = f"{disease}/fold{fold}/seed{seed}"
        self.received[ref] = {"train": set(train_subject_keys), "val": set(val_subject_keys)}
        for k in list(train_subject_keys) + list(val_subject_keys):
            dataset_view.read_windows(k)
            dataset_view.read_label(k)
        d = LO.ref_output_dir(self.out_dir, self.run_id, ref)
        os.makedirs(d, exist_ok=True)
        raw = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
        for path_key in sorted(set(FW.FILE_SOURCE.values())):
            if path_key == "feat_dump_path":                   # dumper writes feat_dump
                continue
            if path_key == "preprocessing_config_path":
                p = os.path.join(d, "preprocessing_config.json")
                content = PC.canonical_json().encode()          # canonical (finalize validates its content)
            else:
                p = os.path.join(d, path_key + ".bin")
                content = f"{ref}:{path_key}".encode()
            with open(p, "wb") as f:
                f.write(content)
            raw[path_key] = p
        tcp = os.path.join(d, "training_config.json")          # training_config sidecar (non-registry; finalize validates)
        with open(tcp, "w") as f:
            f.write(TC.canonical_json())
        raw["training_config_path"] = tcp
        return raw


class FakeFileDumper:
    """Synthetic FILE-emitting label-free dumper (no torch). Reads ALL fold subjects via the label-free embedding view and writes
    ONLY the feat_dump file into the PER-REF output dir. Never reads labels."""

    def __init__(self, out_dir, run_id):
        self.out_dir = out_dir
        self.run_id = run_id
        self.reads = {}

    def dump_embeddings(self, disease, fold, seed, embedding_view, all_fold_subject_keys, train_result, role_by_subject=None):
        import os
        from acar.v5.substrate import stage1b_output_layout as LO
        from acar.v5.substrate import stage1b_feature_dump_writer as FDW
        ref = f"{disease}/fold{fold}/seed{seed}"
        rd = []
        for k in all_fold_subject_keys:
            embedding_view.read_windows(k)
            rd.append(k)
        self.reads[ref] = rd
        d = LO.ref_output_dir(self.out_dir, self.run_id, ref)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "feat_dump.npz")                  # a schema-valid dump (finalize parses it)
        role_by_subject = role_by_subject or {}
        records = [(k, role_by_subject.get(k, "train"), 0, [0.0, 1.0, 2.0]) for k in all_fold_subject_keys]
        FDW.write_feature_dump(p, ref=ref, disease=disease, fold=fold, seed=seed,
                               preprocessing_config_sha256="0" * 64, training_config_sha256="0" * 64,
                               encoder_checkpoint_file_sha256="0" * 64, source_state_file_sha256="0" * 64, records=records)
        n_windows_by_subject = {k: 1 for k in all_fold_subject_keys}   # one record per subject → authoritative count = 1
        return {"ref": ref, "disease": disease, "fold": fold, "seed": seed, "feat_dump_path": p,
                "n_windows_by_subject": n_windows_by_subject}


def synthetic_canonical_artifacts(hashval=None):
    """A dict {ref: artifact_manifest} covering EXACTLY the 30 canonical fold refs, with placeholder 64-hex registry hashes (for
    finalize/registry barrier tests that don't need real files)."""
    from acar.v5 import protocol as P
    from acar.v5.substrate import stage1b_authorization as SA
    hv = hashval or ("a" * 64)
    artifacts = {}
    for ref in SA.CANONICAL_FOLD_REFS:
        disease = ref.split("/")[0]
        fold = int(ref.split("fold")[1].split("/")[0])
        seed = int(ref.split("seed")[1])
        art = {"ref": ref, "disease": disease, "fold": fold, "seed": seed}
        for h in P.REGISTRY_HASH_FIELDS:
            art[h] = hv
        artifacts[ref] = art
    return artifacts


def synthetic_canonical_paths(collide=False):
    """{ref: {path_key: path}} for the 30 canonical refs, each under its own per-ref slug dir (paths need not exist). If collide,
    a second ref reuses the first ref's feat_dump path (to exercise the GLOBAL cross-ref uniqueness guard)."""
    from acar.v5.substrate import stage1b_authorization as SA
    from acar.v5.substrate import stage1b_file_artifact_writer as FW
    refs = sorted(SA.CANONICAL_FOLD_REFS)
    paths = {}
    for ref in refs:
        slug = ref.replace("/", "_")
        paths[ref] = {pk: f"/out/run-syn-0001/{slug}/{pk}.bin" for pk in FW.FILE_SOURCE.values()}
    if collide:
        paths[refs[1]]["feat_dump_path"] = paths[refs[0]]["feat_dump_path"]
    return paths


def modern_channel_names():
    """The 19 canonical montage electrodes in MODERN 10-10 names (T3→T7, T4→T8, T5→P7, T6→P8; others unchanged)."""
    from acar.v5.substrate import preprocessing_config as PC
    m = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return [m.get(c, c) for c in PC.CHANNELS_19]


def make_mne_raw(ch_names, n_times=2048, sfreq=256.0, seed=0, nan_channels=()):
    """A synthetic mne RawArray fixture (real mne numerics, SYNTHETIC signal) for montage-completion tests. `nan_channels` sets
    those channel rows to NaN (to exercise the non-finite fail-closed path). Lazy mne."""
    import numpy as np
    import mne
    mne.set_log_level("ERROR")
    data = (np.random.RandomState(seed).randn(len(ch_names), n_times) * 1e-5).astype("float64")
    for c in nan_channels:
        data[list(ch_names).index(c), :] = np.nan
    info = mne.create_info(list(ch_names), float(sfreq), ["eeg"] * len(ch_names))
    return mne.io.RawArray(data, info, verbose="ERROR")


def channels_tsv_path_for(dirpath, stem):
    """BIDS channels.tsv path for a recording stem (the trailing '_eeg' is replaced, not appended)."""
    import os
    base = stem[: -len("_eeg")] if stem.endswith("_eeg") else stem
    return os.path.join(dirpath, base + "_channels.tsv")


def make_brainvision_triplet(dirpath, stem, ch_names=("Fp1", "Fp2"), n_points=300, sfreq=256.0, with_marker=True,
                             data_file=None, marker_file=None, generic_header=False, write_channels_tsv=False,
                             channels_tsv_names=None, ordinal_prefix_overrides=None):
    """Write a minimal, valid BrainVision triplet (.vhdr + .eeg + optional .vmrk; MULTIPLEXED IEEE_FLOAT_32) into `dirpath` and
    return the .vhdr path. SYNTHETIC signal only. `with_marker=False` omits the MarkerFile line and writes no .vmrk (the marker-less
    defect). `data_file`/`marker_file` override the header's INTERNAL DataFile/MarkerFile pointer values (to synthesize the
    stale-pointer defect). `generic_header=True` writes the [Channel Infos] names as ORDINAL placeholders `<PREFIX>00i` (default
    prefix EEG — the pure ds003944 defect); `ordinal_prefix_overrides={1-based position: prefix}` sets specific positions to a
    different prefix (e.g. {62: 'EOG', 63: 'ECG'} — the type-prefixed ds003944/ds003947 defect). `ch_names` still drives the channel
    COUNT and the channels.tsv rows. `write_channels_tsv=True` writes a BIDS channels.tsv (using `channels_tsv_names` if given, else
    `ch_names`)."""
    import csv
    import os
    import numpy as np
    os.makedirs(dirpath, exist_ok=True)
    nch = len(ch_names)
    data = (np.random.RandomState(abs(hash(stem)) % (2**32)).randn(nch, n_points) * 1e-6).astype("<f4")
    data.T.tofile(os.path.join(dirpath, stem + ".eeg"))          # multiplexed: pt-major, channel-minor
    interval_us = int(round(1_000_000.0 / float(sfreq)))
    ov = ordinal_prefix_overrides or {}
    header_names = ([f"{ov.get(i + 1, 'EEG')}{i+1:03d}" for i in range(nch)] if generic_header else list(ch_names))
    ch_lines = "".join(f"Ch{i+1}={nm},,1,\N{MICRO SIGN}V\n" for i, nm in enumerate(header_names))
    dfl = data_file if data_file is not None else (stem + ".eeg")
    mfl = marker_file if marker_file is not None else (stem + ".vmrk")
    hdr = ["Brain Vision Data Exchange Header File Version 1.0", "", "[Common Infos]", "Codepage=UTF-8",
           f"DataFile={dfl}"]
    if with_marker:
        hdr.append(f"MarkerFile={mfl}")
    hdr += ["DataFormat=BINARY", "DataOrientation=MULTIPLEXED", f"NumberOfChannels={nch}",
            f"DataPoints={n_points}", f"SamplingInterval={interval_us}", "", "[Binary Infos]",
            "BinaryFormat=IEEE_FLOAT_32", "", "[Channel Infos]", ch_lines]
    vhdr_path = os.path.join(dirpath, stem + ".vhdr")
    with open(vhdr_path, "w", encoding="utf-8") as f:
        f.write("\n".join(hdr))
    if with_marker:
        with open(os.path.join(dirpath, stem + ".vmrk"), "w", encoding="utf-8") as f:
            f.write("Brain Vision Data Exchange Marker File, Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
                    f"DataFile={stem}.eeg\n\n[Marker Infos]\nMk1=New Segment,,1,1,0\n")
    if write_channels_tsv:
        rows = channels_tsv_names if channels_tsv_names is not None else list(ch_names)
        with open(channels_tsv_path_for(dirpath, stem), "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["name", "type"])
            for nm in rows:
                w.writerow([nm, "EEG"])
    return vhdr_path


def make_fake_raw(ch_names, n_times=1024):
    """A minimal mne-Raw-like fixture (records DSP calls; deterministic varying data). Duck-typed for real_mne_reader."""
    import numpy as np

    class _FakeRaw:
        def __init__(self, names, nt):
            self._ch = list(names)
            self._data = np.zeros((len(self._ch), nt), dtype=np.float64)
            for i in range(len(self._ch)):
                self._data[i, :] = np.linspace(-1, 1, nt) * (i + 1) + i
            self.calls = []

        @property
        def ch_names(self):
            return list(self._ch)

        def pick(self, names):
            idx = [self._ch.index(n) for n in names]
            self._data = self._data[idx, :]
            self._ch = list(names)
            self.calls.append(("pick", tuple(names)))
            return self

        def set_eeg_reference(self, ref, projection=False):
            self.calls.append(("ref", ref, projection))
            return self

        def filter(self, l_freq, h_freq):
            self.calls.append(("filter", l_freq, h_freq))
            return self

        def resample(self, sfreq):
            self.calls.append(("resample", sfreq))
            return self

        def get_data(self, units=None):
            self.calls.append(("get_data", units))
            return self._data

    return _FakeRaw(ch_names, n_times)


class FakeEegnetBackend:
    """Synthetic numeric backend for real_eegnet_trainer (NO torch). Records determinism/seed + fit/embed calls; returns
    deterministic bytes for the 4 model artifacts and the feature dump."""

    def __init__(self):
        self.seeds = []
        self.fit_calls = []          # list of (n_train, n_val)
        self.embed_calls = []        # list of n_subjects
        self.embed_frozen_refs = []  # frozen-substrate ref each embed_from_artifacts was driven by

    def set_deterministic(self, seed):
        self.seeds.append(int(seed))

    def fit(self, train, val, training_config):
        self.fit_calls.append((len(list(train)), len(list(val))))
        tag = f"{len(train)}:{len(val)}".encode()
        return {"encoder_state_dict": b"enc_sd:" + tag, "encoder_checkpoint_file": b"enc_ckpt:" + tag,
                "source_state_artifact": b"ss_art:" + tag, "source_state_file": b"ss_file:" + tag}

    def embed_from_artifacts(self, windows_by_subject, frozen, training_config):
        import numpy as np
        self.embed_calls.append(len(windows_by_subject))
        self.embed_frozen_refs.append(frozen.ref)             # records which frozen substrate drove the dump
        # deterministic per-subject embedding with ONE ROW PER WINDOW (dim 4); content-independent (synthetic)
        out = {}
        for i, sk in enumerate(sorted(windows_by_subject)):
            nw = int(getattr(windows_by_subject[sk], "n_windows", 1) or 1)
            out[sk] = np.zeros((nw, 4), dtype=np.float32) + float(i)
        return out


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


def synthetic_registry(meta_over=None):
    """A fully-populated 30-ref SubstrateRegistry with placeholder 64-hex hashes + valid meta (for Stage-2 intake tests). No build,
    no torch, no files. `meta_over(ref, disease, fold, seed) -> dict` (or a plain dict) overrides meta per entry (e.g. to inject a
    forbidden site token into cohort_inclusion_list)."""
    from acar.v5 import protocol as P
    from acar.v5.substrate import stage1b_authorization as SA
    from acar.v5.substrate import registry as REG
    reg = REG.SubstrateRegistry()
    for ref in sorted(SA.CANONICAL_FOLD_REFS):
        disease = ref.split("/")[0]
        fold = int(ref.split("fold")[1].split("/")[0])
        seed = int(ref.split("seed")[1])
        hashes = {h: "a" * 64 for h in P.REGISTRY_HASH_FIELDS}
        meta = {"channel_montage": "std_1020_19", "sampling_rate": 128, "windowing_config": "4s_nonoverlap",
                "cohort_inclusion_list": list(P.DEV_COHORTS[disease]), "random_seed": seed,
                "git_commit": "0" * 40, "env_lock_sha256": "b" * 64}
        if meta_over is not None:
            meta.update(meta_over(ref, disease, fold, seed) if callable(meta_over) else meta_over)
        reg.register(disease, fold, seed, hashes=hashes, meta=meta)
    return reg


def stage1b_finalized_package(root, run_id="run-syn-0001", *, registry=None, reg=True, marker=True, sha=None, n_refs=30,
                              status="FINALIZED"):
    """Write a synthetic finalized Stage-1B package (registry.json + FINALIZED.json) under root/run_id, mirroring the finalize
    barrier's marker payload. Returns run_root. Overrides let a test tamper the marker (bad sha / n_refs / status / missing file)."""
    import json
    import os
    from acar.v5.substrate import stage1b_registry_io as RIO
    registry = registry if registry is not None else synthetic_registry()
    run_root = os.path.join(root, run_id)
    os.makedirs(run_root, exist_ok=True)
    reg_sha = RIO.registry_sha256(registry)
    if reg:
        reg_sha = RIO.write_registry(registry, os.path.join(run_root, RIO.REGISTRY_FILE))
    if marker:
        payload = {"status": status, "n_registered": 30, "n_refs": n_refs,
                   "registry_sha256": (sha if sha is not None else reg_sha),
                   "git_commit": "0" * 40, "env_lock_sha256": "b" * 64}
        with open(os.path.join(run_root, RIO.MARKER_FILE), "w") as f:
            json.dump(payload, f, sort_keys=True)
    return run_root


def write_synthetic_feat_dump(path, ref="PD/fold0/seed20260711", disease="PD", fold=0, seed=20260711, records=None):
    """Write a schema-valid, LABEL-FREE feat_dump.npz via the real writer (used by Stage-2 header-peek tests). One record default."""
    from acar.v5.substrate import stage1b_feature_dump_writer as FDW
    recs = records if records is not None else [(f"{disease}/ds-xxx/sub-001", "eval", 0, [0.0, 1.0, 2.0])]
    FDW.write_feature_dump(path, ref=ref, disease=disease, fold=fold, seed=seed,
                           preprocessing_config_sha256="0" * 64, training_config_sha256="0" * 64,
                           encoder_checkpoint_file_sha256="0" * 64, source_state_file_sha256="0" * 64, records=recs)
    return path


# ---- Stage-2B0 synthetic fixtures (engine tests; torch/sklearn-free) ----
def stage2b_auth(**over):
    """A fully-valid SYNTHETIC Stage-2B authorization (override any field via kwargs)."""
    from acar.v5 import stage2b_authorization as SA
    a = {"stage": "Stage-2B", "protocol_tag": SA.PROTOCOL_TAG, "protocol_tag_target_sha": SA.PROTOCOL_TAG_TARGET_SHA_FULL,
         "implementation_base_sha": "0" * 40, "stage1b_run_id": "acar-v5-stage1b-c4412b4-r1",
         "stage1b_registry_sha256": "a" * 64, "allowed_selection_refs": list(SA.CANONICAL_SELECTION_REFS),
         "allowed_candidate_ids": list(SA.CANONICAL_CANDIDATE_IDS), "forbid_s1_refs_for_selection": True,
         "forbid_external_read": True, "forbid_lockbox": True, "run_id": "syn-stage2b",
         "statement": SA.REQUIRED_STAGE2B_STATEMENT}
    a.update(over)
    return a


class DictLabelView:
    """A minimal evaluation-only label view (resolve_label from a dict) for engine tests — no bulk dump, no path attribute."""

    def __init__(self, labels):
        _labels = dict(labels)

        def _resolve(sk):
            return int(_labels[sk])
        self._resolve = _resolve

    def resolve_label(self, subject_key):
        return self._resolve(subject_key)


def stage2b_synthetic_source_state(D=8, seed=0):
    import numpy as np
    r = np.random.RandomState(seed)
    return {"means": (r.randn(2, D) * 0.5), "cov": np.eye(D) + 0.02, "priors": np.array([0.5, 0.5]),
            "classes": np.array([0, 1])}


def stage2b_disease_inputs(n_folds=2, D=8, seed=0, n_windows=16):
    """Synthetic {disease: {"folds": [{by_subject, source_lda, label_view}, ...]}} for the selection engine (torch-free)."""
    import numpy as np
    from acar.v5 import stage2_action_records as AR
    r = np.random.RandomState(seed)

    def fold(disease, cohort, k):
        by_subject, labels = {}, {}
        for i, role in enumerate(["train", "train", "val", "cal", "eval", "eval"]):
            sk = f"{disease}/{cohort}/sub-f{k}i{i}"
            by_subject[sk] = {"embedding": (r.randn(n_windows, D) * 0.3), "split_role": role}
            labels[sk] = i % 2
        return {"by_subject": by_subject,
                "source_lda": AR.SourceLDA(stage2b_synthetic_source_state(D, seed + k + (0 if disease == "PD" else 100))),
                "label_view": DictLabelView(labels)}
    return {"PD": {"folds": [fold("PD", "ds002778", k) for k in range(n_folds)]},
            "SCZ": {"folds": [fold("SCZ", "ds003944", k) for k in range(n_folds)]}}


def has_torch():
    """True iff torch is importable (spdim's real action transform needs it; py3.9 home has no torch)."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def stage2b_rank_deficient_batch(n=32, D=256, rank=5, seed=0, noise=1e-6):
    """A rank-deficient [n, D] batch (n windows spanned by `rank` latent directions + tiny noise) — the regime that made the
    frozen CORAL target covariance near-singular / overflow-prone. rank << min(n, D)."""
    import numpy as np
    r = np.random.RandomState(seed)
    return (r.randn(n, rank) @ r.randn(rank, D)) + noise * r.randn(n, D)


def stage2b_scaled_source_state(D=64, scale=1000.0, seed=0):
    """Synthetic source_state with a LARGE-eigenvalue pooled covariance (to force the raw whiten-color operator gain above the
    SVD cap so the cap is demonstrably active)."""
    import numpy as np
    r = np.random.RandomState(seed)
    return {"means": (r.randn(2, D) * 0.5), "cov": scale * (np.eye(D) + 0.02), "priors": np.array([0.5, 0.5]),
            "classes": np.array([0, 1])}


def stage2b_holm_per(evaluable_p=0.001, nonevaluable_ids=()):
    """Synthetic `per` map for Holm-family tests: {(candidate_id, disease): {"cal_raw": {H1,H2,H3}} or None}. Candidate ids in
    `nonevaluable_ids` get None for BOTH diseases (non-evaluable cells)."""
    from acar.v5 import protocol as P
    from acar.v5 import stage2_selection_engine as ENG
    non = set(nonevaluable_ids)
    per = {}
    for cid in P.CANDIDATE_IDS:
        for d in ENG.DISEASES:
            per[(cid, d)] = None if cid in non else {"cal_raw": {"H1": evaluable_p, "H2": evaluable_p, "H3": evaluable_p}}
    return per
