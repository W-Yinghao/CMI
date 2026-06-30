"""ACAR v4 — fixed-candidate DEV substrate-compatibility replay command (C1: executable body, AUTHORIZATION-gated; FAIL-CLOSED).

The ONE frozen command that re-embeds the OLD SEVEN DEV cohorts with the NEW all-DEV substrate and replays the FIXED candidate
(shift_margin+benefit_ranked+harm_indicator; NO reselection) to decide — via the pre-registered numeric pass-line
`regen_substrate.compatibility_replay_pass` (v2_replay a HARD requirement) — whether external Arm B may run. THIS IS NOT A NEW
DEV SELECTION RUN and emits NO selection/external/binding vocabulary (only the SUBSTRATE_COMPATIBILITY_* taxonomy).

C1 structure (mirrors B1b run_regen_substrate):
- TWO-COMMIT split: the substrate manifest pins `substrate_protocol_commit` (b99fa4f, the frozen substrate-generation code) AND
  `compatibility_protocol_commit` (this C1 replay code). The runner requires HEAD == compatibility_protocol_commit, so the
  b99fa4f substrates stay authoritative while the replay runs under the C1 commit (no dead-lock).
- STDLIB-FIRST preflight (schema + git + clean + output-absent + artifact/dev-input/env-lock FILE-byte hashes). Without a valid
  compatibility AUTHORIZATION manifest it raises SubstrateCompatibilityNotAuthorizedError BEFORE any torch/cmi import or DEV read.
- With a valid, hash-bound authorization → atomic output claim → `_run_compatibility_replay` (gated): runtime==env-lock verify +
  substrate SEMANTIC-hash verify + re-embed (the sha-pinned DEV-dump metadata supplies WindowKeys+labels; windows come from the
  SHA-PINNED scps cache by EXACT KEYED lookup `X[window_index]`, with `cache.subject[gi]==subject` & `cache.cohort[gi]==dataset_id`
  verified per row) + derive-under-frozen-source-state + 1x1x1 exploration + compatibility_replay_pass — all REAL. The only DEV
  read runs here (from the pinned cache), at the authorized C-run.
  C5 RESOLVED — alignment is by EXACT KEYED lookup, NOT by-position trust: the dump's global `window_index_te` IS the scps cache
  row index, so a SAME-COUNT reorder (or wrong cache) fails two ways — the pinned cache sha mismatches at preflight, and/or the
  subject/cohort-at-index check fails. The live `load_cohort` raw reader is NOT used. See
  notes/ACAR_V4_C1_COMPAT_REPLAY_READINESS.md "C5 RESOLVED".

Usage:
    python -m acar.v4.run_substrate_compatibility --substrate-manifest /abs/substrate_manifest.json --output /abs/new_compat_dir
        [--compat-authorization /abs/compat_auth.json]      # omit => fail closed (preflight only)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import sys

from acar.v4 import regen_substrate as RS
from acar.v4.run_regen_substrate import (_repo_root, _verify_commit, _verify_clean, _sha256_file,  # noqa: F401
                                         _verify_runtime_matches_lock)


def run(substrate_manifest_path, output, *, compat_authorization=None):
    """STDLIB-FIRST, FAIL-CLOSED. Validates the substrate manifest (two-commit + fixed candidate + pinned op-point + the 4
    artifact hashes + dev-input-manifest pins) + git(HEAD==compatibility_protocol_commit)/clean/output/file-hash preflight.
    Without a valid compatibility authorization it refuses to replay (raises) — NO torch/cmi import, NO DEV read, NO output.
    With a valid, hash-bound authorization it runs the gated replay under an atomic output claim."""
    with open(substrate_manifest_path, "rb") as f:
        raw = f.read()
    input_manifest_sha256 = hashlib.sha256(raw).hexdigest()
    spec = json.loads(raw.decode())
    RS.validate_substrate_manifest(spec)                              # two-commit + fixed candidate + op-point + 4 hashes
    root = _repo_root()
    _verify_commit(root, spec["compatibility_protocol_commit"])       # HEAD == the C1 replay commit (NOT the substrate commit)
    _verify_clean(root)
    if os.path.exists(output):
        raise FileExistsError(f"output dir already exists (no overwrite): {output}")
    _verify_compat_preflight_hashes(spec)                            # artifact .pt/.npz + dev-input-manifest + env-lock FILE shas
    report = {"input_manifest_sha256": input_manifest_sha256, "candidate": RS.FIXED_CANDIDATE,
              "substrate_protocol_commit": spec["substrate_protocol_commit"],
              "compatibility_protocol_commit": spec["compatibility_protocol_commit"],
              "pass_line": {"coverage_min": RS.COVERAGE_MIN, "budget": RS.BUDGET, "alpha": RS.ALPHA,
                            "v2_replay": "HARD requirement (no waiver)"},
              "result_taxonomy": list(RS.SUBSTRATE_COMPAT_STATUSES),
              "expected_output": RS.expected_compat_output(output),
              "command": shlex.join([sys.executable, "-m", "acar.v4.run_substrate_compatibility",
                                     "--substrate-manifest", substrate_manifest_path, "--output", output])}
    if compat_authorization is None:                                  # COMPATIBILITY GATE — no authorization => fail closed
        _require_compat_authorization(report)                        # raises (no torch/cmi import, no DEV read, no output)
    auth = _load_compat_authorization(compat_authorization, spec, input_manifest_sha256, output)  # validate + bind
    return _authorized_replay_and_write(spec, output, report, auth)   # atomic; calls the gated _run_compatibility_replay


def _verify_compat_preflight_hashes(spec):
    """METADATA/file-byte preflight (stdlib; no torch/DEV signal): each disease's encoder .pt + source-state .npz match their
    *_file_sha256; the dev-input-manifest file matches its pinned sha (pins the exact eligible DEV universe to re-embed); the
    env-lock file matches env_lock_sha256. The canonical SEMANTIC hashes are re-verified inside the authorized replay loader."""
    for d in ("PD", "SCZ"):
        sd = spec["substrates"][d]
        for path_key, sha_key in (("encoder_checkpoint_path", "encoder_checkpoint_file_sha256"),
                                  ("source_state_path", "source_state_file_sha256"),
                                  ("dev_input_manifest_path", "dev_input_manifest_sha256")):
            p = sd[path_key]
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{d}: {path_key} missing: {p}")
            got = _sha256_file(p)
            if got != sd[sha_key]:
                raise ValueError(f"{d}: {path_key} {sha_key} mismatch ({got} != {sd[sha_key]})")
        for cohort, dp in sd["dev_feat_dump_paths"].items():       # DEV feat-dump alignment source of truth (sha-pinned)
            if not os.path.isfile(dp):
                raise FileNotFoundError(f"{d}/{cohort}: dev_feat_dump missing: {dp}")
            got = _sha256_file(dp)
            if got != sd["dev_feat_dump_sha256"][cohort]:
                raise ValueError(f"{d}/{cohort}: dev_feat_dump_sha256 mismatch ({got} != {sd['dev_feat_dump_sha256'][cohort]})")
        cpath = sd["scps_cache_path"]                               # C5: the window source (keyed by the dump global index)
        if not os.path.isfile(cpath):
            raise FileNotFoundError(f"{d}: scps_cache missing: {cpath}")
        if _sha256_file(cpath) != sd["scps_cache_sha256"]:
            raise ValueError(f"{d}: scps_cache_sha256 mismatch ({_sha256_file(cpath)} != {sd['scps_cache_sha256']})")
    elp = spec["env_lock_path"]
    if not os.path.isfile(elp):
        raise FileNotFoundError(f"env_lock_path missing: {elp}")
    if _sha256_file(elp) != spec["env_lock_sha256"]:
        raise ValueError(f"env_lock_sha256 mismatch ({_sha256_file(elp)} != {spec['env_lock_sha256']})")


def _require_compat_authorization(report):
    raise RS.SubstrateCompatibilityNotAuthorizedError(
        "DEV substrate-compatibility replay is NOT authorized — no compatibility authorization manifest supplied. The manifest "
        "validated (two-commit, fixed candidate, pinned operating point, trained-artifact + dev-input + env-lock file hashes) "
        "and the full preflight passed, but re-embedding DEV with the new substrate + the fixed-candidate replay needs an "
        "explicit, hash-bound compatibility authorization manifest (--compat-authorization; "
        "notes/ACAR_V4_SUBSTRATE_REGEN_COMMAND.md). Decision uses regen_substrate.compatibility_replay_pass (v2_replay HARD). "
        "No torch/cmi import, no DEV read, no output written. report=" + json.dumps(report, sort_keys=True))


def _load_compat_authorization(path, spec, input_manifest_sha256, output):
    """Validate + BIND the compatibility authorization to THIS run: schema (RS.validate_compat_authorization) + the
    authorization must match compatibility_protocol_commit, substrate_protocol_commit, substrate_manifest_sha256 (==the file
    sha of THIS manifest), env_lock_sha256, and output_path. Any mismatch raises BEFORE any heavy import / DEV read."""
    with open(path) as f:
        auth = json.load(f)
    RS.validate_compat_authorization(auth)
    checks = (("compatibility_protocol_commit", spec["compatibility_protocol_commit"]),
              ("substrate_protocol_commit", spec["substrate_protocol_commit"]),
              ("substrate_manifest_sha256", input_manifest_sha256),
              ("env_lock_sha256", spec["env_lock_sha256"]), ("output_path", output))
    for k, want in checks:
        if auth[k] != want:
            raise ValueError(f"compatibility authorization {k} != run {k} ({auth[k]!r} != {want!r})")
    return auth


def _verify_substrate_semantic_hashes(spec):                        # pragma: no cover — gated (torch/acar.v3); tests monkeypatch
    """Re-verify the canonical SEMANTIC substrate hashes against the on-disk artifacts (SAFE weights_only load + acar.v3
    self-verifying load_frozen) — the artifacts must be byte-and-semantics identical to the b99fa4f training record before any
    DEV metric. NO unsafe pickle fallback."""
    import numpy as np
    import torch
    from acar.v3.loader import load_frozen_source_state_artifact
    for d in ("PD", "SCZ"):
        sd = spec["substrates"][d]
        state = torch.load(sd["encoder_checkpoint_path"], map_location="cpu", weights_only=True)
        got = RS.canonical_state_dict_sha256({k: v.detach().cpu().numpy() for k, v in state.items()})
        if got != sd["encoder_state_dict_sha256"]:
            raise ValueError(f"{d}: encoder_state_dict_sha256 mismatch ({got} != {sd['encoder_state_dict_sha256']})")
        art = load_frozen_source_state_artifact(dict(np.load(sd["source_state_path"], allow_pickle=False)))
        if art.source_state_sha256 != sd["source_state_artifact_sha256"]:
            raise ValueError(f"{d}: source_state_artifact_sha256 mismatch")


def _dev_input_manifest(spec, disease):
    """Load the disease's pinned DEV input manifest (the B1b regen input manifest) — the authoritative eligible-subject
    universe + cohort source paths. Its file sha was already verified == dev_input_manifest_sha256 in the stdlib preflight."""
    with open(spec["substrates"][disease]["dev_input_manifest_path"]) as f:
        return json.load(f)


def _raw_subjects_by_cohort(spec, disease):
    """METADATA-only: list sub-* dirs per DEV cohort (no signal), for check_eligible_subjects (eligible = raw − excluded)."""
    m = _dev_input_manifest(spec, disease)
    out = {}
    for c in m["dev_cohorts"]:
        cdir = m["source_paths"][c]
        out[c] = [d for d in os.listdir(cdir) if d.startswith("sub-") and os.path.isdir(os.path.join(cdir, d))]
    return out


def _load_disease_cache(spec, disease):                            # pragma: no cover — gated DEV-signal read; tests inject a fake cache
    """C5: read the SHA-PINNED per-condition scps cache (the SAME source that produced the DEV feat dumps) — the replay's window
    source. The cache row index IS the dump's GLOBAL window_index (cmi.run_scps_crossdataset `load()` then te_g = where(te_mask)),
    so a window is fetched by EXACT KEYED lookup `X[window_index]` (NOT by a same-count by-position guess), and the live raw
    reader is NEVER used. Loaded allow_pickle=False (X float, subject/cohort str). Returns {"X","subject","cohort"} as the
    producer's `load(condition, cohorts=DEV_SCOPE[disease])` order; the per-row subject/cohort are re-verified at lookup."""
    import numpy as np
    sd = spec["substrates"][disease]
    with np.load(sd["scps_cache_path"], allow_pickle=False) as d:   # NO pickle (sha-pinned in the preflight)
        X = np.asarray(d["X"], dtype="<f4")
        subject = np.asarray(d["subject"]).astype(str)
        cohort = np.asarray(d["cohort"]).astype(str)
    keep = np.isin(cohort, sorted(RS.DEV_SCOPE[disease]))           # SAME cohort filter as the producer's load()
    return {"X": X[keep], "subject": subject[keep], "cohort": cohort[keep]}


def _load_subject_windows_and_keys(dataset_id, subject, cfg, dump_rows, *, cache):
    """C5 REAL alignment by EXACT KEYED lookup (synthetic-tested; no real raw in tests). The sha-pinned DEV-dump metadata is the
    SOURCE OF TRUTH: `dump_rows` = (recording_id, window_index, label) for THIS subject, where window_index is the GLOBAL cache
    row index. `subject` is the FULL namespaced subject VERBATIM from the dump's subject_id_te (e.g. "ds002778/sub-hc1") — the
    cmi scps cache stores `cache.subject` in that SAME cohort-prefixed form (cmi.run_scps_crossdataset; _dump_audit_fold writes
    subject_id_te[i]=cache.subject[gi]), and v3's load_deployment_batches uses subject_id_te verbatim as the WindowKey subject, so
    the replay's WindowKeys are byte-identical to v3's. `cache` = {"X","subject","cohort"} from the sha-pinned scps cache. For each
    dump row the window is fetched at `cache["X"][window_index]` and VERIFIED: cache.subject[window_index] == subject AND
    cache.cohort[window_index] == dataset_id (so a reordered/wrong cache — which would also change the pinned sha — cannot silently
    mispair; a same-count reorder fails here, not just at the count). FAIL-CLOSED on out-of-range index, subject/cohort mismatch,
    wrong shape, non-finite, or a duplicate (recording_id, window_index). Returns (windows[n,19,512], WindowKeys[n], labels)."""
    import numpy as np
    from acar.v3.set_features import WindowKey
    Xc, csub, ccoh = cache["X"], cache["subject"], cache["cohort"]
    n_rows = Xc.shape[0]
    cfg_win = (cfg["canon_channels"], int(cfg["resample_fs"] * cfg["window_sec"]))
    windows, keys, labels = [], [], {}
    for rec_id, win_idx, label in sorted(dump_rows, key=lambda r: int(r[1])):
        gi = int(win_idx)
        if gi < 0 or gi >= n_rows:
            raise ValueError(f"{subject}: dump global window_index {gi} out of cache range [0,{n_rows})")
        if str(csub[gi]) != subject or str(ccoh[gi]) != dataset_id:
            raise ValueError(f"{subject}: cache row {gi} is cohort={ccoh[gi]} subject={csub[gi]} (subject/cohort mismatch — wrong/reordered cache)")
        w = np.asarray(Xc[gi], dtype="<f4")
        if w.shape != cfg_win:
            raise ValueError(f"{subject}: cache window {gi} shape {w.shape} != {cfg_win}")
        wk = WindowKey(dataset_id, subject, str(rec_id), gi)
        if wk in labels:
            raise ValueError(f"{subject}: duplicate WindowKey ({rec_id},{gi}) in dump rows")
        windows.append(w); keys.append(wk); labels[wk] = int(label)
    RS.assert_finite(np.asarray(windows, dtype="<f4"), f"{subject} windows")
    return np.asarray(windows, dtype="<f4"), keys, labels


def _load_frozen_substrate(spec):                                   # pragma: no cover — gated (torch/acar.v3); tests monkeypatch
    """Load the FROZEN B1b substrate per disease: the encoder state_dict (SAFE weights_only) into an EEGNet backbone, and the
    source-state via acar.v3 load_frozen_source_state_artifact (self-verifying; NEVER refit). REQUIRES the source-state artifact
    (path + canonical hash) — this is the substrate the replay must check, not a re-fitted one. Returns {disease: (encoder, art)}."""
    import numpy as np
    import torch
    from cmi.models.backbones import build_backbone
    from acar.v3.loader import load_frozen_source_state_artifact
    s = RS.TRAINING_SCHEDULE
    out = {}
    for d in ("PD", "SCZ"):
        sd = spec["substrates"][d]
        if not sd.get("source_state_path") or not RS._is_hex(sd.get("source_state_artifact_sha256", ""), 64):
            raise ValueError(f"{d}: frozen source-state artifact (path + source_state_artifact_sha256) is REQUIRED")
        bb = build_backbone(s["model"], n_chans=s["n_chans"], n_times=s["n_times"], n_classes=s["n_classes"], device="cpu")
        bb.load_state_dict(torch.load(sd["encoder_checkpoint_path"], map_location="cpu", weights_only=True)); bb.eval()
        art = load_frozen_source_state_artifact(dict(np.load(sd["source_state_path"], allow_pickle=False)))   # FROZEN; no refit
        if art.source_state_sha256 != sd["source_state_artifact_sha256"] or str(art.disease) != d:
            raise ValueError(f"{d}: frozen source-state artifact mismatch (hash/disease)")
        out[d] = (bb, art)
    return out


def _check_reembed_universe(disease, seen_ns, eligible):
    """FAIL-CLOSED subject-universe reconciliation for the re-embed: the set of subjects actually re-embedded (those present in
    the sha-pinned DEV feat-dump AND eligible) must EQUAL the eligible set AND number EXACTLY EXACT_ELIGIBLE[disease]. Catches a
    dump that silently OMITS an eligible subject (or a whole cohort) — a whole-subject-granularity 'missing window' that the
    per-subject by-key aligner cannot see and that cv_assignment (only needs ≥k folds) would not catch."""
    seen, elig = set(seen_ns), set(eligible)
    if seen != elig:
        missing, extra = sorted(elig - seen), sorted(seen - elig)
        raise ValueError(f"{disease}: re-embedded universe != eligible (missing {missing[:5]}, extra {extra[:5]})")
    if len(seen) != RS.EXACT_ELIGIBLE[disease]:
        raise ValueError(f"{disease}: re-embedded {len(seen)} subjects != EXACT_ELIGIBLE {RS.EXACT_ELIGIBLE[disease]}")
    return True


def _reembed_dev_under_substrate(spec, frozen):                     # pragma: no cover — gated raw read at C-run; tests monkeypatch
    """Re-embed the OLD-SEVEN eligible DEV windows (exact universe pinned by each disease's dev_input_manifest — PD 230 /
    SCZ 225, ds004000/sub-042 excluded, FROZEN_PIPELINE, cohort-aware keys) with the NEW all-DEV encoder, and build per-disease
    v3 DeploymentBatches (z = NEW embeddings; window/subject/recording keys + window_index from the DEV dump metadata;
    source_state_ref = the FROZEN B1b artifact's ref) + labels_by_window. NO source-state is fitted here. C5: windows come from
    the SHA-PINNED scps cache by EXACT KEYED lookup (cache row == dump global window_index), NOT a live raw reader. Returns
    {disease: {"artifact": art, "batches": [...], "labels": {...}}}. The DEV-signal read runs ONLY at the authorized C-run; the
    keyed lookup + per-row subject/cohort verification + v3 integrity machinery fail-close on any mismatch."""
    import numpy as np
    import torch
    from acar.v3.data import build_deployment_batches
    cfg = RS.FROZEN_PIPELINE
    out = {}
    for d in ("PD", "SCZ"):
        bb, art = frozen[d]
        m = _dev_input_manifest(spec, d)
        eligible = set(RS.check_eligible_subjects(d, _raw_subjects_by_cohort(spec, d), m))    # cohort-aware "dsid/sub"
        sd = spec["substrates"][d]
        cache = _load_disease_cache(spec, d)                                                  # C5: sha-pinned scps cache = window source
        rows_by_ds, labels, seen = {}, {}, set()
        for cohort in m["dev_cohorts"]:
            meta = np.load(sd["dev_feat_dump_paths"][cohort], allow_pickle=False)             # METADATA (ids/index/labels); NOT raw signal / NOT old z
            sid = [str(x) for x in meta["subject_id_te"].tolist()]
            rec = [str(x) for x in meta["recording_id_te"].tolist()]
            wix = [int(x) for x in meta["window_index_te"].tolist()]
            yte = [int(x) for x in meta["y_te"].tolist()]
            if not (len(sid) == len(rec) == len(wix) == len(yte)):                            # dump column-length consistency
                raise ValueError(f"{d}/{cohort}: dump metadata columns length mismatch "
                                 f"({len(sid)}/{len(rec)}/{len(wix)}/{len(yte)})")
            if any(y_ not in (0, 1) for y_ in yte):                                           # labels must be in {0,1}
                raise ValueError(f"{d}/{cohort}: dump y_te has labels outside {{0,1}}")
            per_sub = {}                                                                      # eligible subjects only (excluded skipped)
            for s_, r_, w_, y_ in zip(sid, rec, wix, yte):
                ns = s_ if "/" in s_ else f"{cohort}/{s_}"                                    # FULL namespaced "dsid/sub" (== eligible + cache.subject + v3 WindowKey)
                if ns not in eligible:
                    continue
                per_sub.setdefault(ns, []).append((r_, w_, y_))                               # key = VERBATIM namespaced subject (the cache & v3 store it prefixed)
            for sub, dump_rows in per_sub.items():                                            # sub = "dsid/sub" (NOT bare — matches cache.subject + v3 convention)
                X, keys, lab = _load_subject_windows_and_keys(cohort, sub, cfg, dump_rows, cache=cache)  # KEYED cache lookup
                with torch.no_grad():
                    z = bb(torch.as_tensor(np.asarray(X, dtype="<f4")))[1].cpu().numpy()       # NEW-encoder embeddings (forward → (logits, z))
                RS.assert_finite(z, f"{sub} embeddings")
                for wk, zi in zip(keys, z):
                    rows_by_ds.setdefault(cohort, []).append((wk.subject_id, wk.recording_id, wk.window_index, np.asarray(zi, float)))
                labels.update(lab)
                seen.add(sub)                                                                 # sub is already the full namespaced "dsid/sub" == eligible format
        _check_reembed_universe(d, seen, eligible)                                            # re-embedded set == eligible (count == EXACT)
        batches = []
        for ds_id, rows in rows_by_ds.items():
            batches += list(build_deployment_batches(ds_id, d, rows, art.source_state_ref))
        out[d] = {"artifact": art, "batches": batches, "labels": labels}
    return out


def _derive_under_frozen_source_state(reembed_out):                 # pragma: no cover — gated; tests monkeypatch _run_compatibility_replay
    """Derive V4OOFRecords + the v2-replay comparator UNDER the FROZEN B1b source-state — the no-refit path. Mirrors
    real_adapter.derive's body but builds the SourceStateRegistry from the FROZEN artifact and EXECUTES it (disease_exec_cache
    → SourceStateArtifact.execute; run_c0; _emit_records over the v3 CV folds). It DELIBERATELY does NOT call
    real_adapter.build_cohort_inputs / v3 build_cohort_input (which would RE-FIT a per-cohort source-state). Returns
    (records, v2_replay_red_by_disease)."""
    import numpy as np
    from acar.v3 import develop as V3D
    from acar.v3.loader import SourceStateRegistry
    from acar.v3.data import deployment_batch_digest, canon_subject
    from acar.v3.splits import cv_assignment
    from acar.v4.real_adapter import _fold_roles, _emit_records, ACTIONS
    records, v2_replay = [], {}
    for d in ("PD", "SCZ"):
        art, batches, labels = reembed_out[d]["artifact"], reembed_out[d]["batches"], reembed_out[d]["labels"]
        registry = SourceStateRegistry(d); registry.add(art)        # FROZEN B1b artifact ONLY (no DEV-fitted artifact)
        idx = V3D._subject_batches(batches); eligible = V3D._eligible_subjects(idx)
        all_subjects = [v["key"] for v in idx.values()]; elig_canon = {canon_subject(s) for s in eligible}
        assignment, _ = cv_assignment(all_subjects, eligible=elig_canon)        # SAME CV call/seeds as the DEV run
        cache = V3D.disease_exec_cache(registry, batches, labels)   # execute once via the FROZEN artifact — NO refit
        v2_replay[d] = float(V3D.run_c0(d, registry, batches, labels, 0.10, 0.0, cache=cache).red_router)
        cells = {}
        for cc, slot in idx.items():
            key = slot["key"]; elig = []
            for b in slot["eligible"]:
                c = cache[deployment_batch_digest(b)]
                dr = np.array([float(c["dr"][a]) for a in ACTIONS], float)
                feats = np.stack([np.asarray(c["c0feat"][a], float) for a in ACTIONS])
                elig.append((deployment_batch_digest(b), dr, feats))
            cells[cc] = {"dataset": key.dataset_id, "subject": key.subject_id, "eligible": elig,
                         "fallback": [deployment_batch_digest(b) for b in slot["fallback"]]}
        assignment_canon = [{"fold": fa["fold"], "eval": {canon_subject(s) for s in fa["eval"]},
                             "cal": {canon_subject(s) for s in fa["cal"]}, "fit": {canon_subject(s) for s in fa["fit"]}}
                            for fa in assignment]
        records += _emit_records(d, _fold_roles(assignment_canon), cells)
    return records, v2_replay


def _fixed_candidate_per_disease_metrics(reembed_out, spec):       # pragma: no cover — gated; tests monkeypatch _run_compatibility_replay
    """Derive UNDER the FROZEN source-state (no refit) → run the FIXED-candidate exploration pinned to EXACTLY ONE config
    (1x1x1: score=shift_margin, policy=benefit_ranked, loss=harm_indicator) → extract per-disease stats. NO real_adapter.derive
    / build_cohort_inputs (which would re-fit source-state)."""
    from acar.v4.develop import run_dev_exploration, V4DevConfig
    records, v2_replay = _derive_under_frozen_source_state(reembed_out)
    cfg = V4DevConfig(policy_families=(RS.FIXED_CANDIDATE["policy"],), losses=(RS.FIXED_CANDIDATE["loss"],),
                      budget_by_loss={RS.FIXED_CANDIDATE["loss"]: RS.BUDGET}, alpha=RS.ALPHA,
                      coverage_min=RS.COVERAGE_MIN, g3_comparator="v2_replay")
    result = run_dev_exploration(records, config=cfg, score_families=[RS.FIXED_CANDIDATE["score_family"]],
                                 real_mode=True, v2_replay_red_by_disease=v2_replay)
    return _extract_fixed_candidate_stats(result)


def _extract_fixed_candidate_stats(result):
    """REAL deterministic accessor: pull the per-disease stats compatibility_replay_pass needs from a V4DevExplorationResult
    that was run with EXACTLY the fixed candidate. Fail-closed if not exactly one (disease, benefit_ranked, harm_indicator)
    report per disease, if a disease is missing, or if the v2 comparator is absent — never fabricate. Maps:
      lambda_certified = g4_harm_control_pass — the LTT certification that the harm_indicator loss is controlled at the budget;
      L_harm_all_eval  = eval_L_harm_all — the EXACT all-batch-denominator EVAL harm_indicator loss (the object the LTT budget
                         controls), NOT the conditional harm_rate proxy. REQUIRED on the report (None ⇒ fail-closed) for a real
                         compat run. (harm_among_adapted = harm_rate is carried DESCRIPTIVELY only and never gates.)
      v2_replay_red    = c0_red (== the v2_replay comparator because g3_comparator='v2_replay')."""
    import math
    fc = RS.FIXED_CANDIDATE
    per_disease = {}
    for d in ("PD", "SCZ"):
        rs = [r for r in getattr(result, "reports", ())
              if r.disease == d and r.policy_family == fc["policy"] and r.loss == fc["loss"]]
        if len(rs) != 1:
            raise ValueError(f"{d}: expected EXACTLY ONE fixed-candidate report (no reselection grid), got {len(rs)}")
        r = rs[0]
        v2r = r.c0_red
        v2_eval = v2r is not None and math.isfinite(float(v2r))
        elh = getattr(r, "eval_L_harm_all", None)
        if elh is None or not math.isfinite(float(elh)):           # the EXACT all-eval harm is REQUIRED + finite (no proxy fallback)
            raise ValueError(f"{d}: report.eval_L_harm_all must be a finite number (exact all-batch EVAL harm), got {elh!r}")
        hr = float(r.harm_rate)                                     # DESCRIPTIVE only (conditional harm; NaN if nothing adapted)
        per_disease[d] = {"lambda_certified": bool(r.g4_harm_control_pass), "coverage": float(r.coverage),
                          "red": float(r.red), "L_harm_all_eval": float(elh),
                          "harm_among_adapted": (float(hr) if math.isfinite(hr) else None),   # descriptive; not a gate
                          "v2_evaluable": bool(v2_eval), "v2_replay_red": (float(v2r) if v2_eval else None)}
    if set(per_disease) != {"PD", "SCZ"}:
        raise ValueError("fixed-candidate extraction must cover EXACTLY PD and SCZ")
    return per_disease


def _run_compatibility_replay(spec, output):                        # pragma: no cover — gated real orchestration; tests monkeypatch
    """REAL orchestration (NOT an entry-raise). Reached ONLY with a valid, bound compatibility authorization. Steps (all REAL):
    runtime==env-lock verify → substrate SEMANTIC-hash verify → load the FROZEN B1b encoder + source-state (no refit) →
    re-embed old-seven eligible DEV under the new encoder → derive UNDER the frozen source-state (NOT build_cohort_input) →
    FIXED-candidate (1x1x1) exploration → extract per-disease stats → regen_substrate.compatibility_replay_pass → verdict.
    Returns {status, reason, per_disease}. Tests monkeypatch this whole function. The actual DEV raw read happens ONLY here, at
    the authorized C-run; an operational failure propagates (caller cleans the output → OPERATIONALLY_ABORTED_NO_VERDICT)."""
    _verify_runtime_matches_lock(spec)                              # cuda + threads=1 + versions == the substrate env lock
    _verify_substrate_semantic_hashes(spec)                        # canonical encoder/source-state hashes == record
    frozen = _load_frozen_substrate(spec)                          # FROZEN B1b encoder + source-state per disease (no refit)
    reembed_out = _reembed_dev_under_substrate(spec, frozen)       # re-embed eligible DEV under the new encoder (no refit)
    per_disease = _fixed_candidate_per_disease_metrics(reembed_out, spec)
    authorized, reason = RS.compatibility_replay_pass(per_disease)  # FROZEN pre-registered pass-line (v2_replay HARD)
    return {"status": "SUBSTRATE_COMPATIBILITY_PASS" if authorized else "SUBSTRATE_COMPATIBILITY_FAIL",
            "reason": reason, "per_disease": per_disease}


def _authorized_replay_and_write(spec, output, report, auth):
    """Atomic output claim → gated replay (called once) → write compat_manifest.json then compat_RESULT.json (status LAST). An
    operational failure removes the claimed output (no partial; OPERATIONALLY_ABORTED_NO_VERDICT — a partial dir is NEVER read
    as a compatibility FAIL)."""
    os.mkdir(output)                                                # atomic claim
    try:
        verdict = _run_compatibility_replay(spec, output)
        if verdict.get("status") not in ("SUBSTRATE_COMPATIBILITY_PASS", "SUBSTRATE_COMPATIBILITY_FAIL"):
            raise RuntimeError(f"replay returned a non-verdict status: {verdict.get('status')!r}")
        body = {"substrate_protocol_commit": spec["substrate_protocol_commit"],
                "compatibility_protocol_commit": spec["compatibility_protocol_commit"],
                "substrate_manifest_sha256": report.get("input_manifest_sha256"), "command": report.get("command"),
                "candidate": RS.FIXED_CANDIDATE, "pass_line": report["pass_line"],
                "env_lock_sha256": spec["env_lock_sha256"],
                "authorization": {k: auth[k] for k in ("authorized_by", "authorization_time", "statement")},
                "verdict": verdict}
        with open(os.path.join(output, "compat_manifest.json"), "w") as f:
            json.dump(body, f, sort_keys=True, allow_nan=False, indent=2)
        with open(os.path.join(output, "compat_RESULT.json"), "w") as f:   # written LAST = completion sentinel
            json.dump({"status": verdict["status"], "reason": verdict["reason"],
                       "candidate": RS.FIXED_CANDIDATE}, f, sort_keys=True, allow_nan=False, indent=2)
        return body
    except BaseException:
        import shutil
        shutil.rmtree(output, ignore_errors=True)                  # no partial; abort = OPERATIONALLY_ABORTED_NO_VERDICT
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACAR v4 fixed-candidate substrate-compatibility replay (C1; authorization-gated)")
    ap.add_argument("--substrate-manifest", required=True)
    ap.add_argument("--output", required=True, help="must not exist")
    ap.add_argument("--compat-authorization", default=None, help="path to a compatibility authorization manifest (omit => fail closed)")
    args = ap.parse_args(argv)
    return run(args.substrate_manifest, args.output, compat_authorization=args.compat_authorization)


if __name__ == "__main__":
    main()
