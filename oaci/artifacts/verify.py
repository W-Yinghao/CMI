"""Artifact verifier (shallow + deep) and CLI.

Shallow: COMMITTED marker + index integrity, every listed file's sha256/size, no unlisted extra file,
no symlink. Deep additionally decodes every JSON/NPZ, rebuilds the support graph / plans / predictions
/ metrics / leakage / diagnostics and re-verifies their hashes, recomputes the method / level / fold
logical hashes, checks every checkpoint reference resolves, that the source-audit and target signatures
agree across methods and levels, that target fits are empty, and recomputes the artifact scientific
hash. Every error carries its exact relative path.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field

from ..runner.scientific_hash import scientific_value_hash
from . import plan_codec as P
from . import prediction_codec as PR
from . import support_codec as SC
from .atomic import COMMIT_MARKER, INDEX_NAME
from ..runner.keys import canonical_json_hash
from .canonical_json import decode_canonical_json
from .reader import read_artifact, read_doc
from .schema import check_schema_version


@dataclass
class VerificationReport:
    ok: bool
    errors: list = field(default_factory=list)            # (relative_path, message)
    n_indexed_files: int = 0
    n_total_files: int = 0                                 # indexed + artifact_index + COMMITTED
    n_checkpoints: int = 0                                 # indexed .pt count
    n_plans: int = 0                                       # indexed plan count
    n_verified_checkpoints: int = 0                        # actually weights_only-loaded (deep)
    n_verified_plans: int = 0                              # actually decoded (deep)
    artifact_scientific_hash: str = ""
    artifact_index_sha256: str = ""

    def fail(self, path, msg):
        self.ok = False
        self.errors.append((path, msg))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_rel(root):
    out = []
    for r, dirs, files in os.walk(root):
        for d in dirs:
            if os.path.islink(os.path.join(r, d)):
                raise ValueError(f"symlink directory in artifact: {os.path.relpath(os.path.join(r, d), root)}")
        for fn in files:
            ap = os.path.join(r, fn)
            if os.path.islink(ap):
                raise ValueError(f"symlink file in artifact: {os.path.relpath(ap, root)}")
            out.append(os.path.relpath(ap, root))
    return out


def verify_artifact_tree(path, *, deep=True) -> VerificationReport:
    rep = VerificationReport(ok=True)
    root = os.path.abspath(path)
    if os.path.islink(root):
        rep.fail(".", "artifact root is a symlink"); return rep
    marker_p = os.path.join(root, COMMIT_MARKER)
    index_p = os.path.join(root, INDEX_NAME)
    if not os.path.exists(marker_p):
        rep.fail(COMMIT_MARKER, "missing commit marker"); return rep
    if not os.path.exists(index_p):
        rep.fail(INDEX_NAME, "missing index"); return rep
    marker = read_doc(marker_p)
    try:
        check_schema_version(marker.get("schema_version", ""))
    except ValueError as e:
        rep.fail(COMMIT_MARKER, str(e)); return rep
    if _sha256(index_p) != marker.get("artifact_index_sha256"):
        rep.fail(INDEX_NAME, "index sha256 disagrees with commit marker"); return rep

    index = read_doc(index_p)["files"]
    listed = {e["relative_path"]: e for e in index}
    try:
        on_disk = set(_walk_rel(root))
    except ValueError as e:
        rep.fail(".", str(e)); return rep
    extras = on_disk - set(listed) - {COMMIT_MARKER, INDEX_NAME}
    for x in sorted(extras):
        rep.fail(x, "file is not listed in the index")
    for rel, e in listed.items():
        ap = os.path.join(root, rel)
        if not os.path.exists(ap):
            rep.fail(rel, "indexed file is missing"); continue
        if os.path.getsize(ap) != e["byte_size"]:
            rep.fail(rel, "byte size mismatch")
        if _sha256(ap) != e["file_sha256"]:
            rep.fail(rel, "sha256 mismatch (corruption)")
    rep.n_indexed_files = len(listed)
    rep.n_total_files = len(listed) + 2                    # + artifact_index + COMMITTED
    rep.artifact_index_sha256 = _sha256(index_p)
    rep.n_checkpoints = sum(1 for e in index if e["artifact_kind"] == "checkpoint_pt")
    rep.n_plans = sum(1 for e in index if e["artifact_kind"] in (P.TASK_KIND, P.ALIGN_KIND, P.FOLD_KIND,
                                                                 P.BOOTSTRAP_KIND, P.DESIGN_KIND))
    if not rep.ok or not deep:
        rep.artifact_scientific_hash = marker.get("artifact_scientific_hash", "")
        return rep

    _deep(root, rep, marker)
    return rep


def _level_dirs(root):
    base = os.path.join(root, "levels")
    if not os.path.isdir(base):
        return []
    return sorted(os.path.join("levels", d) for d in os.listdir(base) if d.startswith("level-"))


def _deep(root, rep, marker):
    sa_sigs, ta_sigs = set(), set()
    level_hashes = []
    for ld in _level_dirs(root):
        try:
            llogical, lbody, _ = read_artifact(os.path.join(root, ld, "level.json"), "level_result")
        except Exception as e:
            rep.fail(f"{ld}/level.json", f"unreadable: {e}"); continue
        if scientific_value_hash(lbody["payload"]) != llogical:
            rep.fail(f"{ld}/level.json", "level logical hash does not recompute")
        level_hashes.append((_level_num(ld), llogical))
        # support graph
        try:
            slog, sbody, sarr = read_artifact(os.path.join(root, ld, "support.json"), SC.SUPPORT_KIND)
            SC.decode_support_graph({k: v for k, v in sbody.items() if k != "npz"}, sarr)
        except Exception as e:
            rep.fail(f"{ld}/support.json", f"support verification failed: {e}")
        # provenance: target fits empty
        _, pbody, _ = read_artifact(os.path.join(root, ld, "provenance.json"), "provenance")
        if pbody.get("target_fit_ids"):
            rep.fail(f"{ld}/provenance.json", "target_fit_ids is not empty")
        # plans
        _verify_plans(root, ld, rep)
        # checkpoints: weights_only-load and re-verify every physical file
        present = _verify_checkpoints(root, ld, rep)
        # methods
        for name in [d for d in sorted(os.listdir(os.path.join(root, ld, "methods")))]:
            md = f"{ld}/methods/{name}"
            mlogical, mbody, _ = read_artifact(os.path.join(root, md, "method.json"), "method_result")
            if scientific_value_hash(mbody) != mlogical:
                rep.fail(f"{md}/method.json", "method logical hash does not recompute")
            if mbody["selection"]["model_hash"] not in present:
                rep.fail(f"{md}/method.json", "selected checkpoint not present in the store")
            for role in ("source_guard", "source_audit", "target_audit"):
                try:
                    plog, pbody2, parr = read_artifact(os.path.join(root, md, role + ".json"), PR.PREDICTION_KIND)
                    b = PR.decode_prediction({k: v for k, v in pbody2.items() if k != "npz"}, parr)
                except Exception as e:
                    rep.fail(f"{md}/{role}.json", f"prediction verification failed: {e}"); continue
                if role == "source_audit":
                    sa_sigs.add(b.audit_signature_hash)
                if role == "target_audit":
                    ta_sigs.add(b.audit_signature_hash)
            try:
                _, mbody2, _ = read_artifact(os.path.join(root, md, "metrics.json"), PR.METRICS_KIND)
                for role in ("source_guard", "source_audit", "target_audit"):
                    PR.decode_metrics(mbody2["roles"][role])
            except Exception as e:
                rep.fail(f"{md}/metrics.json", f"metrics verification failed: {e}")
            _verify_method_leakage(root, md, rep)
    if len(sa_sigs) > 1:
        rep.fail("levels", "source-audit signatures disagree across methods/levels")
    if len(ta_sigs) > 1:
        rep.fail("levels", "target signatures disagree across methods/levels")
    # fold
    flogical, fbody, _ = read_artifact(os.path.join(root, "fold.json"), "fold_result")
    if scientific_value_hash(fbody["payload"]) != flogical:
        rep.fail("fold.json", "fold logical hash does not recompute")
    expected = {int(l): h for l, h in fbody["payload"]["levels"]}
    if expected != dict(level_hashes):
        rep.fail("fold.json", "fold level hashes disagree with the level files")
    # context provenance + manifest/exec/model payloads recomputed from context/*.json
    ctx_hash = _verify_context(root, rep, level_hashes)
    # artifact scientific hash (recomputed from the manifest hash + context hash + fold hash)
    from .writer import artifact_scientific_hash
    ash = artifact_scientific_hash(flogical, _manifest_hash(root), ctx_hash if ctx_hash else marker.get("context_hash", ""))
    rep.artifact_scientific_hash = ash
    if ash != marker.get("artifact_scientific_hash"):
        rep.fail(COMMIT_MARKER, "artifact scientific hash does not recompute")


def _verify_context(root, rep, level_hashes) -> str:
    """Recompute the context hash from context/manifest|execution_config|model_spec|provenance.json and
    verify the manifest / config / spec payload hashes -- never trust COMMITTED.json alone."""
    import types

    from ..protocol.manifest_v2 import manifest_payload_hash
    from .writer import context_scientific_hash
    mlog, mbody, _ = read_artifact(os.path.join(root, "context", "manifest.json"), "manifest")
    mpay = mbody["manifest"]
    if manifest_payload_hash(mpay) != mlog:
        rep.fail("context/manifest.json", "manifest payload does not recompute the manifest hash")
    _, ecbody, _ = read_artifact(os.path.join(root, "context", "execution_config.json"), "execution_config")
    _, msbody, _ = read_artifact(os.path.join(root, "context", "model_spec.json"), "model_spec")
    ec = [[int(l), m] for l, m in ecbody["levels"]]
    ms = [[int(l), m] for l, m in msbody["levels"]]
    # each level's config/spec payload must hash to that level's stored hash
    for lvl, _h in level_hashes:
        _, lbody, _ = read_artifact(os.path.join(root, f"levels/level-{int(lvl):03d}", "level.json"), "level_result")
        ecp = dict(ec).get(int(lvl)); msp = dict(ms).get(int(lvl))
        if ecp is None or canonical_json_hash(ecp) != lbody["execution_config_hash"]:
            rep.fail("context/execution_config.json", f"level {lvl} execution config payload hash mismatch")
        if msp is None or canonical_json_hash(msp) != lbody["model_spec_hash"]:
            rep.fail("context/model_spec.json", f"level {lvl} model spec payload hash mismatch")
    _, gbody, _ = read_artifact(os.path.join(root, "context", "provenance.json"), "context_provenance")
    git = types.SimpleNamespace(commit=gbody["commit"], tree_hash=gbody["tree_hash"],
                                scientific_paths=tuple(gbody["scientific_paths"]), clean=bool(gbody["clean"]))
    return context_scientific_hash(mpay, ec, ms, git)


def _manifest_hash(root):
    mlog, _, _ = read_artifact(os.path.join(root, "context", "manifest.json"), "manifest")
    return mlog


def _level_num(ld):
    return int(ld.rsplit("-", 1)[-1])


def _verify_checkpoints(root, ld, rep) -> set:
    from .checkpoint import read_checkpoint_file
    ck_dir = os.path.join(root, ld, "checkpoints")
    present = set()
    if not os.path.isdir(ck_dir):
        return present
    pts = {fn[:-3] for fn in os.listdir(ck_dir) if fn.endswith(".pt")}
    jsons = {fn[:-5] for fn in os.listdir(ck_dir) if fn.endswith(".json")}
    for orphan in (pts ^ jsons):
        rep.fail(f"{ld}/checkpoints/{orphan}", "orphan checkpoint (.pt and .json must pair up)")
    for stem in sorted(pts & jsons):
        try:
            logical, meta, _ = read_artifact(os.path.join(ck_dir, f"{stem}.json"), "checkpoint")
            if logical != stem or meta.get("model_hash") != stem:
                rep.fail(f"{ld}/checkpoints/{stem}.json", "checkpoint metadata model_hash != filename stem")
                continue
            read_checkpoint_file(os.path.join(ck_dir, f"{stem}.pt"), meta)   # weights_only load + state_hash check
            present.add(stem); rep.n_verified_checkpoints += 1
        except Exception as e:
            rep.fail(f"{ld}/checkpoints/{stem}.pt", f"checkpoint verification failed: {e}")
    return present


def _verify_plans(root, ld, rep):
    pl = f"{ld}/plans"
    specs = [("stage1_task", P.TASK_KIND, P.decode_task_plan), ("stage2_task", P.TASK_KIND, P.decode_task_plan),
             ("selection_design", P.DESIGN_KIND, P.decode_design),
             ("oaci_alignment", P.ALIGN_KIND, P.decode_alignment_plan),
             ("full_domain_alignment", P.ALIGN_KIND, P.decode_alignment_plan),
             ("selection_fold_plan", P.FOLD_KIND, P.decode_fold_plan),
             ("selection_bootstrap_plan", P.BOOTSTRAP_KIND, P.decode_bootstrap_plan)]
    for fn, kind, dec in specs:
        p = os.path.join(root, pl, fn + ".json")
        if not os.path.exists(p):
            rep.fail(f"{pl}/{fn}.json", "missing plan file"); continue
        try:
            logical, body, arrays = read_artifact(p, kind)
            if body.get("present", True):
                dec({k: v for k, v in body.items() if k not in ("present", "npz")}, arrays)
                rep.n_verified_plans += 1
        except Exception as e:
            rep.fail(f"{pl}/{fn}.json", f"plan verification failed: {e}")


def _verify_method_leakage(root, md, rep):
    for fn in ("selection_leakage", "audit_leakage"):
        p = os.path.join(root, md, fn + ".json")
        try:
            logical, body, _ = read_artifact(p, PR.LEAKAGE_KIND)
            if body.get("present", True):
                PR.decode_leakage({k: v for k, v in body.items() if k != "present"}, logical)
        except Exception as e:
            rep.fail(f"{md}/{fn}.json", f"leakage verification failed: {e}")
    p = os.path.join(root, md, "training_diagnostics.json")
    try:
        logical, body, _ = read_artifact(p, PR.DIAGNOSTICS_KIND)
        PR.decode_diagnostics(body, logical)
    except Exception as e:
        rep.fail(f"{md}/training_diagnostics.json", f"diagnostics verification failed: {e}")


def _main(argv):
    if not argv:
        print("usage: python -m oaci.artifacts.verify <artifact-dir>", file=sys.stderr)
        return 2
    rep = verify_artifact_tree(argv[0], deep=True)
    if rep.ok:
        print(f"OK  indexed_files={rep.n_indexed_files} total_files={rep.n_total_files} "
              f"verified_checkpoints={rep.n_verified_checkpoints} verified_plans={rep.n_verified_plans} "
              f"artifact_scientific_hash={rep.artifact_scientific_hash}")
        return 0
    for path, msg in rep.errors:
        print(f"FAIL  {path}: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
