"""P1.5 metadata sidecar + replay — PREPARATION (build now; binding runs only AFTER Freeze A1).

Goal: recover the row-level metadata that feat_dump_v2/*.npz lacks (domain/subject/recording/sample ids, splits),
WITHOUT modifying the dumps, and bind it ONLY if a deterministic replay proves exact row-level correspondence.

Sidecar schema (per array row): array_key, row_index, sample_id, subject_id, recording_id, cohort_id,
                                domain_role, split, fold, seed.

Binding policy (per directive):
- Recover fields from the dataset loader indices + the runner's seeded split logic (replayed deterministically).
- LABEL ≠ ROW binding: repetitive binary EEG labels mean y_se/y_ev/y_te can be identical even under a within-class
  permutation. So the label pre-check proves only SPLIT + LABEL-SEQUENCE compatibility, NOT that row i's feature
  belongs to sidecar row i's subject/sample. Row-level binding therefore REQUIRES the post-Freeze-A1 feature-hash
  replay — that gate is necessary, not redundant caution.
- LABEL-level pre-check (no model needed, allowed before Freeze A1): recovered y == dump y, exactly.
- FEATURE-level binding (AFTER Freeze A1 only): re-embed with the FROZEN checkpoint, hash features, require an
  exact match to feat_dump_v2; only then write the bound sidecar.
- If metadata cannot be recovered/verified -> status BLOCKED_ON_GROUP_METADATA (NOT scientific INCONCLUSIVE);
  no p15_decision.json. The representation half may still produce a PARTIAL report (no retain/drop).

This module never edits feature-generation code and never overwrites the dumps.
"""
import os, sys, glob, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root on path (script mode)
import numpy as np

DUMP_DIR = "results/feat_dump_v2"
SIDECAR_DIR = "results/feat_dump_v2_sidecar"     # NEW dir; dumps stay untouched
SIDE_FIELDS = ["array_key", "row_index", "window_index", "sample_id", "subject_id", "recording_id",
               "cohort_id", "group_id", "domain_role", "split", "fold", "seed"]

# Binding statuses (per directive): label-match alone is NOT row-binding (repetitive binary EEG labels allow
# within-class permutation with identical y_se/y_ev/y_te). Distinguish the three real outcomes:
FEATURE_BOUND = "FEATURE_BOUND"                       # labels/IDs replay AND frozen-pipeline features match the dump
REQUIRES_FROZEN_REDUMP = "REQUIRES_FROZEN_REDUMP"     # labels/IDs replay BUT frozen features != old dump (version/config)
METADATA_BINDING_FAILED = "METADATA_BINDING_FAILED"   # split/sample-order/subject-mapping itself inconsistent


def _split_indices(n_tr, seed, frac=0.7):
    """Replicate the runner's RANDOM source-internal split (run_scps_crossdataset: default leakage_split=random)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_tr); cut = int(frac * len(idx))
    return idx[:cut], idx[cut:]                  # pi (probe-train), ei (probe-eval)


def reconstruct_fold(y, subj, coh, hold, seed):
    """Given the loader arrays + a held-out cohort, reconstruct the EXACT (se, ev, te) row -> global-index maps and
    the labels the dump SHOULD contain. Deterministic; no model. Returns dict of {split: (global_idx, y, subj, coh)}."""
    te = np.where(coh == hold)[0]
    tr = np.where(coh != hold)[0]
    pi, ei = _split_indices(len(tr), seed)
    se, ev = tr[pi], tr[ei]
    return {"se": dict(gidx=se, y=y[se], subj=subj[se], coh=coh[se]),
            "ev": dict(gidx=ev, y=y[ev], subj=subj[ev], coh=coh[ev]),
            "te": dict(gidx=te, y=y[te], subj=subj[te], coh=coh[te])}


def build_sidecar_rows(cond, hold, fold_map, seed):
    """Sidecar rows for one cohort dump's three arrays (z_se,z_ev,z_te). IDs are GLOBALLY NAMESPACED so that
    same-named subjects across cohorts are never merged into one group:
        group_id  = cohort_id :: subject_id            (the unit for GROUPED splits)
        sample_id = cohort_id :: subject_id :: recording_id :: window_index
    recording_id defaults to subject_id (these EEG cohorts are ~one recording/subject); window_index is the
    sample's global row in the canonical loader order."""
    role = {"se": "source", "ev": "source", "te": "target"}
    rows = []
    for split in ("se", "ev", "te"):
        m = fold_map[split]
        for ri, g in enumerate(m["gidx"]):
            subj, coh = str(m["subj"][ri]), str(m["coh"][ri])
            rows.append(dict(array_key=f"z_{split}", row_index=ri, window_index=int(g),
                             sample_id=f"{coh}::{subj}::{subj}::{int(g)}",
                             subject_id=subj, recording_id=subj, cohort_id=coh,
                             group_id=f"{coh}::{subj}",                # namespaced grouping unit
                             domain_role=role[split], split=split, fold=str(hold), seed=int(seed)))
    return rows


def label_precheck(dump_npz, fold_map):
    """LABEL-level correspondence (allowed pre-Freeze-A1): recovered y == dump y EXACTLY, per array."""
    o = np.load(dump_npz)
    res = {}
    for split, key in (("se", "y_se"), ("ev", "y_ev"), ("te", "y_te")):
        if key not in o:
            res[split] = ("missing", False); continue
        rec = fold_map[split]["y"]
        ok = (rec.shape == o[key].shape) and bool(np.array_equal(rec, o[key]))
        res[split] = (f"n={len(o[key])}", ok)
    return res


def feature_hash(z):
    return hashlib.sha256(np.ascontiguousarray(np.round(np.asarray(z, float), 5), dtype=np.float64).tobytes()).hexdigest()[:16]


def recover_and_precheck(loader, conditions=("PD", "SCZ"), seed=0):
    """PREPARATION entry: replay loader + splits, recover sidecar rows, run the LABEL pre-check against the dumps.
    `loader(cond)` -> (y, subj, coh) in canonical order (X not needed for metadata). Returns a report dict.
    Does NOT bind (feature-level binding is post-Freeze-A1). Does NOT touch the dumps."""
    report = dict(status="PREP", conditions={}, all_labels_match=True)
    for cond in conditions:
        try:
            y, subj, coh = loader(cond)
        except Exception as e:
            report["conditions"][cond] = dict(error=str(e)); report["all_labels_match"] = False; continue
        cohs = sorted(set(coh.tolist()))
        cond_rep = dict(cohorts=cohs, folds={})
        for hold in cohs:
            dump = f"{DUMP_DIR}/feat_{cond}_{hold}_erm_0.npz"
            if not os.path.exists(dump):
                cond_rep["folds"][hold] = dict(dump="missing"); report["all_labels_match"] = False; continue
            fm = reconstruct_fold(y, subj, coh, hold, seed)
            pc = label_precheck(dump, fm)
            rows = build_sidecar_rows(cond, hold, fm, seed)
            cond_rep["folds"][hold] = dict(label_precheck={k: v for k, v in pc.items()},
                                           n_sidecar_rows=len(rows),
                                           ok=all(v[1] for v in pc.values()))
            if not cond_rep["folds"][hold]["ok"]:
                report["all_labels_match"] = False
        report["conditions"][cond] = cond_rep
    return report


def dump_generation_config(cond):
    """The config that produced feat_dump_v2 (r12feat run). Co-source with Freeze A1 requires these to match the
    frozen pipeline EXACTLY (deterministic flag, selection objective, config grid, batch order, drop_last, seed)."""
    import json as _j
    for p in (f"results/r9_dualpc2/r12feat_{cond}.json", f"results/feat_dump_v2/r12feat_{cond}.json"):
        if os.path.exists(p):
            c = _j.load(open(p)).get("config", {})
            return dict(present=True, deterministic=c.get("deterministic", False),
                        select=c.get("select"), select_pipeline=c.get("select_pipeline"),
                        configs=c.get("configs"), seed=c.get("seed"), leakage_split=c.get("leakage_split"))
    return dict(present=False)

def is_cosource(freeze_manifest, gen_cfg):
    """feat_dump_v2 is co-source with Freeze A1 iff its generation used the SAME frozen pipeline settings."""
    if not gen_cfg.get("present"):
        return False, ["no dump-generation manifest"]
    fr = (freeze_manifest or {}).get("frozen", {})
    diffs = []
    if not gen_cfg.get("deterministic"):
        diffs.append("dump used NON-deterministic training (pre-fix)")
    if fr.get("select_pipeline") and gen_cfg.get("select_pipeline") != fr.get("select_pipeline"):
        diffs.append(f"select_pipeline {gen_cfg.get('select_pipeline')} != frozen {fr.get('select_pipeline')}")
    if fr.get("configs") and gen_cfg.get("configs") != fr.get("configs"):
        diffs.append("candidate grid differs")
    return (len(diffs) == 0), diffs

def assess_binding(freeze_manifest, gen_cfg, metadata_ok, feat_match_erm=None, feat_match_lpc=None):
    """Return one of the three statuses + per-branch detail. feat_match_* is the post-replay feature-hash result
    (None until the frozen re-embed has run). ERM and LPC are bound SEPARATELY (passing ERM does NOT imply LPC)."""
    if not metadata_ok:
        return METADATA_BINDING_FAILED, {"reason": "split/sample-order/subject mapping inconsistent"}
    cosrc, diffs = is_cosource(freeze_manifest, gen_cfg)
    if not cosrc:
        return REQUIRES_FROZEN_REDUMP, {"reason": "feat_dump_v2 not co-source with Freeze A1", "diffs": diffs}
    # co-source: require the frozen-pipeline feature replay to match, PER BRANCH
    if feat_match_erm is None or feat_match_lpc is None:
        return REQUIRES_FROZEN_REDUMP, {"reason": "feature replay not yet run (post-Freeze-A1)"}
    if feat_match_erm and feat_match_lpc:
        return FEATURE_BOUND, {"erm": True, "lpc": True}
    return REQUIRES_FROZEN_REDUMP, {"reason": "frozen features != dump", "erm": bool(feat_match_erm), "lpc": bool(feat_match_lpc)}

def real_loader(cond):
    """Lightweight metadata loader: returns (y, subj, coh) in the SAME canonical order as the runner's load()."""
    from cmi.run_scps_crossdataset import load
    X, y, subj, coh, classes = load(cond, None)
    return np.asarray(y), np.asarray(subj), np.asarray(coh)


if __name__ == "__main__":
    # PREPARATION run: recover + label pre-check (loads the dataset metadata; no model, no probe, no binding).
    rep = recover_and_precheck(real_loader)
    os.makedirs(SIDECAR_DIR, exist_ok=True)
    json.dump(rep, open(f"{SIDECAR_DIR}/precheck_report.json", "w"), indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str)[:2000])
    print("\nlabel pre-check all-match:", rep["all_labels_match"],
          "-> sidecar binding still GATED on Freeze A1 feature-hash replay" if rep["all_labels_match"]
          else "-> BLOCKED_ON_GROUP_METADATA (labels do not correspond; investigate split/seed/loader-order)")
