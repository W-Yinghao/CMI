"""C86H field identity (candidate IDs) + SYNTHETIC production-format field generator.

Semantics B is honoured exactly: ONE physical trial -> ONE label -> its 8 context-specific
probability/contribution rows. All randomness is SHA-seeded (never Python ``hash()``), so a
synthetic field replays byte-identically across interpreter processes. The synthetic
generator emits the exact on-disk field format the batch H1 evaluator consumes plus a sealed
held-evaluation field; it is used only by the e2e test and the outcome-free resource
benchmark and is never a substitute for a real authorized field.

Candidate identity is C86-specific: ``c86_ + SHA256(interface_id | field_training_manifest_sha
| panel | seed | level | regime | epoch)`` — same ID for a candidate across both untouched
cohorts, and disjoint from the historical ``c84_`` 20-channel namespace.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

from oaci.theory.c86_active_program import canonical_trial_split
from . import contract as K

_REGIME_EPOCHS = tuple(range(4, 200, 5))    # 40 stage-2 checkpoints (OACI/SRC)


def _sha_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def field_context_keys() -> list:
    return [f"panel={p}|seed={s}|level={lv}"
            for p in K.PANELS for s in K.TRAINING_SEEDS for lv in K.LEVELS]


def _ctx_meta(ctx: str) -> dict:
    parts = dict(kv.split("=") for kv in ctx.split("|"))
    return {"panel": parts["panel"], "seed": int(parts["seed"]), "level": int(parts["level"])}


# --------------------------------------------------------------- C86 candidate identity
def c86_candidate_id(interface_id: str, manifest_sha: str, panel: str, seed: int,
                     level: int, regime: str, epoch: int) -> str:
    payload = f"{interface_id}|{manifest_sha}|{panel}|{seed}|{level}|{regime}|{epoch}"
    return "c86_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def canonical_candidate_ids(panel: str, seed: int, level: int,
                            interface_id: str = K.COMMON_INTERFACE_ID,
                            manifest_sha: str = K.FIELD_TRAINING_MANIFEST_SHA256) -> list:
    """The 81 C86 candidate IDs in canonical order 0..80 (ERM, OACI 1..40, SRC 1..40)."""
    ids = [c86_candidate_id(interface_id, manifest_sha, panel, seed, level, "ERM", 199)]
    ids += [c86_candidate_id(interface_id, manifest_sha, panel, seed, level, "OACI", e)
            for e in _REGIME_EPOCHS]
    ids += [c86_candidate_id(interface_id, manifest_sha, panel, seed, level, "SRC", e)
            for e in _REGIME_EPOCHS]
    return ids


# ----------------------------------------------------------------- synthetic generation
def _synth_labels(dataset, subject, n_trials, seed) -> np.ndarray:
    rng = np.random.default_rng(_sha_seed("C86H_SYN_LABELS_V1", dataset, subject, seed))
    return rng.integers(0, 2, size=n_trials).astype(int)


def _synth_probs(dataset, subject, ctx, labels, seed) -> np.ndarray:
    """Per-context candidate probabilities [n,81,2] given the SHARED per-trial labels."""
    rng = np.random.default_rng(_sha_seed("C86H_SYN_PROBS_V1", dataset, subject, ctx, seed))
    n, Kc = len(labels), K.CANDIDATES_PER_CONTEXT
    skill = np.concatenate([[1.2], np.linspace(0.6, 2.4, 40), np.linspace(0.2, 1.4, 40)])
    logits = skill[None, :] * (2.0 * labels[:, None] - 1.0) + rng.normal(0.0, 0.8, (n, Kc))
    p1 = 1.0 / (1.0 + np.exp(-logits))
    return np.stack([1.0 - p1, p1], axis=2).astype(np.float64)


def _contribs(probs, labels):
    n, Kc, _ = probs.shape
    y = np.asarray(labels).astype(int)
    p_true = probs[np.arange(n)[:, None], np.arange(Kc)[None, :], y[:, None]]
    nll = -np.log(np.clip(p_true, 1e-7, 1.0))
    pred = probs.argmax(axis=2)
    correct = (pred == y[:, None]).astype(np.float64)
    confidence = probs.max(axis=2)
    conf_bin = np.clip((confidence * 15).astype(int), 0, 14).astype(np.float64)
    signed_cal = correct - confidence
    return {"nll": nll, "correct": correct, "confidence": confidence,
            "conf_bin": conf_bin, "signed_calibration": signed_cal}


def synthesize_field(root: str, cohorts: dict, seed: int = 12345) -> dict:
    """Write a small synthetic field in the exact C86 layout. NO real data.

    cohorts = {cohort_name: {"dataset": ds, "subjects": [...], "n_trials": N}}
    One label per physical trial, shared across all 8 contexts (Semantics B). Returns a field
    manifest (targets, per-target pool sizes, context keys).
    """
    pool_root = os.path.join(root, "acquisition_unlabeled_pool")
    oracle_root = os.path.join(root, "acquisition_label_oracle")
    contrib_root = os.path.join(root, "query_contribution_store")
    held_root = os.path.join(root, "held_evaluation_field")
    for d in (pool_root, oracle_root, contrib_root, held_root):
        os.makedirs(d, exist_ok=True)

    contexts = field_context_keys()
    label_rows = []
    target_cohort, pool_sizes = {}, {}

    for cohort, spec in cohorts.items():
        ds = spec["dataset"]
        for subj in spec["subjects"]:
            subj = int(subj)                          # canonical integer target id
            tgt = (ds, subj)
            target_cohort[str(tgt)] = cohort
            trial_ids = [f"{subj}_t{j}" for j in range(spec["n_trials"])]
            labels = _synth_labels(ds, subj, spec["n_trials"], seed)   # ONE label per trial
            pool_ids, held_ids = canonical_trial_split(ds, str(subj), trial_ids,
                                                       salt=K.SPLIT_SALT)
            idx = {t: j for j, t in enumerate(trial_ids)}
            pj = [idx[t] for t in pool_ids]; hj = [idx[t] for t in held_ids]
            pool_sizes[str(tgt)] = len(pj)
            for ctx in contexts:
                probs = _synth_probs(ds, subj, ctx, labels, seed)      # shared labels
                meta = json.dumps({"dataset": ds, "subject": subj, **_ctx_meta(ctx)})
                cm = _ctx_meta(ctx)
                tag = f"{ds}__{subj}__p{cm['panel']}_s{cm['seed']}_l{cm['level']}"
                np.savez(os.path.join(pool_root, tag + ".npz"), meta=meta,
                         trial_ids=np.array(pool_ids), probabilities=probs[pj])
                pc = _contribs(probs[pj], labels[pj])
                np.savez(os.path.join(contrib_root, tag + ".npz"), meta=meta,
                         trial_ids=np.array(pool_ids), true_label=labels[pj], **pc)
                np.savez(os.path.join(held_root, tag + ".npz"), meta=meta,
                         trial_ids=np.array(held_ids), probabilities=probs[hj],
                         true_label=labels[hj])
            for t in pool_ids:                        # oracle: one label per physical trial
                label_rows.append({"dataset": ds, "target_subject_id": subj,
                                   "target_trial_id": t,
                                   "canonical_class_label": int(labels[idx[t]])})

    with open(os.path.join(oracle_root, "labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "target_subject_id",
                                           "target_trial_id", "canonical_class_label"])
        w.writeheader(); w.writerows(label_rows)

    manifest = {"synthetic": True, "semantics": "B_one_label_per_physical_trial",
                "contexts": contexts, "target_cohort": target_cohort,
                "pool_sizes": pool_sizes, "n_targets": len(target_cohort)}
    with open(os.path.join(root, "C86H_SYNTHETIC_FIELD_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_pool(pool_root: str) -> dict:
    """target=(ds,subj) -> {trial: {context: probs[81,2]}} (unlabeled; H1a only)."""
    import glob
    out = {}
    for pf in sorted(glob.glob(os.path.join(pool_root, "*.npz"))):
        z = np.load(pf, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        tgt = (meta["dataset"], meta["subject"])
        ctx = f"panel={meta['panel']}|seed={meta['seed']}|level={meta['level']}"
        probs = z["probabilities"].astype(np.float64)
        for j, t in enumerate(z["trial_ids"]):
            out.setdefault(tgt, {}).setdefault(str(t), {})[ctx] = probs[j]
    return out


def load_held_field(held_root: str) -> dict:
    """Open the sealed held-evaluation field. Called ONLY in H2 after freeze verification."""
    import glob
    out = {}
    for hf in sorted(glob.glob(os.path.join(held_root, "*.npz"))):
        z = np.load(hf, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        tgt = (meta["dataset"], meta["subject"])
        ctx = f"panel={meta['panel']}|seed={meta['seed']}|level={meta['level']}"
        out[(tgt, ctx)] = {"probs": z["probabilities"].astype(np.float64),
                           "labels": z["true_label"].astype(int)}
    return out
