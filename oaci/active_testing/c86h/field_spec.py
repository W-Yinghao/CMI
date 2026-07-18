"""C86H field manifest loader + SYNTHETIC production-format field generator.

The synthetic generator emits the EXACT on-disk field format the frozen C86D isolation
machinery consumes (acquisition pool + sealed oracle + sealed contribution store) plus a
sealed held-evaluation field, so the integrated runner can be exercised end-to-end with
NO real EEG/label. It is used only by the e2e test and the outcome-free resource
benchmark; it is never a substitute for a real authorized field.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from oaci.theory.c86_active_program import canonical_trial_split
from . import contract as K


def field_context_keys() -> list:
    return [f"panel={p}|seed={s}|level={lv}"
            for p in K.PANELS for s in K.TRAINING_SEEDS for lv in K.LEVELS]


def _ctx_meta(ctx: str) -> dict:
    parts = dict(kv.split("=") for kv in ctx.split("|"))
    return {"panel": parts["panel"], "seed": int(parts["seed"]), "level": int(parts["level"])}


def _synth_context(dataset, subject, ctx, n_trials, seed):
    """Deterministic synthetic (labels[n], probs[n,81,2]) with candidate-varying skill."""
    rng = np.random.default_rng(seed)
    K_ = K.CANDIDATES_PER_CONTEXT
    labels = rng.integers(0, 2, size=n_trials)
    # candidate skill: ERM(0) mid, OACI(1..40) rising, SRC(41..80) lower/noisier
    skill = np.concatenate([[1.2], np.linspace(0.6, 2.4, 40), np.linspace(0.2, 1.4, 40)])
    logits = (skill[None, :] * (2.0 * labels[:, None] - 1.0)
              + rng.normal(0.0, 0.8, size=(n_trials, K_)))
    p1 = 1.0 / (1.0 + np.exp(-logits))
    probs = np.stack([1.0 - p1, p1], axis=2)          # [n,81,2]
    return labels.astype(int), probs.astype(np.float64)


def _contribs(probs, labels):
    n, K_, _ = probs.shape
    y = np.asarray(labels).astype(int)
    p_true = probs[np.arange(n)[:, None], np.arange(K_)[None, :], y[:, None]]
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
    Returns a field manifest (targets, per-target pool sizes, context keys).
    """
    pool_root = os.path.join(root, "acquisition_unlabeled_pool")
    oracle_root = os.path.join(root, "acquisition_label_oracle")
    contrib_root = os.path.join(root, "query_contribution_store")
    held_root = os.path.join(root, "held_evaluation_field")
    for d in (pool_root, oracle_root, contrib_root, held_root):
        os.makedirs(d, exist_ok=True)

    contexts = field_context_keys()
    label_rows = []
    target_cohort, target_dataset, pool_sizes = {}, {}, {}

    for cohort, spec in cohorts.items():
        ds = spec["dataset"]
        for subj in spec["subjects"]:
            subj = int(subj)                       # canonical integer target id (frozen server casts int)
            tgt = (ds, subj)
            target_cohort[str(tgt)] = cohort
            for ci, ctx in enumerate(contexts):
                cseed = seed + (abs(hash((ds, subj, ctx))) % 1_000_000)
                labels, probs = _synth_context(ds, subj, ctx, spec["n_trials"], cseed)
                trial_ids = [f"{subj}_t{j}" for j in range(spec["n_trials"])]
                pool_ids, held_ids = canonical_trial_split(ds, str(subj), trial_ids,
                                                           salt=K.SPLIT_SALT)
                idx = {t: j for j, t in enumerate(trial_ids)}
                pj = [idx[t] for t in pool_ids]
                hj = [idx[t] for t in held_ids]
                pool_sizes[str(tgt)] = len(pj)
                meta = json.dumps({"dataset": ds, "subject": subj, **_ctx_meta(ctx)})
                tag = f"{ds}__{subj}__p{_ctx_meta(ctx)['panel']}_s{_ctx_meta(ctx)['seed']}_l{_ctx_meta(ctx)['level']}"

                # acquisition pool (client-visible probs only)
                np.savez(os.path.join(pool_root, tag + ".npz"), meta=meta,
                         trial_ids=np.array(pool_ids), probabilities=probs[pj])
                # sealed contribution store (derived fields + true labels)
                pc = _contribs(probs[pj], labels[pj])
                np.savez(os.path.join(contrib_root, tag + ".npz"), meta=meta,
                         trial_ids=np.array(pool_ids), true_label=labels[pj], **pc)
                # sealed held-evaluation field (probs + labels, opened only in H2)
                np.savez(os.path.join(held_root, tag + ".npz"), meta=meta,
                         trial_ids=np.array(held_ids), probabilities=probs[hj],
                         true_label=labels[hj])
                # oracle labels for the pool trials
                for t in pool_ids:
                    label_rows.append({"dataset": ds, "target_subject_id": subj,
                                       "target_trial_id": t,
                                       "canonical_class_label": int(labels[idx[t]])})

    with open(os.path.join(oracle_root, "labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "target_subject_id",
                                           "target_trial_id", "canonical_class_label"])
        w.writeheader()
        w.writerows(label_rows)

    manifest = {"synthetic": True, "contexts": contexts,
                "target_cohort": target_cohort, "pool_sizes": pool_sizes,
                "n_targets": len(target_cohort)}
    with open(os.path.join(root, "C86H_SYNTHETIC_FIELD_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


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
