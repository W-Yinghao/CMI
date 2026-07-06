"""Fork 2 --- source-only ENVIRONMENT definitions (E0-E6) for source-rich discovery. Each returns a per-source-
trial integer environment label (SOURCE-ONLY; no target, no held-out data), so a leave-one-ENVIRONMENT-out
split generalizes leave-one-subject-out. Environments are coherent SUBJECT groups (a valid domain split), so
clustering is done at subject level and broadcast to trials. See notes/SOURCE_RICH_ENV_DESIGN.md.

Availability: E0/E1(if session meta)/E2/E4 are implemented from the frozen Z + metadata; E3 (spectral) and E5
(augmentation) are data-dependent (need raw signal / perturbation) and return (None, reason) for now; E6
(cross-dataset) is deferred. env_labels(...) returns (labels or None, reason).
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from tos_cmi.eeg.erasure_baselines import _ids

AVAILABLE = ["subject", "session", "subject_session", "covariance_cluster", "margin_cluster"]
DATA_DEPENDENT = ["spectral_cluster", "augmentation_shift"]     # need raw signal / perturbation
DEFERRED = ["cross_dataset"]


def _subject_feats(Zs, subj):
    """Per-subject source-only feature = [mean(Z), std(Z), flattened upper-tri of cov(Z)] (capped)."""
    subs = sorted(set(subj.tolist())); feats = []
    for s in subs:
        Z = Zs[subj == s]
        c = np.cov(Z.T)
        iu = np.triu_indices(c.shape[0])
        feats.append(np.concatenate([Z.mean(0), Z.std(0), c[iu][:200]]))
    F = np.array(feats)
    return subs, (F - F.mean(0)) / (F.std(0) + 1e-8)


def _cluster_subjects(subs, feat, subj, k, seed):
    kk = int(min(k, len(subs)))
    if kk < 2:
        return None, "too few subjects for k=%d clusters (have %d)" % (k, len(subs))
    lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(feat)
    s2c = {s: int(lab[i]) for i, s in enumerate(subs)}
    return np.array([s2c[s] for s in subj], int), None


def env_labels(name, Zs, ys, subj, session=None, k=8, seed=0):
    """Return (per-trial int environment labels, reason). labels=None when unavailable (reason set)."""
    subj = _ids(subj)[0] if subj.dtype.kind not in "iu" else subj
    if name == "subject":
        return subj.copy(), None
    if name in ("session", "subject_session"):
        if session is None:
            return None, "no session metadata in dump"
        sess = _ids(session)[0]
        if name == "session":
            return sess, None
        return _ids(np.array(["%d_%d" % (a, b) for a, b in zip(subj, sess)]))[0], None
    if name == "covariance_cluster":
        subs, F = _subject_feats(Zs, subj)
        return _cluster_subjects(subs, F, subj, k, seed)
    if name == "margin_cluster":
        if len(np.unique(ys)) < 2:
            return None, "single-class source; margin undefined"
        # cross-fit source margins (source labels legal), aggregate per subject, cluster subjects
        proba = cross_val_predict(LogisticRegression(max_iter=200), Zs, ys, cv=5, method="predict_proba")
        p = np.clip(proba.max(1), 1e-6, 1 - 1e-6)
        margin = p - proba.min(1)                      # source-only confidence margin
        ent = -(proba * np.log(np.clip(proba, 1e-9, 1))).sum(1)
        subs = sorted(set(subj.tolist()))
        F = np.array([[margin[subj == s].mean(), ent[subj == s].mean()] for s in subs])
        F = (F - F.mean(0)) / (F.std(0) + 1e-8)
        return _cluster_subjects(subs, F, subj, k, seed)
    if name in DATA_DEPENDENT:
        return None, "data-dependent (needs raw signal / perturbation); deferred to Phase 3 data step"
    if name in DEFERRED:
        return None, "cross-dataset environments deferred (optional, not first round)"
    raise ValueError("unknown environment '%s'" % name)


def leave_one_environment_out(env):
    """Yield (train_mask, held_env_id) over the distinct environment ids (the source-LOEO split)."""
    for e in sorted(set(env.tolist())):
        yield (env != e), e


def random_partition_matched(env, seed):
    """Size-matched RANDOM partition (same #environments and sizes) -- the p-hacking control baseline."""
    rng = np.random.default_rng(seed)
    ids, counts = np.unique(env, return_counts=True)
    perm = rng.permutation(len(env)); out = np.empty(len(env), int); i = 0
    for e, c in zip(ids, counts):
        out[perm[i:i + c]] = e; i += c
    return out
