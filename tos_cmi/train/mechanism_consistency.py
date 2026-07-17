"""Mechanism-Contrast Consistency (MCC) training loss + balanced subject-class episodic sampler.

Direction-normalized cross-subject class-contrast consensus (NOT raw Frobenius): for each source subject d and
unordered class pair (a<b) in the encoder bottleneck, pull each subject's UNIT class-contrast direction toward the
leave-one-subject-out consensus of the others (stop-gradient on the consensus). Scale-insensitive; the task CE
holds the margin. Arm C ("shuffle") permutes the subject assignment within each class to break real within-subject
contrasts while matching batch/class counts/gradient budget. Missing subject-class cells FAIL LOUD (never silently
reweight). Target trials must NEVER enter training. Manuscript FROZEN; only the project owner stops a science line.
"""
from __future__ import annotations
import numpy as np
import torch

EPS = 1e-6


def class_pairs(classes):
    """All unordered class pairs (a<b): 4-class -> 6 pairs, binary -> 1."""
    cs = sorted(int(c) for c in classes)
    return [(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs))]


def _shuffle_subjects_within_class(y, d, generator):
    """Arm C: within each class independently permute the subject assignment (breaks real within-subject contrasts;
    preserves batch composition + per-class subject counts). Returns a permuted subject-label vector."""
    d_new = d.clone()
    for c in torch.unique(y):
        idx = torch.where(y == c)[0]
        # randperm on CPU (generator is CPU) then move to idx's device -> works on both CPU and CUDA tensors
        rp = torch.randperm(idx.numel(), generator=generator).to(idx.device)
        d_new[idx] = d[idx[rp]]
    return d_new


def mcc_loss(Z, y, d, pairs=None, eps=EPS, shuffle_subjects=False, generator=None):
    """L_MCC = mean_d mean_{a<b} [ 1 - <u_d^{a,b}, sg(ubar_{-d}^{a,b})> ].
    Z: (B, p) encoder embeddings (grad flows to the encoder). y: (B,) class ids. d: (B,) subject ids. FAILS LOUD if
    any (subject, class) cell present-subject is missing a class needed for a pair. Returns (loss, info)."""
    if not torch.is_tensor(y):
        y = torch.as_tensor(np.asarray(y), device=Z.device)
    if not torch.is_tensor(d):
        d = torch.as_tensor(np.asarray(d), device=Z.device)
    y = y.long(); d = d.long()
    if shuffle_subjects:
        d = _shuffle_subjects_within_class(y, d, generator)
    subs = torch.unique(d).tolist(); classes = torch.unique(y).tolist()
    if pairs is None:
        pairs = class_pairs(classes)
    # per (subject, class) mean embedding; FAIL LOUD on an empty cell
    mean = {}
    for s in subs:
        for c in classes:
            m = (d == s) & (y == c)
            if int(m.sum()) == 0:
                raise ValueError(f"MCC: subject {s} missing class {c} in batch (balanced sampler must fill every "
                                 f"subject-class cell; do NOT silently reweight)")
            mean[(s, c)] = Z[m].mean(0)
    terms = []; align_sum = 0.0; n = 0
    for (a, b) in pairs:
        U = torch.stack([torch.nn.functional.normalize(mean[(s, a)] - mean[(s, b)], dim=0, eps=eps) for s in subs])  # (m, p)
        Udet = U.detach(); total = Udet.sum(0)
        for i, s in enumerate(subs):
            ubar = torch.nn.functional.normalize(total - Udet[i], dim=0, eps=eps)   # LOSO consensus, stop-grad
            cosang = torch.dot(U[i], ubar)
            terms.append(1.0 - cosang); align_sum += float(cosang.detach()); n += 1
    loss = torch.stack(terms).mean()
    return loss, dict(mean_alignment=align_sum / max(n, 1), n_terms=n, n_subjects=len(subs), n_pairs=len(pairs))


def contrast_norm(Z, y, d):
    """Mean L2 norm of per-subject class contrasts (collapse guard: must not -> 0)."""
    y = torch.as_tensor(np.asarray(y)) if not torch.is_tensor(y) else y.long()
    d = torch.as_tensor(np.asarray(d)) if not torch.is_tensor(d) else d.long()
    subs = torch.unique(d).tolist(); classes = torch.unique(y).tolist(); pairs = class_pairs(classes)
    ns = []
    for s in subs:
        mean = {c: Z[(d == s) & (y == c)].mean(0) for c in classes if int(((d == s) & (y == c)).sum()) > 0}
        for (a, b) in pairs:
            if a in mean and b in mean:
                ns.append(float((mean[a] - mean[b]).norm()))
    return float(np.mean(ns)) if ns else float("nan")


def effective_rank(Z):
    """Entropy-based effective rank of the embedding covariance (collapse guard)."""
    Zc = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Zc.detach())
    p = (s / (s.sum() + 1e-12)); p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


class BalancedSubjectClassSampler:
    """Yields index batches = all source subjects x all classes x K trials/cell (source-side replacement sampling
    when a cell has < K trials). Never emits target trials (caller supplies source-only arrays). Deterministic per
    seed so warm-up/arms are reproducible."""

    def __init__(self, subjects, classes_arr, K=4, n_batches=None, seed=0):
        self.subj = np.asarray(subjects); self.cls = np.asarray(classes_arr); self.K = int(K)
        self.subs = sorted(np.unique(self.subj).tolist()); self.classes = sorted(np.unique(self.cls).tolist())
        self.cells = {(s, c): np.where((self.subj == s) & (self.cls == c))[0] for s in self.subs for c in self.classes}
        missing = [k for k, v in self.cells.items() if v.size == 0]
        if missing:
            raise ValueError(f"BalancedSubjectClassSampler: empty subject-class cells {missing[:5]} "
                             f"(source split must cover every subject x class)")
        self.rng = np.random.default_rng(seed)
        self.batch_size = len(self.subs) * len(self.classes) * self.K
        self.n_batches = int(n_batches) if n_batches else max(1, len(self.subj) // self.batch_size)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            idx = []
            for s in self.subs:
                for c in self.classes:
                    pool = self.cells[(s, c)]
                    replace = pool.size < self.K
                    idx.extend(self.rng.choice(pool, size=self.K, replace=replace).tolist())
            yield np.asarray(idx)
