"""Diagnostic score-shape utilities for source-describability audits."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import audit_utils as au


def rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    denom = max(len(vals) - 1, 1)
    return np.asarray([r / denom for r in ranks], dtype=float)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(set(y.tolist())) < 2:
        return math.nan
    rx = rankdata(x.tolist())
    ry = rankdata(y.tolist())
    sx, sy = float(np.std(rx)), float(np.std(ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(np.abs(p - neg) <= 1e-12))
    return wins / (len(pos) * len(neg))


def auprc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if int(np.sum(labels)) == 0:
        return math.nan
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(labels)), 1)
    prev = 0.0
    area = 0.0
    for p, r in zip(precision, recall):
        area += float(p) * max(float(r) - prev, 0.0)
        prev = float(r)
    return area


def deciles(values, group_keys=None):
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=int)
    if group_keys is None:
        groups = {"all": np.arange(len(values))}
    else:
        groups = defaultdict(list)
        for i, g in enumerate(group_keys):
            groups[g].append(i)
        groups = {k: np.asarray(v, dtype=int) for k, v in groups.items()}
    for idx in groups.values():
        vals = values[idx]
        ranks = rankdata(vals.tolist())
        out[idx] = np.minimum((ranks * 10).astype(int), 9)
    return out


def diagnostic_decile_scores(values, labels, group_keys=None):
    labels = np.asarray(labels)
    dec = deciles(values, group_keys)
    out = np.zeros(len(values), dtype=float)
    if group_keys is None:
        keys = ["all"] * len(values)
    else:
        keys = list(group_keys)
    buckets = defaultdict(list)
    for i, (g, d) in enumerate(zip(keys, dec)):
        buckets[(g, int(d))].append(i)
    fallback = float(np.mean(labels)) if len(labels) else 0.0
    for i, (g, d) in enumerate(zip(keys, dec)):
        idx = buckets[(g, int(d))]
        out[i] = float(np.mean(labels[idx])) if idx else fallback
    return out


def top_hit_by_trajectory(rows, scores):
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[r["trajectory"]].append(i)
    hits = []
    bases = []
    for idx in buckets.values():
        labels = np.asarray([int(rows[i]["query_positive_label"]) for i in idx], dtype=int)
        vals = np.asarray([float(scores[i]) for i in idx], dtype=float)
        top = float(np.max(vals))
        tied = np.where(np.abs(vals - top) <= 1e-12)[0]
        hits.append(float(np.mean(labels[tied])) if len(tied) else math.nan)
        bases.append(float(np.mean(labels)) if len(labels) else math.nan)
    hit = au.finite_mean(hits)
    base = au.finite_mean(bases)
    return hit, au.enrichment(hit, base)
