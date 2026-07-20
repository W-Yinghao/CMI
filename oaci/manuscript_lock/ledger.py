"""C21 — claim-ledger validation. Every claim must be evidence-grounded (report + commit + key number); the
within-target ~0.64 must stay FUTURE-WORK (never an established generalization / estimand replacement); and no
generated text may contain a forbidden over-claim."""
from __future__ import annotations

from . import schema


def by_category() -> dict:
    out = {c: [] for c in schema.CATEGORIES}
    for cl in schema.CLAIMS:
        out[cl["category"]].append(cl)
    return out


def validate_claims() -> None:
    ids = set()
    for cl in schema.CLAIMS:
        for k in ("id", "category", "text", "commit", "report", "key"):
            if not cl.get(k):
                raise ValueError(f"claim {cl.get('id')} missing evidence field {k!r}")
        if cl["category"] not in schema.CATEGORIES:
            raise ValueError(f"claim {cl['id']} bad category {cl['category']}")
        if cl["id"] in ids:
            raise ValueError(f"duplicate claim id {cl['id']}")
        ids.add(cl["id"])


def assert_estimand_not_swapped() -> None:
    """The within-target ~0.64 must be FUTURE-WORK (F1), never an 'established' generalization/success claim,
    and no 'established' claim may assert external validation success."""
    f1 = next((c for c in schema.CLAIMS if c["id"] == "F1"), None)
    if f1 is None or f1["category"] != "future_work":
        raise ValueError("within-target ~0.64 (F1) must be categorized future_work, not a success claim")
    for c in schema.CLAIMS:
        if c["category"] == "established" and ("external" in c["text"].lower() and "not " not in c["text"].lower()
                                               and "not_" not in c["text"].lower()):
            raise ValueError(f"established claim {c['id']} appears to assert external validation success")


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "no deployable", "not a", "not established", "fails", "barred", "instead of")


def assert_no_forbidden(text) -> None:
    """Flag a forbidden phrase only when it is AFFIRMATIVE — i.e. NOT preceded (within ~30 chars) by a negation
    cue. This lets the claim ledger legitimately say 'NO deployable ... selector is established' while still
    catching an affirmative over-claim."""
    low = text.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            before = low[max(0, i - 30):i]
            if not any(cue in before for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim near: ...{low[max(0, i - 30):i + len(s)]!r}")
            i += len(s)


def counts() -> dict:
    bc = by_category()
    return {c: len(bc[c]) for c in schema.CATEGORIES}
