"""Phase 4B guard: the CIGL manuscript draft must stay within the bounded claims.

Checks (on paper/cigl/MANUSCRIPT_DRAFT.md): required disclaimers/terms are present; clear affirmative
overclaim phrasings are absent; and the sensitive terms (state-of-the-art, unbiased CMI, edge-CMI,
"eliminat*") only ever appear in a NEGATED / out-of-scope context (a negation cue within a small window).
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRAFT = REPO / "paper" / "cigl" / "MANUSCRIPT_DRAFT.md"

NEG_CUES = ("no ", "not ", "never", "without", "do not", "does not", "rather than", "out of scope",
            "future work", "disclaim", "no claim", "we make no")


def _draft():
    assert DRAFT.exists(), f"missing {DRAFT}"
    # strip markdown emphasis/code marks and collapse whitespace so literal phrase checks aren't broken by
    # **bold** / `code` or by line-wrapping (e.g. "out of\nscope" -> "out of scope").
    return re.sub(r"\s+", " ", DRAFT.read_text().replace("*", "").replace("`", ""))


def test_required_terms_present():
    t = _draft().lower()
    assert "posterior-kl" in t                       # proxy framing
    assert "evaluation-only" in t                     # target labels evaluation-only
    assert "source-only" in t
    assert "not an unbiased" in t                     # explicit CMI disclaimer
    assert "partial" in t                             # partial, not elimination


def test_no_affirmative_overclaim_phrases():
    t = _draft().lower()
    # ONLY unambiguously-affirmative phrasings (these never appear inside our negated disclaimers).
    forbidden_affirmative = [
        "we achieve state-of-the-art", "achieves state-of-the-art", "new state of the art",
        "state-of-the-art accuracy on", "eliminates leakage", "leakage-free", "removes the leakage",
        "unbiased estimate of cmi", "we provide an unbiased", "generalizes to all", "works for all eeg",
        "at no task cost",                                   # Phase 4C: use the retention-gate wording instead
        "per-sample a(x) is what memorizes",                 # Phase 4C: no causal isolation of A(x)
        "without harming source-task performance",           # Phase 4D: use the retention-gate wording instead
    ]
    hits = [p for p in forbidden_affirmative if p in t]
    assert not hits, f"affirmative overclaim phrasing present: {hits}"


def test_dynamic_edge_wording_is_cautious():
    """If 'task-harmful' appears, the cautious 'not causally isolate' qualifier must be present."""
    t = _draft().lower()
    if "task-harmful" in t:
        assert "not causally isolate" in t or "do not causally isolate" in t
    assert "not causally isolate" in t                       # the cautious dynamic-edge framing is present


def _all_occurrences_negated(text, term, before=90, after=60):
    """Every case-insensitive occurrence of `term` must have a negation/out-of-scope cue within `before`
    chars before OR `after` chars after it (disclaimers may precede, e.g. 'we make no … edge-CMI', or
    follow, e.g. 'edge-CMI is out of scope')."""
    low = text.lower()
    term = term.lower()
    i = 0
    while True:
        j = low.find(term, i)
        if j == -1:
            return True
        ctx = low[max(0, j - before): j + len(term) + after]
        if not any(cue in ctx for cue in NEG_CUES):
            return False
        i = j + len(term)


def test_sensitive_terms_only_in_negated_context():
    t = _draft()
    for term in ("state-of-the-art", "unbiased CMI", "edge-CMI", "eliminat"):
        assert _all_occurrences_negated(t, term), f"'{term}' appears outside a negated/out-of-scope context"


def test_paper_files_present_and_nontrivial():
    for f in ("README.md", "MANUSCRIPT_DRAFT.md", "CLAIMS_AUDIT.md", "TABLES_AND_FIGURES_PLAN.md",
              "RELATED_WORK_MATRIX.md", "OPEN_PAPER_BLOCKERS.md"):
        p = REPO / "paper" / "cigl" / f
        assert p.exists() and len(p.read_text().splitlines()) >= 20, f"{f} missing/too short"
