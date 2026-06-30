"""Phase 4F guard: the cigl_latex package is structurally complete, citation-honest, and claim-bounded.

Checks: required .tex/.bib files exist; no committed PDF; unresolved citation TODOs stay visible; and the
sensitive terms (SOTA, unbiased CMI estimator, edge-CMI, dynamic-edge "method", beyond-MI generalization,
"eliminat*") appear ONLY in a negated / out-of-scope context, never as an affirmative claim; and no
affirmative target-label training/selection claim appears.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LATEX = REPO / "paper" / "cigl_latex"
BIB = REPO / "paper" / "cigl" / "REFERENCES_DRAFT.bib"

SECTIONS = [
    "01_intro", "02_related_work", "03_method", "04_protocol", "05_results",
    "06_analysis_negative_results", "07_limitations_conclusion",
]
TABLES = [
    "table1_method_protocol", "table2_leakage_audit", "table3_bnci2014",
    "table4_bnci2015", "table5_negative_results",
]

NEG_CUES = ("no ", "not ", "never", "without", "do not", "does not", "rather than", "out of scope",
            "unsupported", "future work", "disclaim", "make no", "we make no", "excluded")


def _tex_files():
    files = [LATEX / "main.tex"]
    files += [LATEX / "sections" / f"{s}.tex" for s in SECTIONS]
    files += [LATEX / "tables" / f"{t}.tex" for t in TABLES]
    return files


def _all_tex():
    # strip LaTeX emphasis/braces noise and collapse whitespace so phrase checks aren't broken by markup.
    raw = "\n".join(p.read_text() for p in _tex_files())
    return re.sub(r"\s+", " ", raw.replace("\\textbf{", "").replace("\\emph{", "").replace("}", "")
                  .replace("\\texttt{", ""))


def test_required_files_exist():
    assert (LATEX / "main.tex").exists(), "missing main.tex"
    for s in SECTIONS:
        assert (LATEX / "sections" / f"{s}.tex").exists(), f"missing section {s}.tex"
    for t in TABLES:
        assert (LATEX / "tables" / f"{t}.tex").exists(), f"missing table {t}.tex"
    assert BIB.exists(), "missing REFERENCES_DRAFT.bib"
    assert (LATEX / "figures" / "FIGURE_ASSET_PLAN.md").exists(), "missing FIGURE_ASSET_PLAN.md"


def test_no_committed_pdf():
    pdfs = list(LATEX.rglob("*.pdf"))
    assert not pdfs, f"PDF(s) present (no compile authorized): {pdfs}"


def test_unresolved_citation_todos_visible():
    # 6 citations remain unverified -> the visible [TODO: verify citation] marker must be present.
    t = _all_tex()
    assert "[TODO: verify citation]" in t, "expected visible unresolved-citation markers in the .tex"


def test_bib_has_no_fabricated_doi_for_unverified_entries():
    # entries known to still be TODO (DGCNN, Schirrmeister, Li, CCMI) must NOT carry a doi = {...} field.
    bib = BIB.read_text()
    # crude entry split on '@'
    for entry in re.split(r"(?=@)", bib):
        head = entry[:80].lower()
        if any(k in head for k in ("song2018dgcnn", "schirrmeister2017deep",
                                   "li2018conditional", "mukherjee2020ccmi")):
            assert not re.search(r"\bdoi\s*=", entry), f"unverified entry has a doi field: {head!r}"


def _all_occurrences_negated(text, term, before=90, after=60):
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
    t = _all_tex()
    for term in ("state-of-the-art", "sota", "unbiased cmi estimator", "edge-cmi",
                 "beyond-mi", "eliminat"):
        assert _all_occurrences_negated(t, term), f"'{term}' appears outside a negated/out-of-scope context"


def test_no_affirmative_target_label_or_method_claims():
    t = _all_tex().lower()
    forbidden = [
        "eliminates leakage", "leakage-free", "removes the leakage",
        "dynamic-edge method works", "edge-cmi works", "achieves state-of-the-art",
        "trained on target labels", "target labels for selection",
        "using target labels for training", "select on target",
    ]
    hits = [p for p in forbidden if p in t]
    assert not hits, f"forbidden affirmative claim(s) present: {hits}"
    # and the positive firewall statement must be present
    assert "evaluation-only" in t, "target-labels-evaluation-only statement missing from the package"
