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


_BUILD = "_build"  # gitignored compile-smoke dir; artifacts there are allowed, never committed


def _outside_build(paths):
    return [p for p in paths if _BUILD not in p.parts]


def test_no_committed_pdf():
    # PDFs may exist only under the gitignored _build/ (compile smoke); never in a tracked path.
    pdfs = _outside_build(LATEX.rglob("*.pdf"))
    assert not pdfs, f"PDF(s) in a committable path (no compile output may be committed): {pdfs}"


def test_no_generated_latex_artifacts_outside_build():
    bad = []
    for ext in ("*.aux", "*.bbl", "*.blg", "*.log", "*.fls", "*.fdb_latexmk", "*.synctex.gz", "*.out"):
        bad += _outside_build(LATEX.rglob(ext))
    assert not bad, f"generated LaTeX artifact(s) outside _build/: {bad}"


def test_no_fabricated_doi_placeholder():
    # citations are resolved or honestly omitted; no placeholder DOI value may appear.
    bib = BIB.read_text().lower()
    for bad in ("todo-doi", "doi = {todo", "doi={todo", "doi = {xxx", "doi = {0000", "doi = {tbd"):
        assert bad not in bib, f"fabricated/placeholder DOI value present: {bad!r}"


def test_bib_parses_and_cited_entries_complete():
    bib = BIB.read_text()
    assert bib.count("{") == bib.count("}"), "unbalanced braces in REFERENCES_DRAFT.bib"
    entries = dict(re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, flags=re.S))
    assert len(entries) >= 11, f"expected >= 11 bib entries, found {len(entries)}"
    # every cited key must resolve to an entry carrying at least author/title (or howpublished) + year.
    for key in _cited_keys():
        assert key in entries, f"cited key not defined in bib: {key}"
        body = entries[key].lower()
        assert "title" in body and "year" in body, f"entry {key} missing title/year"
        assert ("author" in body) or ("howpublished" in body), f"entry {key} missing author"


def _cited_keys():
    keys = set()
    for p in _tex_files():
        for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", p.read_text()):
            for k in m.group(1).split(","):
                if k.strip():
                    keys.add(k.strip())
    return keys


def test_main_uses_only_defined_citation_keys():
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    undefined = sorted(_cited_keys() - bib_keys)
    assert not undefined, f"cited keys not present in REFERENCES_DRAFT.bib: {undefined}"


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
