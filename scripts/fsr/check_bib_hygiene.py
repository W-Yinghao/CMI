#!/usr/bin/env python
"""FSR Phase 5A — fail-closed BibTeX hygiene checker for paper/fsr/refs.bib.

Fails (exit 1) on: the word "verify"; "ICCV/TPAMI"; "others" in an author field; "Anonymous" in an
EXTERNAL entry (one whose note does not say "Internal"); "preprint" in an entry with no eprint/arXiv
id; unused bib keys (defined but never \\cite-d); undefined bib keys (\\cite-d but not in the .bib).
Writes results/fsr_phase5a/bib_hygiene_report.txt.

    python scripts/fsr/check_bib_hygiene.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIB = REPO / "paper/fsr/refs.bib"
TEXDIR = REPO / "paper/fsr"
OUT = REPO / "results/fsr_phase5a/bib_hygiene_report.txt"


def parse_entries(text):
    """Return list of (key, body) for each @type{key, ... } entry (brace-balanced)."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        key = m.group(2).strip()
        start = m.end()
        depth = 1
        i = m.start(0)
        # find opening brace of the entry
        ob = text.index("{", m.start(0))
        j = ob + 1
        depth = 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        entries.append((key, text[ob + 1:j - 1]))
    return entries


def field(body, name):
    m = re.search(rf"{name}\s*=\s*", body, re.I)
    if not m:
        return None
    i = m.end()
    # value may be {..} or "..";
    if body[i] == "{":
        depth = 1
        j = i + 1
        while j < len(body) and depth > 0:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        return body[i + 1:j - 1]
    if body[i] == '"':
        j = body.index('"', i + 1)
        return body[i + 1:j]
    m2 = re.match(r"([^,\n]+)", body[i:])
    return m2.group(1).strip() if m2 else None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = BIB.read_text()
    entries = parse_entries(text)
    keys = {k for k, _ in entries}
    fails, notes = [], []

    for key, body in entries:
        low = body.lower()
        is_internal = "internal" in (field(body, "note") or "").lower()
        auth = field(body, "author") or ""
        note = field(body, "note") or ""
        journal = field(body, "journal") or ""
        has_eprint = bool(field(body, "eprint") or field(body, "archivePrefix"))
        if "verify" in low:
            fails.append(f"[{key}] contains 'verify'")
        if "iccv/tpami" in low:
            fails.append(f"[{key}] contains 'ICCV/TPAMI'")
        if re.search(r"\bothers\b", auth, re.I):
            fails.append(f"[{key}] author field contains 'others'")
        if "anonymous" in auth.lower() and not is_internal:
            fails.append(f"[{key}] 'Anonymous' author in an EXTERNAL entry (mark note=Internal or fix author)")
        if "preprint" in (note + " " + journal).lower() and not has_eprint:
            fails.append(f"[{key}] says 'preprint' but has no eprint/arXiv id")

    # citation cross-reference
    cited = set()
    for tex in TEXDIR.rglob("*.tex"):
        t = tex.read_text()
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", t):
            for k in m.group(1).split(","):
                if k.strip():
                    cited.add(k.strip())
    undefined = sorted(cited - keys)
    unused = sorted(keys - cited)
    if undefined:
        fails.append(f"undefined bib keys (cited, not in .bib): {undefined}")
    if unused:
        fails.append(f"unused bib keys (in .bib, never cited): {unused}")

    notes.append(f"entries: {len(entries)}; cited keys: {len(cited)}; internal-marked: "
                 f"{sum('internal' in (field(b,'note') or '').lower() for _,b in entries)}")
    report = ["FSR bib hygiene check", "=" * 40, *notes, ""]
    report += (["PASS — no issues"] if not fails else ["FAIL:"] + [f"  - {x}" for x in fails])
    OUT.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
