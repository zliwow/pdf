import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # pymupdf


# Matches a ToC line like:
#   "1 Introduction ............ 16"
#   "3.2.1  Boot Sequence                 42"
#   "2  Scope"
# Groups: section_number, title, optional page_number.
TOC_LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<num>\d+(?:\.\d+)*)          # section number: 1, 1.2, 3.2.1
    [\s\.]+                         # dot leaders or spaces
    (?P<title>.+?)                  # title (non-greedy)
    (?:[\s\.]+(?P<page>\d{1,4}))?   # optional trailing page number
    \s*$
    """,
    re.VERBOSE,
)


def parse_toc_pages(pdf_path: Path, page_range: tuple[int, int]) -> list[dict]:
    doc = fitz.open(pdf_path)
    lo = max(0, page_range[0] - 1)
    hi = min(doc.page_count - 1, page_range[1] - 1)

    entries: list[dict] = []
    for pno in range(lo, hi + 1):
        text = doc[pno].get_text("text")
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = TOC_LINE_RE.match(line)
            if not m:
                continue
            num = m.group("num")
            title = m.group("title").strip().strip(".").strip()
            page = m.group("page")
            entries.append({
                "section_number": num,
                "title": title,
                "page": int(page) if page else None,
                "level": num.count(".") + 1,
                "source_toc_page": pno + 1,
            })

    # De-dupe while preserving order (same section can appear multiple times
    # if it wraps across lines; keep the entry that has a page number).
    seen: dict[str, dict] = {}
    for e in entries:
        key = e["section_number"]
        if key not in seen:
            seen[key] = e
        elif seen[key]["page"] is None and e["page"] is not None:
            seen[key] = e
    return list(seen.values())


def attach_page_ranges(entries: list[dict], total_pages: int) -> None:
    """For entries with page numbers, set end_page = next entry's page - 1."""
    numbered = [e for e in entries if e["page"] is not None]
    numbered.sort(key=lambda e: e["page"])
    for i, e in enumerate(numbered):
        nxt = numbered[i + 1]["page"] if i + 1 < len(numbered) else total_pages + 1
        e["end_page"] = max(e["page"], nxt - 1)


def main() -> int:
    p = argparse.ArgumentParser(description="Parse a printed Table of Contents from the PDF.")
    p.add_argument("pdf", type=Path)
    p.add_argument("--toc-pages", required=True,
                   help="1-indexed page range where the printed ToC lives, e.g. '14-15' or '14'")
    p.add_argument("-o", "--out", type=Path, default=Path("toc.json"))
    args = p.parse_args()

    if "-" in args.toc_pages:
        a, b = args.toc_pages.split("-", 1)
        page_range = (int(a), int(b))
    else:
        n = int(args.toc_pages)
        page_range = (n, n)

    doc = fitz.open(args.pdf)
    total_pages = doc.page_count
    doc.close()

    entries = parse_toc_pages(args.pdf, page_range)
    attach_page_ranges(entries, total_pages)

    with_pages = sum(1 for e in entries if e["page"] is not None)
    payload = {
        "source": str(args.pdf),
        "toc_pages": list(page_range),
        "entry_count": len(entries),
        "entries_with_page_numbers": with_pages,
        "entries": entries,
    }

    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"parsed {len(entries)} ToC entries ({with_pages} with page numbers) -> {args.out}")
    print("first 20 entries:")
    for e in entries[:20]:
        pg = f"p{e['page']}" if e["page"] is not None else "   "
        print(f"  L{e['level']}  {e['section_number']:<10} {pg:<6} {e['title']}")

    if with_pages == 0:
        print("\nNOTE: no page numbers parsed. Either the extractor dropped them (common with "
              "right-aligned ToC layouts) or the regex didn't match. Re-run with --toc-pages pointing "
              "at the right page(s), or we'll add a fallback parser.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
