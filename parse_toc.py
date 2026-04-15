import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # pymupdf


# A ToC row, visually reconstructed. The number may be alone at the start,
# title in the middle, page number at the end. Dot leaders become spaces.
TOC_LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<num>\d+(?:\.\d+)*)           # section number
    \s+
    (?P<title>.+?)                   # title
    \s+
    (?P<page>\d{1,4})                # trailing page number (required here)
    \s*$
    """,
    re.VERBOSE,
)

# Fallback for rows without a page number (still useful for section map).
TOC_LINE_NO_PAGE_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)*)\s+(?P<title>.+?)\s*$"
)


def reconstruct_rows(page: fitz.Page, y_tol: float = 2.5) -> list[str]:
    """Rebuild visual rows from word positions so 3-column ToC layouts don't
    get flattened into separate lines for number / title / page."""
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word)
    if not words:
        return []
    # Bucket by rounded y-midpoint so words on the same visual row group together.
    rows: dict[int, list[tuple[float, str]]] = {}
    for x0, y0, x1, y1, text, *_ in words:
        if not text.strip():
            continue
        y_mid = (y0 + y1) / 2
        key = int(round(y_mid / y_tol))
        rows.setdefault(key, []).append((x0, text))
    out: list[str] = []
    for key in sorted(rows):
        row_words = sorted(rows[key], key=lambda w: w[0])
        # Collapse runs of dots (leaders) down to a single space.
        joined = " ".join(w for _, w in row_words)
        joined = re.sub(r"[\.\s]*\.{2,}[\.\s]*", " ", joined)
        joined = re.sub(r"\s+", " ", joined).strip()
        if joined:
            out.append(joined)
    return out


def parse_toc_pages(pdf_path: Path, page_range: tuple[int, int]) -> list[dict]:
    doc = fitz.open(pdf_path)
    lo = max(0, page_range[0] - 1)
    hi = min(doc.page_count - 1, page_range[1] - 1)

    entries: list[dict] = []
    for pno in range(lo, hi + 1):
        page = doc[pno]
        for line in reconstruct_rows(page):
            m = TOC_LINE_RE.match(line)
            page_num: int | None = None
            if m:
                num = m.group("num")
                title = m.group("title").strip().strip(".").strip()
                page_num = int(m.group("page"))
            else:
                m2 = TOC_LINE_NO_PAGE_RE.match(line)
                if not m2:
                    continue
                num = m2.group("num")
                title = m2.group("title").strip().strip(".").strip()
                # Skip garbage like "1 ." where title collapsed to nothing.
                if not title or title.isdigit():
                    continue
            entries.append({
                "section_number": num,
                "title": title,
                "page": page_num,
                "level": num.count(".") + 1,
                "source_toc_page": pno + 1,
            })

    # De-dupe by section number; prefer the entry that has a page.
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
