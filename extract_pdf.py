import argparse
import json
import sys
from pathlib import Path

import fitz  # pymupdf


def get_toc(doc: fitz.Document) -> list[dict]:
    raw = doc.get_toc(simple=False)
    entries = []
    for i, item in enumerate(raw):
        level, title, page = item[0], item[1], item[2]
        entries.append({
            "index": i,
            "level": level,
            "title": title.strip(),
            "start_page": page - 1,  # pymupdf is 0-indexed internally
        })
    for i, e in enumerate(entries):
        nxt = entries[i + 1]["start_page"] if i + 1 < len(entries) else doc.page_count
        e["end_page"] = max(e["start_page"], nxt - 1)
    return entries


def page_text(doc: fitz.Document, start: int, end: int) -> str:
    chunks = []
    for pno in range(start, end + 1):
        chunks.append(doc[pno].get_text("text"))
    return "\n".join(chunks).strip()


def extract(pdf_path: Path, out_path: Path, preview_chars: int,
            start_page: int | None, end_page: int | None) -> None:
    doc = fitz.open(pdf_path)
    toc = get_toc(doc)

    # Normalize page range (1-indexed input -> 0-indexed internal). Inclusive on both ends.
    lo = (start_page - 1) if start_page else 0
    hi = (end_page - 1) if end_page else (doc.page_count - 1)
    lo = max(0, lo)
    hi = min(doc.page_count - 1, hi)

    sections = []
    if toc:
        for e in toc:
            if e["end_page"] < lo or e["start_page"] > hi:
                continue
            s_lo = max(e["start_page"], lo)
            s_hi = min(e["end_page"], hi)
            text = page_text(doc, s_lo, s_hi)
            sections.append({
                **e,
                "start_page": s_lo,
                "end_page": s_hi,
                "char_count": len(text),
                "text": text,
                "preview": text[:preview_chars],
            })
    else:
        # Fallback: no bookmarked ToC. Dump per-page so you can still eyeball it,
        # and we can parse a printed ToC in a later pass.
        print("WARNING: no embedded ToC/bookmarks found. Dumping per-page instead.", file=sys.stderr)
        for pno in range(lo, hi + 1):
            text = doc[pno].get_text("text").strip()
            sections.append({
                "index": pno,
                "level": 0,
                "title": f"[page {pno + 1}]",
                "start_page": pno,
                "end_page": pno,
                "char_count": len(text),
                "text": text,
                "preview": text[:preview_chars],
            })

    payload = {
        "source": str(pdf_path),
        "page_count": doc.page_count,
        "extracted_range": [lo + 1, hi + 1],  # 1-indexed, human-readable
        "has_embedded_toc": bool(toc),
        "section_count": len(sections),
        "sections": sections,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {len(sections)} sections from {doc.page_count} pages -> {out_path}")
    if toc:
        print("first 10 ToC entries:")
        for e in toc[:10]:
            print(f"  L{e['level']} p{e['start_page'] + 1}-{e['end_page'] + 1}  {e['title']}")


def main() -> int:
    p = argparse.ArgumentParser(description="Extract PDF ToC + section text to JSON for inspection.")
    p.add_argument("pdf", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("pdf.json"))
    p.add_argument("--preview", type=int, default=200, help="preview chars per section (default: 200)")
    p.add_argument("--start-page", type=int, default=None,
                   help="1-indexed first page to include (skip cover/history/ToC). Inclusive.")
    p.add_argument("--end-page", type=int, default=None,
                   help="1-indexed last page to include (skip glossary/legal). Inclusive.")
    args = p.parse_args()

    extract(args.pdf, args.out, args.preview, args.start_page, args.end_page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
