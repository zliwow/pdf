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


def extract(pdf_path: Path, out_path: Path, preview_chars: int) -> None:
    doc = fitz.open(pdf_path)
    toc = get_toc(doc)

    sections = []
    if toc:
        for e in toc:
            text = page_text(doc, e["start_page"], e["end_page"])
            sections.append({
                **e,
                "char_count": len(text),
                "text": text,
                "preview": text[:preview_chars],
            })
    else:
        # Fallback: no bookmarked ToC. Dump per-page so you can still eyeball it,
        # and we can parse a printed ToC in a later pass.
        print("WARNING: no embedded ToC/bookmarks found. Dumping per-page instead.", file=sys.stderr)
        for pno in range(doc.page_count):
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
    args = p.parse_args()

    extract(args.pdf, args.out, args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
