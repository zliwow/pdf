import argparse
import json
from pathlib import Path
from statistics import mean, median


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize pdf.json: page/section counts, char distribution, previews.")
    p.add_argument("json_path", type=Path, nargs="?", default=Path("pdf.json"))
    p.add_argument("-p", "--page", type=int, default=None,
                   help="print full text of a specific section/page (1-indexed in print, 0-indexed in json)")
    p.add_argument("--previews", type=int, default=5, help="number of section previews to print (default: 5)")
    p.add_argument("--low-threshold", type=int, default=100,
                   help="flag sections with fewer than this many chars (default: 100)")
    args = p.parse_args()

    data = json.loads(args.json_path.read_text())
    sections = data["sections"]

    print(f"source: {data.get('source')}")
    print(f"page_count: {data.get('page_count')}")
    print(f"has_embedded_toc: {data.get('has_embedded_toc')}")
    print(f"section_count: {len(sections)}")
    print()

    if args.page is not None:
        idx = args.page
        if 0 <= idx < len(sections):
            s = sections[idx]
            print(f"=== section index {idx} (pages {s['start_page'] + 1}-{s['end_page'] + 1}) ===")
            print(f"title: {s.get('title')}")
            print(f"chars: {s.get('char_count')}")
            print()
            print(s.get("text", ""))
        else:
            print(f"index out of range: {idx} (have 0..{len(sections) - 1})")
        return 0

    char_counts = [s["char_count"] for s in sections]
    if char_counts:
        print("char count distribution:")
        print(f"  min:    {min(char_counts)}")
        print(f"  median: {int(median(char_counts))}")
        print(f"  mean:   {int(mean(char_counts))}")
        print(f"  max:    {max(char_counts)}")
        print(f"  total:  {sum(char_counts)}")
        print()

        low = [s for s in sections if s["char_count"] < args.low_threshold]
        if low:
            print(f"sections under {args.low_threshold} chars ({len(low)} total — likely blank/cover/ToC pages):")
            for s in low[:20]:
                print(f"  [{s['index']}] p{s['start_page'] + 1}  {s['char_count']} chars  {s.get('title', '')[:60]}")
            if len(low) > 20:
                print(f"  ... +{len(low) - 20} more")
            print()

    print(f"first {args.previews} section previews:")
    for s in sections[:args.previews]:
        print(f"--- [{s['index']}] p{s['start_page'] + 1}-{s['end_page'] + 1}  ({s['char_count']} chars)  {s.get('title', '')}")
        print(s.get("preview", "")[:400])
        print()

    print("to dump a full section:  python inspect_pdf.py -p <index>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
