import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize excel.json: column list, row count, value counts for a chosen column.")
    p.add_argument("json_path", type=Path, nargs="?", default=Path("excel.json"))
    p.add_argument("-c", "--column", default=None,
                   help="column name to count values for (e.g. 'Item Type'). "
                        "If omitted, prints column list and tries common type columns.")
    p.add_argument("--top", type=int, default=50, help="show top-N values (default: 50)")
    args = p.parse_args()

    data = json.loads(args.json_path.read_text())
    rows = data["rows"]
    cols = data["columns"]

    print(f"source: {data.get('source')}")
    print(f"rows: {len(rows)}")
    print(f"columns ({len(cols)}):")
    for c in cols:
        print(f"  - {c!r}")
    print()

    candidates = [args.column] if args.column else [
        c for c in cols if c and any(k in c.lower() for k in ("type", "category", "kind"))
    ]

    if not candidates:
        print("no column specified and no 'type'-like column found. "
              "Re-run with -c '<Column Name>' to see value counts.")
        return 0

    for col in candidates:
        if col not in cols:
            print(f"column not found: {col!r}")
            continue
        counts = Counter((r.get(col) or "").strip() for r in rows)
        print(f"value counts for {col!r} (top {args.top}):")
        for val, n in counts.most_common(args.top):
            label = val if val else "<empty>"
            print(f"  {n:>6}  {label}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
