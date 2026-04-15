import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def extract(xlsx_path: Path, sheet: str | int, out_path: Path) -> None:
    df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
    df = df.fillna("")
    rows = df.to_dict(orient="records")

    payload = {
        "source": str(xlsx_path),
        "sheet": sheet,
        "columns": list(df.columns),
        "row_count": len(rows),
        "rows": rows,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} rows, {len(df.columns)} columns -> {out_path}")
    print(f"columns: {list(df.columns)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Dump a Jama Excel export to JSON for inspection.")
    p.add_argument("xlsx", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("excel.json"))
    p.add_argument("-s", "--sheet", default=0, help="sheet name or 0-based index (default: 0)")
    args = p.parse_args()

    sheet: str | int
    try:
        sheet = int(args.sheet)
    except (TypeError, ValueError):
        sheet = args.sheet

    extract(args.xlsx, sheet, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
