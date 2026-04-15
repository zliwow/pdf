# pdf

Step 1: extract Excel and PDF into inspectable JSON before any comparison logic.

## setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## run

```
python extract_excel.py path/to/jama_export.xlsx -o excel.json
python extract_pdf.py   path/to/spec.pdf          -o pdf.json
```

Then eyeball `excel.json` and `pdf.json` — especially:
- Excel: which columns matter (likely `ID`, `Name`, `Description`).
- PDF: did `has_embedded_toc` come back true? If false we need to parse the printed ToC.
- PDF: do section boundaries look roughly right? Check `preview` fields.
