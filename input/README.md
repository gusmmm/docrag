# Input pipeline

This folder manages ingest of PDFs into a simple JSON registry and stable filenames.

What it does
- Extracts title and DOI from PDFs using a no-LLM pipeline (`check_pdf.py`).
- If a DOI is found, fetches full bibliographic metadata (CSL-JSON) from Crossref.
- Generates a Better BibTeX–style citation key (authorYearShortTitle).
- Renames PDFs to `citation_key.pdf` under `input/pdf/`.
- Deduplicates primarily by `citation_key`, then `doi`, then legacy `title`.
- Persists to `input/input_pdf.json` with fields: `citation_key`, `title`, `doi`, `csl`.

Layout
- `pdf/` – put your PDFs here (the orchestrator will also move stray PDFs into this folder).
- `check_pdf.py` – no-LLM extractor + CSL fetch + citation key generator.
- `input.py` – orchestrator: scan, extract, rename, dedup, persist.
- `input_pdf.json` – the registry/database for all PDFs in this project.

Quick start
1. Drop one or more PDFs into `input/pdf/`.
2. Run the orchestrator:
   - via uv: `uv run python -m input.input`
   - or: `uv run python input/input.py`
3. Inspect `input/input_pdf.json` and renamed files under `input/pdf/`.

Configuration
- Crossref polite headers: set `CROSSREF_UA` (e.g., `docrag/0.1 (+https://example.org; mailto:you@example.org)`).
- OCR fallback for scanned PDFs requires `pytesseract` and `Pillow` (optional; enabled via `--ocr` when using `check_pdf.py` CLI).

Notes
- Citation key collisions are handled by unique filenames (`... (1).pdf`). If you prefer letter-suffixed keys (`a`, `b`, ...), we can add that later.
- Existing entries are merged in place to add missing `doi`, `title`, or `csl`.
