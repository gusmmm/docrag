# workflow pipeline guidelines
- implement in a stepwise fashion according to the instructions in this file

# step 1 - input pdfs
- Put PDFs under `input/pdf/`. If you drop PDFs directly into `input/`, the orchestrator will move them to `input/pdf/` automatically.
- Run `uv run python -m input.input` to:
	- extract Title/DOI without LLMs,
	- fetch full bibliographic metadata (CSL-JSON) from Crossref when DOI present,
	- generate a Better BibTeX–style citation key,
	- deduplicate by citation_key → DOI → title,
	- rename PDFs to `citation_key.pdf` (in `input/pdf/`).
- The registry (database) is `input/input_pdf.json` with fields: `citation_key`, `title`, `doi`, `csl`.

Notes:
- Set `CROSSREF_UA` for polite Crossref requests.
- OCR is optional and requires `pytesseract` + `Pillow` (use the `--ocr` flag with `check_pdf.py` when needed).

