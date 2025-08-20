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


# step 2 - pdf -> markdown with images
- Run `uv run python main.py md-with-images` to convert each `input/pdf/<citation_key>.pdf` into a Markdown with referenced images under `output/papers/<key>/md_with_images/`.
- Images are stored alongside the Markdown for each paper.
- OCR defaults: engine auto-selects tesseract-cli if available; language defaults to `eng` (override with `DOCRAG_OCR_LANGS`).


# step 3 - prepare Markdown for RAG
- Run `uv run python main.py prepare-rag` to:
	- Extract references to `references.json` using `agents/references_agent.py`.
	- Strip the references section from the Markdown (based on `src/02_1_strip_refs.py`).
	- Clean boilerplate (authors, affiliations, disclosures, etc.) using `agents/md_clean_agent.py`.
	- Save the result as `<original-stem>-RAG.md` in the same folder.
- Idempotent: if a `-RAG.md` exists for a given input Markdown, the step is skipped for that file.


# step 4 - add metadata to RAG Markdown
- Run `uv run python main.py add-metadata` to prepend a YAML front matter to each `*-RAG.md` using entries in `input/input_pdf.json`.
- Fields include: title, authors, doi, citation_key, journal, volume, issue, pages, issued (ISO), url, and source provenance.

# step 5 - index into Milvus (Gemini embeddings)
- Run `docker compose up -d` to ensure Milvus/Attu are up.
- Run `uv run python main.py index` to:
	- Use DB `journal_papers`.
	- Ensure collections: `papers_meta` (one row per paper) and `paper_chunks` (chunked content + vectors).
	- Skip papers that already exist by DOI or citation_key in `papers_meta`.
	- Chunk by headings/paragraphs and embed with Gemini (`gemini-embedding-001`).
	- Insert chunks referencing `paper_id`, `doi`, and `citation_key`.

