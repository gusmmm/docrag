# docRAG (Docling + Gemini + Milvus)

End-to-end pipeline to turn scientific PDFs into a multimodal RAG index using:
- Docling (PDF → Markdown with image references)
- Google Gemini (embeddings + agents)
- Milvus (vector DB) with Attu UI
- uv for Python env and scripts

This README reflects the current pipeline scripts under `src/10..14_*.py` and the input orchestrator in `input/`.

## TL;DR — Process a new PDF
Prereqs: Docker, uv, and a Gemini API key exported as GEMINI_API_KEY (or GOOGLE_API_KEY).

1) Start Milvus stack
```bash
docker compose up -d
```

2) Install deps
```bash
uv sync
```

3) Drop your PDF(s)
- Put files into `input/pdf/` (or just `input/`; the orchestrator will move them).

4) Build/refresh the input registry and prepare per-paper folders
```bash
uv run python -m input.input              # extract title/doi, fetch CSL, create citation_key, rename PDFs, update input/input_pdf.json
uv run python src/10_prepare_output_dirs.py
```

5) Create Markdown with image references (Docling)
```bash
uv run python src/11_create_md_with_images.py
```

6) Create cleaned RAG Markdown (-RAG.md): extract/strip references and clean text
```bash
uv run python src/12_remove_refs_clean.py           # process all
# or a single file: --file output/papers/<key>/md_with_images/<key>-with-image-refs.md
```

7) Add YAML metadata to each -RAG.md from `input/input_pdf.json`
```bash
uv run python src/13_add_metada.py                  # process all -RAG.md
# or a single file: --file output/papers/<key>/md_with_images/<key>-RAG.md
```

8) Index into Milvus (Gemini embeddings)
```bash
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
uv run python src/14_index.py \
  --db-name journal_papers \
  --meta-collection papers_meta \
  --collection paper_chunks \
  --embed-model gemini-embedding-001 \
  --embed-batch 64 \
  --insert-batch 256 \
  --show 3
```

9) Query or inspect
```bash
uv run python RAG_milvus_demo/demo.py --collection paper_chunks --query-only --query "What was the primary outcome?" --show 3
# Attu UI: http://localhost:8000 (connect to localhost:19530)
```

Notes:
- Use `uv run python src/14_index.py --dry-run --show 3` to preview chunking without embeddings/inserts.
- To index a single paper, pass `--file output/papers/<key>/md_with_images/<key>-RAG.md` to step 8.

## What lands where
- Input PDFs: `input/pdf/<citation_key>.pdf`
- Registry: `input/input_pdf.json` (records with citation_key, title, doi, csl)
- Per-paper folder: `output/papers/<citation_key>/`
- Markdown with image refs: `output/papers/<key>/md_with_images/<key>-with-image-refs.md`
- Cleaned RAG Markdown: `output/papers/<key>/md_with_images/<key>-RAG.md`

## Milvus layout (DB: journal_papers)
- papers_meta (1 row per paper)
  - id (PK auto), doi, citation_key, title, journal, issued, url, source_path
- paper_chunks (RAG content)
  - id (PK auto), paper_id, doi, citation_key, section, chunk_index, hash, image_refs, text, vector

Collections use COSINE metric with AUTOINDEX. Re-indexing is idempotent at the paper level: if a paper exists in `papers_meta` (by DOI or citation_key), its chunks are skipped.

## Why Markdown (with YAML)
- Human-readable and easy to audit
- Preserves image links for multimodal use
- YAML carries bibliographic context (DOI, authors, journal, etc.)

## Troubleshooting
- API key: export `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- Milvus: `docker compose up -d`; Attu at http://localhost:8000
- No rows: ensure steps 6–8 completed and collection names match
- Long text: the indexer splits paragraphs to fit Milvus VARCHAR limits

## Related scripts
- `src/10_prepare_output_dirs.py` — makes `output/papers/<key>/`
- `src/11_create_md_with_images.py` — Docling PDF→MD with images
- `src/12_remove_refs_clean.py` — references extraction and cleaning → `-RAG.md`
- `src/13_add_metada.py` — YAML metadata from `input/input_pdf.json`
- `src/14_index.py` — chunk + embed + insert into Milvus (`journal_papers` DB)
- `RAG_milvus_demo/demo.py` — quick query CLI
