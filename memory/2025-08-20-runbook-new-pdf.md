# Runbook: Process a new PDF into Milvus (Docling + Gemini)

Date: 2025-08-20

Scope
- Where to place a new PDF
- Exact commands to generate `-RAG.md` and index into Milvus
- Key behaviors (dedup by doi/citation_key; collections; env)

Where to save the PDF
- Place files in `input/pdf/` (or `input/` root — orchestrator moves them into `input/pdf/`).

Commands (uv)
1) Start Milvus stack
   - `docker compose up -d`

2) Install deps
   - `uv sync`

3) Build/refresh input registry and prepare per-paper folders
   - `uv run python -m input.input`
   - `uv run python src/10_prepare_output_dirs.py`

4) Create Markdown with image refs per paper (Docling)
   - `uv run python src/11_create_md_with_images.py`

5) Create cleaned RAG Markdown (extract/strip refs, clean)
   - `uv run python src/12_remove_refs_clean.py`
   - Optional single-file: `--file output/papers/<key>/md_with_images/<key>-with-image-refs.md`

6) Add YAML metadata to each `-RAG.md` from `input/input_pdf.json`
   - `uv run python src/13_add_metada.py`
   - Optional single-file: `--file output/papers/<key>/md_with_images/<key>-RAG.md`

7) Index into Milvus with Gemini embeddings
   - Export API key: `export GEMINI_API_KEY=...` (or `GOOGLE_API_KEY`)
   - `uv run python src/14_index.py \
       --db-name journal_papers \
       --meta-collection papers_meta \
       --collection paper_chunks \
       --embed-model gemini-embedding-001 \
       --embed-batch 64 \
       --insert-batch 256 \
       --show 3`
   - Optional: `--dry-run --show 3` to preview chunks only
   - Single file: `--file output/papers/<key>/md_with_images/<key>-RAG.md`

Outcomes
- Input registry: `input/input_pdf.json`
- Per-paper dir: `output/papers/<key>/`
- RAG Markdown: `output/papers/<key>/md_with_images/<key>-RAG.md`
- Milvus DB: journal_papers with collections `papers_meta` and `paper_chunks`
  - Dedup: skip if `doi` or `citation_key` already present in `papers_meta`

Notes
- Attu UI at http://localhost:8000 (connect to localhost:19530)
- Embedding model: `gemini-embedding-001`
- Use `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- All commands are uv-based as per project guidelines
