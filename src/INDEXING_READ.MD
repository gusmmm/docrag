# Indexing to Milvus with Gemini (Markdown + YAML + Images)

This guide shows how to merge citation metadata into Markdown, chunk semantically, embed with Google Gemini, and index into Milvus for RAG. It uses uv for Python execution and the included docker-compose stack for Milvus.

## Prerequisites
- Docker + Docker Compose running locally
- uv installed (https://docs.astral.sh/uv/)
- A Google Gemini API key exported as `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- Milvus stack from this repo’s `docker-compose.yml`

## 1) Start Milvus stack
```bash
docker compose up -d
```

Milvus gRPC will be on `localhost:19530` and Attu UI on `http://localhost:8000`.

## 2) (One-time) install deps with uv
```bash
uv sync
```

## 3) Export your Gemini API key
```bash
export GEMINI_API_KEY="<your_api_key>"
```

## 4) Merge citation JSON into Markdown (with image refs)
The script reads your citation JSON and prepends YAML front matter to the Markdown, writing a `-merged.md` file.
```bash
uv run python src/02_indexing.py \
	--md output/md_with_images/jama_summers_2025-with-image-refs.md \
	--citations output/citations/jama_summers_2025-with-image-refs-genai.json
```
Output: `output/md_with_images/jama_summers_2025-with-image-refs-merged.md`

## 5) Dry-run indexing (preview chunks only)
This validates parsing and chunking without requiring embeddings or Milvus inserts.
```bash
uv run python src/03_indexing.py --dry-run --show 3
```

## 6) Index into Milvus with Gemini embeddings
```bash
uv run python src/03_indexing.py \
	--input output/md_with_images/jama_summers_2025-with-image-refs-merged.md \
	--collection doc_md_multimodal \
	--embed-model gemini-embedding-001 \
	--embed-batch 64 \
	--insert-batch 256
```

Flags you may find useful:
- `--drop-before` drop and recreate the Milvus collection
- `--milvus-host` and `--milvus-port` if not local
- `--show N` preview first N chunks before embedding

## 7) Verify in Attu UI
Open `http://localhost:8000`, connect to `milvus:19530` (or `localhost:19530`), and inspect the `doc_md_multimodal` collection. You should see scalar fields (doi, source, section, chunk_index, hash, image_refs, text) and the `vector` field.

## 8) Optional: quick querying
You can use the demo query script to embed a question and search the existing collection:
```bash
uv run python RAG_milvus_demo/demo.py \
	--collection doc_md_multimodal \
	--query-only \
	--query "What was the primary outcome?" \
	--show 3
```

## What this pipeline stores
- Chunk text (semantic: headings → paragraphs; safe splitting for long text)
- Gemini vector embedding
- DOI from YAML front matter for provenance
- Source file path
- Hierarchical section string
- Chunk index (position within source)
- Image references linked in the chunk

## Tips & Troubleshooting
- Connection refused: ensure `docker compose up -d` and Milvus container is healthy.
- Missing API key: export `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
- Attu shows no data: refresh connection, confirm collection name `doc_md_multimodal`.
- Very large paragraphs: the indexer splits long text to fit Milvus VARCHAR limits.

## Why Markdown (with YAML) for ingest?
- Human-readable, easy to audit
- Preserves image links for multimodal pipelines
- YAML front matter provides clean provenance (DOI, journal, title, etc.)

