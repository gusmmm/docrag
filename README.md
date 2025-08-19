# docRAG (Gemini + Milvus + Docling)

End-to-end pipeline to turn papers into a multimodal RAG index using:
- Docling (PDF to Markdown with image refs)
- Google Gemini (embeddings + agents)
- Milvus (vector DB) with Attu UI
- uv for Python env and scripts

## Workflow
1) PDF -> Markdown with image references (Docling)
2) Extract citation metadata with a Gemini agent (optional)
3) Merge citation metadata into the Markdown as YAML front matter
4) Extract references/bibliography to structured JSON and exclude them from embedding
5) Chunk Markdown semantically (headings -> paragraphs), keep image refs
6) Embed chunks with Gemini and index in Milvus
7) Query the Milvus collection (CLI or Attu UI)

## Components
- docker-compose.yml: Milvus stack + Attu UI
- src/02_merger.py (aka 02_indexing.py): merge citation JSON into MD (YAML front matter)
- agents/references_agent.py: extract references -> references.json
- src/03_indexing.py: chunk + embed (Gemini) + insert into Milvus
- RAG_milvus_demo/demo.py: query-only CLI using Gemini

## Quick start
Prereqs: Docker, uv, Gemini API key exported as GEMINI_API_KEY (or GOOGLE_API_KEY).

1) Start Milvus
```bash
docker compose up -d
```

2) Install deps
```bash
uv sync
```

3) Merge citation metadata into Markdown
```bash
uv run python src/02_merger.py \
  output/md_with_images/jama_summers_2025-with-image-refs.md \
  --citations output/citations/jama_summers_2025-with-image-refs-genai.json
# Output -> output/md_with_images/jama_summers_2025-with-image-refs-merged.md
```

4) Extract references/bibliography (structured JSON) to exclude from embedding
```bash
uv run python agents/references_agent.py \
  output/md_with_images/jama_summers_2025-with-image-refs-merged.md
# Output -> output/md_with_images/references.json
```

5) Preview chunking (no embeddings)
```bash
uv run python src/03_indexing.py --dry-run --show 3
```

6) Index into Milvus (Gemini embeddings)
```bash
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY
uv run python src/03_indexing.py \
  --input output/md_with_images/jama_summers_2025-with-image-refs-merged.md \
  --collection doc_md_multimodal \
  --embed-model gemini-embedding-001 \
  --embed-batch 64 \
  --insert-batch 256
```

7) Query
```bash
uv run python RAG_milvus_demo/demo.py \
  --collection doc_md_multimodal \
  --query-only \
  --query "What was the primary outcome?" \
  --show 3
```

8) Inspect in Attu
- Open http://localhost:8000
- Connect to milvus:19530 (or localhost:19530)
- Explore collection `doc_md_multimodal`

## Milvus schema per chunk
- doi (from YAML front matter)
- source (file path)
- section (hierarchical heading path)
- chunk_index (position)
- hash (sha256 of text)
- image_refs (pipe-separated list)
- text (chunk content)
- vector (FLOAT_VECTOR; COSINE; AUTOINDEX)

## Why Markdown (with YAML)
- Human-readable, easy to audit
- Preserves image links for multimodal use
- YAML front matter carries bibliographic context

## Implemented
- Persistent Milvus + Attu via docker-compose
- Merge citation JSON -> YAML in Markdown
- References extractor (Gemini structured output) -> references.json
- Semantic chunking (headings/paragraphs) with safe splitting for VARCHAR limits
- Gemini embeddings and Milvus indexing
- Query-only CLI that embeds queries with Gemini

## Missing / Next
- Exclude references during indexing automatically (use references.json or cut at "## References")
- Dedup/idempotent re-index (hash pre-check + pagination)
- Add chunk-level metadata (page numbers, figure ids) if available
- Image embeddings / multimodal fusion (optional)
- Reranking + answer generation with Gemini LLM
- Tests/CI and docling PDF automation script

## Troubleshooting
- API key: export GEMINI_API_KEY or GOOGLE_API_KEY
- Milvus up: docker compose up -d; check Attu at http://localhost:8000
- No hits: confirm collection name and that indexing completed
- Long text: indexer splits long paragraphs to fit Milvus VARCHAR(8192)
