# 2025-08-19: Merge citations into Markdown script

- Added `src/02_indexing.py` to merge citation JSON into Markdown-with-image-refs.
- Produces a merged Markdown with YAML front matter: `<md-stem>-merged.md` in the same folder.
- Default example:
  - MD: `output/md_with_images/jama_summers_2025-with-image-refs.md`
  - Citation JSON: `output/citations/jama_summers_2025-with-image-refs-genai.json`
  - Output: `output/md_with_images/jama_summers_2025-with-image-refs-merged.md`
- Rationale: keep Markdown as the primary ingest format with YAML front matter for metadata; images remain linked for multi-modal RAG.
- Next: optional JSONL export per chunk if we need chunk-level metadata before indexing.
