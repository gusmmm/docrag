# 2025-08-20: Add metadata to -RAG.md from registry

- Implemented `src/13_add_metada.py` to prepend YAML front matter to each `*-RAG.md` using `input/input_pdf.json`.
- Orchestrator `main.py` got a new `add-metadata` subcommand.
- `PIPELINE.MD` updated with Step 4 details.

YAML fields: title, authors (bounded), doi, citation_key, journal, volume, issue, pages, issued (ISO), url, source_registry.
