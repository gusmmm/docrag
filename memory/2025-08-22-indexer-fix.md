# 2025-08-22: Indexer loop indentation fix and DB fallback

- Fixed indentation/scope errors at the tail of src/14_index.py inside the for-loop.
- Ensured topic-aware DB routing executes within the loop:
  - Determine target_db from input/input_pdf.json topic, else fallback to --db-name.
  - Use milvus_connect_db_with_fallback to detect named DB support; when unsupported, use suffixed collection paper_chunks__<topic> in default DB.
  - Insert meta into --meta-db-name, then chunks into the effective collection.
- Dry-run validated successfully (no syntax errors; chunk previews printed).
- Next: run full pipeline to populate collections and verify no more "database not found" errors.
