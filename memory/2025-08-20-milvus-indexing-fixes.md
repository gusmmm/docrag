# 2025-08-20 Milvus indexing fixes

- Fixed invalid vector dimension (1) in `papers_meta` dummy vector field; set to dim=2 and insert `[0.0, 0.0]`.
- More robust DB handling: try to create `journal_papers` using `db.list_databases()`/`list_database()` and `create_database()`. If unsupported or unavailable, fall back to default DB with a warning.
- Avoided query-before-load errors: load collections before queries and prefer primary keys from insert result.
- Ensured meta collection can load on strict servers by creating `AUTOINDEX` on `vector_meta`.
- Verified end-to-end: first run indexed both papers; subsequent run deduped with `[skip] Already indexed`.
