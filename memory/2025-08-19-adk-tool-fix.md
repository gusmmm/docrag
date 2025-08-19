# 2025-08-19 - ADK tool schema fix for milvus_semantic_search

- Changed chatbot/agent.py: simplified milvus_semantic_search signature to `def milvus_semantic_search(query: str) -> Dict[str, Any]`.
- Reason: ADK automatic function calling failed to parse `collection: str | None = None` with defaults/unions.
- Behavior: `top_k` and `collection` now read from env:
  - ADK_TOP_K (default 5)
  - ADK_COLLECTION (default doc_md_multimodal_json)
- Verified: File imports without errors; ADK web startup reached application initialization. Port 8010 was busy; run with another port if needed.
- Next: Optionally add a separate config tool to adjust `top_k`/`collection` at runtime.
