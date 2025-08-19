from __future__ import annotations

"""Dedicated Milvus RAG agent for Google ADK.

This agent encapsulates the Milvus semantic search tool using Gemini embeddings.
It can be used standalone or as a sub-agent of the main chatbot agent.

Env knobs:
- GOOGLE_API_KEY or GEMINI_API_KEY: API key for google-genai
- GEMINI_EMBED_MODEL or ADK_EMBED_MODEL: embedding model id (default gemini-embedding-001)
- MILVUS_HOST/MILVUS_PORT: Milvus connection (defaults 127.0.0.1:19530)
- ADK_COLLECTION: Milvus collection name (default doc_md_multimodal)
- ADK_TOP_K: number of results (default 5)
"""

import os
from typing import Any, Dict, List

from google.adk.agents import Agent
import time


def _embed_query(text: str) -> List[float]:
    """Return a Gemini embedding vector for the query text with simple retry/backoff."""
    last_err = None
    for attempt in range(3):
        try:
            from google import genai  # type: ignore
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for embeddings")
            client = genai.Client(api_key=api_key)
            model_id = os.getenv("ADK_EMBED_MODEL") or os.getenv("GEMINI_EMBED_MODEL") or "gemini-embedding-001"
            resp = client.models.embed_content(model=model_id, contents=text)
            if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
                return list(resp.embedding.values)
            if hasattr(resp, "values"):
                return list(resp.values)
            if hasattr(resp, "embeddings") and resp.embeddings:
                return list(resp.embeddings[0].values)
            v = getattr(resp, "vector", None)
            if v is not None:
                return list(getattr(v, "values", v))
            raise RuntimeError("Unexpected embedding response from google-genai")
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Embedding error after retries: {last_err}")


def milvus_semantic_search(query: str) -> Dict[str, Any]:
    """Retrieve relevant passages from Milvus for a scientific question.

    Returns a dict with 'status' and 'results' or 'error_message'.
    """
    try:
        from pymilvus import connections, utility, Collection  # type: ignore
    except Exception as e:  # pragma: no cover
        return {"status": "error", "error_message": f"pymilvus not available: {e}"}

    try:
        qvec = _embed_query(query)
    except Exception as e:
        return {"status": "error", "error_message": f"Embedding error: {e}"}

    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    coll_name = os.getenv("ADK_COLLECTION") or "doc_md_multimodal"
    top_k_env = os.getenv("ADK_TOP_K", "5")
    try:
        _top_k = max(1, int(top_k_env))
    except Exception:
        _top_k = 5

    try:
        connections.connect(alias="default", host=host, port=port)
        if not utility.has_collection(coll_name):
            return {"status": "error", "error_message": f"Collection '{coll_name}' not found"}
        coll = Collection(coll_name)
        schema_fields = {f.name for f in coll.schema.fields}
        output_fields = [f for f in ["text", "section", "doi", "source", "chunk_index"] if f in schema_fields]
        coll.load()
        params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        res = coll.search(data=[qvec], anns_field="vector", param=params, limit=int(_top_k), output_fields=output_fields)
        out: List[Dict[str, Any]] = []
        for hit in res[0]:
            item = {
                "score": float(hit.distance),
                "text": hit.entity.get("text"),
            }
            if "section" in output_fields:
                item["section"] = hit.entity.get("section")
            if "doi" in output_fields:
                item["doi"] = hit.entity.get("doi")
            if "source" in output_fields:
                item["source"] = hit.entity.get("source")
            if "chunk_index" in output_fields:
                item["chunk_index"] = hit.entity.get("chunk_index")
            out.append(item)
        return {"status": "success", "results": out}
    except Exception as e:
        return {"status": "error", "error_message": f"Milvus search failed: {e}"}


_MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash")

milvus_rag_agent = Agent(
    name="milvus_rag_agent",
    model=_MODEL,
    description=(
        "Specialist agent that retrieves passages from Milvus (Gemini embeddings) and returns concise, cited answers."
    ),
    instruction=(
        "You are a retrieval specialist. Always call milvus_semantic_search with the user's question, "
        "then summarize the best passages into a concise answer with inline citations like "
        "[doi:... | Section]. If nothing relevant is found, say so briefly."
    ),
)

# Register tool with this agent
milvus_rag_agent.tools.append(milvus_semantic_search)
