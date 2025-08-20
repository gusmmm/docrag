from __future__ import annotations

"""Dedicated Milvus RAG agent for Google ADK.

This agent encapsulates the Milvus semantic search tool using Gemini embeddings.
It can be used standalone or as a sub-agent of the main chatbot agent.

Env knobs:
- GOOGLE_API_KEY or GEMINI_API_KEY: API key for google-genai
- GEMINI_EMBED_MODEL or ADK_EMBED_MODEL: embedding model id (default gemini-embedding-001)
- MILVUS_HOST/MILVUS_PORT: Milvus connection (defaults 127.0.0.1:19530)
- ADK_COLLECTION: Milvus collection name (default paper_chunks)
- ADK_TOP_K: number of results (default 5)
"""

import os
from typing import Any, Dict, List, Optional

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
    coll_name = os.getenv("ADK_COLLECTION") or "paper_chunks"
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
        # Align with our chunk schema: includes citation_key; no 'source' field
        output_fields = [f for f in ["text", "section", "doi", "citation_key", "chunk_index"] if f in schema_fields]
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
            if "citation_key" in output_fields:
                item["citation_key"] = hit.entity.get("citation_key")
            if "chunk_index" in output_fields:
                item["chunk_index"] = hit.entity.get("chunk_index")
            out.append(item)
        return {"status": "success", "results": out}
    except Exception as e:
        return {"status": "error", "error_message": f"Milvus search failed: {e}"}


def milvus_meta_info(question: Optional[str] = None) -> Dict[str, Any]:
    """Query papers metadata from the papers_meta collection.

    Supports:
    - Count total papers
    - List papers (citation_key, doi, title, journal, issued)
    - Filter by citation_key or doi if present in the question
    """
    try:
        from pymilvus import connections, utility, Collection  # type: ignore
    except Exception as e:  # pragma: no cover
        return {"status": "error", "error_message": f"pymilvus not available: {e}"}

    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    meta_name = os.getenv("ADK_META_COLLECTION") or "papers_meta"

    try:
        connections.connect(alias="default", host=host, port=port)
        if not utility.has_collection(meta_name):
            return {"status": "error", "error_message": f"Collection '{meta_name}' not found"}
        meta = Collection(meta_name)
        # Load is needed for some ops and to avoid not-loaded errors on strict servers
        try:
            meta.load()
        except Exception:
            pass
        fields = {f.name for f in meta.schema.fields}
        select = [f for f in ["citation_key", "doi", "title", "journal", "issued"] if f in fields]
        # Detect intent
        q = (question or "").lower()
        want_count = any(k in q for k in ["how many", "count", "number of"])
        # Extract simple filters
        flt_expr = ""
        # Very basic term grabs; users can type citation_key:xxx or doi:10.
        if "citation_key" in q:
            try:
                val = q.split("citation_key", 1)[1].strip().lstrip(":= ")
                val = val.split()[0]
                flt_expr = f'citation_key == "{val}"'
            except Exception:
                pass
        if not flt_expr and "doi" in q:
            try:
                # naive DOI token
                token = next((t for t in q.split() if t.startswith("10.")), "")
                if token:
                    flt_expr = f'doi == "{token}"'
            except Exception:
                pass

        out: Dict[str, Any] = {"status": "success"}
        # Count
        if want_count and not flt_expr:
            try:
                total = meta.num_entities
                out["count"] = int(total)
                return out
            except Exception:
                # fallback via query count
                rows = meta.query(expr="id >= 0", output_fields=["id"], limit=1_000_000)
                out["count"] = len(rows)
                return out

        # List or filtered fetch
        if flt_expr:
            rows = meta.query(expr=flt_expr, output_fields=select, limit=100)
            out["results"] = rows
            return out
        # default list
        rows = meta.query(expr="id >= 0", output_fields=select, limit=50)
        out["results"] = rows
        out["note"] = "Showing up to 50; refine with citation_key:... or doi:..."
        return out
    except Exception as e:
        return {"status": "error", "error_message": f"Milvus meta query failed: {e}"}


_MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash")

milvus_rag_agent = Agent(
    name="milvus_rag_agent",
    model=_MODEL,
    description=(
    "Specialist agent that retrieves passages from Milvus (Gemini embeddings) and returns concise, cited answers."
    ),
    instruction=(
    "You are a retrieval specialist sub-agent for a scientific RAG over Milvus.\n"
        "- For scientific journal papers' metadata, use the papers_meta database (collection).\n"
        "- For scientific paper content, use the paper_chunks database (collection).\n"
        "- When answering questions about the database itself (e.g., which papers it contains, how many, titles), use the papers_meta database.\n"
        "- When referencing sources, include only citation_key and doi in citations.\n\n"
    "Process: If the user asks about the database or metadata, call milvus_meta_info first. "
    "If the user asks content questions, call milvus_semantic_search with the user's question, then summarize the best passages. "
    "Cite as [citation_key | doi]. If nothing relevant is found, say so briefly.\n\n"
    "You are a SUB-AGENT. Do not address the user directly. When you finish, RETURN your final answer back to the root orchestrator 'doc_rag_chatbot' so it can present the response."
    ),
)

# Register tools with this agent
milvus_rag_agent.tools.append(milvus_semantic_search)
milvus_rag_agent.tools.append(milvus_meta_info)
