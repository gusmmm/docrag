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
from typing import Any, Dict, List, Optional, Tuple

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
    # Topic-aware databases: ADK_DB_LIST=journal_papers,icu,neuro ...
    # If provided, iterate and merge results. If not, search default connection only.
    db_list_env = os.getenv("ADK_DB_LIST", "").strip()
    db_names = [d.strip() for d in db_list_env.split(",") if d.strip()] or [os.getenv("ADK_DB_NAME", "journal_papers")]
    top_k_env = os.getenv("ADK_TOP_K", "20")
    try:
        _top_k = max(1, int(top_k_env))
    except Exception:
        _top_k = 20

    try:
        all_hits: List[Dict[str, Any]] = []
        # Iterate across DBs
        for db_name in db_names:
            # New connection alias per DB to avoid cross-DB state issues
            alias = f"db_{db_name}"
            try:
                try:
                    connections.disconnect(alias)
                except Exception:
                    pass
                connections.connect(alias=alias, host=host, port=port, db_name=db_name)
                collection_to_use = coll_name
            except Exception:
                # Fallback for servers without DB support: collections may be suffixed
                try:
                    connections.disconnect(alias)
                except Exception:
                    pass
                connections.connect(alias=alias, host=host, port=port)
                collection_to_use = f"{coll_name}__{db_name}"

            if not utility.has_collection(collection_to_use):
                continue
            coll = Collection(collection_to_use)
            schema_fields = {f.name for f in coll.schema.fields}
            output_fields = [f for f in ["text", "section", "doi", "citation_key", "chunk_index"] if f in schema_fields]
            try:
                coll.load()
            except Exception:
                pass
            params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            res = coll.search(data=[qvec], anns_field="vector", param=params, limit=int(_top_k), output_fields=output_fields)
            for hit in res[0]:
                item = {
                    "score": float(hit.distance),
                    "text": hit.entity.get("text"),
                    "db": db_name,
                }
                if "section" in output_fields:
                    item["section"] = hit.entity.get("section")
                if "doi" in output_fields:
                    item["doi"] = hit.entity.get("doi")
                if "citation_key" in output_fields:
                    item["citation_key"] = hit.entity.get("citation_key")
                if "chunk_index" in output_fields:
                    item["chunk_index"] = hit.entity.get("chunk_index")
                all_hits.append(item)
        # Sort by score desc, keep top_k overall
        all_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return {"status": "success", "results": all_hits[:_top_k]}
    except Exception as e:
        return {"status": "error", "error_message": f"Milvus search failed: {e}"}


def _discover_search_targets(host: str, port: str, coll_base: str = "paper_chunks") -> Tuple[bool, List[Dict[str, str]]]:
    """Discover available Milvus search targets.

    Returns (named_db_supported, targets), where each target is a dict:
      - mode: 'db' | 'suffix' | 'default'
      - db: database name or logical topic label
      - collection: collection name to search
    """
    try:
        from pymilvus import connections, utility  # type: ignore
        # First try named DBs via db API
        named_supported = False
        targets: List[Dict[str, str]] = []
        try:
            connections.connect(alias="bootstrap", host=host, port=port)
            try:
                from pymilvus import db  # type: ignore
                list_dbs = getattr(db, "list_databases", None) or getattr(db, "list_database", None)
                if callable(list_dbs):
                    dbs = list_dbs() or []
                    for d in dbs:
                        alias = f"disco_{d}"
                        try:
                            try:
                                connections.disconnect(alias)
                            except Exception:
                                pass
                            connections.connect(alias=alias, host=host, port=port, db_name=d)
                            # Check for base collection
                            if utility.has_collection(coll_base):
                                targets.append({"mode": "db", "db": d, "collection": coll_base})
                                named_supported = True
                        except Exception:
                            # Ignore DBs we cannot access
                            pass
            finally:
                try:
                    connections.disconnect("bootstrap")
                except Exception:
                    pass
        except Exception:
            pass

        # If none found or named DBs unsupported, fall back to default DB collections
        try:
            connections.connect(alias="default", host=host, port=port)
            cols = utility.list_collections()
            if coll_base in cols:
                targets.append({"mode": "default", "db": "default", "collection": coll_base})
            # Discover suffixed collections as logical topics
            prefix = f"{coll_base}__"
            for c in cols:
                if c.startswith(prefix):
                    topic = c[len(prefix):]
                    targets.append({"mode": "suffix", "db": topic, "collection": c})
        except Exception:
            pass
        return named_supported, targets
    except Exception:
        return False, []


def milvus_smart_search(query: str) -> Dict[str, Any]:
    """Auto-discover DBs/collections, pick relevant targets for the query, and search them.

    Selection heuristic:
      - If the query mentions a topic/DB name (substring match), prioritize those targets.
      - Otherwise, search all discovered targets.
    Returns merged, deduped hits across targets with provenance.
    """
    # Embedding
    try:
        qvec = _embed_query(query)
    except Exception as e:
        return {"status": "error", "error_message": f"Embedding error: {e}"}

    # Discover targets
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    coll_base = os.getenv("ADK_COLLECTION") or "paper_chunks"
    named_supported, targets = _discover_search_targets(host, port, coll_base=coll_base)
    if not targets:
        return {"status": "error", "error_message": "No collections or databases discovered for search."}

    # Select targets based on query keywords
    ql = query.lower()
    selected = [t for t in targets if t.get("db", "").lower() and t["db"].lower() in ql]
    if not selected:
        selected = list(targets)

    # Top-K handling
    try:
        overall_top_k = max(1, int(os.getenv("ADK_TOP_K", "20")))
    except Exception:
        overall_top_k = 20
    per_target_k = max(1, overall_top_k // max(1, min(len(selected), 4)))

    # Search each selected target
    try:
        from pymilvus import connections, utility, Collection  # type: ignore
        all_hits: List[Dict[str, Any]] = []
        for t in selected:
            mode = t["mode"]
            db_label = t["db"]
            coll_name = t["collection"]
            alias = f"srch_{mode}_{db_label}"
            try:
                try:
                    connections.disconnect(alias)
                except Exception:
                    pass
                if mode == "db":
                    connections.connect(alias=alias, host=host, port=port, db_name=db_label)
                else:
                    connections.connect(alias=alias, host=host, port=port)
            except Exception:
                # Skip if cannot connect
                continue

            if not utility.has_collection(coll_name):
                continue
            coll = Collection(coll_name)
            schema_fields = {f.name for f in coll.schema.fields}
            output_fields = [f for f in ["text", "section", "doi", "citation_key", "chunk_index"] if f in schema_fields]
            try:
                coll.load()
            except Exception:
                pass
            params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            try:
                res = coll.search(data=[qvec], anns_field="vector", param=params, limit=int(per_target_k), output_fields=output_fields)
            except Exception:
                continue
            for hit in (res[0] if res else []):
                item = {
                    "score": float(hit.distance),
                    "text": hit.entity.get("text"),
                    "db": db_label,
                    "mode": mode,
                    "collection": coll_name,
                }
                if "section" in output_fields:
                    item["section"] = hit.entity.get("section")
                if "doi" in output_fields:
                    item["doi"] = hit.entity.get("doi")
                if "citation_key" in output_fields:
                    item["citation_key"] = hit.entity.get("citation_key")
                if "chunk_index" in output_fields:
                    item["chunk_index"] = hit.entity.get("chunk_index")
                all_hits.append(item)

        # Deduplicate by (citation_key, chunk_index, text)
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for h in sorted(all_hits, key=lambda x: x.get("score", 0.0), reverse=True):
            key = (h.get("citation_key"), h.get("chunk_index"), (h.get("text") or "")[:64])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)

        return {
            "status": "success",
            "named_db_supported": named_supported,
            "targets_considered": targets,
            "targets_selected": selected,
            "results": deduped[:overall_top_k],
        }
    except Exception as e:
        return {"status": "error", "error_message": f"Milvus smart search failed: {e}"}


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
    "Specialist agent that discovers Milvus databases/collections, retrieves relevant passages (Gemini embeddings), and returns comprehensive, cited answers."
    ),
    instruction=(
    "You are a retrieval specialist sub-agent for a scientific RAG over Milvus.\n"
    "- First, discover all available Milvus databases/collections automatically.\n"
    "- Choose the most appropriate one(s) for the user's query (by topic keywords or domain terms); if unclear, search across all.\n"
    "- For metadata questions (what's in the DB, counts, titles), call milvus_meta_info.\n"
    "- For content questions, call milvus_smart_search with the user's question.\n"
    "- When referencing sources, include only citation_key and doi in citations.\n\n"
    "Process: 1) If the query is about the database contents, use milvus_meta_info. 2) Otherwise, call milvus_smart_search. In your final response:\n"
    "   - Briefly list the databases/collections discovered and which were selected for the query.\n"
    "   - Analyze the retrieved passages and synthesize a comprehensive answer grounded in the database content.\n"
    "   - Cite as [citation_key | doi]. If nothing relevant is found, say so briefly.\n\n"
    "You are a SUB-AGENT. Do not address the user directly. When you finish, RETURN your final answer back to the root orchestrator 'doc_rag_chatbot' so it can present the response."
    ),
)

# Register tools with this agent
milvus_rag_agent.tools.append(milvus_semantic_search)
milvus_rag_agent.tools.append(milvus_meta_info)
milvus_rag_agent.tools.append(milvus_smart_search)
