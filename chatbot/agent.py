from __future__ import annotations

"""Google ADK chatbot agent.

Follow the ADK quickstart: https://google.github.io/adk-docs/get-started/quickstart/

This module defines a minimal multi-tool agent that can be discovered by the ADK CLI
and used via `adk web`, `adk run`, or `adk api_server` from the repo root.

Auth:
- Prefer GOOGLE_API_KEY in .env at project root or export in shell.
- Set GOOGLE_GENAI_USE_VERTEXAI=FALSE for AI Studio keys (default path here).

Note: Keep model strings current with ADK docs. For general chat, gemini-2.0-flash is a good default.
"""

import os
import time
import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any

from google.adk.agents import Agent
from .db_agent import milvus_rag_agent, milvus_meta_info

# Optional dependencies loaded lazily inside tools
# - google.genai for embeddings
# - pymilvus for vector search



# Root agent discovered by ADK
# Default to a more robust model to mitigate occasional 500 INTERNAL errors; override via ADK_MODEL.
_MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash")

root_agent = Agent(
	name="doc_rag_chatbot",
	model=_MODEL,
	description=(
		"Agent that can answer scientific questions and, when needed, retrieve supporting passages "
		"from a Milvus vector DB indexed with Gemini embeddings."
	),
	instruction=(
		"You are a helpful scientific assistant for a Milvus-backed RAG.\n"
		"- For scientific journal papers' metadata, use the papers_meta database (collection).\n"
		"- For scientific paper content, use the paper_chunks database (collection).\n"
		"- When answering questions about the database itself (e.g., which papers it contains, how many, titles/authors), use the papers_meta database via milvus_meta_info.\n"
		"- When referencing sources, cite using only citation_key and doi (e.g., [citation_key | doi]).\n\n"
		"When a question requires facts from the indexed paper(s), call milvus_semantic_search to retrieve passages, then synthesize a concise answer with the citations above. "
		"If nothing relevant is found, state that briefly and, if useful, ask a short clarifying question."
	),
	sub_agents=[milvus_rag_agent],
 # milvus tool appended below after definition
)


# -----------------------------
# Milvus semantic search tool
# -----------------------------

def _embed_query(text: str) -> List[float]:
	"""Return a Gemini embedding vector for the query text.

	Uses GOOGLE_API_KEY or GEMINI_API_KEY; model defaults to gemini-embedding-001.
	"""
	last_err = None
	for attempt in range(3):
		try:
			# Lazy import to avoid import cost when tool not used
			from google import genai  # type: ignore
			api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
			if not api_key:
				raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for embeddings")
			client = genai.Client(api_key=api_key)
			model_id = os.getenv("ADK_EMBED_MODEL") or os.getenv("GEMINI_EMBED_MODEL") or "gemini-embedding-001"
			resp = client.models.embed_content(model=model_id, contents=text)
			# Adapt to SDK variants
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
		except Exception as e:  # pragma: no cover
			last_err = e
			time.sleep(0.5 * (2 ** attempt))
	# If we reach here, all retries failed
	raise RuntimeError(f"Embedding error after retries: {last_err}")


def milvus_semantic_search(query: str) -> Dict[str, Any]:
	"""Retrieve relevant passages from Milvus for a scientific question.

	Args:
		query: The natural language question.

	Returns:
		dict with 'status' and either 'results' (list) or 'error_message'. Each result has
		score, text, section, doi, citation_key, chunk_index.
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
	# Top-k is configurable via env; keep simple signature for ADK auto-calling
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
		# Determine available scalar fields dynamically (aligned with our schema)
		schema_fields = {f.name for f in coll.schema.fields}
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


# Register tools with the agent
root_agent.tools.append(milvus_semantic_search)
root_agent.tools.append(milvus_meta_info)

