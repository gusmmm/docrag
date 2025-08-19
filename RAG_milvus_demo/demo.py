"""Milvus RAG demo using Google Gemini embeddings.

- Vector store: Milvus (pymilvus)
- Embeddings: Google GenAI (Gemini), default model: gemini-embedding-001
- Demo data: paragraphs from a Markdown file or built-in samples

Env vars:
- MILVUS_HOST (default: localhost)
- MILVUS_PORT (default: 19530)
- GOOGLE_API_KEY or GEMINI_API_KEY (required for embeddings)
- GEMINI_EMBED_MODEL (default: gemini-embedding-001)

Usage examples:
- Index only from a Markdown file and show first 5 rows:
	uv run python RAG_milvus_demo/demo.py --file output/md_with_images/jama_summers_2025-with-image-refs.md --index-only --show 5
- Use sample texts to index only:
	uv run python RAG_milvus_demo/demo.py --index-only --show 5
- Index and then query:
	uv run python RAG_milvus_demo/demo.py --file output/md_with_images/jama_summers_2025-with-image-refs.md --query "What was the primary outcome?"
"""

from __future__ import annotations

import os
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any

from dotenv import load_dotenv

load_dotenv()

try:
	from google import genai
except Exception:  # pragma: no cover
	genai = None

try:
	from pymilvus import (
		connections,
		utility,
		FieldSchema,
		CollectionSchema,
		DataType,
		Collection,
	)
except Exception:  # pragma: no cover
	connections = None
	utility = None
	FieldSchema = None
	CollectionSchema = None
	DataType = None
	Collection = None


def get_genai_client():
	if genai is None:
		raise RuntimeError("google-genai SDK not installed; run 'uv add google-genai'.")
	api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required for embeddings.")
	return genai.Client(api_key=api_key)


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
	"""Return embeddings for a list of texts using Gemini embeddings.

	Tries batch embedding first; falls back to per-item if needed.
	"""
	client = get_genai_client()
	# Prefer latest default per project guidance
	model_id = model or os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

	# Batch attempt
	try:
		resp = client.models.embed_content(model=model_id, contents=texts)
		if hasattr(resp, "embeddings") and resp.embeddings:
			return [e.values for e in resp.embeddings]
	except Exception:
		pass

	# Fallback: single calls
	vecs: List[List[float]] = []
	for t in texts:
		r = client.models.embed_content(model=model_id, contents=t)
		vecs.append(r.embeddings[0].values)
	return vecs


def split_markdown_paragraphs(md: str, min_len: int = 40, max_paragraphs: int = 200) -> List[str]:
	# Split on blank lines; strip headings and image refs for cleaner chunks
	parts = re.split(r"\n\s*\n", md)
	cleaned: List[str] = []
	for p in parts:
		p = p.strip()
		if not p:
			continue
		# Remove markdown images and links clutter
		p = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", p)
		p = re.sub(r"\[[^\]]*\]\([^)]*\)", "", p)
		# Skip very short or pure headings
		if len(p) < min_len or p.startswith("#"):
			continue
		cleaned.append(p)
		if len(cleaned) >= max_paragraphs:
			break
	return cleaned


def ensure_milvus_connection(host: str, port: str):
	if connections is None:
		raise RuntimeError("pymilvus not installed; run 'uv add pymilvus'.")
	try:
		connections.connect(alias="default", host=host, port=port)
	except Exception as e:
		raise RuntimeError(f"Failed to connect to Milvus at {host}:{port}. Is the server running? {e}")


def create_collection(name: str, dim: int) -> Any:
	if utility.has_collection(name):
		return Collection(name)
	fields = [
		FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
		FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
		FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
	]
	schema = CollectionSchema(fields=fields, description="Demo RAG collection")
	coll = Collection(name=name, schema=schema)
	# Create an AUTOINDEX for simplicity; COSINE metric is common for embeddings
	coll.create_index(
		field_name="vector",
		index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
	)
	return coll


def insert_documents(coll: Any, texts: List[str], vectors: List[List[float]]):
	assert len(texts) == len(vectors)
	# Use dict to avoid field-order pitfalls with auto_id
	coll.insert({"text": texts, "vector": vectors})
	coll.flush()


def search(coll: Any, query_vec: List[float], top_k: int = 5) -> List[Tuple[float, str]]:
	coll.load()
	search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
	results = coll.search(
		data=[query_vec],
		anns_field="vector",
		param=search_params,
		limit=top_k,
		output_fields=["text"],
	)
	hits: List[Tuple[float, str]] = []
	for hit in results[0]:
		text = hit.entity.get("text")
		hits.append((hit.distance, text))
	return hits


def _print_collection_preview(coll: Any, limit: int = 5):
	try:
		coll.load()
		rows = coll.query(expr="id >= 0", output_fields=["id", "text"], limit=limit)
		print(f"Showing {len(rows)} row(s):")
		for r in rows:
			rid = r.get("id")
			txt = (r.get("text") or "")[:180].replace("\n", " ")
			print(f"- id={rid} | {txt}")
	except Exception as e:
		print(f"Preview failed: {e}")


def demo(md_file: Path | None, query: str | None, collection_name: str = "demo_rag", embed_model: str | None = None, *, index_only: bool = False, show: int = 0):
	host = os.getenv("MILVUS_HOST", "localhost")
	port = os.getenv("MILVUS_PORT", "19530")
	ensure_milvus_connection(host, port)

	# Prepare corpus
	if md_file and Path(md_file).exists():
		md_text = Path(md_file).read_text(encoding="utf-8")
		corpus = split_markdown_paragraphs(md_text)
		if not corpus:
			print("No suitable paragraphs found in markdown; using sample texts.")
	else:
		corpus = []

	if not corpus:
		corpus = [
			"This randomized trial evaluated augmented enteral protein in critically ill ICU patients.",
			"The primary outcome was days alive and free from hospital at day 90.",
			"Protein delivery in the augmented group was higher with similar caloric intake.",
			"No significant improvement was observed in primary or key secondary outcomes.",
			"Subgroup analysis suggested heterogeneity related to mechanical ventilation and kidney therapy.",
		]

	# Get a sample embedding to determine dimension
	sample_vec = embed_texts([corpus[0]], embed_model)[0]
	dim = len(sample_vec)
	coll = create_collection(collection_name, dim)

	# If empty, insert documents
	if coll.num_entities == 0:
		vectors = embed_texts(corpus, embed_model)
		insert_documents(coll, corpus, vectors)
		print(f"Inserted {len(corpus)} documents into collection '{collection_name}'.")
	else:
		print(f"Collection '{collection_name}' already has {coll.num_entities} entities; skipping insert.")

	# Show preview of stored rows if requested
	if show and show > 0:
		print(f"Collection '{collection_name}' entities={coll.num_entities}")
		_print_collection_preview(coll, limit=show)

	# Stop here if index-only
	if index_only or not query:
		return

	# Query flow
	qvec = embed_texts([query], embed_model)[0]
	hits = search(coll, qvec, top_k=5)
	print("Top results:")
	for score, text in hits:
		print(f"- score={score:.4f} | {text[:180].replace('\n',' ')}")


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Milvus RAG demo using Gemini embeddings")
	parser.add_argument("--file", type=str, default=None, help="Path to a Markdown file to index")
	parser.add_argument("--query", type=str, default=None, help="Query text. Omit or use --index-only to skip querying.")
	parser.add_argument("--collection", type=str, default="demo_rag", help="Milvus collection name")
	parser.add_argument("--embed-model", type=str, default=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"), help="Gemini embedding model")
	parser.add_argument("--index-only", action="store_true", help="Run chunking+embedding only; skip query")
	parser.add_argument("--show", type=int, default=0, help="After indexing, print N stored rows from the collection")
	args = parser.parse_args()

	md_file = Path(args.file) if args.file else None
	demo(
		md_file,
		args.query,
		collection_name=args.collection,
		embed_model=args.embed_model,
		index_only=args.index_only,
		show=args.show,
	)


if __name__ == "__main__":
	main()

