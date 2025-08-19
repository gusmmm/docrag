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
import hashlib
from pathlib import Path
from typing import List, Tuple, Any, Iterable

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


def embed_texts(texts: List[str], model: str, batch_size: int = 64) -> List[List[float]]:
	"""Return embeddings for a list of texts using Gemini embeddings, batched.

	Falls back to per-item on unexpected batch errors.
	"""
	client = get_genai_client()
	model_id = model or os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
	out: List[List[float]] = []
	n = len(texts)
	i = 0
	while i < n:
		batch = texts[i : i + batch_size]
		try:
			resp = client.models.embed_content(model=model_id, contents=batch)
			if hasattr(resp, "embeddings") and resp.embeddings:
				out.extend([e.values for e in resp.embeddings])
			else:
				# Fallback to per-item if response not as expected
				for t in batch:
					r = client.models.embed_content(model=model_id, contents=t)
					out.append(r.embeddings[0].values)
		except Exception:
			# Fallback to per-item on exceptions
			for t in batch:
				r = client.models.embed_content(model=model_id, contents=t)
				out.append(r.embeddings[0].values)
		i += len(batch)
	return out


def _chunk_by_length(text: str, max_len: int = 6000) -> List[str]:
	"""Split text into chunks <= max_len, preferring whitespace boundaries.

	Keeps chunks comfortably under Milvus VARCHAR(8192) default limit.
	"""
	chunks: List[str] = []
	i = 0
	n = len(text)
	if n <= max_len:
		return [text]
	while i < n:
		end = min(i + max_len, n)
		probe_start = min(n, i + int(max_len * 0.8))
		j = text.rfind(" ", probe_start, end)
		if j == -1 or j <= i:
			j = end
		chunk = text[i:j].strip()
		if chunk:
			chunks.append(chunk)
		i = j
	return chunks


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
		if p.startswith("#"):
			continue
		# Ensure each chunk fits VARCHAR limit headroom
		for piece in _chunk_by_length(p, max_len=6000):
			if len(piece) < min_len:
				continue
			cleaned.append(piece)
			if len(cleaned) >= max_paragraphs:
				return cleaned
	return cleaned


def ensure_milvus_connection(host: str, port: str):
	if connections is None:
		raise RuntimeError("pymilvus not installed; run 'uv add pymilvus'.")
	try:
		connections.connect(alias="default", host=host, port=port)
	except Exception as e:
		raise RuntimeError(f"Failed to connect to Milvus at {host}:{port}. Is the server running? {e}")


def create_collection(name: str, dim: int, *, with_hash: bool = False, drop_before: bool = False) -> Any:
	if utility.has_collection(name):
		if drop_before:
			utility.drop_collection(name)
		else:
			return Collection(name)
	fields = [
		FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
	]
	if with_hash:
		fields.append(FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64))
	fields.extend([
		FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
		FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
	])
	schema = CollectionSchema(fields=fields, description="Demo RAG collection")
	coll = Collection(name=name, schema=schema)
	coll.create_index(
		field_name="vector",
		index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
	)
	return coll


def _get_non_pk_fields(coll: Any) -> List[str]:
	names: List[str] = []
	for f in coll.schema.fields:
		if getattr(f, "is_primary", False):
			continue
		names.append(f.name)
	return names


def insert_documents(coll: Any, texts: List[str], vectors: List[List[float]], batch_size: int = 256, hashes: List[str] | None = None):
	assert len(texts) == len(vectors)
	n = len(texts)
	i = 0
	field_order = _get_non_pk_fields(coll)
	while i < n:
		tb = texts[i : i + batch_size]
		vb = vectors[i : i + batch_size]
		payload: List[Any] = []
		for field in field_order:
			if field == "text":
				payload.append(tb)
			elif field == "vector":
				payload.append(vb)
			elif field == "hash":
				hb = (hashes or [""] * n)[i : i + batch_size]
				payload.append(hb)
			else:
				# Unknown extra field; insert empty placeholders if needed
				payload.append([None] * len(tb))
		coll.insert(payload)
		i += len(tb)
	coll.flush()


def search(coll: Any, query_vec: List[float], top_k: int = 5) -> List[Tuple[float, str, str | None]]:
	coll.load()
	# Choose available output fields
	field_names = [f.name for f in coll.schema.fields]
	ofields = ["text"]
	if "section" in field_names:
		ofields.append("section")
	search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
	results = coll.search(
		data=[query_vec],
		anns_field="vector",
		param=search_params,
		limit=top_k,
		output_fields=ofields,
	)
	out: List[Tuple[float, str, str | None]] = []
	for hit in results[0]:
		text = hit.entity.get("text")
		section = hit.entity.get("section") if "section" in ofields else None
		out.append((hit.distance, text, section))
	return out


def _rerank_hits_by_substring(
	hits: List[Tuple[float, str, str | None]],
	prefs_text: List[str] | None,
	prefs_section: List[str] | None = None,
) -> List[Tuple[float, str, str | None]]:
	if not prefs_text and not prefs_section:
		return hits
	prefs_text_l = [p.lower() for p in (prefs_text or [])]
	prefs_sec_l = [p.lower() for p in (prefs_section or [])]
	def _key(item: Tuple[float, str, str | None]) -> Tuple[int, int, float]:
		score, txt, sec = item
		t = (txt or "").lower()
		s = (sec or "").lower()
		boost_sec = any(p in s for p in prefs_sec_l) if prefs_sec_l else False
		boost_txt = any(p in t for p in prefs_text_l) if prefs_text_l else False
		# Prefer section boost most, then text boost, then score desc
		return (0 if boost_sec else 1, 0 if boost_txt else 1, -score)
	return sorted(hits, key=_key)


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


def _iter_markdown_files(dir_path: Path) -> Iterable[Path]:
	for p in dir_path.rglob("*.md"):
		if p.is_file():
			yield p


def _sha256(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def demo(
	md_file: Path | None,
	query: str | None,
	collection_name: str = "demo_rag",
	embed_model: str | None = None,
	*,
	index_only: bool = False,
	show: int = 0,
	md_dir: Path | None = None,
	embed_batch: int = 64,
	insert_batch: int = 256,
	drop_before: bool = False,
	dedup: bool = False,
	query_only: bool = False,
	top_k: int = 5,
	prefer_substr: List[str] | None = None,
	prefer_section: List[str] | None = None,
):
	host = os.getenv("MILVUS_HOST", "localhost")
	port = os.getenv("MILVUS_PORT", "19530")
	ensure_milvus_connection(host, port)

	# Query-only path: do not index; just preview and/or search existing collection
	if query_only:
		if not utility.has_collection(collection_name):
			print(f"Collection '{collection_name}' does not exist. Nothing to query.")
			return
		coll = Collection(collection_name)
		if show and show > 0:
			print(f"Collection '{collection_name}' entities={coll.num_entities}")
			_print_collection_preview(coll, limit=show)
		if not query:
			return
		# Embed and search
		qvec = embed_texts([query], embed_model, batch_size=1)[0]
		hits = search(coll, qvec, top_k=top_k)
		hits = _rerank_hits_by_substring(hits, prefer_substr, prefer_section)
		print("Top results:")
		for score, text, section in hits:
			snippet = (text or "").replace("\n", " ")
			if len(snippet) > 180:
				snippet = snippet[:180]
			prefix = f"[{section}] " if section else ""
			print(f"- score={score:.4f} | {prefix}{snippet}")
		return

	# Prepare corpus
	corpus: List[str] = []
	if md_dir and Path(md_dir).exists():
		total_files = 0
		for f in _iter_markdown_files(Path(md_dir)):
			total_files += 1
			md_text = f.read_text(encoding="utf-8", errors="ignore")
			parts = split_markdown_paragraphs(md_text)
			if parts:
				corpus.extend(parts)
		if total_files:
			print(f"Scanned {total_files} markdown file(s) under {md_dir} -> {len(corpus)} chunk(s)")
	elif md_file and Path(md_file).exists():
		md_text = Path(md_file).read_text(encoding="utf-8")
		corpus = split_markdown_paragraphs(md_text)
		if not corpus:
			print("No suitable paragraphs found in markdown; using sample texts.")

	if not corpus:
		corpus = [
			"This randomized trial evaluated augmented enteral protein in critically ill ICU patients.",
			"The primary outcome was days alive and free from hospital at day 90.",
			"Protein delivery in the augmented group was higher with similar caloric intake.",
			"No significant improvement was observed in primary or key secondary outcomes.",
			"Subgroup analysis suggested heterogeneity related to mechanical ventilation and kidney therapy.",
		]

	# Get a sample embedding to determine dimension
	sample_vec = embed_texts([corpus[0]], embed_model, batch_size=embed_batch)[0]
	dim = len(sample_vec)
	coll = create_collection(collection_name, dim, with_hash=True, drop_before=drop_before)

	# Build dedup set (existing hashes) if requested and hash field exists
	existing_hashes: set[str] = set()
	non_pk_fields = _get_non_pk_fields(coll)
	if dedup and "hash" in non_pk_fields:
		try:
			coll.load()
			rows = coll.query(expr="id >= 0", output_fields=["hash"], limit=100000)
			for r in rows:
				h = r.get("hash")
				if h:
					existing_hashes.add(h)
		except Exception as e:
			print(f"WARN: Could not fetch existing hashes for dedup: {e}")

	# Prepare hashes and optional dedup
	hashes = [_sha256(t) for t in corpus]
	if dedup and existing_hashes:
		filtered_corpus: List[str] = []
		filtered_hashes: List[str] = []
		for t, h in zip(corpus, hashes):
			if h in existing_hashes:
				continue
			filtered_corpus.append(t)
			filtered_hashes.append(h)
		corpus = filtered_corpus
		hashes = filtered_hashes

	if not corpus:
		print("No new documents to insert after dedup.")
	else:
		vectors = embed_texts(corpus, embed_model, batch_size=embed_batch)
		insert_documents(coll, corpus, vectors, batch_size=insert_batch, hashes=hashes)
		print(f"Inserted {len(corpus)} documents into collection '{collection_name}'. Entities now: {coll.num_entities}")

	# Show preview of stored rows if requested
	if show and show > 0:
		print(f"Collection '{collection_name}' entities={coll.num_entities}")
		_print_collection_preview(coll, limit=show)

	# Stop here if index-only
	if index_only or not query:
		return

	# Query flow
	qvec = embed_texts([query], embed_model)[0]
	hits = search(coll, qvec, top_k=top_k)
	hits = _rerank_hits_by_substring(hits, prefer_substr, prefer_section)
	print("Top results:")
	for score, text, section in hits:
		snippet = (text or "").replace("\n", " ")
		if len(snippet) > 180:
			snippet = snippet[:180]
		prefix = f"[{section}] " if section else ""
		print(f"- score={score:.4f} | {prefix}{snippet}")


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Milvus RAG demo using Gemini embeddings")
	parser.add_argument("--file", type=str, default=None, help="Path to a Markdown file to index")
	parser.add_argument("--dir", type=str, default=None, help="Path to a directory of Markdown files (recursive)")
	parser.add_argument("--query", type=str, default=None, help="Query text. Omit or use --index-only to skip querying.")
	parser.add_argument("--collection", type=str, default="demo_rag", help="Milvus collection name")
	parser.add_argument("--embed-model", type=str, default=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"), help="Gemini embedding model")
	parser.add_argument("--index-only", action="store_true", help="Run chunking+embedding only; skip query")
	parser.add_argument("--query-only", action="store_true", help="Only run a query against an existing collection (no indexing)")
	parser.add_argument("--show", type=int, default=0, help="After indexing, print N stored rows from the collection")
	parser.add_argument("--embed-batch", type=int, default=64, help="Embedding batch size")
	parser.add_argument("--insert-batch", type=int, default=256, help="Insert batch size")
	parser.add_argument("--drop-before", action="store_true", help="Drop collection first (recreate with hash field)")
	parser.add_argument("--dedup", action="store_true", help="Skip inserting chunks that already exist (by text hash)")
	parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
	parser.add_argument(
		"--prefer-substr",
		action="append",
		default=[],
		help="Boost results whose text contains this substring (case-insensitive). Can be repeated.",
	)
	parser.add_argument(
		"--prefer-section",
		action="append",
		default=[],
		help="Boost results whose section contains this substring (case-insensitive). Can be repeated.",
	)
	args = parser.parse_args()

	md_file = Path(args.file) if args.file else None
	md_dir = Path(args.dir) if args.dir else None
	demo(
		md_file,
		args.query,
		collection_name=args.collection,
		embed_model=args.embed_model,
		index_only=args.index_only,
		query_only=args.query_only,
		show=args.show,
		md_dir=md_dir,
		embed_batch=args.embed_batch,
		insert_batch=args.insert_batch,
		drop_before=args.drop_before,
		dedup=args.dedup,
		top_k=args.top_k,
		prefer_substr=args.prefer_substr,
		prefer_section=args.prefer_section,
	)


if __name__ == "__main__":
	main()

