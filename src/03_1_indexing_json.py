"""
Index cleaned JSON (from md_clean_agent) into Milvus using Gemini embeddings.

Input JSON structure (produced by agents/md_clean_agent.py):
{
  "metadata": { ... front matter ... },
  "content": [
	{ "type": "section", "level": 2, "title": "Results", "text": "..." },
	...
  ]
}

Features:
- Preserve DOI and source path as provenance fields.
- Chunk each content block into safe lengths, preserving section.
- Embed with Google Gemini (gemini-embedding-001 by default).
- Store in Milvus: doi, source, section, chunk_index, hash, image_refs, text, vector.

This is analogous to 03_indexing.py but takes JSON instead of Markdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# --- Helpers ---

IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^\)]+)\)")


@dataclass
class Chunk:
	text: str
	section: str
	chunk_index: int
	image_refs: List[str]


def _sha256(s: str) -> str:
	return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _smart_split(text: str, max_len: int) -> List[str]:
	if len(text) <= max_len:
		return [text]
	# sentence-aware split
	parts: List[str] = []
	cur = ""
	for s in re.split(r"(?<=[.!?])\s+", text):
		if not cur:
			cur = s
		elif len(cur) + 1 + len(s) <= max_len:
			cur = cur + " " + s
		else:
			parts.append(cur)
			cur = s
	if cur:
		parts.append(cur)
	# force split any oversize
	out: List[str] = []
	for p in parts:
		if len(p) <= max_len:
			out.append(p)
		else:
			for i in range(0, len(p), max_len):
				out.append(p[i : i + max_len])
	return out


def _split_paragraphs(text: str) -> List[str]:
	paras = re.split(r"\n\s*\n", text)
	return [p.strip() for p in paras if p.strip()]


def load_clean_json(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
	data = json.loads(path.read_text(encoding="utf-8"))
	meta = data.get("metadata") or {}
	content = data.get("content") or []
	if not isinstance(meta, dict) or not isinstance(content, list):
		raise ValueError("Invalid cleaned JSON format: expected {metadata: {}, content: []}")
	return meta, content


def extract_doi(meta: Dict[str, Any]) -> str:
	doi = meta.get("doi") or meta.get("DOI") or ""
	return doi.strip() if isinstance(doi, str) else ""


def build_chunks(content: List[Dict[str, Any]], *, max_text_len: int = 7000) -> List[Chunk]:
	chunks: List[Chunk] = []
	idx = 0
	for i, item in enumerate(content):
		# content blocks may have title/level; be robust if absent
		title = item.get("title") if isinstance(item, dict) else None
		if not title:
			# Try to infer from leading all-caps label lines (e.g., KEY POINTS, METHODS)
			title = item.get("section") or f"block_{i:03d}"
		text = (item.get("text") or "") if isinstance(item, dict) else ""
		if not text.strip():
			continue
		# Collect image refs inside the block
		imgs = IMAGE_RE.findall(text)
		# Paragraph-level then length-safe splitting
		for para in _split_paragraphs(text):
			for part in _smart_split(para, max_text_len):
				chunks.append(Chunk(text=part, section=str(title), chunk_index=idx, image_refs=imgs))
				idx += 1
	return chunks


# --- Gemini embeddings ---

def get_genai_client():
	from google import genai  # lazy import
	# .env support if present
	try:
		from dotenv import load_dotenv  # type: ignore
		load_dotenv()
	except Exception:
		pass
	api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
	if not api_key:
		raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment")
	return genai.Client(api_key=api_key)


def embed_texts(texts: Sequence[str], model: str = "gemini-embedding-001", batch_size: int = 64) -> List[List[float]]:
	client = get_genai_client()
	vecs: List[List[float]] = []
	model_id = model or os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
	for i in range(0, len(texts), batch_size):
		batch = list(texts[i : i + batch_size])
		try:
			resp = client.models.embed_content(model=model_id, contents=batch)
			if hasattr(resp, "embeddings") and resp.embeddings:
				for e in resp.embeddings:
					vecs.append(list(e.values))
				continue
		except Exception:
			pass
		for t in batch:
			r = client.models.embed_content(model=model_id, contents=t)
			if hasattr(r, "embedding") and hasattr(r.embedding, "values"):
				vecs.append(list(r.embedding.values))
			elif hasattr(r, "values"):
				vecs.append(list(r.values))
			else:
				v = getattr(r, "embedding", None) or getattr(r, "vector", None) or getattr(r, "values", None)
				if v is None:
					raise RuntimeError("Unexpected embedding response format from google-genai")
				if hasattr(v, "values"):
					vecs.append(list(v.values))
				else:
					vecs.append(list(v))
	return vecs


# --- Milvus helpers ---

def milvus_connect(host: str, port: int):
	from pymilvus import connections

	connections.connect(alias="default", host=host, port=str(port))


def ensure_collection(name: str, dim: int, drop_before: bool = False):
	from pymilvus import (
		Collection,
		CollectionSchema,
		DataType,
		FieldSchema,
		utility,
	)

	if drop_before and utility.has_collection(name):
		utility.drop_collection(name)

	if utility.has_collection(name):
		return Collection(name)

	fields = [
		FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
		FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=128),
		FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
		FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
		FieldSchema(name="chunk_index", dtype=DataType.INT64),
		FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
		FieldSchema(name="image_refs", dtype=DataType.VARCHAR, max_length=2048),
		FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
		FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
	]
	schema = CollectionSchema(fields, description="Cleaned JSON chunks with Gemini embeddings")
	coll = Collection(name, schema)
	coll.create_index(
		field_name="vector",
		index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
	)
	return coll


def insert_chunks(coll, doi: str, source_path: Path, chunks: List[Chunk], vectors: List[List[float]], insert_batch: int = 256):
	assert len(chunks) == len(vectors)
	cols: Dict[str, List[Any]] = {
		"doi": [],
		"source": [],
		"section": [],
		"chunk_index": [],
		"hash": [],
		"image_refs": [],
		"text": [],
		"vector": [],
	}
	for ch, vec in zip(chunks, vectors):
		cols["doi"].append(doi)
		cols["source"].append(str(source_path))
		cols["section"].append(ch.section)
		cols["chunk_index"].append(int(ch.chunk_index))
		cols["hash"].append(_sha256(ch.text))
		cols["image_refs"].append("|".join(ch.image_refs) if ch.image_refs else "")
		cols["text"].append(ch.text)
		cols["vector"].append(vec)

	n = len(chunks)
	for i in range(0, n, insert_batch):
		batch = [
			cols["doi"][i : i + insert_batch],
			cols["source"][i : i + insert_batch],
			cols["section"][i : i + insert_batch],
			cols["chunk_index"][i : i + insert_batch],
			cols["hash"][i : i + insert_batch],
			cols["image_refs"][i : i + insert_batch],
			cols["text"][i : i + insert_batch],
			cols["vector"][i : i + insert_batch],
		]
		coll.insert(batch)
	coll.flush()


def preview_chunks(chunks: List[Chunk], n: int = 3) -> None:
	for ch in chunks[:n]:
		print(f"[{ch.chunk_index}] Section: {ch.section}")
		if ch.image_refs:
			print(f"  Images: {', '.join(ch.image_refs)}")
		print(ch.text[:200].replace("\n", " ") + ("..." if len(ch.text) > 200 else ""))
		print()


def main():
	parser = argparse.ArgumentParser(description="Index cleaned JSON into Milvus using Gemini embeddings")
	parser.add_argument(
		"--input",
		type=str,
		default="output/md_with_images/jama_summers_2025-with-image-refs-merged-no-ref-clean.json",
		help="Path to cleaned JSON file (from md_clean_agent)",
	)
	parser.add_argument("--collection", type=str, default="doc_md_multimodal", help="Milvus collection name")
	parser.add_argument("--milvus-host", type=str, default="127.0.0.1", help="Milvus host")
	parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
	parser.add_argument("--embed-model", type=str, default="gemini-embedding-001", help="Gemini embedding model")
	parser.add_argument("--embed-batch", type=int, default=64, help="Embedding batch size")
	parser.add_argument("--insert-batch", type=int, default=256, help="Insert batch size")
	parser.add_argument("--drop-before", action="store_true", help="Drop and recreate collection")
	parser.add_argument("--show", type=int, default=3, help="Preview first N chunks")
	parser.add_argument("--dry-run", action="store_true", help="Only parse and preview; do not embed or insert")
	args = parser.parse_args()

	in_path = Path(args.input)
	if not in_path.exists():
		print(f"Input not found: {in_path}", file=sys.stderr)
		sys.exit(1)

	meta, content = load_clean_json(in_path)
	doi = extract_doi(meta)
	if not doi:
		print("Warning: DOI not found in metadata; proceeding without DOI")

	chunks = build_chunks(content)
	print(f"Parsed chunks: {len(chunks)} from {in_path}")
	if args.show > 0:
		preview_chunks(chunks, args.show)

	if args.dry_run:
		print("Dry run complete. Skipping embeddings and Milvus insert.")
		return

	texts = [c.text for c in chunks]
	vectors = embed_texts(texts, model=args.embed_model, batch_size=args.embed_batch)
	if not vectors:
		print("No embeddings produced.", file=sys.stderr)
		sys.exit(2)
	dim = len(vectors[0])

	milvus_connect(args.milvus_host, args.milvus_port)
	coll = ensure_collection(args.collection, dim=dim, drop_before=args.drop_before)
	insert_chunks(coll, doi=doi, source_path=in_path, chunks=chunks, vectors=vectors, insert_batch=args.insert_batch)
	print(f"Inserted {len(chunks)} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
	main()

