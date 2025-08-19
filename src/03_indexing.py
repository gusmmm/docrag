"""
Index merged Markdown (with YAML metadata and image refs) into Milvus using Gemini embeddings.

Features:
- Parse YAML front matter for DOI and other provenance.
- Chunk Markdown semantically by headings and paragraphs, preserving image refs.
- Embed using Google Gemini embeddings (model='gemini-embedding-001').
- Store in Milvus with scalar fields for DOI, source, section, chunk_index, hash, image_refs, and text.

Default input example:
  output/md_with_images/jama_summers_2025-with-image-refs-merged.md

Environment:
- GEMINI_API_KEY or GOOGLE_API_KEY for Gemini.
- Milvus running from docker-compose at localhost:19530.

Note: We avoid external YAML deps by parsing a simple subset of front matter.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# --- Front matter parsing (minimal YAML subset) ---

FM_BOUNDARY = re.compile(r"^---\s*$")
KEY_VAL_RE = re.compile(r"^(?P<key>[A-Za-z0-9_\-]+):\s*(?P<val>.*)$")
LIST_ITEM_RE = re.compile(r"^\s*\-\s*(.*)$")


def parse_front_matter(md_text: str) -> Tuple[Dict[str, Any], str]:
	"""Parse a simple YAML-like front matter and return (metadata, body_md).

	Supports:
	- key: value (value may be quoted with "...")
	- key: (followed by indented list of - items)
	- Stops at the second '---' boundary
	"""
	lines = md_text.splitlines()
	if not lines or not FM_BOUNDARY.match(lines[0]):
		return {}, md_text

	meta: Dict[str, Any] = {}
	i = 1
	current_key: Optional[str] = None
	current_list: Optional[List[str]] = None
	while i < len(lines):
		line = lines[i]
		if FM_BOUNDARY.match(line):
			# close any open list
			if current_key and current_list is not None:
				meta[current_key] = current_list
			body = "\n".join(lines[i + 1:])
			return meta, body
		if line.strip() == "":
			i += 1
			continue
		m = KEY_VAL_RE.match(line)
		if m:
			# close previous list
			if current_key and current_list is not None:
				meta[current_key] = current_list
				current_list = None
			key = m.group("key").strip()
			val = m.group("val").strip()
			if val == "" or val == "|" or val == ">":
				# start list or multi-line (we only support list here)
				current_key = key
				current_list = []
			else:
				# strip quotes if present
				if (val.startswith('"') and val.endswith('"')) or (
					val.startswith("'") and val.endswith("'")
				):
					val = val[1:-1]
				meta[key] = val
				current_key = None
				current_list = None
		else:
			m2 = LIST_ITEM_RE.match(line)
			if m2 and current_key is not None:
				item = m2.group(1).strip()
				if (item.startswith('"') and item.endswith('"')) or (
					item.startswith("'") and item.endswith("'")
				):
					item = item[1:-1]
				if current_list is None:
					current_list = []
				current_list.append(item)
			else:
				# Unknown content in front matter; ignore gracefully
				pass
		i += 1
	# If no closing boundary, treat as no front matter
	return {}, md_text


# --- Markdown chunking ---

HEADING_RE = re.compile(r"^(?P<h>#{1,6})\s+(?P<title>.+?)\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^\)]+)\)")


@dataclass
class Chunk:
	text: str
	section: str
	chunk_index: int
	image_refs: List[str]


def _split_paragraphs(lines: List[str]) -> List[List[str]]:
	paras: List[List[str]] = []
	buf: List[str] = []
	def flush():
		nonlocal buf
		if buf and any(s.strip() for s in buf):
			paras.append(buf)
		buf = []
	for ln in lines:
		if ln.strip() == "":
			flush()
		else:
			buf.append(ln)
	flush()
	return paras


def _smart_split(text: str, max_len: int) -> List[str]:
	"""Split text into <= max_len chunks, preferably on sentence or newline boundaries."""
	if len(text) <= max_len:
		return [text]
	# Try by sentences
	sentences = re.split(r"(?<=[.!?])\s+", text)
	parts: List[str] = []
	cur = ""
	for s in sentences:
		if not cur:
			cur = s
		elif len(cur) + 1 + len(s) <= max_len:
			cur = cur + " " + s
		else:
			parts.append(cur)
			cur = s
	if cur:
		parts.append(cur)
	# If still some parts too long (long sentences), force split
	out: List[str] = []
	for p in parts:
		if len(p) <= max_len:
			out.append(p)
		else:
			for i in range(0, len(p), max_len):
				out.append(p[i:i+max_len])
	return out


def chunk_markdown(md_body: str, max_text_len: int = 7000) -> List[Chunk]:
	"""Chunk MD by headings and paragraphs, preserving image refs and section titles.

	max_text_len ensures we stay within Milvus VARCHAR(8192) after adding some headroom.
	"""
	lines = md_body.splitlines()
	section_stack: List[str] = []
	chunks: List[Chunk] = []
	chunk_idx = 0

	# First, split by top-level heading blocks to keep semantics tighter
	i = 0
	block_lines: List[str] = []
	block_sections: List[str] = []

	def current_section() -> str:
		return " / ".join(section_stack)

	def process_block(blines: List[str], section_title: str):
		nonlocal chunk_idx
		if not blines:
			return
		for para in _split_paragraphs(blines):
			para_text = "\n".join(para).strip()
			if not para_text:
				continue
			imgs = IMAGE_RE.findall(para_text)
			# split if needed
			for part in _smart_split(para_text, max_text_len):
				chunks.append(Chunk(text=part, section=section_title, chunk_index=chunk_idx, image_refs=imgs))
				chunk_idx += 1

	while i < len(lines):
		m = HEADING_RE.match(lines[i])
		if m:
			# process previous block
			if block_lines:
				process_block(block_lines, current_section())
				block_lines = []
			level = len(m.group("h"))
			title = m.group("title").strip()
			# adjust stack
			while len(section_stack) >= level:
				section_stack.pop()
			section_stack.append(title)
		else:
			block_lines.append(lines[i])
		i += 1
	if block_lines:
		process_block(block_lines, current_section())

	return chunks


# --- Gemini embeddings ---

def get_genai_client():
	from google import genai
	# Try to load environment from .env if available
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
		batch = list(texts[i:i+batch_size])
		try:
			# Try batch embedding
			resp = client.models.embed_content(model=model_id, contents=batch)
			# google-genai returns an object with .embeddings for batch
			if hasattr(resp, "embeddings") and resp.embeddings:
				for e in resp.embeddings:
					vecs.append(list(e.values))
				continue
			# If response is single (SDK variant), fall back
		except Exception:
			# Fall back to per-item
			pass
		# Per-item fallback
		for t in batch:
			r = client.models.embed_content(model=model_id, contents=t)
			# single embedding response has .embedding or .values depending on SDK
			if hasattr(r, "embedding") and hasattr(r.embedding, "values"):
				vecs.append(list(r.embedding.values))
			elif hasattr(r, "values"):
				vecs.append(list(r.values))
			else:
				# Attempt generic access
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
	schema = CollectionSchema(fields, description="Multimodal MD chunks with Gemini embeddings")
	coll = Collection(name, schema)
	# AUTOINDEX for simplicity
	coll.create_index(
		field_name="vector",
		index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
	)
	return coll


def _sha256(s: str) -> str:
	return hashlib.sha256(s.encode("utf-8")).hexdigest()


def insert_chunks(
	coll, doi: str, source_path: str, chunks: List[Chunk], vectors: List[List[float]], insert_batch: int = 256
):
	assert len(chunks) == len(vectors)
	# Prepare columnar lists
	all_doi: List[str] = []
	all_source: List[str] = []
	all_section: List[str] = []
	all_chunk_idx: List[int] = []
	all_hash: List[str] = []
	all_image_refs: List[str] = []
	all_text: List[str] = []
	all_vecs: List[List[float]] = []

	for ch, vec in zip(chunks, vectors):
		all_doi.append(doi)
		all_source.append(str(source_path))
		all_section.append(ch.section)
		all_chunk_idx.append(int(ch.chunk_index))
		all_text.append(ch.text)
		# Join image refs into a single field (| separated)
		img_field = "|".join(ch.image_refs) if ch.image_refs else ""
		all_image_refs.append(img_field)
		all_hash.append(_sha256(ch.text))
		all_vecs.append(vec)

	# Insert in batches
	for i in range(0, len(all_text), insert_batch):
		batch = [
			all_doi[i:i+insert_batch],
			all_source[i:i+insert_batch],
			all_section[i:i+insert_batch],
			all_chunk_idx[i:i+insert_batch],
			all_hash[i:i+insert_batch],
			all_image_refs[i:i+insert_batch],
			all_text[i:i+insert_batch],
			all_vecs[i:i+insert_batch],
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


def extract_doi(meta: Dict[str, Any]) -> str:
	doi = meta.get("doi") or meta.get("DOI") or ""
	if not doi or not isinstance(doi, str):
		return ""
	return doi.strip()


def main():
	parser = argparse.ArgumentParser(description="Index merged Markdown into Milvus with Gemini embeddings")
	parser.add_argument(
		"--input",
		type=str,
		default="output/md_with_images/jama_summers_2025-with-image-refs-merged.md",
		help="Path to merged Markdown (with YAML front matter)",
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

	md_path = Path(args.input)
	if not md_path.exists():
		print(f"Input Markdown not found: {md_path}", file=sys.stderr)
		sys.exit(1)

	raw = md_path.read_text(encoding="utf-8")
	meta, body = parse_front_matter(raw)
	doi = extract_doi(meta)
	if not doi:
		print("Warning: DOI not found in front matter; proceeding without DOI")

	chunks = chunk_markdown(body)
	print(f"Parsed chunks: {len(chunks)} from {md_path}")
	if args.show > 0:
		preview_chunks(chunks, args.show)

	if args.dry_run:
		print("Dry run complete. Skipping embeddings and Milvus insert.")
		return

	# Compute embeddings
	texts = [c.text for c in chunks]
	vectors = embed_texts(texts, model=args.embed_model, batch_size=args.embed_batch)
	if not vectors:
		print("No embeddings produced.", file=sys.stderr)
		sys.exit(2)
	dim = len(vectors[0])

	# Milvus
	milvus_connect(args.milvus_host, args.milvus_port)
	coll = ensure_collection(args.collection, dim=dim, drop_before=args.drop_before)
	insert_chunks(coll, doi=doi, source_path=md_path, chunks=chunks, vectors=vectors, insert_batch=args.insert_batch)
	print(f"Inserted {len(chunks)} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
	main()

