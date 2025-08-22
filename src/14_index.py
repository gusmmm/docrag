from __future__ import annotations

"""
Step 5: Chunk, embed, and index -RAG.md files into Milvus.

Database layout:
- papers_meta (one row per paper) in the metadata DB (default: journal_papers):
  id (PK, auto), doi, citation_key, title, journal, issued, url, source_path
- paper_chunks (content + vectors) in the target chunks DB:
  id (PK, auto), paper_id (FK-like), doi, citation_key, section, chunk_index, hash, image_refs, text, vector

Behavior:
- If a paper (by doi or citation_key) already exists in papers_meta, skip indexing its chunks.
- Chunking preserves headings context and image references; embeddings use Gemini (gemini-embedding-001).

Usage:
    uv run python src/14_index.py --dry-run --show 3
    uv run python src/14_index.py --collection paper_chunks --meta-collection papers_meta
    # Topic-aware indexing (automatic):
    # - Reads input/input_pdf.json for each citation_key to find optional "topic".
    # - Inserts bibliographic row into --meta-db-name (papers_meta),
    #   and chunk rows into the DB named after the topic (or --db-name if no topic).
"""

import argparse
import os
import re
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cache whether Milvus named databases are supported/usable in this run
_NAMED_DB_SUPPORTED: Optional[bool] = None


# --- Front matter parsing (minimal YAML subset) ---

FM_BOUNDARY = re.compile(r"^---\s*$")
KEY_VAL_RE = re.compile(r"^(?P<key>[A-Za-z0-9_\-]+):\s*(?P<val>.*)$")
LIST_ITEM_RE = re.compile(r"^\s*\-\s*(.*)$")


def parse_front_matter(md_text: str) -> Tuple[Dict[str, Any], str]:
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
            if current_key and current_list is not None:
                meta[current_key] = current_list
            body = "\n".join(lines[i + 1 :])
            return meta, body
        if line.strip() == "":
            i += 1
            continue
        m = KEY_VAL_RE.match(line)
        if m:
            if current_key and current_list is not None:
                meta[current_key] = current_list
                current_list = None
            key = m.group("key").strip()
            val = m.group("val").strip()
            if val == "" or val in ("|", ">"):
                current_key = key
                current_list = []
            else:
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                meta[key] = val
                current_key = None
                current_list = None
        else:
            m2 = LIST_ITEM_RE.match(line)
            if m2 and current_key is not None:
                item = m2.group(1).strip()
                if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                    item = item[1:-1]
                if current_list is None:
                    current_list = []
                current_list.append(item)
        i += 1
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
    if len(text) <= max_len:
        return [text]
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
    out: List[str] = []
    for p in parts:
        if len(p) <= max_len:
            out.append(p)
        else:
            for i in range(0, len(p), max_len):
                out.append(p[i : i + max_len])
    return out


def chunk_markdown(md_body: str, max_text_len: int = 7000) -> List[Chunk]:
    lines = md_body.splitlines()
    section_stack: List[str] = []
    chunks: List[Chunk] = []
    chunk_idx = 0

    i = 0
    block_lines: List[str] = []

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
            for part in _smart_split(para_text, max_text_len):
                chunks.append(Chunk(text=part, section=section_title, chunk_index=chunk_idx, image_refs=imgs))
                chunk_idx += 1

    while i < len(lines):
        m = HEADING_RE.match(lines[i])
        if m:
            if block_lines:
                process_block(block_lines, current_section())
                block_lines = []
            level = len(m.group("h"))
            title = m.group("title").strip()
            while len(section_stack) >= level:
                section_stack.pop()
            section_stack.append(title)
        else:
            block_lines.append(lines[i])
        i += 1
    if block_lines:
        process_block(block_lines, current_section())
    return chunks


# --- Embeddings (Gemini) ---

def get_genai_client():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    from google import genai
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

def milvus_connect_or_create_db(host: str, port: int, db_name: str):
    """Connect to a Milvus database, creating it if supported and missing.

    Avoids initial connect with db_name to reduce RPC error logs on servers without DB support.
    """
    from pymilvus import connections
    global _NAMED_DB_SUPPORTED
    # Preflight on default connection for DB API support and creation
    try:
        connections.connect(alias="bootstrap", host=host, port=str(port))
        try:
            from pymilvus import db
            list_dbs = getattr(db, "list_databases", None) or getattr(db, "list_database", None)
            create_db = getattr(db, "create_database", None)
            if callable(list_dbs) and callable(create_db):
                try:
                    if db_name not in (list_dbs() or []):
                        create_db(db_name)
                except Exception:
                    # Ignore and fallback later
                    pass
            else:
                # No DB API; connect to default only
                try:
                    connections.disconnect("bootstrap")
                except Exception:
                    pass
                connections.connect(alias="default", host=host, port=str(port))
                print("[warn] Milvus named databases unsupported; using default database.")
                _NAMED_DB_SUPPORTED = False
                return
        finally:
            try:
                connections.disconnect("bootstrap")
            except Exception:
                pass
        # Try to connect to the requested DB now
        try:
            connections.connect(alias="default", host=host, port=str(port), db_name=db_name)
            _NAMED_DB_SUPPORTED = True
            return
        except Exception:
            # Fallback to default
            connections.connect(alias="default", host=host, port=str(port))
            print("[warn] Milvus server/client couldn't open named database; using default database.")
            _NAMED_DB_SUPPORTED = False
            return
    except Exception:
        # As a last resort, connect to default
        connections.connect(alias="default", host=host, port=str(port))
        print("[warn] Milvus connection fallback to default database.")
        if _NAMED_DB_SUPPORTED is None:
            _NAMED_DB_SUPPORTED = False


def milvus_connect_db_with_fallback(host: str, port: int, db_name: str) -> bool:
    """Connect to a specific DB if supported, else to the default DB.

    Returns True if connected to the named DB, False if fell back to default.
    """
    from pymilvus import connections
    global _NAMED_DB_SUPPORTED
    if _NAMED_DB_SUPPORTED is False:
        # Skip trying named DB connects entirely
        try:
            connections.connect(alias="default", host=host, port=str(port))
        except Exception:
            pass
        return False
    # Preflight using default to avoid RPC errors
    try:
        connections.connect(alias="bootstrap", host=host, port=str(port))
        try:
            from pymilvus import db
            list_dbs = getattr(db, "list_databases", None) or getattr(db, "list_database", None)
            create_db = getattr(db, "create_database", None)
            if callable(list_dbs) and callable(create_db):
                try:
                    if db_name not in (list_dbs() or []):
                        create_db(db_name)
                except Exception:
                    pass
                # Try named DB connect
                try:
                    connections.connect(alias="default", host=host, port=str(port), db_name=db_name)
                    _NAMED_DB_SUPPORTED = True
                    return True
                except Exception:
                    connections.connect(alias="default", host=host, port=str(port))
                    print(f"[warn] Could not open DB '{db_name}'; using default DB.")
                    _NAMED_DB_SUPPORTED = False
                    return False
            else:
                # No DB API
                connections.connect(alias="default", host=host, port=str(port))
                print(f"[warn] Named databases unsupported; using default DB for '{db_name}'.")
                _NAMED_DB_SUPPORTED = False
                return False
        finally:
            try:
                connections.disconnect("bootstrap")
            except Exception:
                pass
    except Exception:
        connections.connect(alias="default", host=host, port=str(port))
        print(f"[warn] Milvus connection fallback to default DB for '{db_name}'.")
        if _NAMED_DB_SUPPORTED is None:
            _NAMED_DB_SUPPORTED = False
        return False


def ensure_database(db_name: str):
    """Deprecated: database creation is handled in milvus_connect_or_create_db."""
    return


def ensure_meta_collection(name: str):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
    created = False
    if utility.has_collection(name):
        coll = Collection(name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="citation_key", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="journal", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="issued", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
            # Some Milvus/pymilvus versions require at least one vector field in a collection.
            # Add a tiny dummy vector to satisfy schema requirements; not used for search.
            # Milvus requires dim in [2, 32768]
            FieldSchema(name="vector_meta", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields, description="Papers metadata (one row per paper)")
        coll = Collection(name, schema)
        created = True
    # Ensure an index exists for vector_meta so that load() works on some Milvus versions
    try:
        coll.create_index(
            field_name="vector_meta",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        )
    except Exception:
        # Ignore if index already exists or server doesn't require it
        pass
    return coll


def ensure_chunks_collection(name: str, dim: int):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
    if utility.has_collection(name):
        coll = Collection(name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="paper_id", dtype=DataType.INT64),
            FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="citation_key", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="image_refs", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="Paper chunks with Gemini embeddings")
        coll = Collection(name, schema)
        coll.create_index(
            field_name="vector",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        )
    return coll


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def paper_exists(meta_coll, doi: str, citation_key: str) -> bool:
    expr_parts: List[str] = []
    def esc(s: str) -> str:
        return s.replace('"', '\\"')
    if doi:
        expr_parts.append(f'doi == "{esc(doi)}"')
    if citation_key:
        expr_parts.append(f'citation_key == "{esc(citation_key)}"')
    if not expr_parts:
        return False
    expr = " or ".join(expr_parts)
    try:
        # Ensure collection is loaded before query
        try:
            meta_coll.load()
        except Exception:
            pass
        res = meta_coll.query(expr=expr, output_fields=["id"], limit=1)
        return bool(res)
    except Exception:
        return False


def insert_paper_meta(meta_coll, doi: str, citation_key: str, title: str, journal: str, issued: str, url: str, source_path: str) -> int:
    data = [
        [doi],
        [citation_key],
        [title],
        [journal],
        [issued],
        [url],
    [source_path],
    [[0.0, 0.0]],  # vector_meta dummy (dim=2)
    ]
    # Order must match schema without PK
    mr = meta_coll.insert(data)
    meta_coll.flush()
    # Prefer primary keys from mutation result when available
    try:
        pks = getattr(mr, "primary_keys", None)
        if pks and len(pks) > 0:
            return int(pks[0])
    except Exception:
        pass
    # Fallback: query to fetch the id (ensure collection is loaded)
    try:
        meta_coll.load()
    except Exception:
        pass
    expr = ""
    if doi:
        expr = f'doi == "{doi.replace("\"", "\\\"")}"'
    else:
        expr = f'citation_key == "{citation_key.replace("\"", "\\\"")}"'
    res = meta_coll.query(expr=expr, output_fields=["id"], limit=1)
    return int(res[0]["id"]) if res else -1


def insert_chunks(chunks_coll, paper_id: int, doi: str, citation_key: str, chunks: List[Chunk], vectors: List[List[float]], insert_batch: int = 256):
    assert len(chunks) == len(vectors)
    cols = {
        "paper_id": [],
        "doi": [],
        "citation_key": [],
        "section": [],
        "chunk_index": [],
        "hash": [],
        "image_refs": [],
        "text": [],
        "vector": [],
    }
    for ch, vec in zip(chunks, vectors):
        cols["paper_id"].append(int(paper_id))
        cols["doi"].append(doi)
        cols["citation_key"].append(citation_key)
        cols["section"].append(ch.section)
        cols["chunk_index"].append(int(ch.chunk_index))
        cols["hash"].append(_sha256(ch.text))
        cols["image_refs"].append("|".join(ch.image_refs) if ch.image_refs else "")
        cols["text"].append(ch.text)
        cols["vector"].append(vec)
    # Insert in batches
    n = len(cols["text"])
    for i in range(0, n, insert_batch):
        batch = [
            cols["paper_id"][i:i+insert_batch],
            cols["doi"][i:i+insert_batch],
            cols["citation_key"][i:i+insert_batch],
            cols["section"][i:i+insert_batch],
            cols["chunk_index"][i:i+insert_batch],
            cols["hash"][i:i+insert_batch],
            cols["image_refs"][i:i+insert_batch],
            cols["text"][i:i+insert_batch],
            cols["vector"][i:i+insert_batch],
        ]
        chunks_coll.insert(batch)
    chunks_coll.flush()


def discover_rag_files() -> List[Path]:
    base = ROOT / "output" / "papers"
    out: List[Path] = []
    if not base.exists():
        return out
    for key_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        md_dir = key_dir / "md_with_images"
        if not md_dir.exists():
            continue
        out.extend(md for md in md_dir.glob("*-RAG.md"))
    return out


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Index -RAG.md into Milvus (topic-aware: metadata in meta DB; chunks in per-topic DB)")
    ap.add_argument("--collection", default="paper_chunks", help="Chunks collection name")
    ap.add_argument("--meta-collection", default="papers_meta", help="Papers metadata collection name")
    ap.add_argument("--milvus-host", default="127.0.0.1")
    ap.add_argument("--milvus-port", type=int, default=19530)
    ap.add_argument("--db-name", default="journal_papers", help="Default chunks DB name when no topic is specified")
    ap.add_argument("--meta-db-name", default="journal_papers", help="Metadata DB name where papers_meta lives")
    ap.add_argument("--embed-model", default="gemini-embedding-001")
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--insert-batch", type=int, default=256)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-prepend-section", action="store_false", dest="prepend_section")
    ap.set_defaults(prepend_section=True)
    ap.add_argument("--file", default=None, help="Index a single -RAG.md file")
    ap.add_argument("--force-reindex-chunks", action="store_true", help="When a paper already exists in metadata, reinsert chunks into the target collection (do not duplicate metadata)")
    args = ap.parse_args(argv)

    files = [Path(args.file)] if args.file else discover_rag_files()
    if not files:
        print("No -RAG.md files found to index.")
        return

    # Load registry to map citation_key -> topic
    reg_path = ROOT / "input" / "input_pdf.json"
    reg_map: Dict[str, Dict[str, Any]] = {}
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        for r in reg:
            k = str(r.get("citation_key") or "").strip()
            if k:
                reg_map[k] = r
    except Exception:
        pass

    # If dry-run, just preview chunking without touching Milvus
    if args.dry_run:
        for md_path in files:
            raw = md_path.read_text(encoding="utf-8")
            meta, body = parse_front_matter(raw)
            chunks = chunk_markdown(body)
            print(f"Parsed chunks: {len(chunks)} from {md_path}")
            if args.show > 0:
                for ch in chunks[:args.show]:
                    print(f"[{ch.chunk_index}] Section: {ch.section}")
                    if ch.image_refs:
                        print(f"  Images: {', '.join(ch.image_refs)}")
                    print(ch.text[:200].replace('\n', ' ') + ("..." if len(ch.text) > 200 else ""))
                    print()
        return

    # Helper: get key from path and topic from registry
    def _key_from_rag_path(p: Path) -> str:
        return p.parent.parent.name

    indexed = 0
    for md_path in files:
        raw = md_path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        doi = (meta.get("doi") or meta.get("DOI") or "").strip()
        citation_key = (meta.get("citation_key") or "").strip()
        title = (meta.get("title") or "").strip()
        journal = (meta.get("journal") or "").strip()
        issued = (meta.get("issued") or "").strip()
        url = (meta.get("url") or "").strip()

        # Connect to meta DB and ensure meta collection
        milvus_connect_or_create_db(args.milvus_host, args.milvus_port, db_name=args.meta_db_name)
        from pymilvus import Collection  # local import after connection
        meta_coll = ensure_meta_collection(args.meta_collection)
        try:
            meta_coll.load()
        except Exception:
            pass

        # Handle non-journal PDFs (no DOI): assign a stable synthetic document ID and mock citation_key
        def _is_real_doi(s: str) -> bool:
            return bool(s) and s.startswith("10.")
        if not _is_real_doi(doi):
            # Use content hash of the RAG body as synthetic ID so it stays stable across runs
            try:
                synthetic = "doc:" + _sha256(body)[:16]
            except Exception:
                synthetic = "doc:" + _sha256(str(md_path))[:16]
            doi = synthetic
            if not citation_key:
                # Create a mock citation key based on filename and hash suffix
                base = md_path.stem.lower()
                base = re.sub(r"[^a-z0-9]+", "", base) or "doc"
                citation_key = f"{base}{synthetic[-8:]}"
            if not title:
                title = md_path.stem
            if not journal:
                journal = "grey-literature"
            if not url:
                url = f"file://{md_path}"

        existing_paper_id: Optional[int] = None
        if paper_exists(meta_coll, doi, citation_key):
            if not args.force_reindex_chunks:
                print(f"[skip] Already indexed: doi={doi or '-'} key={citation_key or '-'}")
                continue
            # Query existing paper_id to avoid duplicating metadata
            try:
                try:
                    meta_coll.load()
                except Exception:
                    pass
                expr = f'doi == "{doi.replace("\"", "\\\"")}"' if doi else f'citation_key == "{citation_key.replace("\"", "\\\"")}"'
                res = meta_coll.query(expr=expr, output_fields=["id"], limit=1)
                if res:
                    existing_paper_id = int(res[0]["id"])  # type: ignore[index]
            except Exception:
                existing_paper_id = None

        chunks = chunk_markdown(body)
        print(f"Parsed chunks: {len(chunks)} from {md_path}")
        if args.show > 0:
            for ch in chunks[:args.show]:
                print(f"[{ch.chunk_index}] Section: {ch.section}")
                if ch.image_refs:
                    print(f"  Images: {', '.join(ch.image_refs)}")
                print(ch.text[:200].replace('\n', ' ') + ("..." if len(ch.text) > 200 else ""))
                print()

        texts = [(f"{c.section}\n\n{c.text}" if args.prepend_section and c.section else c.text) for c in chunks]
        vectors = embed_texts(texts, model=args.embed_model, batch_size=args.embed_batch)
        if not vectors:
            print("[warn] No embeddings produced; skipping file")
            continue
        dim = len(vectors[0])
        # Determine target DB for chunks: topic db or default
        key = _key_from_rag_path(md_path)
        topic = (reg_map.get(key, {}).get("topic") or "").strip()
        target_db = topic if topic else args.db_name
        # If named DBs unsupported, fallback to suffixed collection name
        db_supported = milvus_connect_db_with_fallback(args.milvus_host, args.milvus_port, db_name=target_db)
        effective_collection = args.collection if db_supported else f"{args.collection}__{target_db}"
        chunks_coll = ensure_chunks_collection(effective_collection, dim=dim)

        # Insert meta first to get paper_id (meta DB) unless reusing existing
        milvus_connect_or_create_db(args.milvus_host, args.milvus_port, db_name=args.meta_db_name)
        meta_coll = ensure_meta_collection(args.meta_collection)
        if existing_paper_id is not None:
            paper_id = existing_paper_id
        else:
            paper_id = insert_paper_meta(
                meta_coll,
                doi=doi,
                citation_key=citation_key,
                title=title,
                journal=journal,
                issued=issued,
                url=url,
                source_path=str(md_path),
            )
        # Insert chunks in target DB
        if db_supported:
            milvus_connect_or_create_db(args.milvus_host, args.milvus_port, db_name=target_db)
        else:
            milvus_connect_or_create_db(args.milvus_host, args.milvus_port, db_name=args.db_name)
        chunks_coll = ensure_chunks_collection(effective_collection, dim=dim)
        insert_chunks(chunks_coll, paper_id=paper_id, doi=doi, citation_key=citation_key, chunks=chunks, vectors=vectors, insert_batch=args.insert_batch)
        indexed += 1
        loc = f"DB='{target_db}'" if db_supported else f"collection='{effective_collection}' (default DB)"
        print(f"[ok] Indexed {len(chunks)} chunks for paper_id={paper_id} into {loc}")

    if indexed == 0:
        print("No new papers indexed.")


if __name__ == "__main__":
    main()
