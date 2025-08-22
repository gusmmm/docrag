from __future__ import annotations

"""
Input PDF orchestrator

Behavior
- List PDFs in input/pdf/ and in topic folders input/topics/<topic>/.
- Prefer citation-key-based deduplication and filenames.
- For each PDF, if its stem equals an existing citation_key, treat as existing.
- Else, run extractor (check_pdf.check_pdf) → title, doi, citation_key, csl.
- Rename the PDF to citation_key.pdf within its directory (input/pdf/ or the topic subfolder).
- Deduplicate by citation_key first, then DOI. Merge in missing fields when possible.
- Records from topic folders are annotated with {"topic": "<topic>"} for topic-aware indexing.

JSON format: list of objects like:
{
	"citation_key": str,
	"title": str,
	"doi": str,
	"csl": { ... CSL-JSON ... },
	"topic": str | null
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
	# When run as a module: python -m input.input
	from input.check_pdf import check_pdf, normalize_doi  # type: ignore
	from input.utils import (
		load_db,
		save_db,
		find_by_key,
		find_by_doi,
		find_by_title,
		safe_filename_from_key,
		unique_path,
	)
except Exception:  # When run as a script: python input/input.py
	from check_pdf import check_pdf, normalize_doi  # type: ignore
	from utils import (
		load_db,
		save_db,
		find_by_key,
		find_by_doi,
		find_by_title,
		safe_filename_from_key,
		unique_path,
	)


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "pdf"
JSON_PATH = ROOT / "input_pdf.json"
TOPICS_DIR = ROOT / "topics"


def _ensure_pdf_dir() -> None:
	INPUT_DIR.mkdir(parents=True, exist_ok=True)
	TOPICS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
	_ensure_pdf_dir()
	items = load_db(JSON_PATH)
	changed = False

	# Also move stray PDFs in input/ into input/pdf (do not move topic PDFs)
	stray = [p for p in ROOT.glob("*.pdf")]
	for p in stray:
		target = (INPUT_DIR / p.name)
		if not target.exists():
			try:
				p.rename(target)
				print(f"Moved stray PDF to pdf/: {p.name}")
			except Exception:
				pass

	# Collect PDFs: standard input/pdf and topic folders input/topics/<topic>
	pdfs = sorted(p for p in INPUT_DIR.glob("*.pdf"))
	topic_pdfs: list[Path] = []
	if TOPICS_DIR.exists():
		for tdir in sorted(d for d in TOPICS_DIR.iterdir() if d.is_dir()):
			topic_pdfs.extend(sorted(p for p in tdir.glob("*.pdf")))
	all_pdfs = pdfs + topic_pdfs
	if not all_pdfs:
		print("No PDFs found in input/ or input/topics/.")
		return

	for pdf in all_pdfs:
		topic_name: str | None = None
		try:
			# Determine topic if under input/topics/<topic>
			rel = pdf.relative_to(TOPICS_DIR)
			topic_name = rel.parts[0] if len(rel.parts) >= 2 else None
		except Exception:
			topic_name = None
		stem = pdf.stem.strip()
		# If the filename already looks like a citation key, check by key first
		existing_by_key = find_by_key(items, stem)
		if existing_by_key:
			print(f"Found existing entry for '{pdf.name}' by citation_key match:")
			print(json.dumps(existing_by_key, indent=2, ensure_ascii=False))
			continue

		print(f"Extracting from: {pdf.name} ...")
		res = check_pdf(pdf)

		# If no real DOI was found, assign a stable synthetic one based on PDF content
		def _is_real_doi(s: str) -> bool:
			return bool(s) and s.startswith("10.")
		if not _is_real_doi(res.doi):
			try:
				import hashlib
				h = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
				synthetic_doi = f"doc:{h}"
			except Exception:
				synthetic_doi = "doc:unknown"
			# Patch result fields to propagate to registry and YAML later
			res.doi = synthetic_doi
			if res.title.lower() == "unknown":
				res.title = pdf.stem

		# Compose record
		record: Dict[str, Any] = {
			"citation_key": res.citation_key,
			"title": res.title,
			"doi": res.doi,
			"csl": res.csl,
			"topic": topic_name or "",
		}

		# Rename PDF to citation_key in-place within its directory
		key_stem = safe_filename_from_key(res.citation_key or stem)
		target = unique_path(pdf.parent, key_stem, ext=pdf.suffix)
		if target.name != pdf.name:
			try:
				pdf.rename(target)
				print(f"Renamed '{pdf.name}' -> '{target.name}'")
				pdf = target
			except Exception as e:
				print(f"[warn] Failed to rename '{pdf.name}': {e}")

		# Dedup by citation_key first
		existing_k = find_by_key(items, res.citation_key)
		if existing_k:
			# Merge missing fields (upgrade path for older entries)
			merged = False
			if not existing_k.get("doi") and res.doi and res.doi != "N/A":
				existing_k["doi"] = res.doi
				merged = True
			if not existing_k.get("title") and res.title and res.title.lower() != "unknown":
				existing_k["title"] = res.title
				merged = True
			if res.csl and not existing_k.get("csl"):
				existing_k["csl"] = res.csl
				merged = True
			# Annotate topic if provided and not already set
			if topic_name and not existing_k.get("topic"):
				existing_k["topic"] = topic_name
				merged = True
			if merged:
				changed = True
				print(f"Merged fields into existing entry for key '{res.citation_key}'.")
			else:
				print(f"Entry already exists for key '{res.citation_key}'.")
			continue

		# Then dedup by DOI (authoritative)
		existing_d = find_by_doi(items, res.doi, normalizer=normalize_doi)
		if existing_d:
			# If DOI exists but citation_key missing or different, attach/normalize
			if not existing_d.get("citation_key"):
				existing_d["citation_key"] = res.citation_key
				changed = True
			# Merge csl and better title
			if res.csl and not existing_d.get("csl"):
				existing_d["csl"] = res.csl
				changed = True
			if res.title and res.title.lower() != "unknown" and not existing_d.get("title"):
				existing_d["title"] = res.title
				changed = True
			if topic_name and not existing_d.get("topic"):
				existing_d["topic"] = topic_name
				changed = True
			print(f"Linked existing DOI entry to key '{res.citation_key}'.")
			continue

		# As a final fallback, attempt title-based match (for legacy entries)
		existing_t = find_by_title(items, res.title)
		if existing_t:
			if not existing_t.get("citation_key"):
				existing_t["citation_key"] = res.citation_key
				changed = True
			if res.csl and not existing_t.get("csl"):
				existing_t["csl"] = res.csl
				changed = True
			if res.doi and res.doi != "N/A" and not existing_t.get("doi"):
				existing_t["doi"] = res.doi
				changed = True
			if topic_name and not existing_t.get("topic"):
				existing_t["topic"] = topic_name
				changed = True
			print(f"Linked legacy title entry to key '{res.citation_key}'.")
			continue

		# New entry
		print(f"Adding new entry: {record}")
		items.append(record)
		changed = True

	if changed:
		save_db(JSON_PATH, items)
		print(f"Updated JSON: {JSON_PATH}")
	else:
		print("No updates needed; JSON unchanged.")


if __name__ == "__main__":
	main()

