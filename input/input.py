from __future__ import annotations

"""
Input PDF orchestrator

Behavior
- List all PDFs in input/ folder.
- Prefer citation-key-based deduplication and filenames.
- For each PDF, if its stem equals an existing citation_key, treat as existing.
- Else, run extractor (check_pdf.check_pdf) → title, doi, citation_key, csl.
- Rename the PDF to the citation_key.pdf (unique when necessary).
- Deduplicate by citation_key first, then DOI. Merge in missing fields when possible.

JSON format: list of objects like:
{
	"citation_key": str,
	"title": str,
	"doi": str,
	"csl": { ... CSL-JSON ... }
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


def _ensure_pdf_dir() -> None:
	INPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
	_ensure_pdf_dir()
	items = load_db(JSON_PATH)
	changed = False

	# Also move stray PDFs in input/ into input/pdf
	stray = [p for p in ROOT.glob("*.pdf")]
	for p in stray:
		target = (INPUT_DIR / p.name)
		if not target.exists():
			try:
				p.rename(target)
				print(f"Moved stray PDF to pdf/: {p.name}")
			except Exception:
				pass

	pdfs = sorted(p for p in INPUT_DIR.glob("*.pdf"))
	if not pdfs:
		print("No PDFs found in input/.")
		return

	for pdf in pdfs:
		stem = pdf.stem.strip()
		# If the filename already looks like a citation key, check by key first
		existing_by_key = find_by_key(items, stem)
		if existing_by_key:
			print(f"Found existing entry for '{pdf.name}' by citation_key match:")
			print(json.dumps(existing_by_key, indent=2, ensure_ascii=False))
			continue

		print(f"Extracting from: {pdf.name} ...")
		res = check_pdf(pdf)

		# Compose record
		record: Dict[str, Any] = {
			"citation_key": res.citation_key,
			"title": res.title,
			"doi": res.doi,
			"csl": res.csl,
		}

		# Rename PDF to citation_key
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

