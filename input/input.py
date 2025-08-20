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
except Exception:  # When run as a script: python input/input.py
	from check_pdf import check_pdf, normalize_doi  # type: ignore


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT
JSON_PATH = ROOT / "input_pdf.json"


def normalize_title(s: str) -> str:
	# Remove extension-like tails and normalize separators/whitespace
	name = s
	if "." in name:
		name = name.rsplit(".", 1)[0]
	name = name.replace("_", " ").replace("-", " ")
	name = " ".join(name.split()).strip().casefold()
	return name


def safe_filename_from_key(key: str) -> str:
	# Citation keys are expected to be [a-z0-9], but sanitize anyway
	s = "".join(ch for ch in key.lower() if ch.isalnum())
	if not s:
		s = "key"
	if len(s) > 80:
		s = s[:80]
	return s


def unique_path(base_dir: Path, stem: str, ext: str = ".pdf") -> Path:
	cand = base_dir / f"{stem}{ext}"
	if not cand.exists():
		return cand
	i = 1
	while True:
		cand = base_dir / f"{stem} ({i}){ext}"
		if not cand.exists():
			return cand
		i += 1


def load_db(path: Path) -> List[Dict[str, Any]]:
	if not path.exists() or path.stat().st_size == 0:
		return []
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
		if isinstance(data, list):
			return data
		# If object with items array, allow it; else fallback
		if isinstance(data, dict) and isinstance(data.get("items"), list):
			return data["items"]
	except Exception:
		pass
	return []


def save_db(path: Path, items: List[Dict[str, Any]]) -> None:
	path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def find_by_title(items: List[Dict[str, Any]], title: str) -> Optional[Dict[str, Any]]:
	key = normalize_title(title)
	for obj in items:
		t = str(obj.get("title", ""))
		if normalize_title(t) == key:
			return obj
	return None


def find_by_doi(items: List[Dict[str, Any]], doi: Optional[str]) -> Optional[Dict[str, Any]]:
	if not doi:
		return None
	try:
		key = normalize_doi(str(doi))
	except Exception:
		key = str(doi).strip()
	for obj in items:
		v = obj.get("doi")
		if not v:
			continue
		try:
			vd = normalize_doi(str(v))
		except Exception:
			vd = str(v).strip()
		if vd.casefold() == key.casefold():
			return obj
	return None


def find_by_key(items: List[Dict[str, Any]], key: Optional[str]) -> Optional[Dict[str, Any]]:
	if not key:
		return None
	k = str(key).strip().lower()
	for obj in items:
		v = str(obj.get("citation_key", "")).strip().lower()
		if v and v == k:
			return obj
	return None


def main() -> None:
	items = load_db(JSON_PATH)
	changed = False

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
		existing_d = find_by_doi(items, res.doi)
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

