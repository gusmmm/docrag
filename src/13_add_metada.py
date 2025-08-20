from __future__ import annotations

"""
Step 4: Add bibliographic metadata (YAML front matter) to each -RAG.md using input/input_pdf.json.

For each file output/papers/<key>/md_with_images/*-RAG.md:
- Look up <key> in input/input_pdf.json.
- Build a YAML header from the record (title, authors, doi, journal, volume, issue, pages, issued, citation_key, url).
- Prepend YAML to the Markdown (replace any existing YAML block if present).

Usage:
  uv run python src/13_add_metada.py            # process all -RAG.md files
  uv run python src/13_add_metada.py --file <path-to-RAG.md>
"""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


def _load_registry() -> List[Dict[str, Any]]:
	path = ROOT / "input" / "input_pdf.json"
	if not path.exists():
		raise FileNotFoundError(f"Registry not found: {path}")
	return json.loads(path.read_text(encoding="utf-8"))


def _key_from_rag_path(p: Path) -> str:
	# .../output/papers/<key>/md_with_images/<stem>-RAG.md
	return p.parent.parent.name


def _strip_existing_front_matter(md_text: str) -> Tuple[str, bool]:
	lines = md_text.splitlines()
	if len(lines) >= 3 and lines[0].strip() == "---":
		for i in range(1, len(lines)):
			if lines[i].strip() == "---":
				body = "\n".join(lines[i + 1 :])
				if md_text.endswith("\n"):
					body += "\n"
				return (body, True)
	return (md_text, False)


def _yaml_escape(value: str) -> str:
	return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _csl_iso_date(csl: Dict[str, Any]) -> str:
	try:
		issued = csl.get("issued") or {}
		parts = issued.get("date-parts") or issued.get("date_parts")
		if not parts:
			return ""
		arr = parts[0]
		y = int(arr[0])
		m = int(arr[1]) if len(arr) > 1 else 1
		d = int(arr[2]) if len(arr) > 2 else 1
		return date(y, m, d).isoformat()
	except Exception:
		return ""


def _format_authors_from_csl(csl: Dict[str, Any]) -> List[str]:
	out: List[str] = []
	for a in csl.get("author", []) or []:
		if isinstance(a, dict):
			given = (a.get("given") or "").strip()
			family = (a.get("family") or "").strip()
			full = (a.get("full") or "").strip()
			name = full or (f"{given} {family}".strip())
			if name:
				out.append(name)
	return out


def build_yaml_from_record(rec: Dict[str, Any]) -> str:
	title = rec.get("title") or rec.get("csl", {}).get("title", "")
	doi = rec.get("doi") or rec.get("csl", {}).get("DOI", "")
	csl = rec.get("csl", {}) or {}
	journal = csl.get("container-title", "")
	volume = csl.get("volume", "")
	issue = csl.get("issue", "")
	pages = csl.get("page", "") or csl.get("pages", "")
	url = csl.get("URL", "")
	issued_iso = _csl_iso_date(csl)
	authors = _format_authors_from_csl(csl)
	key = rec.get("citation_key", "")

	lines: List[str] = ["---"]
	if title:
		lines.append(f"title: {_yaml_escape(title)}")
	if authors:
		lines.append("authors:")
		for a in authors[:50]:  # keep list bounded
			lines.append(f"  - {_yaml_escape(a)}")
	if doi:
		lines.append(f"doi: {_yaml_escape(doi)}")
	if key:
		lines.append(f"citation_key: {_yaml_escape(key)}")
	if journal:
		lines.append(f"journal: {_yaml_escape(journal)}")
	if volume:
		lines.append(f"volume: {_yaml_escape(volume)}")
	if issue:
		lines.append(f"issue: {_yaml_escape(issue)}")
	if pages:
		lines.append(f"pages: {_yaml_escape(pages)}")
	if issued_iso:
		lines.append(f"issued: {_yaml_escape(issued_iso)}")
	if url:
		lines.append(f"url: {_yaml_escape(url)}")
	# provenance
	lines.append(f"source_registry: {_yaml_escape(str((ROOT / 'input' / 'input_pdf.json').as_posix()))}")
	lines.append("---")
	return "\n".join(lines) + "\n\n"


def _find_record_by_key(reg: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
	for r in reg:
		if r.get("citation_key") == key:
			return r
	return None


def process_file(md_path: Path, registry: List[Dict[str, Any]]) -> bool:
	if not md_path.exists():
		print(f"[warn] Not found: {md_path}")
		return False
	key = _key_from_rag_path(md_path)
	rec = _find_record_by_key(registry, key)
	if not rec:
		print(f"[skip] No registry entry for key={key} -> {md_path}")
		return False
	text = md_path.read_text(encoding="utf-8", errors="ignore")
	body, _ = _strip_existing_front_matter(text)
	yaml_block = build_yaml_from_record(rec)
	md_path.write_text(yaml_block + body.lstrip("\n"), encoding="utf-8")
	print(f"[ok] Added metadata to: {md_path}")
	return True


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
	ap = argparse.ArgumentParser(description="Add YAML metadata to -RAG.md from input/input_pdf.json")
	ap.add_argument("--file", type=str, default=None, help="Process a single RAG Markdown path")
	args = ap.parse_args(argv)

	registry = _load_registry()
	files = [Path(args.file)] if args.file else discover_rag_files()
	if not files:
		print("No -RAG.md files found to update.")
		return
	updated = 0
	for p in files:
		if process_file(p, registry):
			updated += 1
	if updated == 0:
		print("No files updated.")


if __name__ == "__main__":
	main()

