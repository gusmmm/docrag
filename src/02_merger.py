"""
Merge citation metadata into a Markdown file (with image refs) and write a merged Markdown.

- Reads a citation JSON (as generated in output/citations/...-genai.json)
- Reads the Markdown with image refs (output/md_with_images/*.md)
- Prepends a YAML front matter block with bibliographic metadata
- Writes a new file alongside the MD: <original-stem>-merged.md

This merged MD is intended for downstream chunking/embedding/indexing in a RAG pipeline,
preserving image links and adding rich bibliographic context as structured YAML.

Usage (defaults match the example in this repo):
	uv run python src/02_indexing.py \
		--md output/md_with_images/jama_summers_2025-with-image-refs.md \
		--citations output/citations/jama_summers_2025-with-image-refs-genai.json

If no args are provided, the above defaults are used.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _yaml_escape(value: str) -> str:
	"""Escape a string for safe single-line YAML emission.

	Uses double quotes and escapes existing backslashes and quotes.
	"""
	escaped = value.replace("\\", "\\\\").replace('"', '\\"')
	return f'"{escaped}"'


def _format_authors(authors: List[Dict[str, Any]]) -> List[str]:
	"""Return a list of author display names, preferring 'full' then 'given family'."""
	display = []
	for a in authors or []:
		if isinstance(a, dict):
			name = a.get("full") or " ".join(
				[p for p in [a.get("given"), a.get("family")] if p]
			).strip()
			if name:
				display.append(name)
	return display


def _iso_date(issued: Dict[str, Any]) -> str:
	"""Build an ISO date string (YYYY-MM-DD) from CSL-like 'issued' mapping if possible."""
	try:
		y = int(issued.get("year"))
		m = int(issued.get("month", 1))
		d = int(issued.get("day", 1))
		return date(y, m, d).isoformat()
	except Exception:
		return ""


def build_yaml_front_matter(citation: Dict[str, Any], source_json: Path) -> str:
	"""Create a YAML front matter block from citation metadata.

	Only uses stdlib; avoids requiring PyYAML.
	"""
	title = citation.get("title", "")
	authors = _format_authors(citation.get("authors", []))
	doi = citation.get("doi", "")
	container_title = citation.get("container_title", "")
	volume = citation.get("volume", "")
	issue = citation.get("issue", "")
	pages = citation.get("pages", "")
	issued_iso = _iso_date(citation.get("issued", {}))

	lines: List[str] = ["---"]
	if title:
		lines.append(f"title: {_yaml_escape(title)}")
	if authors:
		lines.append("authors:")
		for a in authors:
			lines.append(f"  - {_yaml_escape(a)}")
	if doi:
		lines.append(f"doi: {_yaml_escape(doi)}")
	if container_title:
		lines.append(f"journal: {_yaml_escape(container_title)}")
	if volume:
		lines.append(f"volume: {_yaml_escape(volume)}")
	if issue:
		lines.append(f"issue: {_yaml_escape(issue)}")
	if pages:
		lines.append(f"pages: {_yaml_escape(pages)}")
	if issued_iso:
		lines.append(f"issued: {_yaml_escape(issued_iso)}")

	# Provenance
	lines.append(f"source_citation_json: {_yaml_escape(str(source_json))}")
	lines.append("---")
	return "\n".join(lines) + "\n\n"


def strip_existing_front_matter(md_text: str) -> Tuple[str, bool]:
	"""Remove an existing YAML front matter block if present.

	Returns (stripped_md, removed_bool)
	"""
	lines = md_text.splitlines()
	if len(lines) >= 3 and lines[0].strip() == "---":
		# find closing '---'
		for i in range(1, len(lines)):
			if lines[i].strip() == "---":
				return ("\n".join(lines[i + 1:]) + ("\n" if md_text.endswith("\n") else ""), True)
	return (md_text, False)


def merge_citation_into_md(md_path: Path, citation_json_path: Path, out_path: Path | None = None) -> Path:
	"""Merge citation metadata into the Markdown file and return the output path."""
	if not md_path.exists():
		raise FileNotFoundError(f"Markdown file not found: {md_path}")
	if not citation_json_path.exists():
		raise FileNotFoundError(f"Citation JSON not found: {citation_json_path}")

	with citation_json_path.open("r", encoding="utf-8") as f:
		citation = json.load(f)

	with md_path.open("r", encoding="utf-8") as f:
		md_text = f.read()

	md_body, removed = strip_existing_front_matter(md_text)
	yaml_block = build_yaml_front_matter(citation, citation_json_path)

	merged = yaml_block + md_body.lstrip("\n")

	if out_path is None:
		out_path = md_path.with_name(md_path.stem + "-merged.md")

	with out_path.open("w", encoding="utf-8") as f:
		f.write(merged)

	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Merge citation JSON into Markdown with image refs.")
	parser.add_argument(
		"--md",
		type=str,
		default="output/md_with_images/jama_summers_2025-with-image-refs.md",
		help="Path to the Markdown file with image refs.",
	)
	parser.add_argument(
		"--citations",
		type=str,
		default="output/citations/jama_summers_2025-with-image-refs-genai.json",
		help="Path to the citation JSON file to merge.",
	)
	parser.add_argument(
		"--out",
		type=str,
		default=None,
		help="Optional output path for the merged Markdown. Defaults to <md-stem>-merged.md",
	)
	args = parser.parse_args()

	md_path = Path(args.md)
	citation_json_path = Path(args.citations)
	out_path = Path(args.out) if args.out else None

	result_path = merge_citation_into_md(md_path, citation_json_path, out_path)
	# Print a simple success line for CLI users
	rel = os.path.relpath(result_path, Path.cwd())
	print(f"Merged file written to: {rel}")


if __name__ == "__main__":
	main()

