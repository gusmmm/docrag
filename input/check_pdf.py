from __future__ import annotations

"""
No-LLM PDF checker: extract DOI and Title from a scientific journal article PDF.

Strategy
- Try PDF metadata (XMP) for title.
- Scan first 1–2 pages of text for a DOI via regex; normalize it.
- If DOI is found, call Crossref to fetch authoritative title.
- If no DOI, heuristically pick the largest-font line on page 1 as the title.
- Optional: OCR page 1 if the PDF is scanned (requires pytesseract + Pillow).

Outputs
- Prints informative messages to the console (what was detected and how).
- Emits final JSON: {"title": str, "doi": str}
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
import fitz  # PyMuPDF

# DOI bibliographic client (Crossref → CSL-JSON)
try:  # when running as module: python -m input.check_pdf
	from src.DOI_lookup import DoiBibliographyClient  # type: ignore
except Exception:  # fallback when running as script from repo root
	try:
		from DOI_lookup import DoiBibliographyClient  # type: ignore
	except Exception:
		DoiBibliographyClient = None  # type: ignore


DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def normalize_doi(raw: str) -> str:
	s = raw.strip()
	# Strip URL prefixes if present
	s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
	# Remove surrounding punctuation
	s = s.strip(" \t\r\n.,;)>")
	# Lowercase doi except keep original slash parts
	return s


def extract_pdf_metadata(doc: fitz.Document) -> Dict[str, Optional[str]]:
	md = doc.metadata or {}
	title = md.get("title") or md.get("Title") or None
	# DOI is rarely in metadata; keep placeholder None
	return {"title": title, "doi": None}


def first_pages_text(doc: fitz.Document, pages: int = 2) -> str:
	n = min(pages, doc.page_count)
	parts = []
	for i in range(n):
		try:
			txt = doc.load_page(i).get_text()
			parts.append(txt or "")
		except Exception:
			continue
	return "\n".join(parts)


def find_doi_in_text(text: str) -> Optional[str]:
	m = DOI_REGEX.search(text or "")
	if not m:
		return None
	return normalize_doi(m.group(0))


def first_page_largest_font_title(doc: fitz.Document) -> Optional[str]:
	if doc.page_count == 0:
		return None
	page = doc.load_page(0)
	try:
		data = page.get_text("dict")
	except Exception:
		return None
	best: Tuple[float, str] = (0.0, "")
	for block in data.get("blocks", []):
		for line in block.get("lines", []):
			for span in line.get("spans", []):
				text = (span.get("text") or "").strip()
				size = float(span.get("size") or 0)
				if not text or len(text) < 5:
					continue
				# Skip lines likely not titles
				lower = text.lower()
				if any(k in lower for k in ["doi:", "www.", "http://", "https://", "copyright"]):
					continue
				if size > best[0]:
					best = (size, text)
	return best[1] or None


def render_page1_and_ocr(doc: fitz.Document) -> Optional[str]:
	try:
		import pytesseract  # type: ignore
		from PIL import Image  # type: ignore
	except Exception:
		return None
	if doc.page_count == 0:
		return None
	page = doc.load_page(0)
	pix = page.get_pixmap(dpi=200)
	img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
	try:
		return pytesseract.image_to_string(img)
	except Exception:
		return None


def crossref_title_for_doi(doi: str, timeout: float = 8.0) -> Optional[str]:
	url = f"https://api.crossref.org/works/{doi}"
	headers = {
		"User-Agent": os.getenv(
			"CROSSREF_UA",
			"docrag/0.1 (+https://example.org; mailto:you@example.org)",
		)
	}
	try:
		with httpx.Client(headers=headers, timeout=timeout) as client:
			resp = client.get(url)
			if resp.status_code != 200:
				return None
			js = resp.json()
			msg = (js or {}).get("message", {})
			titles = msg.get("title") or []
			if titles:
				return titles[0]
	except Exception:
		return None
	return None


@dataclass
class PDFCheckResult:
	title: str
	doi: str
	title_source: str
	doi_source: str
	csl: Optional[Dict[str, Any]]
	citation_key: str


def check_pdf(path: Path, pages_to_scan: int = 2, use_ocr: bool = False) -> PDFCheckResult:
	doc = fitz.open(path)

	# 1) Metadata
	meta = extract_pdf_metadata(doc)
	meta_title = meta.get("title")

	# 2) Extract text from first pages (or OCR if requested)
	text = first_pages_text(doc, pages=pages_to_scan)
	if use_ocr and (not text or len(text.strip()) < 40):
		ocr_text = render_page1_and_ocr(doc)
		if ocr_text:
			text = (text + "\n" + ocr_text).strip()

	# 3) Find DOI
	doi = find_doi_in_text(text)
	doi_source = "first_pages_regex" if doi else "none"

	# 4) Title resolution (and CSL metadata via Crossref when DOI exists)
	title = None
	title_source = "none"
	csl: Optional[Dict[str, Any]] = None
	if doi:
		# Prefer full CSL-JSON via Crossref (authoritative title, authors, year, etc.)
		if DoiBibliographyClient is not None:
			try:
				csl = DoiBibliographyClient().fetch_csl(doi)
				if isinstance(csl, dict):
					csl_title = csl.get("title")
					if isinstance(csl_title, list):
						csl_title = csl_title[0] if csl_title else None
					if csl_title:
						title, title_source = str(csl_title), "crossref_csl"
			except Exception:
				# Fall back to direct Crossref title endpoint if CSL fetch fails
				csl = None
		if not title:
			cr_title = crossref_title_for_doi(doi)
			if cr_title:
				title, title_source = cr_title, "crossref"
	if not title and meta_title:
		title, title_source = meta_title, "pdf_metadata"
	if not title:
		heuristic = first_page_largest_font_title(doc)
		if heuristic:
			title, title_source = heuristic, "largest_font_heuristic"

	if not title:
		title = "Unknown"
	if not doi:
		doi = "N/A"

	# 5) Build a robust citation key (Better BibTeX-style: authorYearShortTitle)
	citation_key = _build_citation_key(csl=csl, fallback_title=title)

	return PDFCheckResult(
		title=title,
		doi=doi,
		title_source=title_source,
		doi_source=doi_source,
		csl=csl,
		citation_key=citation_key,
	)


# --- Citation key helpers ---
STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "with"}


def _slug(s: str, max_len: int = 30) -> str:
	s = s.lower()
	s = re.sub(r"[^a-z0-9]+", " ", s)
	s = "".join(part for part in s.split())  # collapse to alphanum contiguous
	if len(s) > max_len:
		s = s[:max_len]
	return s or "key"


def _year_from_csl(csl: Optional[Dict[str, Any]]) -> str:
	if not csl:
		return "0000"
	try:
		for field in ("issued", "published-print", "published-online", "created", "deposited"):
			part = csl.get(field)
			if isinstance(part, dict):
				dp = part.get("date-parts") or part.get("'date-parts")
				if isinstance(dp, list) and dp and isinstance(dp[0], list) and dp[0]:
					y = dp[0][0]
					if isinstance(y, int):
						return str(y)
	except Exception:
		pass
	return "0000"


def _first_author_family(csl: Optional[Dict[str, Any]]) -> str:
	if not csl:
		return "anon"
	try:
		authors = csl.get("author") or []
		if isinstance(authors, list) and authors:
			a0 = authors[0]
			fam = a0.get("family") or a0.get("last") or a0.get("surname")
			if isinstance(fam, str) and fam.strip():
				return re.sub(r"[^A-Za-z0-9]", "", fam.strip().lower()) or "anon"
	except Exception:
		pass
	# Fallback to publisher or container-title if no authors
	for k in ("publisher", "container-title", "institution"):
		v = csl.get(k) if csl else None
		if isinstance(v, list):
			v = v[0] if v else None
		if isinstance(v, str) and v.strip():
			return re.sub(r"[^A-Za-z0-9]", "", v.strip().split()[0].lower()) or "anon"
	return "anon"


def _short_title_component(title: str, max_len: int = 20) -> str:
	words = re.findall(r"[A-Za-z0-9]+", (title or "").lower())
	words = [w for w in words if w not in STOPWORDS]
	if not words:
		return "art"
	s = "".join(words)  # concatenate words
	return s[:max_len] if len(s) > max_len else s


def _build_citation_key(csl: Optional[Dict[str, Any]], fallback_title: str) -> str:
	fam = _first_author_family(csl)
	year = _year_from_csl(csl)
	title = None
	if csl:
		t = csl.get("title")
		if isinstance(t, list):
			t = t[0] if t else None
		if isinstance(t, str):
			title = t
	if not title:
		title = fallback_title or "Untitled"
	short = _short_title_component(title, max_len=24)
	key = f"{fam}{year}{short}"
	key = re.sub(r"[^a-z0-9]", "", key.lower())
	return key or _slug(title)


def _cli():
	import argparse

	parser = argparse.ArgumentParser(description="Check a PDF and extract Title and DOI without LLMs")
	parser.add_argument("pdf", help="Path to PDF file")
	parser.add_argument("--pages", type=int, default=2, help="Number of initial pages to scan for DOI (default 2)")
	parser.add_argument("--ocr", action="store_true", help="Try OCR on page 1 if text is minimal")
	args = parser.parse_args()

	pdf_path = Path(args.pdf)
	if not pdf_path.exists():
		print(f"[error] File not found: {pdf_path}")
		return

	res = check_pdf(pdf_path, pages_to_scan=args.pages, use_ocr=args.ocr)
	print(f"PDF: {pdf_path.name}")
	print(f"DOI: {res.doi} (source: {res.doi_source})")
	print(f"Title: {res.title} (source: {res.title_source})")
	print("JSON:")
	payload = {
		"title": res.title,
		"doi": res.doi,
		"citation_key": res.citation_key,
		"csl": res.csl,
	}
	print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	_cli()

