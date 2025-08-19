"""DOI Bibliographic fetcher using Crossref.

Features:
- Fetch bibliographic metadata by DOI using Crossref API.
- Return data in CSL-JSON (Citation Style Language) open standard when available.
- Save results to `output/citations/{safe-doi}.json`.
- Simple CLI for ad-hoc use.

Notes:
- Primary source: Crossref REST API (https://api.crossref.org/works/{doi}).
- Fallback: DOI.org content negotiation for CSL-JSON.
- Rate limits: Be respectful; set a mailto contact header.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


CSL_CONTENT_TYPE = "application/vnd.citationstyles.csl+json"


def _safe_filename_from_doi(doi: str) -> str:
    # Replace unsafe path characters with '_'
    return re.sub(r"[^A-Za-z0-9._-]", "_", doi.strip().lower())


@dataclass
class DoiBibliographyClient:
    """Client to fetch and persist bibliographic metadata for a DOI.

    Contract:
    - input: DOI string (may include url prefix); optional output dir
    - output: dict representing CSL-JSON metadata, saved to file as well
    - errors: raises ValueError for invalid DOI; httpx.HTTPStatusError for HTTP failures
    """

    user_agent: str = "docrag/0.1 (mailto:{email})"
    contact_email: str = os.getenv("CONTACT_EMAIL", "example@example.com")
    timeout_s: float = 20.0
    citations_dir: Path = Path("output/citations")

    def __post_init__(self) -> None:
        self.citations_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_doi(doi: str) -> str:
        if not doi:
            raise ValueError("Empty DOI")
        doi = doi.strip()
        # Remove protocol/host if present
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        # Basic validation heuristic: must contain a '/'
        if "/" not in doi:
            raise ValueError(f"Invalid DOI format: {doi}")
        return doi

    def _headers(self, accept: Optional[str] = None) -> Dict[str, str]:
        ua = self.user_agent.format(email=self.contact_email)
        headers = {"User-Agent": ua}
        if accept:
            headers["Accept"] = accept
        mailto = os.getenv("CROSSREF_MAILTO") or self.contact_email
        if mailto:
            headers["mailto"] = mailto
        return headers

    def fetch_csl(self, doi: str) -> Dict[str, Any]:
        """Fetch CSL-JSON metadata for a DOI.

        Tries Crossref first; falls back to DOI.org content negotiation.
        """
        norm = self.normalize_doi(doi)
        # Try Crossref REST: https://api.crossref.org/works/{doi}
        crossref_url = f"https://api.crossref.org/works/{httpx.URL('/'+norm).raw_path.decode().lstrip('/')}"
        # The response is a Crossref wrapper: { status, message-type, message-version, message: {...} }
        with httpx.Client(timeout=self.timeout_s, headers=self._headers()) as client:
            r = client.get(crossref_url)
            if r.status_code == 200:
                data = r.json()
                message = data.get("message")
                if isinstance(message, dict):
                    # Convert Crossref message to CSL-JSON-like where possible
                    csl = self._crossref_message_to_csl(message)
                    if csl:
                        return csl
            else:
                r.raise_for_status()

        # Fallback to DOI.org content negotiation for CSL-JSON
        doi_url = f"https://doi.org/{norm}"
        with httpx.Client(timeout=self.timeout_s, headers=self._headers(CSL_CONTENT_TYPE)) as client:
            r2 = client.get(doi_url)
            r2.raise_for_status()
            # Many resolvers return CSL JSON directly
            try:
                return r2.json()
            except json.JSONDecodeError:
                # Some responses might be text; attempt to parse when possible
                raise ValueError("DOI.org did not return valid CSL-JSON")

    def save_csl(self, doi: str, data: Dict[str, Any]) -> Path:
        filename = _safe_filename_from_doi(doi) + ".json"
        path = self.citations_dir / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return path

    # --- BibTeX export ---
    @staticmethod
    def _bibtex_escape(value: str) -> str:
        if value is None:
            return ""
        # Minimal escaping for BibTeX safety
        return (
            value.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )

    @staticmethod
    def _bibtex_type_from_csl(csl_type: Optional[str]) -> str:
        mapping = {
            "journal-article": "article",
            "proceedings-article": "inproceedings",
            "paper-conference": "inproceedings",
            "book": "book",
            "book-chapter": "incollection",
            "chapter": "incollection",
            "report": "techreport",
            "thesis": "phdthesis",
            "dissertation": "phdthesis",
            "dataset": "misc",
            "posted-content": "misc",
        }
        return mapping.get((csl_type or "").lower(), "misc")

    @staticmethod
    def _bibtex_key_from_csl(csl: Dict[str, Any]) -> str:
        doi = csl.get("DOI")
        if doi:
            return "doi_" + _safe_filename_from_doi(doi)
        title = csl.get("title") or "untitled"
        year = None
        try:
            parts = csl.get("issued", {}).get("date-parts", [])
            if parts and parts[0]:
                year = parts[0][0]
        except Exception:
            pass
        key = re.sub(r"[^A-Za-z0-9]+", "", (title or "untitled").lower())[:30]
        if year:
            return f"{key}{year}"
        return key or "key"

    @staticmethod
    def _bibtex_author_field(csl: Dict[str, Any]) -> Optional[str]:
        authors = csl.get("author") or []
        if not isinstance(authors, list) or not authors:
            return None
        names = []
        for a in authors:
            given = a.get("given")
            family = a.get("family")
            if family and given:
                names.append(f"{family}, {given}")
            elif family:
                names.append(family)
            elif given:
                names.append(given)
        return " and ".join(names) if names else None

    def csl_to_bibtex(self, csl: Dict[str, Any]) -> str:
        entry_type = self._bibtex_type_from_csl(csl.get("type"))
        key = self._bibtex_key_from_csl(csl)

        fields: Dict[str, Optional[str]] = {}
        fields["title"] = csl.get("title")
        fields["author"] = self._bibtex_author_field(csl)
        # Date
        year = None
        month = None
        try:
            parts = csl.get("issued", {}).get("date-parts", [])
            if parts and parts[0]:
                year = str(parts[0][0])
                if len(parts[0]) > 1:
                    month = str(parts[0][1])
        except Exception:
            pass
        if year:
            fields["year"] = year
        if month:
            fields["month"] = month
        # Venue & location
        if entry_type == "article":
            fields["journal"] = csl.get("container-title")
        else:
            # Use booktitle for inproceedings/incollection when available
            if csl.get("container-title"):
                if entry_type in ("inproceedings", "incollection"):
                    fields["booktitle"] = csl.get("container-title")
                else:
                    fields["howpublished"] = csl.get("container-title")
        if csl.get("volume"):
            fields["volume"] = str(csl["volume"])
        if csl.get("issue"):
            fields["number"] = str(csl["issue"])
        if csl.get("page"):
            fields["pages"] = str(csl["page"])
        if csl.get("publisher"):
            fields["publisher"] = csl.get("publisher")
        if csl.get("DOI"):
            fields["doi"] = csl.get("DOI")
        if csl.get("URL"):
            fields["url"] = csl.get("URL")

        # Compose BibTeX
        lines = [f"@{entry_type}{{{key},"]
        for k in [
            "author",
            "title",
            "journal",
            "booktitle",
            "publisher",
            "year",
            "month",
            "volume",
            "number",
            "pages",
            "doi",
            "url",
            "howpublished",
        ]:
            v = fields.get(k)
            if v:
                lines.append(f"  {k} = {{{self._bibtex_escape(v)}}},")
        lines.append("}")
        return "\n".join(lines)

    def save_bibtex(self, doi: str, csl: Dict[str, Any]) -> Path:
        bib = self.csl_to_bibtex(csl)
        filename = _safe_filename_from_doi(doi) + ".bib"
        path = self.citations_dir / filename
        path.write_text(bib)
        return path

    # --- helpers ---
    @staticmethod
    def _join_names(authors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        csl_authors: list[dict[str, Any]] = []
        for a in authors:
            given = a.get("given") or a.get("given-name") or a.get("first")
            family = a.get("family") or a.get("surname") or a.get("last")
            name_dict: dict[str, Any] = {}
            if given:
                name_dict["given"] = given
            if family:
                name_dict["family"] = family
            if name_dict:
                csl_authors.append(name_dict)
        return csl_authors

    def _crossref_message_to_csl(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Map a Crossref work message to a CSL-JSON-like dict.

        Crossref doesn't return pure CSL-JSON, but fields are close.
        We map common fields for broad compatibility.
        """
        csl: Dict[str, Any] = {}
        csl["DOI"] = msg.get("DOI")
        csl["type"] = msg.get("type")
        csl["title"] = msg.get("title", [None])[0] if isinstance(msg.get("title"), list) else msg.get("title")
        csl["container-title"] = (
            msg.get("container-title", [None])[0] if isinstance(msg.get("container-title"), list) else msg.get("container-title")
        )
        csl["publisher"] = msg.get("publisher")
        csl["issued"] = {"date-parts": msg.get("issued", {}).get("'date-parts", msg.get("issued", {}).get("date-parts"))}
        csl["author"] = self._join_names(msg.get("author", [])) if isinstance(msg.get("author"), list) else []
        # Identifiers
        if msg.get("ISSN"):
            csl["ISSN"] = msg["ISSN"]
        if msg.get("ISBN"):
            csl["ISBN"] = msg["ISBN"]
        # Pagination & location
        for k in ("page", "volume", "issue"):
            if msg.get(k):
                csl[k] = msg[k]
        # URLs
        if msg.get("URL"):
            csl["URL"] = msg["URL"]
        # Fallback sanity: require at least DOI or title
        if not csl.get("DOI") and not csl.get("title"):
            return {}
        return csl


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and save CSL-JSON citation for a DOI")
    parser.add_argument("doi", help="DOI string (e.g., 10.1000/xyz123 or https://doi.org/10.1000/xyz123)")
    parser.add_argument("--out-dir", default=str(DoiBibliographyClient().citations_dir), help="Directory to save the JSON (default: output/citations)")
    parser.add_argument("--bibtex", action="store_true", help="Also save a BibTeX file next to the JSON output")
    args = parser.parse_args()

    client = DoiBibliographyClient(citations_dir=Path(args.out_dir))
    data = client.fetch_csl(args.doi)
    path = client.save_csl(args.doi, data)
    print(f"Saved CSL-JSON to: {path}")
    if args.bibtex:
        bpath = client.save_bibtex(args.doi, data)
        print(f"Saved BibTeX to: {bpath}")


if __name__ == "__main__":
    _cli()
# Use Crossref API to get bibliographic information from DOI
# check https://api.crossref.org/swagger-ui/index.html
# example DOI for testing
# doi:10.1001/jama.2025.9110


