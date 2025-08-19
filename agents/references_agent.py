"""Google GenAI agent: extract references/bibliography from a merged Markdown file.

Outputs a structured JSON array (BibTeX-like fields) saved as references.json
in the same folder as the input Markdown.

Env:
- GOOGLE_API_KEY or GEMINI_API_KEY required
- Default model: gemini-2.5-flash-lite (configurable)

Usage:
  uv run python agents/references_agent.py \
    output/md_with_images/jama_summers_2025-with-image-refs-merged.md
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None


DEFAULT_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash-lite")


def _slice_references_section(md_text: str) -> str:
    """Try to slice the Markdown starting at a 'References'/'Bibliography' section.

    Falls back to returning the full text if a clear section isn't found.
    """
    # Look for common headings indicating references
    headings = [
        r"^##+\s+references\b",
        r"^##+\s+bibliography\b",
        r"^##+\s+works\s+cited\b",
        r"^#\s+references\b",
        r"^#\s+bibliography\b",
    ]
    pattern = re.compile("|".join(headings), flags=re.IGNORECASE | re.MULTILINE)
    m = pattern.search(md_text)
    if m:
        return md_text[m.start():]
    # If no heading found, try last 25% of the document as a heuristic
    n = len(md_text)
    return md_text[int(n * 0.75) :]


@dataclass
class ReferencesExtractorAgent:
    model: str = DEFAULT_MODEL

    def _client(self):
        if genai is None:
            raise RuntimeError("google-genai SDK not installed. See https://googleapis.github.io/python-genai/")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required.")
        return genai.Client(api_key=api_key)

    def extract(self, markdown_text: str) -> List[Dict[str, Any]]:
        client = self._client()

        # Limit to references section when possible
        refs_text = _slice_references_section(markdown_text)

        prompt = (
            "Identify and extract all bibliography/references entries at the end of this Markdown. "
            "Return an array of JSON objects, BibTeX-like, using these fields when available: "
            "entry_type (e.g., article, book, inproceedings), key, title, authors (array of {given,family,full}), "
            "year, month, day, journal, booktitle, volume, issue, pages, publisher, editors (array like authors), "
            "doi, url. If a field is unknown, omit it. Do not include non-reference content."
        )

        schema = genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "entry_type": genai_types.Schema(type=genai_types.Type.STRING),
                    "key": genai_types.Schema(type=genai_types.Type.STRING),
                    "title": genai_types.Schema(type=genai_types.Type.STRING),
                    "authors": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            properties={
                                "given": genai_types.Schema(type=genai_types.Type.STRING),
                                "family": genai_types.Schema(type=genai_types.Type.STRING),
                                "full": genai_types.Schema(type=genai_types.Type.STRING),
                            },
                        ),
                    ),
                    "year": genai_types.Schema(type=genai_types.Type.INTEGER),
                    "month": genai_types.Schema(type=genai_types.Type.INTEGER),
                    "day": genai_types.Schema(type=genai_types.Type.INTEGER),
                    "journal": genai_types.Schema(type=genai_types.Type.STRING),
                    "booktitle": genai_types.Schema(type=genai_types.Type.STRING),
                    "volume": genai_types.Schema(type=genai_types.Type.STRING),
                    "issue": genai_types.Schema(type=genai_types.Type.STRING),
                    "pages": genai_types.Schema(type=genai_types.Type.STRING),
                    "publisher": genai_types.Schema(type=genai_types.Type.STRING),
                    "editors": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            properties={
                                "given": genai_types.Schema(type=genai_types.Type.STRING),
                                "family": genai_types.Schema(type=genai_types.Type.STRING),
                                "full": genai_types.Schema(type=genai_types.Type.STRING),
                            },
                        ),
                    ),
                    "doi": genai_types.Schema(type=genai_types.Type.STRING),
                    "url": genai_types.Schema(type=genai_types.Type.STRING),
                },
            ),
        )

        response = client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [
                {"text": prompt},
                {"text": refs_text},
            ]}],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        data = json.loads(response.text)
        if not isinstance(data, list):
            raise ValueError("Structured output did not return a JSON array.")
        # Normalize: ensure types
        norm: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                norm.append(item)
        return norm

    def extract_from_file(self, md_path: Path) -> List[Dict[str, Any]]:
        text = Path(md_path).read_text(encoding="utf-8")
        return self.extract(text)

    def save_json(self, md_path: Path, refs: List[Dict[str, Any]]) -> Path:
        out = Path(md_path).parent / "references.json"
        out.write_text(json.dumps(refs, indent=2, ensure_ascii=False))
        return out


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Extract references from Markdown using Google GenAI structured output")
    parser.add_argument("path", help="Path to merged Markdown file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id (default: gemini-2.5-flash-lite)")
    args = parser.parse_args()

    agent = ReferencesExtractorAgent(model=args.model)
    md_path = Path(args.path)
    refs = agent.extract_from_file(md_path)
    out = agent.save_json(md_path, refs)
    print(f"Saved references JSON to: {out}")


if __name__ == "__main__":
    _cli()
