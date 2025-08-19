"""Google GenAI agent: extract citation data from Markdown using structured output.

Requirements:
- Use google-genai (Gemini) structured output per docs: https://ai.google.dev/gemini-api/docs/structured-output
- Default model: gemini-2.5-flash-lite
- Save extracted citation JSON to output/citations/

Env:
- GOOGLE_API_KEY required (GEMINI_API_KEY also accepted)

Notes:
- We keep schema minimal but useful: doi, title, authors (array of {given,family,full?}), container_title, publisher, issued {year,month,day}, volume, issue, pages, url.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    # google genai sdk per current docs
    from google import genai
    from google.genai import types as genai_types
except Exception as e:  # pragma: no cover
    genai = None
    genai_types = None


DEFAULT_MODEL = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash-lite")


SCHEMA = {
    "type": "object",
    "properties": {
        "doi": {"type": "string"},
        "title": {"type": "string"},
        "authors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "given": {"type": "string"},
                    "family": {"type": "string"},
                    "full": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "container_title": {"type": "string"},
        "publisher": {"type": "string"},
        "issued": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "month": {"type": "integer"},
                "day": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        "volume": {"type": "string"},
        "issue": {"type": "string"},
        "pages": {"type": "string"},
        "url": {"type": "string"},
    },
    "required": ["title"],
    "additionalProperties": False,
}


@dataclass
class CitationExtractorAgent:
    model: str = DEFAULT_MODEL
    out_dir: Path = Path("output/citations")

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _client(self):
        if genai is None:
            raise RuntimeError("google-genai SDK not installed. See https://googleapis.github.io/python-genai/")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required.")
        return genai.Client(api_key=api_key)

    def extract(self, markdown_text: str) -> Dict[str, Any]:
        client = self._client()
        prompt = (
            "Extract the primary article citation from the following Markdown."
            " Return a single JSON object matching the provided JSON schema."
            " If a field is unknown, omit it."
        )
        schema = genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "doi": genai_types.Schema(type=genai_types.Type.STRING),
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
                "container_title": genai_types.Schema(type=genai_types.Type.STRING),
                "publisher": genai_types.Schema(type=genai_types.Type.STRING),
                "issued": genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "year": genai_types.Schema(type=genai_types.Type.INTEGER),
                        "month": genai_types.Schema(type=genai_types.Type.INTEGER),
                        "day": genai_types.Schema(type=genai_types.Type.INTEGER),
                    },
                ),
                "volume": genai_types.Schema(type=genai_types.Type.STRING),
                "issue": genai_types.Schema(type=genai_types.Type.STRING),
                "pages": genai_types.Schema(type=genai_types.Type.STRING),
                "url": genai_types.Schema(type=genai_types.Type.STRING),
            },
            required=["title"],
        )

        response = client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [
                {"text": prompt},
                {"text": markdown_text},
            ]}],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        # response.text is JSON when response_mime_type=application/json
        data = json.loads(response.text)
        if not isinstance(data, dict):
            raise ValueError("Structured output was not a JSON object.")
        return data

    def extract_from_file(self, md_path: Path) -> Dict[str, Any]:
        text = Path(md_path).read_text(encoding="utf-8")
        return self.extract(text)

    def save_json(self, md_path: Path, data: Dict[str, Any]) -> Path:
        stem = Path(md_path).stem
        out = self.out_dir / f"{stem}-genai.json"
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return out


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Extract citation from Markdown using Google GenAI structured output")
    parser.add_argument("path", help="Path to Markdown file (e.g., output/md_with_images/xxx.md)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id (default: gemini-2.5-flash-lite)")
    args = parser.parse_args()

    agent = CitationExtractorAgent(model=args.model)
    md_path = Path(args.path)
    data = agent.extract_from_file(md_path)
    out = agent.save_json(md_path, data)
    print(f"Saved citation JSON to: {out}")


if __name__ == "__main__":
    _cli()
# this agent takes the md file from output/md_with_images
# extracts the citation data from the md, such as title, authors, year, and DOI
# the output from the agent will be used to fetch the full citation data from Crossref

