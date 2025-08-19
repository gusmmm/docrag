from __future__ import annotations

"""
Markdown cleaner agent

Goal: Given a merged-no-ref Markdown file, output a cleaned JSON that:
- Keeps YAML front matter (unchanged)
- Keeps scientific text, tables, figures/images
- Removes author info, author/group contributions, funding/support/roles, disclosures, article info boilerplate

Optional: If GOOGLE_API_KEY/GEMINI_API_KEY is present, a lightweight Gemini pass can refine
borderline classifications. Defaults to rule-based cleaning for determinism.

Output: <input-stem>-clean.json in the same folder as the input file.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from dotenv import load_dotenv
import yaml

load_dotenv()

try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None


# -----------------------------
# Utilities
# -----------------------------

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def split_front_matter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    m = FRONT_MATTER_RE.match(text)
    if not m:
        return None, text
    fm_text = m.group(1)
    rest = text[m.end():]
    try:
        meta = yaml.safe_load(fm_text) or {}
    except Exception:
        meta = None
    return meta, rest


@dataclass
class Section:
    level: int
    title: str
    text: str


def parse_sections(md: str) -> List[Section]:
    lines = md.splitlines()
    sections: List[Section] = []
    current: Optional[Section] = None
    for line in lines:
        h = HEADING_RE.match(line.strip())
        if h:
            # Start a new section
            level = len(h.group(1))
            title = h.group(2).strip()
            if current:
                sections.append(current)
            current = Section(level=level, title=title, text="")
        else:
            if current is None:
                # Pre-body text (before first heading) -> put into a synthetic section
                current = Section(level=1, title="_preamble_", text="")
            current.text += ("\n" if current.text else "") + line
    if current:
        sections.append(current)
    return sections


# -----------------------------
# Rules for keep/drop
# -----------------------------

DROP_SECTION_PATTERNS = [
    r"article information",
    r"author affiliations",
    r"author contributions",
    r"additional contributions",
    r"group information",
    r"investigators",
    r"clinical trials group",
    r"funder",
    r"sponsor",
    r"funding",
    r"support",
    r"role of the funder",
    r"role of the sponsor",
    r"corresponding author",
    r"conflict",
    r"competing interest",
    r"financial disclosure",
    r"acknowledg?ments?",
]

KEEP_SECTION_HINTS = [
    r"key points",
    r"importance",
    r"objective",
    r"design",
    r"participants",
    r"intervention",
    r"interventions",
    r"main outcomes? and measures",
    r"outcome measures",
    r"outcomes?",
    r"methods?",
    r"trial procedures",
    r"randomization",
    r"statistical analysis",
    r"results",
    r"biochemical outcomes",
    r"subgroup analyses",
    r"adverse events",
    r"readmissions",
    r"discussion",
    r"conclusions?( and relevance)?",
    r"enteral nutrition delivery",
    r"patients",
    r"safety",
    r"sample size calculation",
    r"table ",
    r"figure ",
]


def _matches_any(s: str, patterns: List[str]) -> bool:
    s_l = s.lower()
    return any(re.search(p, s_l) for p in patterns)


def classify_section(title: str) -> str:
    """Return 'keep' or 'drop' using heuristics."""
    if title == "_preamble_":
        # Keep preamble; later we'll prune known boilerplate lines
        return "keep"
    if _matches_any(title, DROP_SECTION_PATTERNS):
        return "drop"
    if _matches_any(title, KEEP_SECTION_HINTS):
        return "keep"
    # Default to keep (conservative)
    return "keep"


LINE_DROP_PATTERNS = [
    r"^\s*jama\.com\s*$",
    r"^\s*accepted for publication:.*$",
    r"^\s*published online:.*$",
    r"^\s*doi:\s*10\.[0-9]{4,9}/.*$",
    r"^\s*corresponding author:.*$",
]


def prune_lines(text: str) -> str:
    lines = text.splitlines()
    kept: List[str] = []
    for ln in lines:
        l = ln.strip()
        if any(re.search(p, l, flags=re.IGNORECASE) for p in LINE_DROP_PATTERNS):
            continue
        kept.append(ln)
    # Collapse excessive blank lines
    out: List[str] = []
    blank = 0
    for ln in kept:
        if ln.strip() == "":
            blank += 1
            if blank > 2:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out).strip()


# -----------------------------
# Optional Gemini refinement
# -----------------------------

def get_genai_client():
    if genai is None:
        return None
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def llm_refine_decision(client, title: str, text: str, default: str) -> str:
    """Ask Gemini very briefly whether to keep or drop, bounded by our intent.
    Returns 'keep' or 'drop'.
    """
    if client is None:
        return default
    try:
        prompt = (
            "You are cleaning a scientific manuscript Markdown for RAG indexing. "
            "Answer with a single word: KEEP or DROP. KEEP scientific content (methods, results, outcomes, conclusions, discussion, tables, figures/images). "
            "DROP author info, affiliations, contributions, groups, funding/support/roles, disclosures, acknowledgments, article information, and boilerplate.\n\n"
            f"Section title: {title}\n\nSnippet:\n{text[:800]}"
        )
        model = os.getenv("GEMINI_AGENT_MODEL", "gemini-2.5-flash-lite")
        resp = client.models.generate_content(model=model, contents=prompt)
        out = (getattr(resp, "text", None) or "").strip().upper()
        if "DROP" in out and "KEEP" not in out:
            return "drop"
        if "KEEP" in out and "DROP" not in out:
            return "keep"
        return default
    except Exception:
        return default


# -----------------------------
# Main cleaning pipeline
# -----------------------------

def clean_markdown(md_text: str, use_llm: bool = False) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    meta, body = split_front_matter(md_text)
    sections = parse_sections(body)
    client = get_genai_client() if use_llm else None

    kept_blocks: List[Dict[str, Any]] = []
    for sec in sections:
        decision = classify_section(sec.title)
        if use_llm:
            decision = llm_refine_decision(client, sec.title, sec.text, decision)
        if decision == "drop":
            continue
        cleaned_text = prune_lines(sec.text)
        if not cleaned_text:
            continue
        kept_blocks.append({
            "type": "section",
            "level": sec.level,
            "title": sec.title if sec.title != "_preamble_" else None,
            "text": cleaned_text,
        })
    return meta, kept_blocks


def save_clean_json(input_path: Path, meta: Optional[Dict[str, Any]], blocks: List[Dict[str, Any]]) -> Path:
    out_path = input_path.with_suffix("")
    out_path = out_path.with_name(out_path.name + "-clean.json")
    payload = {
        "metadata": meta or {},
        "content": blocks,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Clean merged-no-ref Markdown into a JSON suitable for scientific RAG.")
    parser.add_argument("input", type=str, help="Path to the merged-no-ref Markdown file")
    parser.add_argument("--use-llm", action="store_true", help="Use Gemini to refine keep/drop decisions (requires API key)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    md_text = in_path.read_text(encoding="utf-8", errors="ignore")
    meta, blocks = clean_markdown(md_text, use_llm=args.use_llm)
    out_path = save_clean_json(in_path, meta, blocks)
    print(f"Saved cleaned JSON to: {out_path}")


if __name__ == "__main__":
    main()
