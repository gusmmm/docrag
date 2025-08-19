"""
Strip references/bibliography from a merged Markdown file (with YAML front matter).

- Preserves the YAML front matter intact.
- Finds a heading like '## References' or '## Bibliography' (case-insensitive) and removes everything from there to the end.
- Writes a new file next to the input: <stem>-no-ref.md (unless --out is provided).

Usage:
  uv run python src/04_strip_refs.py \
    --input output/md_with_images/jama_summers_2025-with-image-refs-merged.md

Optional:
  --out <path>  # custom output path
  --extra "Appendix|Notes"  # pipe-separated extra heading patterns to strip from
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


FM_BOUNDARY = re.compile(r"^---\s*$")


def split_front_matter(text: str) -> Tuple[str, str]:
    """Return (front_matter_with_markers_or_empty, body)."""
    lines = text.splitlines()
    if not lines or not FM_BOUNDARY.match(lines[0]):
        return "", text
    for i in range(1, len(lines)):
        if FM_BOUNDARY.match(lines[i]):
            fm = "\n".join(lines[: i + 1]) + "\n"
            body = "\n".join(lines[i + 1 :])
            return fm, body
    return "", text


def strip_references(body_md: str, extra_patterns: List[str] | None = None) -> str:
    """Strip from the first References/Bibliography heading to end. If not found, return unchanged."""
    headings = [
        r"^#{1,6}\s+references\b",
        r"^#{1,6}\s+bibliography\b",
        r"^#{1,6}\s+works\s+cited\b",
        r"^#{1,6}\s+references\s+and\s+notes\b",
    ]
    if extra_patterns:
        for p in extra_patterns:
            p = p.strip()
            if p:
                headings.append(rf"^#{'{'}1,6{'}'}\s+{p}\b")
    pattern = re.compile("|".join(headings), flags=re.IGNORECASE | re.MULTILINE)
    m = pattern.search(body_md)
    if not m:
        return body_md
    # Keep everything before the heading start
    return body_md[: m.start()].rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove references/bibliography section from Markdown")
    ap.add_argument(
        "--input",
        type=str,
        default="output/md_with_images/jama_summers_2025-with-image-refs-merged.md",
        help="Path to merged Markdown with YAML front matter",
    )
    ap.add_argument("--out", type=str, default=None, help="Output path (default: <stem>-no-ref.md)")
    ap.add_argument(
        "--extra",
        type=str,
        default="",
        help="Optional additional heading names (pipe-separated) to trigger stripping, e.g., 'Appendix|Notes'",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    text = inp.read_text(encoding="utf-8")
    fm, body = split_front_matter(text)

    extra = [s for s in args.extra.split("|") if s] if args.extra else []
    cleaned_body = strip_references(body, extra_patterns=extra)
    out_path = Path(args.out) if args.out else inp.with_name(inp.stem + "-no-ref.md")
    out_path.write_text(fm + cleaned_body, encoding="utf-8")
    print(f"Wrote cleaned Markdown to: {out_path}")


if __name__ == "__main__":
    main()
