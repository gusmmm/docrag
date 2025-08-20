from __future__ import annotations

"""
Step 3: Prepare Markdown for RAG (extract references, strip them, and clean scientific content).

For each Markdown created by Step 2 under output/papers/<key>/md_with_images/:
- If a "-RAG.md" file already exists in the same folder, skip entirely.
- Else:
  1) Extract references with the Google GenAI agent (agents/references_agent.py) and save references.json.
  2) Strip references from the Markdown (src/02_1_strip_refs.py logic).
  3) Clean non-scientific boilerplate (agents/md_clean_agent.py) and save <stem>-RAG.md.

Usage:
  uv run python src/12_remove_refs_clean.py           # process all papers
  uv run python src/12_remove_refs_clean.py --file <path-to-md>
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))


def _import_references_agent():
	import importlib.util
	import sys as _sys
	mod_path = ROOT / "agents" / "references_agent.py"
	spec = importlib.util.spec_from_file_location("references_agent_mod", mod_path)
	if not spec or not spec.loader:
		raise RuntimeError("Cannot load agents/references_agent.py")
	mod = importlib.util.module_from_spec(spec)
	_sys.modules[spec.name] = mod  # register for libraries relying on sys.modules
	spec.loader.exec_module(mod)  # type: ignore[attr-defined]
	return mod


def _import_strip_refs():
	import importlib.util
	import sys as _sys
	mod_path = ROOT / "src" / "02_1_strip_refs.py"
	spec = importlib.util.spec_from_file_location("strip_refs_mod", mod_path)
	if not spec or not spec.loader:
		raise RuntimeError("Cannot load src/02_1_strip_refs.py")
	mod = importlib.util.module_from_spec(spec)
	_sys.modules[spec.name] = mod
	spec.loader.exec_module(mod)  # type: ignore[attr-defined]
	return mod


def _import_md_clean_agent():
	import importlib.util
	import sys as _sys
	mod_path = ROOT / "agents" / "md_clean_agent.py"
	spec = importlib.util.spec_from_file_location("md_clean_agent_mod", mod_path)
	if not spec or not spec.loader:
		raise RuntimeError("Cannot load agents/md_clean_agent.py")
	mod = importlib.util.module_from_spec(spec)
	_sys.modules[spec.name] = mod
	spec.loader.exec_module(mod)  # type: ignore[attr-defined]
	return mod


def discover_md_files() -> List[Path]:
	base = ROOT / "output" / "papers"
	if not base.exists():
		return []
	out: List[Path] = []
	for key_dir in sorted(p for p in base.iterdir() if p.is_dir()):
		md_dir = key_dir / "md_with_images"
		if not md_dir.exists():
			continue
		for md in md_dir.glob("*.md"):
			# Skip already RAG-ified files
			if md.name.endswith("-RAG.md"):
				continue
			out.append(md)
	return out


def rag_output_path(md_path: Path) -> Path:
	return md_path.with_name(md_path.stem + "-RAG.md")


def process_single(md_path: Path) -> Tuple[bool, Path | None]:
	"""Process one Markdown file. Returns (created, rag_path or None)."""
	if not md_path.exists():
		print(f"[warn] Input Markdown not found: {md_path}")
		return (False, None)
	out_rag = rag_output_path(md_path)
	if out_rag.exists():
		print(f"[skip] RAG already exists: {out_rag}")
		return (False, out_rag)

	# 1) Extract references with agent and save references.json
	try:
		refs_mod = _import_references_agent()
		agent = refs_mod.ReferencesExtractorAgent()  # type: ignore[attr-defined]
		refs = agent.extract_from_file(md_path)  # type: ignore[attr-defined]
		refs_out = agent.save_json(md_path, refs)  # type: ignore[attr-defined]
		print(f"[refs] Saved references to: {refs_out}")
	except Exception as e:
		print(f"[refs] Skipping reference extraction ({e})")

	# 2) Strip references from Markdown (in-memory)
	strip_mod = _import_strip_refs()
	text = md_path.read_text(encoding="utf-8", errors="ignore")
	fm, body = strip_mod.split_front_matter(text)  # type: ignore[attr-defined]
	body_no_refs = strip_mod.strip_references(body)  # type: ignore[attr-defined]
	stripped_md = (fm or "") + body_no_refs

	# 3) Clean Markdown using md_clean_agent (rule-based by default)
	clean_mod = _import_md_clean_agent()
	meta, blocks = clean_mod.clean_markdown(stripped_md, use_llm=False)  # type: ignore[attr-defined]
	raw_fm, _ = clean_mod.extract_front_matter_raw(stripped_md)  # type: ignore[attr-defined]
	cleaned_md = clean_mod.render_clean_markdown(raw_fm, blocks)  # type: ignore[attr-defined]
	out_rag.write_text(cleaned_md, encoding="utf-8")
	print(f"[ok] Wrote RAG Markdown: {out_rag}")
	return (True, out_rag)


def main(argv: list[str] | None = None) -> None:
	ap = argparse.ArgumentParser(description="Prepare Markdown for RAG: extract refs, strip, and clean")
	ap.add_argument("--file", type=str, default=None, help="Path to a single Markdown file to process")
	args = ap.parse_args(argv)

	to_process: List[Path]
	if args.file:
		to_process = [Path(args.file)]
	else:
		to_process = discover_md_files()
		if not to_process:
			print("No Markdown files found under output/papers/*/md_with_images/ to process.")
			return

	created_any = False
	for md in to_process:
		created, _ = process_single(md)
		created_any = created_any or created

	if not created_any:
		print("No new RAG Markdown generated.")


if __name__ == "__main__":
	main()

