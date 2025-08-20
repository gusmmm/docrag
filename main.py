from __future__ import annotations

"""
Project orchestrator CLI

Subcommands:
- ingest-pdfs       Run the input orchestrator to extract metadata, CSL, and citation keys.
- prepare-outputs   Create per-PDF output folders under output/papers/<citation_key>/.
- md-with-images    Generate Markdown with images per paper under output/papers/<key>/md_with_images/.
- prepare-rag       Extract references, strip them, and clean MD to produce -RAG.md per paper.
- add-metadata      Add YAML metadata from input/input_pdf.json to each -RAG.md.
- all               Run ingest-pdfs then prepare-outputs.

Examples:
- uv run python main.py ingest-pdfs
- uv run python main.py prepare-outputs
- uv run python main.py md-with-images
- uv run python main.py all
"""

import argparse
import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_prepare_module():
    """Load src/10_prepare_output_dirs.py as a module and return it."""
    path = ROOT / "src" / "10_prepare_output_dirs.py"
    spec = importlib.util.spec_from_file_location("prepare_output_dirs_mod", path)
    if not spec or not spec.loader:
        raise RuntimeError("Could not load prepare_output_dirs module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def cmd_ingest_pdfs() -> None:
    # Import here to avoid circulars on tooling
    from input.input import main as ingest_main  # type: ignore
    ingest_main()


def cmd_prepare_outputs() -> None:
    mod = _load_prepare_module()
    created = mod.prepare_output_dirs()  # type: ignore[attr-defined]
    if created:
        for p in created:
            print(f"Created: {p}")
    else:
        print("No new output folders needed.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="docrag", description="Project orchestrator")
    ap.add_argument(
    "command",
    choices=["ingest-pdfs", "prepare-outputs", "md-with-images", "prepare-rag", "add-metadata", "all"],
        help="Which step to run",
    )
    args = ap.parse_args(argv)

    if args.command == "ingest-pdfs":
        cmd_ingest_pdfs()
    elif args.command == "prepare-outputs":
        cmd_prepare_outputs()
    elif args.command == "md-with-images":
        # Lazy import and run
        import importlib.util
        path = ROOT / "src" / "11_create_md_with_images.py"
        spec = importlib.util.spec_from_file_location("mdimgs_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 11_create_md_with_images.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main()  # type: ignore[attr-defined]
    elif args.command == "prepare-rag":
        import importlib.util
        path = ROOT / "src" / "12_remove_refs_clean.py"
        spec = importlib.util.spec_from_file_location("ragprep_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 12_remove_refs_clean.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main()  # type: ignore[attr-defined]
    elif args.command == "add-metadata":
        import importlib.util
        path = ROOT / "src" / "13_add_metada.py"
        spec = importlib.util.spec_from_file_location("addmeta_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 13_add_metada.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main()  # type: ignore[attr-defined]
    elif args.command == "all":
        cmd_ingest_pdfs()
        cmd_prepare_outputs()
        # Run md-with-images last
        import importlib.util
        path = ROOT / "src" / "11_create_md_with_images.py"
        spec = importlib.util.spec_from_file_location("mdimgs_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 11_create_md_with_images.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
