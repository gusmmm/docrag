from __future__ import annotations

"""
Project orchestrator CLI

Subcommands:
- ingest-pdfs       Run the input orchestrator to extract metadata, CSL, and citation keys.
- prepare-outputs   Create per-PDF output folders under output/papers/<citation_key>/.
- md-with-images    Generate Markdown with images per paper under output/papers/<key>/md_with_images/.
- prepare-rag       Extract references, strip them, and clean MD to produce -RAG.md per paper.
- add-metadata      Add YAML metadata from input/input_pdf.json to each -RAG.md.
- index             Chunk, embed, and index -RAG.md files into Milvus (journal_papers DB).
- all               Run ingest-pdfs then prepare-outputs.
- full              Run the complete pipeline end-to-end: ingest → prepare-outputs → md-with-images → prepare-rag → add-metadata → index.

Examples:
- uv run python main.py ingest-pdfs
- uv run python main.py prepare-outputs
- uv run python main.py md-with-images
- uv run python main.py all
- uv run python main.py full --show 3
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


def cmd_full_pipeline(rest: list[str] | None = None) -> None:
    """Run the full pipeline from ingest to indexing.

    Steps:
    1) ingest-pdfs      (input/input.py)
    2) prepare-outputs  (src/10_prepare_output_dirs.py)
    3) md-with-images   (src/11_create_md_with_images.py)
    4) prepare-rag      (src/12_remove_refs_clean.py)
    5) add-metadata     (src/13_add_metada.py)
    6) index            (src/14_index.py)

    Extra args (rest) are forwarded to the final index step, e.g., --show 3.
    """
    # 1) ingest-pdfs
    cmd_ingest_pdfs()

    # 2) prepare-outputs
    cmd_prepare_outputs()

    # 3) md-with-images
    import importlib.util as _ilu
    import sys as _sys
    path = ROOT / "src" / "11_create_md_with_images.py"
    spec = _ilu.spec_from_file_location("mdimgs_mod", path)
    if not spec or not spec.loader:
        raise RuntimeError("Could not load 11_create_md_with_images.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    mod.main()  # type: ignore[attr-defined]

    # 4) prepare-rag (process all)
    path = ROOT / "src" / "12_remove_refs_clean.py"
    spec = _ilu.spec_from_file_location("ragprep_mod", path)
    if not spec or not spec.loader:
        raise RuntimeError("Could not load 12_remove_refs_clean.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    mod.main([])  # type: ignore[attr-defined]

    # 5) add-metadata (process all)
    path = ROOT / "src" / "13_add_metada.py"
    spec = _ilu.spec_from_file_location("addmeta_mod", path)
    if not spec or not spec.loader:
        raise RuntimeError("Could not load 13_add_metada.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    mod.main([])  # type: ignore[attr-defined]

    # 6) index (forward any extra args to allow --show/--dry-run/etc.)
    path = ROOT / "src" / "14_index.py"
    spec = _ilu.spec_from_file_location("index_mod", path)
    if not spec or not spec.loader:
        raise RuntimeError("Could not load 14_index.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    mod.main(rest or [])  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="docrag", description="Project orchestrator", add_help=True)
    ap.add_argument(
        "command",
    choices=["ingest-pdfs", "prepare-outputs", "md-with-images", "prepare-rag", "add-metadata", "index", "all", "full"],
        help="Which step to run",
    )
    # Accept and forward unknown args to sub-commands (e.g., --dry-run)
    args, rest = ap.parse_known_args(argv)

    if args.command == "ingest-pdfs":
        cmd_ingest_pdfs()
    elif args.command == "prepare-outputs":
        cmd_prepare_outputs()
    elif args.command == "md-with-images":
        # Lazy import and run
        import importlib.util
        import sys as _sys
        path = ROOT / "src" / "11_create_md_with_images.py"
        spec = importlib.util.spec_from_file_location("mdimgs_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 11_create_md_with_images.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = mod  # register for libraries relying on sys.modules
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main()  # type: ignore[attr-defined]
    elif args.command == "prepare-rag":
        import importlib.util
        import sys as _sys
        path = ROOT / "src" / "12_remove_refs_clean.py"
        spec = importlib.util.spec_from_file_location("ragprep_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 12_remove_refs_clean.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main(rest)  # type: ignore[attr-defined]
    elif args.command == "add-metadata":
        import importlib.util
        import sys as _sys
        path = ROOT / "src" / "13_add_metada.py"
        spec = importlib.util.spec_from_file_location("addmeta_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 13_add_metada.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main(rest)  # type: ignore[attr-defined]
    elif args.command == "index":
        import importlib.util
        import sys as _sys
        path = ROOT / "src" / "14_index.py"
        spec = importlib.util.spec_from_file_location("index_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 14_index.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main(rest)  # type: ignore[attr-defined]
    elif args.command == "all":
        cmd_ingest_pdfs()
        cmd_prepare_outputs()
        # Run md-with-images last
        import importlib.util
        import sys as _sys
        path = ROOT / "src" / "11_create_md_with_images.py"
        spec = importlib.util.spec_from_file_location("mdimgs_mod", path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not load 11_create_md_with_images.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.main(rest)  # type: ignore[attr-defined]
    elif args.command == "full":
        cmd_full_pipeline(rest)


if __name__ == "__main__":
    main()
