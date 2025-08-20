from __future__ import annotations

"""
Project orchestrator CLI

Subcommands:
- ingest-pdfs       Run the input orchestrator to extract metadata, CSL, and citation keys.
- prepare-outputs   Create per-PDF output folders under output/papers/<citation_key>/.
- all               Run ingest-pdfs then prepare-outputs.

Examples:
- uv run python main.py ingest-pdfs
- uv run python main.py prepare-outputs
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
        choices=["ingest-pdfs", "prepare-outputs", "all"],
        help="Which step to run",
    )
    args = ap.parse_args(argv)

    if args.command == "ingest-pdfs":
        cmd_ingest_pdfs()
    elif args.command == "prepare-outputs":
        cmd_prepare_outputs()
    elif args.command == "all":
        cmd_ingest_pdfs()
        cmd_prepare_outputs()


if __name__ == "__main__":
    main()
