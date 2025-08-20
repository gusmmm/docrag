from __future__ import annotations

"""
Step 1: Prepare per-PDF output folders.

For each PDF under input/pdf/, create an output folder under output/papers/<citation_key>/.
If the PDF filename isn't a citation key yet, we'll derive it by running the input orchestrator once.

This script is idempotent.
"""

from pathlib import Path
from typing import List
import sys

ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parent.parent
# Ensure repo root is on sys.path so that `input` package can be imported when running this script
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
try:
    from input.input import main as run_orchestrator  # ensures citation keys + registry
except Exception:
    run_orchestrator = None

from input.utils import load_db, save_db  # type: ignore
INPUT_DIR = ROOT / "input" / "pdf"
REGISTRY = ROOT / "input" / "input_pdf.json"
PAPERS_DIR = ROOT / "output" / "papers"


def ensure_registry_with_keys() -> None:
    """Ensure input/input_pdf.json has entries with citation_key for each PDF.

    Runs the orchestrator if available.
    """
    if run_orchestrator is not None:
        run_orchestrator()


def prepare_output_dirs() -> List[Path]:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_registry_with_keys()

    items = load_db(REGISTRY)
    created: List[Path] = []
    for it in items:
        key = str(it.get("citation_key", "")).strip().lower()
        if not key:
            # Skip items without a key; orchestrator should have added it
            continue
        d = PAPERS_DIR / key
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    return created


def main() -> None:
    created = prepare_output_dirs()
    if created:
        for p in created:
            print(f"Created: {p}")
    else:
        print("No new output folders needed.")


if __name__ == "__main__":
    main()
