from __future__ import annotations

"""
Step 2: Create Markdown with images per paper.

For each paper folder under output/papers/<citation_key>/:
- If md_with_images/ exists, skip.
- Else, create md_with_images/ and run the Docling pipeline from 01_image_captions
  to produce a Markdown with referenced images and save into that folder.

Requirements:
- PDFs named <citation_key>.pdf must exist under input/pdf/.
"""

from pathlib import Path
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_convert():
    """Import convert_with_image_annotation from src/01_image_captions.py safely."""
    import importlib.util
    mod_path = ROOT / "src" / "01_image_captions.py"
    spec = importlib.util.spec_from_file_location("imgcaps_mod", mod_path)
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load 01_image_captions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "convert_with_image_annotation"):
        raise RuntimeError("convert_with_image_annotation not found in 01_image_captions.py")
    return getattr(mod, "convert_with_image_annotation")


def _find_pdf_for_key(key: str) -> Path | None:
    base_input = ROOT / "input"
    # 1) Standard location
    p = base_input / "pdf" / f"{key}.pdf"
    if p.exists():
        return p
    # 2) Topics: input/topics/<topic>/<key>.pdf
    topics_dir = base_input / "topics"
    if topics_dir.exists():
        for tdir in sorted(d for d in topics_dir.iterdir() if d.is_dir()):
            tp = tdir / f"{key}.pdf"
            if tp.exists():
                return tp
    return None


def create_md_for_paper(key_dir: Path) -> Tuple[bool, Path | None]:
    """Create md_with_images for a single paper dir if missing.

    Returns (created, md_path or None)
    """
    key = key_dir.name
    md_dir = key_dir / "md_with_images"
    if md_dir.exists():
        print(f"[skip] {key}: md_with_images/ already exists")
        return (False, None)

    pdf_path = _find_pdf_for_key(key)
    if not pdf_path:
        print(f"[warn] {key}: PDF not found under input/pdf/ or input/topics/, skipping")
        return (False, None)

    md_dir.mkdir(parents=True, exist_ok=True)

    convert = _import_convert()
    result = convert(pdf_path)

    # Save Markdown with referenced images into md_dir
    from docling_core.types.doc import ImageRefMode  # imported lazily

    md_path = md_dir / f"{key}-with-image-refs.md"
    result.document.save_as_markdown(
        md_path,
        artifacts_dir=md_dir,
        image_mode=ImageRefMode.REFERENCED,
        include_annotations=True,
    )
    print(f"[ok] {key}: wrote {md_path}")
    return (True, md_path)


def process_all() -> List[Path]:
    papers_root = ROOT / "output" / "papers"
    if not papers_root.exists():
        print("No papers directory found; run prepare-outputs first.")
        return []
    created: List[Path] = []
    for key_dir in sorted(p for p in papers_root.iterdir() if p.is_dir()):
        did, md_path = create_md_for_paper(key_dir)
        if did and md_path is not None:
            created.append(md_path)
    return created


def main() -> None:
    created = process_all()
    if not created:
        print("No new md_with_images generated.")


if __name__ == "__main__":
    main()
