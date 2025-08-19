# BASIC docling pdf processing to md
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
import sys

pdf_dir = Path("input/")
md_dir = Path("output/md")

files = list(Path(pdf_dir).glob("*"))
print(f"Found {len(files)} files in {pdf_dir}")
print(files)

if not files:
	print("No files found in input/. Aborting.")
	sys.exit(1)

# example pdf
source = files[0]

# create converter
converter = DocumentConverter()
doc = converter.convert(source).document

# Save DoclingDocument as JSON in output/doc
doc_dir = Path("output/doc")
doc_dir.mkdir(parents=True, exist_ok=True)
out_path = doc_dir / f"{Path(source).stem}.json"
doc.save_as_json(out_path)
print(f"\n===== Saved DoclingDocument to {out_path} =====")

# Save DoclingDocument as Markdown in output/md
md_dir.mkdir(parents=True, exist_ok=True)
md_out_path = md_dir / f"{Path(source).stem}.md"
doc.save_as_markdown(md_out_path)
print(f"\n===== Saved DoclingDocument to {md_out_path} =====")
print(doc.export_to_markdown())