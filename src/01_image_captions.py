"""Add image captions to the Markdown output and save to output/md_with_images.

Now supports CLI flags:
- --ocr-lang: comma-separated list (e.g., "eng" or "eng,por"). Defaults to env DOCRAG_OCR_LANGS or "auto".
- --ocr-engine: auto|tesseract-cli|easyocr (default: auto)
- --captions / --no-captions: enable/disable remote image descriptions (default: enabled if OPENROUTER_API_KEY present)
- --input: path to a single input PDF (default: first file in input/)
"""

import os
import shutil
import argparse
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
	PdfPipelineOptions,
	PictureDescriptionApiOptions,
	TesseractCliOcrOptions,
	EasyOcrOptions,
)
from docling_core.types.doc import ImageRefMode
import sys
from dotenv import load_dotenv

load_dotenv()


pdf_dir = Path("input/pdf")
out_dir = Path("output/md_with_images")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert PDF to Markdown with images and optional captions using Docling.")
	parser.add_argument("--input", type=str, default=None, help="Path to a single input PDF. Defaults to the first file in input/.")
	parser.add_argument("--ocr-lang", type=str, default=None, help="Comma-separated OCR languages (e.g., 'eng' or 'eng,por'). Overrides DOCRAG_OCR_LANGS.")
	parser.add_argument("--ocr-engine", type=str, choices=["auto", "tesseract-cli", "easyocr"], default="auto", help="OCR engine selection. Default: auto (prefer tesseract-cli if available, else easyocr).")
	captions_group = parser.add_mutually_exclusive_group()
	captions_group.add_argument("--captions", dest="captions", action="store_true", help="Enable remote image captions (requires OPENROUTER_API_KEY).")
	captions_group.add_argument("--no-captions", dest="captions", action="store_false", help="Disable remote image captions.")
	parser.set_defaults(captions=None)  # None => auto based on env
	return parser.parse_args()

def convert_with_image_annotation(input_file: Path, *, ocr_engine: str | None = None, ocr_langs_cli: list[str] | None = None, captions: bool | None = None):
	api_key = os.getenv("OPENROUTER_API_KEY")
	# OCR languages: CLI overrides ENV, else default to ["eng"] for quality/stability
	if ocr_langs_cli is not None:
		ocr_langs = ocr_langs_cli
	else:
		ocr_langs_env = os.getenv("DOCRAG_OCR_LANGS", "eng").strip()
		ocr_langs = [s.strip() for s in ocr_langs_env.split(",") if s.strip()] if ocr_langs_env else ["eng"]
	# Avoid tesseract 'Latin' auto-detect warnings; fall back to English if 'auto' is present
	if ocr_langs == ["auto"]:
		ocr_langs = ["eng"]

	# Determine OCR engine
	chosen_engine = ocr_engine or "auto"
	ocr_options = None
	if chosen_engine == "auto":
		if shutil.which("tesseract"):
			chosen_engine = "tesseract-cli"
		else:
			chosen_engine = "easyocr"
	if chosen_engine == "tesseract-cli":
		ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=ocr_langs)
	elif chosen_engine == "easyocr":
		if ocr_langs and ocr_langs != ["auto"]:
			ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=ocr_langs)
		else:
			ocr_options = EasyOcrOptions(force_full_page_ocr=True)
	else:
		raise ValueError(f"Unsupported ocr engine: {chosen_engine}")

	# Captions toggle: default True if API key present, else False; CLI can override
	do_picture_description = bool(api_key)
	if captions is not None:
		do_picture_description = captions and bool(api_key)
	if do_picture_description and not api_key:
		print("--captions requested but OPENROUTER_API_KEY not set; captions will be disabled.")

	picture_desc_api_option = None
	enable_remote_services = False
	generate_picture_images = False

	if do_picture_description and api_key:
		model = "google/gemini-2.5-flash-lite"
		picture_desc_api_option = PictureDescriptionApiOptions(
			url="https://openrouter.ai/api/v1/chat/completions",
			prompt="Describe the image in detail, use several sentences in a single paragraph.",
			params=dict(model=model),
			headers={"Authorization": f"Bearer {api_key}"},
			timeout=60,
		)
		enable_remote_services = True
		generate_picture_images = True
	else:
		if not api_key:
			print("OPENROUTER_API_KEY not found. Proceeding without picture descriptions.")

	print(f"[docling] OCR engine={chosen_engine} langs={','.join(ocr_langs)} captions={do_picture_description}")

	pipeline_options = PdfPipelineOptions(
		do_ocr=True,
		ocr_options=ocr_options,
		do_picture_description=do_picture_description,
		picture_description_options=picture_desc_api_option,
		enable_remote_services=enable_remote_services,
		generate_picture_images=generate_picture_images,
		# Increase image scale for higher quality figures/tables in Markdown
		images_scale=3,
	)
	converter = DocumentConverter(
		format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
	)
	conv_result = converter.convert(source=input_file)
	return conv_result
def main():
	args = parse_args()

	# Resolve input file
	if args.input:
		source = Path(args.input)
		if not source.exists():
			print(f"Input file not found: {source}")
			sys.exit(1)
	else:
		files = list(Path(pdf_dir).glob("*"))
		print(f"Found {len(files)} files in {pdf_dir}")
		print(files)
		if not files:
			print("No files found in input/. Aborting.")
			sys.exit(1)
		source = files[0]

	# Parse OCR langs from CLI
	ocr_langs_cli = None
	if args.ocr_lang:
		ocr_langs_cli = [s.strip() for s in args.ocr_lang.split(",") if s.strip()]

	result = convert_with_image_annotation(
		Path(source),
		ocr_engine=args.ocr_engine,
		ocr_langs_cli=ocr_langs_cli,
		captions=args.captions,
	)

	# Ensure output directories exist
	stem = Path(source).stem
	out_dir.mkdir(parents=True, exist_ok=True)
	images_dir = out_dir / f"{stem}_images"
	images_dir.mkdir(parents=True, exist_ok=True)

	# Save Markdown with externally referenced images into a dedicated folder
	md_path = out_dir / f"{stem}-with-image-refs.md"
	result.document.save_as_markdown(
		md_path,
		artifacts_dir=images_dir,
		image_mode=ImageRefMode.REFERENCED,
		include_annotations=True,
	)

	print(f"[docling] Wrote Markdown: {md_path}")
	print(f"[docling] Images folder: {images_dir}")


if __name__ == "__main__":
	main()




