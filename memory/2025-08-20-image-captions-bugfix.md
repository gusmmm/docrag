# 2025-08-20: Fix image captions script and regenerate MD

- Fixed a NameError in `src/01_image_captions.py` where `stem` was referenced before assignment. Now defined as `Path(source).stem` before building output paths.
- Defaults adjusted previously are confirmed working:
  - OCR languages default to `eng` (env `DOCRAG_OCR_LANGS` or CLI can override).
  - OCR engine auto-selects `tesseract-cli` when available, else `easyocr`.
  - Increased `images_scale` to 3 for higher-quality figures/tables.
  - Images are saved into a dedicated folder per Markdown output.
- Regenerated Step 2 outputs via `src/11_create_md_with_images.py`:
  - Wrote MD and images for `lewis2025focusedupdateclinicalpra` and `summers2025augmentedenteralproteind` under `output/papers/<key>/md_with_images/`.

Next:
- Optionally suppress noisy OCR warnings from Tesseract.
- If desired, tune Docling options for complex tables.
