Step 2: Create Markdown with images per paper

What it does
- For each `output/papers/<citation_key>/`, if `md_with_images/` is missing, it will:
	- create `md_with_images/`
	- run the Docling conversion (from `src/01_image_captions.py`) on `input/pdf/<citation_key>.pdf`
	- save `<citation_key>-with-image-refs.md` and images into that folder

How to run
- After running step 1 (prepare-outputs):
	- `uv run python src/11_create_md_with_images.py`
	- or via orchestrator: `uv run python main.py md-with-images`

Notes
- Requires `docling` and dependencies. Optional captions use `OPENROUTER_API_KEY`.
