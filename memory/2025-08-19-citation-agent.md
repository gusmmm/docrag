# 2025-08-19 Citation Agent

- Added `agents/citation_agent.py` to extract citation metadata from Markdown using Google GenAI structured output.
- Model default: `gemini-2.5-flash-lite` (configurable via `--model` or `GOOGLE_GENAI_MODEL`).
- Output saved to `output/citations/<md-stem>-genai.json`.
- Requires `GOOGLE_API_KEY` in environment.
- Dependencies: `google-genai` added to `pyproject.toml`.
- Usage example:
  - `uv run python agents/citation_agent.py output/md_with_images/jama_summers_2025-with-image-refs.md`
