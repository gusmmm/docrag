# 2025-08-19 - md_clean_agent outputs Markdown

- Updated agents/md_clean_agent.py to default to Markdown output preserving YAML front matter verbatim.
- Added --output-format {md|json} (default md).
- Preserves original heading lines and exact body line formatting; removes non-scientific sections/lines via heuristics.
- Saved file as <input-stem>-clean.md in the same folder.
