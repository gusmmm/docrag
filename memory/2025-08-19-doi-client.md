# 2025-08-19 DOI Client

- Added `src/DOI_lookup.py` with class `DoiBibliographyClient` to fetch bibliographic metadata by DOI.
- Returns CSL-JSON (Citation Style Language) and saves to `output/citations/{safe-doi}.json`.
- Primary source: Crossref REST (`/works/{doi}`); fallback to DOI.org content negotiation with `Accept: application/vnd.citationstyles.csl+json`.
- Added dependency: `httpx` in `pyproject.toml`.
- Optional env: `CONTACT_EMAIL` or `CROSSREF_MAILTO` used in headers for polite API usage.
- Quick CLI: `uv run python src/DOI_lookup.py <doi>` writes the CSL-JSON file.
