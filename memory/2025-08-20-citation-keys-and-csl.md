Title: Citation keys + CSL metadata integrated into input pipeline
Date: 2025-08-20

Summary
- Enhanced input/check_pdf.py to fetch full CSL-JSON metadata from Crossref via DOI_lookup.DoiBibliographyClient when a DOI is detected.
- Implemented a robust citation key generator (authorYearShortTitle) and exposed it in PDFCheckResult.
- Refactored input/input.py orchestrator to:
  - Deduplicate primarily by citation_key, then DOI, then legacy title.
  - Rename PDFs to citation_key.pdf (using unique suffixes when needed).
  - Persist full records to input/input_pdf.json: { citation_key, title, doi, csl }.
- Ran orchestrator; verified rename and JSON augmentation.

Notes
- Citation key style approximates Zotero Better BibTeX: first author family + 4-digit year + short alphanumeric title (stopwords removed).
- Collisions are handled by filesystem-level uniquing (… (1).pdf). Consider future enhancement to add disambiguation letters (a, b, c) into the key itself if preferred.
- Existing entries are merged in-place to add missing fields (csl, citation_key, doi/title) without duplication.

Impacts
- input/input_pdf.json now contains richer metadata and a stable primary key (citation_key) for downstream use across the project.
- No breaking changes for code that only reads title/doi; fields remain present.
