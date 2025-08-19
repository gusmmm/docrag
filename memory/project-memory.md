# Project overview
- the pdf documents to be processed are scientific journal papers

# RAG pipeline
- process the pdf using docling
- extract the text using a semantic approach, by sections and headings
- extract images
- extract tables
- extract references at the end of the paper
- extract the bibliographic reference of the paper containing the DOI

# Implementation notes
- OCR tuned to improve spacing/quality: prefer Tesseract CLI with force_full_page_ocr=True and lang=["auto"], auto-detecting the binary; fallback to EasyOCR(force_full_page_ocr=True) when Tesseract is unavailable.
- Picture captions via remote VLM (Gemini on OpenRouter) are optional; pipeline runs without captions if OPENROUTER_API_KEY is not set.

