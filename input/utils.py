from __future__ import annotations

"""
Utilities for the input pipeline.

Responsibilities
- Safe filename and path helpers
- JSON DB load/save
- Deduplication helpers (by citation_key, DOI, title)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- filename/path helpers ---

def safe_filename_from_key(key: str) -> str:
    """Sanitize a citation key for filesystem usage.

    Keys are expected to be [a-z0-9]; we enforce and cap length.
    """
    s = "".join(ch for ch in (key or "").lower() if ch.isalnum())
    if not s:
        s = "key"
    if len(s) > 80:
        s = s[:80]
    return s


def unique_path(base_dir: Path, stem: str, ext: str = ".pdf") -> Path:
    """Return a unique path under base_dir for stem+ext, adding (n) if needed."""
    cand = base_dir / f"{stem}{ext}"
    if not cand.exists():
        return cand
    i = 1
    while True:
        cand = base_dir / f"{stem} ({i}){ext}"
        if not cand.exists():
            return cand
        i += 1


def normalize_title(s: str) -> str:
    """Normalize a title or filename for comparison (case/spacing-insensitive)."""
    name = s
    if "." in name:
        name = name.rsplit(".", 1)[0]
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split()).strip().casefold()
    return name


# --- JSON DB helpers ---

def load_db(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
    except Exception:
        pass
    return []


def save_db(path: Path, items: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


# --- dedup helpers ---

def find_by_key(items: List[Dict[str, Any]], key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    k = str(key).strip().lower()
    for obj in items:
        v = str(obj.get("citation_key", "")).strip().lower()
        if v and v == k:
            return obj
    return None


def find_by_title(items: List[Dict[str, Any]], title: str) -> Optional[Dict[str, Any]]:
    key = normalize_title(title)
    for obj in items:
        t = str(obj.get("title", ""))
        if normalize_title(t) == key:
            return obj
    return None


def find_by_doi(items: List[Dict[str, Any]], doi: Optional[str], *, normalizer) -> Optional[Dict[str, Any]]:
    if not doi:
        return None
    try:
        key = normalizer(str(doi))
    except Exception:
        key = str(doi).strip()
    for obj in items:
        v = obj.get("doi")
        if not v:
            continue
        try:
            vd = normalizer(str(v))
        except Exception:
            vd = str(v).strip()
        if vd.casefold() == key.casefold():
            return obj
    return None
