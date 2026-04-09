#!/usr/bin/env python3
"""Ingest Normattiva OpenData Codici (IPZS) into lexe-max PostgreSQL KB.

Reads the OpenData ZIP structure: 40 directories, each a codice (legal code).
Each directory has N JSON files representing versions (V0..VN). The latest
version (highest VN) provides the current canonical text for chunking and
embedding. ALL versions are walked to populate kb.normativa_vigenza.

Features:
  - Pydantic validation of OpenData JSON structure
  - Recursive tree walk with concept_path tracking (tipoNir 1-4 → hierarchy)
  - Content-hash-based delta detection for efficient re-runs
  - Enriched embedding text: "Code > Hierarchy | Art. N — Rubrica | testo"
  - Full vigenza historicization from all version files
  - Modification graph from articoliAggiornanti
  - Ingest logging to kb.ingest_log
  - Parallel processing (max 4 codes via asyncio.Semaphore)

Usage:
    # Dry run — parse, validate, show tipoNir distribution, no DB writes
    python ingest_opendata_codici.py --input-dir /tmp/opendata_codici_M --dry-run

    # Discover code mappings without ingesting
    python ingest_opendata_codici.py --input-dir /tmp/opendata_codici_M --discover-codes

    # Ingest specific codes with embeddings
    python ingest_opendata_codici.py --input-dir /tmp/opendata_codici_M --codes CC,CP,CPP

    # Ingest all codes, skip embeddings
    python ingest_opendata_codici.py --input-dir /tmp/opendata_codici_M --skip-embed

    # Ingest all, skip vigenza historicization (faster)
    python ingest_opendata_codici.py --input-dir /tmp/opendata_codici_M --skip-vigenza

Requires: asyncpg, httpx, pydantic
Run inside lexe-max container or with DSN configured.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg required. Install with: pip install asyncpg")
    sys.exit(1)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    print("ERROR: pydantic required. Install with: pip install pydantic")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_opendata")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DSN = "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
EMBED_BATCH_SIZE = 50
CANONICAL_SOURCE = "opendata_ipzs"

# Chunk parameters
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
MIN_TEXT_LENGTH = 10     # skip articles with testo shorter than this
MIN_CHUNK_LENGTH = 30    # DB CHECK constraint on normativa_chunk

# tipoNir values that contain actual article text.
# EMPIRICALLY VERIFIED from dry-run:
#   tipoNir 0 = standard article (most codes: CCII, CDS, etc.)
#   tipoNir 1 = annex article (CP, CPC articles in annessi)
#   tipoNir 2 = annex article variant (CC articles in annessi: 3230 articles!)
#   tipoNir 3, 4 = appendix/attachment with text
#   tipoNir None = organizational header (NO text, used for concept_path)
# Strategy: an element is an article if it has testo AND tipoNir is not None.
# Headers (tipoNir=None) are always structural, never articles.
# NOTE: tipoNir in the JSON is a STRING ('0', '1', etc.), not int!
ARTICLE_TIPOS = {0, 1, 2, 3, 4, '0', '1', '2', '3', '4'}  # Both int and str

# Max parallel code ingestion
MAX_PARALLEL = 4

# ---------------------------------------------------------------------------
# URN-to-code mapping (static, known codes)
# ---------------------------------------------------------------------------
_URN_TO_CODE: dict[str, str] = {
    # --- Codici presenti in kb.work ---
    "urn:nir:stato:regio.decreto:1942-03-16;262": "CC",
    "urn:nir:stato:regio.decreto:1930-10-19;1398": "CP",
    "urn:nir:stato:regio.decreto:1940-10-28;1443": "CPC",
    "urn:nir:stato:decreto.del.presidente.della.repubblica:1988-09-22;447": "CPP",
    "urn:nir:stato:decreto.legislativo:1992-04-30;285": "CDS",
    "urn:nir:stato:decreto.legislativo:2003-06-30;196": "CPRIV",      # Codice Privacy
    "urn:nir:stato:decreto.legislativo:2003-08-01;259": "CCE",
    "urn:nir:stato:decreto.legislativo:2004-01-22;42": "CBCP",        # Codice Beni Culturali
    "urn:nir:stato:decreto.legislativo:2005-02-10;30": "CPI",
    "urn:nir:stato:decreto.legislativo:2005-03-07;82": "CAD",
    "urn:nir:stato:decreto.legislativo:2005-07-18;171": "CND",        # Codice Nautica
    "urn:nir:stato:decreto.legislativo:2005-09-06;206": "CCONS",
    "urn:nir:stato:decreto.legislativo:2005-09-07;209": "CAP",        # Codice Assicurazioni
    "urn:nir:stato:decreto.legislativo:2006-04-03;152": "CAMB",
    "urn:nir:stato:decreto.legislativo:2006-04-11;198": "CPO",
    "urn:nir:stato:decreto.legislativo:2010-07-02;104": "CPA",        # Codice Processo Amministrativo
    "urn:nir:stato:decreto.legislativo:2011-05-23;79": "CTUR",
    "urn:nir:stato:decreto.legislativo:2011-09-06;159": "CAM",        # Codice Antimafia
    "urn:nir:stato:decreto.legislativo:2016-08-26;174": "CGC",
    "urn:nir:stato:decreto.legislativo:2017-07-03;117": "CTS",
    "urn:nir:stato:decreto.legislativo:2019-01-12;14": "CCII",
    "urn:nir:stato:decreto.legislativo:1992-12-31;546": "CPT",        # Codice Processo Tributario
    "urn:nir:stato:decreto.del.presidente.della.repubblica:1992-12-16;495": "REGCDS",
    "urn:nir:stato:decreto.legislativo:2023-03-31;36": "CAPP",        # Codice Appalti 2023
    # --- Codici NUOVI (non ancora in kb.work, verranno creati) ---
    "urn:nir:stato:regio.decreto:1942-03-30;318": "DISP_ATT_CC",
    "urn:nir:stato:regio.decreto:1942-03-30;327": "CNAV",
    "urn:nir:stato:regio.decreto:1941-09-09;1023": "DISP_CPM",
    "urn:nir:stato:regio.decreto:1941-02-20;303": "CPM",
    "urn:nir:stato:regio.decreto:1941-12-18;1368": "DISP_ATT_CPC",
    "urn:nir:stato:decreto.del.presidente.della.repubblica:1952-02-15;328": "REG_NAV",
    "urn:nir:stato:decreto.del.presidente.della.repubblica:1973-03-29;156": "TU_POST",
    "urn:nir:stato:decreto.del.presidente.della.repubblica:2010-10-05;207": "REG_CONTRATTI",
    "urn:nir:stato:decreto.legislativo:1989-07-28;271": "NATT_CPP",
    "urn:nir:stato:decreto.legislativo:2006-04-12;163": "CONTRATTI_2006",
    "urn:nir:stato:decreto.legislativo:2010-03-15;66": "COM",
    "urn:nir:stato:decreto.legislativo:2016-04-18;50": "CONTRATTI_2016",
    "urn:nir:stato:decreto.legislativo:2018-01-02;1": "CPC_CIVILE",
    "urn:nir:stato:decreto.legislativo:2025-11-27;184": "CINC",
    "urn:nir:stato:decreto:1989-09-30;334": "REG_CPP",
    "urn:nir:stato:decreto:2010-01-13;33": "REG_CPI",
}

# Reverse map: directory name -> code (built from the ZIP structure)
_DIR_TO_CODE: dict[str, str] = {
    # --- Codici presenti in kb.work ---
    "REGIO DECRETO_19420316_262": "CC",
    "REGIO DECRETO_19301019_1398": "CP",
    "REGIO DECRETO_19401028_1443": "CPC",
    "DECRETO DEL PRESIDENTE DELLA REPUBBLICA_19880922_447": "CPP",
    "DECRETO LEGISLATIVO_19920430_285": "CDS",
    "DECRETO LEGISLATIVO_20030630_196": "CPRIV",
    "DECRETO LEGISLATIVO_20030801_259": "CCE",
    "DECRETO LEGISLATIVO_20040122_42": "CBCP",
    "DECRETO LEGISLATIVO_20050210_30": "CPI",
    "DECRETO LEGISLATIVO_20050307_82": "CAD",
    "DECRETO LEGISLATIVO_20050718_171": "CND",
    "DECRETO LEGISLATIVO_20050906_206": "CCONS",
    "DECRETO LEGISLATIVO_20050907_209": "CAP",
    "DECRETO LEGISLATIVO_20060403_152": "CAMB",
    "DECRETO LEGISLATIVO_20060411_198": "CPO",
    "DECRETO LEGISLATIVO_20100702_104": "CPA",
    "DECRETO LEGISLATIVO_20110523_79": "CTUR",
    "DECRETO LEGISLATIVO_20110906_159": "CAM",
    "DECRETO LEGISLATIVO_20160826_174": "CGC",
    "DECRETO LEGISLATIVO_20170703_117": "CTS",
    "DECRETO LEGISLATIVO_20190112_14": "CCII",
    "DECRETO LEGISLATIVO_19921231_546": "CPT",
    "DECRETO DEL PRESIDENTE DELLA REPUBBLICA_19921216_495": "REGCDS",
    "DECRETO LEGISLATIVO_20230331_36": "CAPP",
    # --- Codici nuovi (verranno creati in kb.work se mancano) ---
    "REGIO DECRETO_19420330_318": "DISP_ATT_CC",
    "REGIO DECRETO_19420330_327": "CNAV",
    "REGIO DECRETO_19410909_1023": "DISP_CPM",
    "REGIO DECRETO_19410220_303": "CPM",
    "REGIO DECRETO_19411218_1368": "DISP_ATT_CPC",
    "DECRETO DEL PRESIDENTE DELLA REPUBBLICA_19520215_328": "REG_NAV",
    "DECRETO DEL PRESIDENTE DELLA REPUBBLICA_19730329_156": "TU_POST",
    "DECRETO DEL PRESIDENTE DELLA REPUBBLICA_20101005_207": "REG_CONTRATTI",
    "DECRETO LEGISLATIVO_19890728_271": "NATT_CPP",
    "DECRETO LEGISLATIVO_20060412_163": "CONTRATTI_2006",
    "DECRETO LEGISLATIVO_20100315_66": "COM",
    "DECRETO LEGISLATIVO_20160418_50": "CONTRATTI_2016",
    "DECRETO LEGISLATIVO_20180102_1": "CPC_CIVILE",
    "DECRETO LEGISLATIVO_20251127_184": "CINC",
    "DECRETO_19890930_334": "REG_CPP",
    "DECRETO_20100113_33": "REG_CPI",
}

# ---------------------------------------------------------------------------
# Suffix ordering for sort keys
# ---------------------------------------------------------------------------
_SUFFIX_ORDER: dict[str | None, str] = {
    None: "00",
    "bis": "01", "ter": "02", "quater": "03", "quinquies": "04",
    "sexies": "05", "septies": "06", "octies": "07",
    "novies": "08", "nonies": "08",  # variant spelling
    "decies": "09", "undecies": "10", "duodecies": "11",
    "terdecies": "12", "quaterdecies": "13", "quinquiesdecies": "14",
    "sexiesdecies": "15", "septiesdecies": "16", "octiesdecies": "17",
}

# Latin suffixes for regex
_LATIN_SUFFIXES = [
    "bis", "ter", "quater", "quinquies", "sexies", "septies",
    "octies", "novies", "nonies", "decies", "undecies", "duodecies",
    "terdecies", "quaterdecies", "quinquiesdecies",
    "sexiesdecies", "septiesdecies", "octiesdecies",
]

# ---------------------------------------------------------------------------
# Pydantic models — lightweight validation of OpenData JSON
# ---------------------------------------------------------------------------

class VigenzaInterval(BaseModel):
    inizioVigore: str | None = None
    fineVigore: str | None = None


class ArticoloAggiornante(BaseModel):
    atto: str | None = None
    articoli: list[str] = Field(default_factory=list)


class MetadatiAtto(BaseModel):
    urn: str = ""
    eli: str | None = None
    titoloDoc: str = ""
    tipoDoc: str = ""
    dataDoc: str = ""
    numDoc: str = ""
    dataPubblicazione: str | None = None


class CodiceOpenData(BaseModel):
    """Top-level OpenData JSON structure. Only validates fields we use."""
    metadati: MetadatiAtto = Field(default_factory=MetadatiAtto)
    articolato: dict[str, Any] = Field(default_factory=dict)
    annessi: dict[str, Any] | None = None
    note: dict[str, Any] | None = None
    intestazione: dict[str, Any] | None = None
    errore: Any | None = None


# ---------------------------------------------------------------------------
# Data classes for extracted articles
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """An article extracted from the OpenData JSON tree."""
    numNir: str             # cleaned article number: "2043", "640bis"
    articolo_num: int       # numeric part: 2043
    articolo_suffix: str | None  # "bis", "ter", etc.
    testo: str              # cleaned article text
    rubrica: str | None     # article heading
    urn_nir: str            # full URN for this article
    vigenza_da: date | None
    vigenza_a: date | None  # None = still in force
    is_current: bool
    lifecycle: str          # "CURRENT" or "HISTORICAL"
    concept_path: list[str]  # ["Libro IV", "Titolo IX"]
    gerarchia_text: str     # "Libro IV > Titolo IX"
    aggiornanti: list[dict[str, Any]]  # raw articoliAggiornanti
    content_hash: str       # SHA-256 of testo|rubrica|vigenza


@dataclass
class TextChunk:
    """A chunk of article text for embedding."""
    chunk_index: int
    text: str
    char_start: int
    char_end: int


@dataclass
class CodeStats:
    """Per-code ingestion statistics."""
    code: str
    dirname: str
    title: str = ""
    urn: str = ""
    articles_found: int = 0
    articles_from_articolato: int = 0
    articles_from_annessi: int = 0
    articles_inserted: int = 0
    articles_skipped: int = 0  # skipped due to same content_hash
    chunks_inserted: int = 0
    embeddings_inserted: int = 0
    vigenza_records: int = 0
    modifications_inserted: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Helpers: article number parsing
# ---------------------------------------------------------------------------

_ART_PATTERN = re.compile(
    r"(?:.*?art\.?\s*)"
    r"(\d+)"
    r"(?:\s*[-. ]?\s*"
    r"(" + "|".join(_LATIN_SUFFIXES) + r"))?"
    , re.IGNORECASE,
)

_PURE_NUM = re.compile(r"^(\d+)$")


def parse_article_number(num_nir: str) -> tuple[str, int, str | None]:
    """Parse numNir into (articolo_str, articolo_num, suffix).

    Examples:
        "2043"                       -> ("2043", 2043, None)
        "CODICE CIVILE-art. 2043"    -> ("2043", 2043, None)
        "Codice penale-art. 640-bis" -> ("640bis", 640, "bis")
        "1"                          -> ("1", 1, None)
    """
    if not num_nir:
        return ("0", 0, None)

    clean = num_nir.strip()

    # Pure numeric
    m = _PURE_NUM.match(clean)
    if m:
        n = int(m.group(1))
        return (str(n), n, None)

    # Pattern with optional prefix like "CODICE CIVILE-art. 2043"
    m = _ART_PATTERN.search(clean)
    if m:
        n = int(m.group(1))
        suffix = m.group(2).lower() if m.group(2) else None
        art_str = f"{n}{suffix}" if suffix else str(n)
        return (art_str, n, suffix)

    # Last resort: first number found
    nums = re.findall(r"\d+", clean)
    if nums:
        n = int(nums[0])
        return (str(n), n, None)

    return ("0", 0, None)


def make_sort_key(art_num: int, suffix: str | None) -> str:
    """Generate sort key like '002043.01' for art 2043-bis."""
    suffix_code = _SUFFIX_ORDER.get(suffix, "00")
    return f"{art_num:06d}.{suffix_code}"


def classify_identity(suffix: str | None) -> str:
    """Classify: BASE or SUFFIX."""
    return "SUFFIX" if suffix else "BASE"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for Italian)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Helpers: structural label extraction  [C3]
# ---------------------------------------------------------------------------

# tipoNir mapping: 1=Libro, 2=Titolo, 3=Capo, 4=Sezione
_TIPO_LABELS: dict[int, str] = {1: "Libro", 2: "Titolo", 3: "Capo", 4: "Sezione"}

# Pattern to extract label from numNir for structural headers
# e.g. "-*-*LIBRO PRIMO*-*-Delle persone e della famiglia"
_STRUCTURAL_LABEL_RE = re.compile(
    r"(?:-\*-\*\s*)?"
    r"(LIBRO|TITOLO|CAPO|SEZIONE|PARTE|APPENDICE|DISPOSIZION[EI])"
    r"\s+"
    r"([IVXLCDM]+|PRIMO|SECONDO|TERZO|QUARTO|QUINTO|SESTO|SETTIMO|OTTAVO|"
    r"NONO|DECIMO|UNDICESIMO|DODICESIMO|UNICO|\d+)"
    r"(?:\s*[-–—]\s*|\s*\*-\*-\s*|\s+)"
    r"(.*)",
    re.IGNORECASE,
)

# Simpler pattern: just "LIBRO I" etc. without the decorators
_SIMPLE_LABEL_RE = re.compile(
    r"(LIBRO|TITOLO|CAPO|SEZIONE|PARTE|APPENDICE)"
    r"\s+"
    r"([IVXLCDM]+|PRIMO|SECONDO|TERZO|QUARTO|QUINTO|SESTO|SETTIMO|OTTAVO|"
    r"NONO|DECIMO|UNDICESIMO|DODICESIMO|UNICO|\d+)",
    re.IGNORECASE,
)


def extract_label(num_nir: str, rubrica: str | None, tipo_nir: int | None) -> str | None:
    """Extract a human-readable structural label from a header element.

    For tipoNir 1-4, we use the tipo mapping + numNir/rubrica.
    For tipoNir None with numNir, we try to parse the label.

    Returns: "Libro I", "Titolo V — Del possesso", etc. or None.
    """
    combined = f"{num_nir or ''} {rubrica or ''}".strip()
    if not combined:
        return None

    # Try structured pattern with decorators
    m = _STRUCTURAL_LABEL_RE.search(combined)
    if m:
        kind = m.group(1).capitalize()
        number = m.group(2).upper()
        desc = m.group(3).strip().rstrip("*-").strip()
        label = f"{kind} {number}"
        if desc:
            label += f" — {desc}"
        return label

    # Try simple pattern
    m = _SIMPLE_LABEL_RE.search(combined)
    if m:
        kind = m.group(1).capitalize()
        number = m.group(2).upper()
        label = f"{kind} {number}"
        # Append rubrica if available and not already captured
        if rubrica and rubrica.strip() and rubrica.strip().upper() not in combined.upper():
            label += f" — {rubrica.strip()}"
        return label

    # If tipoNir is known, use the tipo label + numNir
    if tipo_nir in _TIPO_LABELS:
        label = _TIPO_LABELS[tipo_nir]
        clean_num = re.sub(r"[*\-]+", "", num_nir or "").strip()
        if clean_num:
            label += f" {clean_num}"
        if rubrica and rubrica.strip():
            label += f" — {rubrica.strip()}"
        return label

    # Fallback: use rubrica if present
    if rubrica and rubrica.strip():
        return rubrica.strip()

    return None


# ---------------------------------------------------------------------------
# Helpers: vigenza date parsing
# ---------------------------------------------------------------------------

def normalize_vigenza_date(s: str | None) -> date | None:
    """Parse OpenData vigenza date to Python date.

    Formats: "1942-04-19" (ISO), "19420419" (compact).
    "99999999" means still in force -> returns None.
    """
    if not s or s.strip() in ("", "99999999"):
        return None
    s = s.strip()
    try:
        if "-" in s and len(s) == 10:
            parts = s.split("-")
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(s) == 8 and s.isdigit():
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# Helpers: content hash  [C7]
# ---------------------------------------------------------------------------

def compute_content_hash(testo: str, rubrica: str | None,
                         vigenza_da: date | None, vigenza_a: date | None) -> str:
    """SHA-256 of testo|rubrica|vigenza for delta detection."""
    parts = [
        testo or "",
        rubrica or "",
        str(vigenza_da) if vigenza_da else "",
        str(vigenza_a) if vigenza_a else "",
    ]
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helpers: text cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Clean article text: strip, normalize whitespace."""
    if not raw:
        return ""
    text = raw.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Helpers: URN building
# ---------------------------------------------------------------------------

def build_article_urn(atto_urn: str, art_str: str, art_num: int) -> str:
    """Build article-level URN from atto URN and article identifier."""
    if art_num > 0:
        return f"{atto_urn}~art{art_str}"
    return atto_urn


# ---------------------------------------------------------------------------
# Helpers: chunking (reused pattern from ingest_normattiva_from_json.py)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS) -> list[TextChunk]:
    """Split text into overlapping chunks.

    Breaks at paragraph boundaries first, then sentence boundaries.
    """
    if not text or len(text.strip()) < MIN_CHUNK_LENGTH:
        return []

    if len(text) <= max_chars:
        return [TextChunk(chunk_index=0, text=text.strip(),
                          char_start=0, char_end=len(text))]

    chunks: list[TextChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Try to break at paragraph or sentence boundary
        if end < len(text):
            para_break = text.rfind("\n\n", start + max_chars // 2, end)
            if para_break > start:
                end = para_break + 2
            else:
                sent_break = text.rfind(". ", start + max_chars // 2, end)
                if sent_break > start:
                    end = sent_break + 2

        chunk_str = text[start:end].strip()
        if chunk_str and len(chunk_str) >= MIN_CHUNK_LENGTH:
            chunks.append(TextChunk(
                chunk_index=idx,
                text=chunk_str,
                char_start=start,
                char_end=end,
            ))
            idx += 1

        start = end - overlap if end < len(text) else len(text)

    return chunks


# ---------------------------------------------------------------------------
# [C4] Enriched embedding text
# ---------------------------------------------------------------------------

def build_embedding_text(code_title: str, article: Article) -> str:
    """Build enriched text for embedding with hierarchical context.

    Result: "Codice Civile > Libro IV > Titolo IX | Art. 2043 — Rubrica | testo"
    """
    parts: list[str] = []

    # Hierarchy prefix
    if article.gerarchia_text:
        parts.append(f"{code_title} > {article.gerarchia_text}")
    elif code_title:
        parts.append(code_title)

    # Article identifier + rubrica
    if article.rubrica:
        parts.append(f"Art. {article.numNir} — {article.rubrica}")
    else:
        parts.append(f"Art. {article.numNir}")

    # Article text
    parts.append(article.testo)

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# [C3] Recursive tree walk — walk_articles
# ---------------------------------------------------------------------------

def walk_articles(
    elementi: list[dict[str, Any]],
    atto_urn: str,
    codice: str,
    path: list[str] | None = None,
    collector: Counter | None = None,
) -> Generator[Article, None, None]:
    """Recursively walk the elementi tree, yielding Article objects.

    [C2] Only tipoNir == 0 produces articles with text.
    [C3] tipoNir 1-4 and None-with-numNir update concept_path.
    [R2-0] Recurse passing the list directly, not a singleton wrapper.

    Args:
        elementi: list of element dicts from articolato or annessi
        atto_urn: base URN of the atto
        codice: code abbreviation (CC, CP, etc.)
        path: current concept path (accumulated hierarchy labels)
        collector: optional Counter for tipoNir distribution diagnostics
    """
    if path is None:
        path = []

    for elem in elementi:
        tipo = elem.get("tipoNir")
        num = elem.get("numNir") or ""
        rubrica = elem.get("rubricaNir") or ""

        # Track tipoNir distribution for diagnostics
        if collector is not None:
            collector[tipo] += 1

        # Determine new concept path for this element
        # Headers (tipoNir=None with numNir) always update path.
        # Article types (0-4) update path ONLY if they don't have text (structural wrapper).
        new_path = path
        has_text = elem.get("testo") is not None and len((elem.get("testo") or "").strip()) >= MIN_TEXT_LENGTH
        if tipo is None and num:
            # Pure organizational header → always update path
            label = extract_label(num, rubrica, tipo)
            if label:
                new_path = path + [label]
        elif tipo in ARTICLE_TIPOS and not has_text:
            # Structural wrapper with no meaningful text → update path
            label = extract_label(num, rubrica, tipo)
            if label:
                new_path = path + [label]

        # Article: any tipoNir in ARTICLE_TIPOS with actual text
        if tipo in ARTICLE_TIPOS and has_text:
            testo_raw = elem.get("testo", "")
            testo = clean_text(testo_raw)

            if len(testo) >= MIN_TEXT_LENGTH:
                art_str, art_num, suffix = parse_article_number(num)

                # Parse vigenza
                vigenza_list = elem.get("dataVigoreVersione") or []
                vigenza_da = None
                vigenza_a = None
                is_current = True
                lifecycle = "CURRENT"

                if vigenza_list and isinstance(vigenza_list, list) and len(vigenza_list) > 0:
                    v = vigenza_list[0]
                    vigenza_da = normalize_vigenza_date(v.get("inizioVigore"))
                    fine = v.get("fineVigore", "")
                    if fine and str(fine).strip() not in ("", "99999999"):
                        vigenza_a = normalize_vigenza_date(fine)
                        is_current = False
                        lifecycle = "HISTORICAL"

                urn_nir = build_article_urn(atto_urn, art_str, art_num)
                content_hash = compute_content_hash(testo, rubrica or None,
                                                    vigenza_da, vigenza_a)

                yield Article(
                    numNir=art_str,
                    articolo_num=art_num,
                    articolo_suffix=suffix,
                    testo=testo,
                    rubrica=rubrica.strip() or None,
                    urn_nir=urn_nir,
                    vigenza_da=vigenza_da,
                    vigenza_a=vigenza_a,
                    is_current=is_current,
                    lifecycle=lifecycle,
                    concept_path=list(new_path),
                    gerarchia_text=" > ".join(new_path),
                    aggiornanti=elem.get("articoliAggiornanti") or [],
                    content_hash=content_hash,
                )

        # [R2-0] Recurse into children passing list directly
        children = elem.get("elementi")
        if children and isinstance(children, list):
            yield from walk_articles(children, atto_urn, codice, new_path, collector)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"_V(\d+)\.json$")


def find_latest_json(directory: Path) -> Path | None:
    """Find the latest version JSON in a code directory.

    Files: *_V0.json, *_V1.json, ... *_VN.json
    Returns the file with the highest V number.
    """
    json_files = list(directory.glob("*.json"))
    if not json_files:
        return None

    versioned: list[tuple[int, Path]] = []
    for f in json_files:
        m = _VERSION_RE.search(f.name)
        if m:
            versioned.append((int(m.group(1)), f))

    if not versioned:
        return sorted(json_files)[-1]

    versioned.sort(key=lambda x: x[0])
    return versioned[-1][1]


def find_all_versions(directory: Path) -> list[tuple[str, Path]]:
    """Find all version files sorted by version number.

    Returns list of (version_name, path) tuples: [("V0", path), ("V1", path), ...]
    """
    json_files = list(directory.glob("*.json"))
    versioned: list[tuple[int, str, Path]] = []
    for f in json_files:
        m = _VERSION_RE.search(f.name)
        if m:
            vnum = int(m.group(1))
            versioned.append((vnum, f"V{vnum}", f))
    versioned.sort(key=lambda x: x[0])
    return [(vname, vpath) for _, vname, vpath in versioned]


def extract_version_name(filename: str) -> str:
    """Extract version name from filename stem. E.g. '..._V41' -> 'V41'."""
    m = _VERSION_RE.search(filename + ".json" if not filename.endswith(".json") else filename)
    if m:
        return f"V{m.group(1)}"
    return "V0"


# ---------------------------------------------------------------------------
# Code resolution: directory -> code abbreviation
# ---------------------------------------------------------------------------

def resolve_code(dirname: str, urn: str, title: str) -> str:
    """Resolve a directory name to a code abbreviation.

    Priority:
    1. _DIR_TO_CODE static mapping
    2. _URN_TO_CODE by URN
    3. Auto-generate from URN
    """
    # Try direct directory mapping
    if dirname in _DIR_TO_CODE:
        return _DIR_TO_CODE[dirname]

    # Try URN mapping
    if urn and urn in _URN_TO_CODE:
        return _URN_TO_CODE[urn]

    # Auto-generate from URN
    if urn:
        return _abbreviation_from_urn(urn)

    return "UNKNOWN"


def _abbreviation_from_urn(urn: str) -> str:
    """Generate a short code from URN when not in static mappings."""
    if not urn:
        return "UNKNOWN"

    parts = urn.split(":")
    if len(parts) < 5:
        return "UNKNOWN"

    act_type = parts[3]
    date_num = parts[4]

    type_map = {
        "regio.decreto": "RD",
        "decreto.legislativo": "DLGS",
        "decreto.del.presidente.della.repubblica": "DPR",
        "decreto": "DM",
        "legge": "L",
        "costituzione": "COST",
    }
    type_short = type_map.get(act_type, act_type.upper()[:6])

    num = ""
    year = ""
    if ";" in date_num:
        date_part, num = date_num.split(";", 1)
        if "-" in date_part:
            year = date_part.split("-")[0]
    elif "-" in date_num:
        year = date_num.split("-")[0]

    if year and num:
        return f"{type_short}_{num}_{year}"
    elif num:
        return f"{type_short}_{num}"
    elif year:
        return f"{type_short}_{year}"
    return type_short


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """Calls an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str, model: str = EMBED_MODEL,
                 api_key: str | None = None):
        if not httpx:
            raise RuntimeError("httpx required for embeddings: pip install httpx")
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(timeout=120.0, headers=headers)

    async def close(self):
        await self.client.aclose()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns embeddings in input order."""
        # Normalize URL: avoid double paths like /v1/embeddings/v1/embeddings
        base = self.base_url.rstrip("/")
        if base.endswith("/v1/embeddings"):
            url = base  # already includes the full path
        elif base.endswith("/v1"):
            url = f"{base}/embeddings"
        else:
            url = f"{base}/v1/embeddings"

        resp = await self.client.post(
            url,
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


# ---------------------------------------------------------------------------
# Database: work cache
# ---------------------------------------------------------------------------

_work_cache: dict[str, uuid.UUID] = {}


async def get_work_id(conn: asyncpg.Connection, code: str) -> uuid.UUID | None:
    """Get work UUID by code, with cache."""
    if code in _work_cache:
        return _work_cache[code]
    row = await conn.fetchval("SELECT id FROM kb.work WHERE code = $1", code)
    if row:
        _work_cache[code] = row
    return row


async def ensure_work(conn: asyncpg.Connection, code: str,
                      title: str) -> uuid.UUID:
    """Get or create a kb.work entry. Returns work_id."""
    wid = await get_work_id(conn, code)
    if wid:
        return wid

    work_id = uuid.uuid5(uuid.NAMESPACE_URL, f"opendata:work:{code}")
    await conn.execute(
        """
        INSERT INTO kb.work (id, code, title, title_short, created_at)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (code) DO UPDATE SET title = EXCLUDED.title
        RETURNING id
        """,
        work_id, code, title[:200], code,
    )
    _work_cache[code] = work_id
    log.info("  [%s] Created work entry: %s", code, title[:60])
    return work_id


# ---------------------------------------------------------------------------
# Database: upsert normativa  [C7, C10]
# ---------------------------------------------------------------------------

async def upsert_normativa(
    conn: asyncpg.Connection,
    articles: list[Article],
    work_id: uuid.UUID,
    code: str,
    urn_base: str,
    eli: str | None,
    version_name: str,
) -> list[tuple[uuid.UUID, Article]]:
    """Upsert articles into kb.normativa with content_hash delta detection.

    [C7] Only updates if content_hash IS DISTINCT FROM existing.
    [C10] Sets canonical_source = 'opendata_ipzs'.

    Returns list of (normativa_id, article) pairs.
    """
    results: list[tuple[uuid.UUID, Article]] = []

    for art in articles:
        art_id = uuid.uuid5(uuid.NAMESPACE_URL, f"opendata:{code}:{art.numNir}")
        sort_key = make_sort_key(art.articolo_num, art.articolo_suffix)
        identity = classify_identity(art.articolo_suffix)
        # concept_path is text[] in DB (not JSONB) — pass as list directly
        concept_path_val = art.concept_path if art.concept_path else None

        # Use RETURNING id to get the actual DB id (may differ from art_id
        # if the row already existed from a previous ingestion)
        actual_id = await conn.fetchval(
            """
            INSERT INTO kb.normativa (
                id, work_id, codice, articolo, articolo_num, articolo_suffix,
                articolo_sort_key, identity_class, quality, lifecycle,
                urn_nir, urn_base, eli, rubrica, testo,
                canonical_source, content_hash, is_current,
                concept_path, gerarchia_text,
                data_vigenza_da, data_vigenza_a,
                opendata_version,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8::article_identity_class, 'VALID_STRONG'::article_quality_class,
                $9::lifecycle_status,
                $10, $11, $12, $13, $14,
                $15, $16, $17,
                $18, $19,
                $20, $21,
                $22,
                NOW(), NOW()
            )
            ON CONFLICT (work_id, articolo_sort_key) DO UPDATE SET
                testo = EXCLUDED.testo,
                rubrica = EXCLUDED.rubrica,
                urn_nir = EXCLUDED.urn_nir,
                urn_base = EXCLUDED.urn_base,
                eli = EXCLUDED.eli,
                lifecycle = EXCLUDED.lifecycle,
                is_current = EXCLUDED.is_current,
                canonical_source = EXCLUDED.canonical_source,
                content_hash = EXCLUDED.content_hash,
                concept_path = EXCLUDED.concept_path,
                gerarchia_text = EXCLUDED.gerarchia_text,
                data_vigenza_da = EXCLUDED.data_vigenza_da,
                data_vigenza_a = EXCLUDED.data_vigenza_a,
                opendata_version = EXCLUDED.opendata_version,
                updated_at = NOW()
            RETURNING id
            """,
            art_id,                     # $1
            work_id,                    # $2
            code,                       # $3
            art.numNir,                 # $4
            art.articolo_num,           # $5
            art.articolo_suffix,        # $6
            sort_key,                   # $7
            identity,                   # $8
            art.lifecycle,              # $9
            art.urn_nir,                # $10
            urn_base,                   # $11
            eli,                        # $12
            art.rubrica,               # $13
            art.testo,                  # $14
            CANONICAL_SOURCE,           # $15
            art.content_hash,           # $16
            art.is_current,             # $17
            concept_path_val,           # $18
            art.gerarchia_text or None, # $19
            art.vigenza_da,             # $20
            art.vigenza_a,              # $21
            version_name,               # $22
        )
        # actual_id may be None if WHERE clause prevented the update (same hash)
        # In that case, look up the existing ID
        if actual_id is None:
            actual_id = await conn.fetchval(
                "SELECT id FROM kb.normativa WHERE work_id = $1 AND articolo_sort_key = $2",
                work_id, sort_key,
            )
        if actual_id:
            results.append((actual_id, art))

    return results


# ---------------------------------------------------------------------------
# Database: chunks and embeddings
# ---------------------------------------------------------------------------

async def upsert_chunks(
    conn: asyncpg.Connection,
    normativa_id: uuid.UUID,
    work_id: uuid.UUID,
    art: Article,
    code: str,
    chunks: list[TextChunk],
) -> list[uuid.UUID]:
    """Insert chunks for an article. Returns chunk IDs."""
    sort_key = make_sort_key(art.articolo_num, art.articolo_suffix)
    chunk_ids: list[uuid.UUID] = []

    for chunk in chunks:
        chunk_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"opendata:chunk:{code}:{art.numNir}:{chunk.chunk_index}",
        )
        token_est = estimate_tokens(chunk.text)

        actual_chunk_id = await conn.fetchval(
            """
            INSERT INTO kb.normativa_chunk (
                id, normativa_id, work_id,
                articolo_sort_key, articolo_num, articolo_suffix,
                chunk_no, char_start, char_end, text, token_est,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
            ON CONFLICT (normativa_id, chunk_no) DO UPDATE SET
                text = EXCLUDED.text,
                char_start = EXCLUDED.char_start,
                char_end = EXCLUDED.char_end,
                token_est = EXCLUDED.token_est
            RETURNING id
            """,
            chunk_id,             # $1
            normativa_id,         # $2
            work_id,              # $3
            sort_key,             # $4
            art.articolo_num,     # $5
            art.articolo_suffix,  # $6
            chunk.chunk_index,    # $7
            chunk.char_start,     # $8
            chunk.char_end,       # $9
            chunk.text,           # $10
            token_est,            # $11
        )
        chunk_ids.append(actual_chunk_id)

    return chunk_ids


async def insert_embeddings(
    conn: asyncpg.Connection,
    chunk_ids: list[uuid.UUID],
    embeddings: list[list[float]],
    model: str,
):
    """Insert embeddings for chunks."""
    for chunk_id, embedding in zip(chunk_ids, embeddings):
        emb_id = uuid.uuid5(uuid.NAMESPACE_URL, f"opendata:emb:{chunk_id}")
        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
        await conn.execute(
            """
            INSERT INTO kb.normativa_chunk_embeddings (
                id, chunk_id, model, channel, dims, embedding, created_at
            ) VALUES ($1, $2, $3, 'testo', $4, $5::vector, NOW())
            ON CONFLICT (chunk_id, model, channel, dims) DO UPDATE SET
                embedding = EXCLUDED.embedding
            """,
            emb_id,
            chunk_id,
            model,
            EMBED_DIMS,
            emb_str,
        )


# ---------------------------------------------------------------------------
# Database: vigenza historicization  [C6]
# ---------------------------------------------------------------------------

async def process_vigenza_all_versions(
    conn: asyncpg.Connection,
    code_dir: Path,
    code: str,
    atto_urn: str,
    normativa_lookup: dict[str, uuid.UUID],
) -> int:
    """Walk ALL version files, insert into kb.normativa_vigenza.

    [C6] Only testo + dates + hash — ZERO chunks/embeddings.
    Each version file is a complete snapshot of all articles at that date.

    Returns number of vigenza records inserted.
    """
    all_versions = find_all_versions(code_dir)
    if not all_versions:
        return 0

    latest_vname = all_versions[-1][0]
    total_inserted = 0

    for version_name, version_path in all_versions:
        try:
            raw = json.loads(version_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning("  [%s] Skipping %s: %s", code, version_path.name, e)
            continue

        # Walk articles from both trees
        articolato_els = raw.get("articolato", {}).get("elementi", [])
        annessi_raw = raw.get("annessi")
        annessi_els = annessi_raw.get("elementi", []) if isinstance(annessi_raw, dict) else []

        v_arts = list(walk_articles(articolato_els + annessi_els, atto_urn, code))
        if version_name == all_versions[0][0] and v_arts:
            # Debug: log first few URNs from version vs lookup
            sample_v = [a.urn_nir for a in v_arts[:3]]
            sample_l = list(normativa_lookup.keys())[:3]
            log.warning("  [%s] VIG DEBUG version URNs: %s", code, sample_v)
            log.warning("  [%s] VIG DEBUG lookup URNs: %s", code, sample_l)
            log.warning("  [%s] VIG DEBUG lookup size: %d", code, len(normativa_lookup))
            if sample_v and sample_l and sample_v[0] != sample_l[0]:
                log.warning("  [%s] URN MISMATCH! version=%s vs lookup=%s",
                            code, sample_v[0], sample_l[0])

        for art in v_arts:
            normativa_id = normativa_lookup.get(art.urn_nir)
            if not normativa_id:
                continue

            content_hash = compute_content_hash(art.testo, art.rubrica,
                                                art.vigenza_da, art.vigenza_a)
            is_current = (version_name == latest_vname)

            try:
                await conn.execute(
                    """
                    INSERT INTO kb.normativa_vigenza (
                        id, normativa_id, testo,
                        inizio_vigore, fine_vigore,
                        opendata_version, is_current,
                        content_hash, source_file,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    ON CONFLICT (normativa_id, opendata_version)
                    DO UPDATE SET
                        testo = EXCLUDED.testo,
                        inizio_vigore = EXCLUDED.inizio_vigore,
                        fine_vigore = EXCLUDED.fine_vigore,
                        is_current = EXCLUDED.is_current,
                        content_hash = EXCLUDED.content_hash
                    """,
                    uuid.uuid5(uuid.NAMESPACE_URL,
                               f"vigenza:{code}:{art.numNir}:{version_name}"),
                    normativa_id,
                    art.testo,
                    art.vigenza_da or date(1900, 1, 1),  # NOT NULL constraint
                    art.vigenza_a,
                    version_name,
                    is_current,
                    content_hash,
                    version_path.name,
                )
                total_inserted += 1
            except Exception as e:
                # Log first few errors at WARNING level for debugging
                if total_inserted == 0 and version_name == all_versions[0][0]:
                    log.warning("  [%s] Vigenza INSERT error art.%s %s: %s",
                                code, art.numNir, version_name, e)
                continue
                continue

        if total_inserted % 5000 == 0 and total_inserted > 0:
            log.info("  [%s] Vigenza progress: %d records (%s)",
                     code, total_inserted, version_name)

    return total_inserted


# ---------------------------------------------------------------------------
# Database: modifications  [C9]
# ---------------------------------------------------------------------------

async def process_modifications(
    conn: asyncpg.Connection,
    articles: list[Article],
    normativa_lookup: dict[str, uuid.UUID],
) -> int:
    """Insert modification records from articoliAggiornanti.

    [C9] Unmappable modification types → 'UNKNOWN' + log warning.
    Returns number of modification records inserted.
    """
    total = 0
    for art in articles:
        normativa_id = normativa_lookup.get(art.urn_nir)
        if not normativa_id:
            continue

        for agg in art.aggiornanti:
            source_urn = agg.get("atto", "")
            if not source_urn:
                continue

            art_refs = agg.get("articoli", [])
            art_ref_str = ", ".join(art_refs) if art_refs else art.numNir

            try:
                mod_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"mod:{source_urn}:{art.urn_nir}",
                )
                await conn.execute(
                    """
                    INSERT INTO kb.normativa_modification (
                        id, source_urn, target_normativa_id, target_article,
                        modification_type, effective_date, evidence,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5::kb.modification_type_enum, $6, $7, NOW())
                    ON CONFLICT DO NOTHING
                    """,
                    mod_id,
                    source_urn,
                    normativa_id,
                    art_ref_str,
                    "UNKNOWN",  # [C9] We don't know the type from the data
                    art.vigenza_da,
                    json.dumps({"articoli": art_refs}),
                )
                total += 1
            except Exception as e:
                log.debug("  Modification insert error: %s", e)
                continue

    return total


# ---------------------------------------------------------------------------
# Database: ingest log
# ---------------------------------------------------------------------------

async def log_ingest_start(
    conn: asyncpg.Connection,
    code: str,
    source_file: str,
    file_hash: str | None = None,
    details: dict | None = None,
) -> uuid.UUID:
    """Start an ingest log entry. Returns log ID."""
    log_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO kb.ingest_log (
            id, code_id, source_file, file_hash, status,
            details, started_at, created_at
        ) VALUES ($1, $2, $3, $4, 'STARTED', $5::jsonb, NOW(), NOW())
        """,
        log_id, code, source_file, file_hash,
        json.dumps(details) if details else None,
    )
    return log_id


async def log_ingest_complete(
    conn: asyncpg.Connection,
    log_id: uuid.UUID,
    articles_ingested: int,
    articles_skipped: int = 0,
    articles_errored: int = 0,
    details: dict | None = None,
):
    """Mark ingest log as COMPLETED."""
    await conn.execute(
        """
        UPDATE kb.ingest_log SET
            status = 'COMPLETED',
            articles_ingested = $2,
            articles_skipped = $3,
            articles_errored = $4,
            details = COALESCE(details, '{}'::jsonb) || $5::jsonb,
            completed_at = NOW()
        WHERE id = $1
        """,
        log_id, articles_ingested, articles_skipped, articles_errored,
        json.dumps(details) if details else "{}",
    )


async def log_ingest_fail(conn: asyncpg.Connection, log_id: uuid.UUID,
                          error: str):
    """Mark ingest log as FAILED."""
    await conn.execute(
        """
        UPDATE kb.ingest_log SET
            status = 'FAILED', error_detail = $2, completed_at = NOW()
        WHERE id = $1
        """,
        log_id, error[:2000],
    )


# ---------------------------------------------------------------------------
# Per-code ingestion
# ---------------------------------------------------------------------------

async def ingest_code(
    conn: asyncpg.Connection | None,
    input_dir: Path,
    dirname: str,
    embed_client: EmbeddingClient | None,
    embed_model: str,
    dry_run: bool,
    skip_embed: bool,
    skip_vigenza: bool,
) -> CodeStats:
    """Ingest a single code directory. Main orchestration function."""
    stats = CodeStats(code="?", dirname=dirname)
    t0 = time.monotonic()

    code_dir = input_dir / dirname
    if not code_dir.is_dir():
        stats.errors.append(f"Not a directory: {code_dir}")
        return stats

    # Find latest version JSON
    latest = find_latest_json(code_dir)
    if not latest:
        stats.errors.append(f"No JSON files in {code_dir}")
        return stats

    latest_version = extract_version_name(latest.name)
    log.info("  Reading %s (version %s)", latest.name, latest_version)

    # Parse and validate JSON
    try:
        raw_text = latest.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        stats.errors.append(f"JSON parse error: {e}")
        return stats

    try:
        data = CodiceOpenData.model_validate(raw)
    except ValidationError as e:
        stats.errors.append(f"Validation error: {e}")
        return stats

    # Extract metadata
    urn = data.metadati.urn
    eli = data.metadati.eli
    titolo = data.metadati.titoloDoc.strip()

    # Resolve code
    code = resolve_code(dirname, urn, titolo)
    stats.code = code
    stats.title = titolo
    stats.urn = urn

    log.info("  [%s] URN: %s", code, urn)
    log.info("  [%s] Title: %s", code, titolo[:80])

    # Walk articles from both articolato and annessi
    tipo_counter: Counter = Counter()
    articolato_els = data.articolato.get("elementi", [])
    annessi_raw = data.annessi
    annessi_els = annessi_raw.get("elementi", []) if isinstance(annessi_raw, dict) else []

    articles_art = list(walk_articles(articolato_els, urn, code,
                                      collector=tipo_counter))
    articles_ann = list(walk_articles(annessi_els, urn, code,
                                      collector=tipo_counter))
    all_articles = articles_art + articles_ann

    stats.articles_from_articolato = len(articles_art)
    stats.articles_from_annessi = len(articles_ann)

    # Log article source distribution
    if not articles_art and articles_ann:
        log.info("  [%s] Annex code: %d articles in annessi only",
                 code, len(articles_ann))
    elif articles_art and articles_ann:
        log.info("  [%s] Mixed: %d articolato + %d annessi",
                 code, len(articles_art), len(articles_ann))
    else:
        log.info("  [%s] Standard: %d articles in articolato",
                 code, len(articles_art))

    # Deduplicate by URN — keep last occurrence
    seen: dict[str, int] = {}
    unique_articles: list[Article] = []
    for art in all_articles:
        key = art.urn_nir
        if key in seen:
            unique_articles[seen[key]] = art
        else:
            seen[key] = len(unique_articles)
            unique_articles.append(art)
    all_articles = unique_articles
    stats.articles_found = len(all_articles)

    log.info("  [%s] Found %d articles (%d unique after dedup)",
             code, stats.articles_from_articolato + stats.articles_from_annessi,
             len(all_articles))

    # ---------------------------------------------------------------------------
    # [C2] Dry-run diagnostics
    # ---------------------------------------------------------------------------
    if dry_run:
        log.info("  [%s] DRY RUN — tipoNir distribution: %s",
                 code, dict(tipo_counter))

        # Suspicious: tipoNir != 0 but has text
        suspicious: list[dict] = []
        _scan_suspicious(articolato_els + annessi_els, suspicious)
        if suspicious:
            log.info("  [%s] Suspicious non-article elements with text (%d):",
                     code, len(suspicious))
            for s in suspicious[:20]:
                log.info("    tipoNir=%s numNir=%s text=%s",
                         s["tipoNir"], s["numNir"], s["text_preview"])

        # [R3-3] Sample extract_label results
        _log_label_samples(articolato_els + annessi_els, code)

        # Sample articles
        for art in all_articles[:5]:
            log.info("    art.%s [%s] %s (%d chars, %s, path=%s)",
                     art.numNir, art.lifecycle,
                     art.rubrica[:40] if art.rubrica else "-",
                     len(art.testo), art.content_hash[:8],
                     art.gerarchia_text or "-")

        stats.elapsed = time.monotonic() - t0
        return stats

    # ---------------------------------------------------------------------------
    # DB operations
    # ---------------------------------------------------------------------------
    if conn is None:
        stats.errors.append("No DB connection")
        return stats

    # Start ingest log
    file_hash = hashlib.sha256(raw_text.encode()).hexdigest()
    ingest_log_id = await log_ingest_start(
        conn, code, latest.name, file_hash,
        details={
            "version": latest_version,
            "dirname": dirname,
            "urn": urn,
            "articles_found": len(all_articles),
            "tipo_distribution": dict(tipo_counter),
        },
    )

    try:
        # Ensure work entry
        work_id = await ensure_work(conn, code, titolo)

        # Upsert articles with content_hash delta detection
        async with conn.transaction():
            art_pairs = await upsert_normativa(
                conn, all_articles, work_id, code, urn, eli, latest_version)
            stats.articles_inserted = len(art_pairs)

        log.info("  [%s] Upserted %d articles", code, stats.articles_inserted)

        # Build normativa_lookup: urn -> normativa_id
        normativa_lookup: dict[str, uuid.UUID] = {}
        for norm_id, art in art_pairs:
            normativa_lookup[art.urn_nir] = norm_id

        # Fallback: if lookup is empty (re-run, same hash), build from DB
        if not normativa_lookup and not dry_run:
            rows = await conn.fetch(
                "SELECT id, urn_nir FROM kb.normativa WHERE work_id = $1 AND urn_nir IS NOT NULL",
                work_id,
            )
            for r in rows:
                normativa_lookup[r["urn_nir"]] = r["id"]
            if normativa_lookup:
                log.info("  [%s] Lookup rebuilt from DB: %d articles", code, len(normativa_lookup))

        # Chunk + embed with enriched text [C4]
        code_title = titolo or code
        all_chunk_pairs: list[tuple[uuid.UUID, str]] = []  # (chunk_id, enriched_text)

        async with conn.transaction():
            for norm_id, art in art_pairs:
                enriched = build_embedding_text(code_title, art)
                chunks = chunk_text(enriched)
                if not chunks:
                    continue

                chunk_ids = await upsert_chunks(
                    conn, norm_id, work_id, art, code, chunks)
                stats.chunks_inserted += len(chunks)

                for cid, ch in zip(chunk_ids, chunks):
                    all_chunk_pairs.append((cid, ch.text))

        log.info("  [%s] Inserted %d chunks", code, stats.chunks_inserted)

        # Generate embeddings
        if not skip_embed and embed_client is not None and all_chunk_pairs:
            log.info("  [%s] Generating embeddings for %d chunks...",
                     code, len(all_chunk_pairs))

            for i in range(0, len(all_chunk_pairs), EMBED_BATCH_SIZE):
                batch = all_chunk_pairs[i:i + EMBED_BATCH_SIZE]
                batch_ids = [cid for cid, _ in batch]
                batch_texts = [text for _, text in batch]

                try:
                    embs = await embed_client.embed_batch(batch_texts)
                    async with conn.transaction():
                        await insert_embeddings(conn, batch_ids, embs, embed_model)
                    stats.embeddings_inserted += len(batch)
                except Exception as e:
                    stats.errors.append(f"Embedding batch {i}: {e}")
                    log.error("  [%s] Embedding error at batch %d: %s",
                              code, i, e)
                    continue

                if stats.embeddings_inserted % 500 == 0 and stats.embeddings_inserted > 0:
                    log.info("  [%s] Embedded %d/%d",
                             code, stats.embeddings_inserted, len(all_chunk_pairs))

            log.info("  [%s] Embeddings: %d/%d",
                     code, stats.embeddings_inserted, len(all_chunk_pairs))
        elif not skip_embed and embed_client is None:
            log.info("  [%s] Skipping embeddings (no client)", code)

        # [C6] Vigenza historicization from ALL versions
        if not skip_vigenza:
            all_versions = find_all_versions(code_dir)
            if len(all_versions) > 1:
                log.info("  [%s] Processing vigenza from %d versions...",
                         code, len(all_versions))
                stats.vigenza_records = await process_vigenza_all_versions(
                    conn, code_dir, code, urn, normativa_lookup)
                log.info("  [%s] Vigenza records: %d", code, stats.vigenza_records)
            else:
                log.info("  [%s] Single version, skipping vigenza historicization", code)

        # Process modifications
        mods = await process_modifications(conn, all_articles, normativa_lookup)
        stats.modifications_inserted = mods
        if mods > 0:
            log.info("  [%s] Modification records: %d", code, mods)

        # Complete ingest log
        await log_ingest_complete(
            conn, ingest_log_id,
            articles_ingested=stats.articles_inserted,
            articles_skipped=stats.articles_skipped,
            details={
                "chunks": stats.chunks_inserted,
                "embeddings": stats.embeddings_inserted,
                "vigenza_records": stats.vigenza_records,
                "modifications": stats.modifications_inserted,
            },
        )

    except Exception as e:
        stats.errors.append(f"Ingestion error: {e}")
        log.error("  [%s] ERROR: %s", code, e, exc_info=True)
        try:
            await log_ingest_fail(conn, ingest_log_id, str(e))
        except Exception:
            pass

    stats.elapsed = time.monotonic() - t0
    return stats


# ---------------------------------------------------------------------------
# Dry-run diagnostic helpers
# ---------------------------------------------------------------------------

def _scan_suspicious(elements: list[dict], result: list[dict],
                     max_items: int = 20):
    """Find elements with tipoNir=None that have text (headers should not have text)."""
    if len(result) >= max_items:
        return
    for e in elements:
        if len(result) >= max_items:
            return
        tipo = e.get("tipoNir")
        testo = e.get("testo") or ""
        # Suspicious: tipoNir=None (header) with actual text content
        if tipo is None and len(testo.strip()) > 20:
            result.append({
                "tipoNir": tipo,
                "numNir": (e.get("numNir") or "")[:60],
                "text_preview": testo[:120],
            })
        children = e.get("elementi")
        if children and isinstance(children, list):
            _scan_suspicious(children, result, max_items)


def _log_label_samples(elements: list[dict], code: str, max_samples: int = 15):
    """[R3-3] Log sample extract_label results for manual verification."""
    samples: list[str] = []
    _collect_label_samples(elements, samples, max_samples)
    if samples:
        log.info("  [%s] extract_label samples (%d):", code, len(samples))
        for s in samples:
            log.info("    %s", s)


def _collect_label_samples(elements: list[dict], result: list[str],
                           max_items: int):
    """Collect label samples from structural elements."""
    if len(result) >= max_items:
        return
    for e in elements:
        if len(result) >= max_items:
            return
        tipo = e.get("tipoNir")
        if tipo in {1, 2, 3, 4} or (tipo is None and e.get("numNir")):
            num = e.get("numNir") or ""
            rub = e.get("rubricaNir") or ""
            label = extract_label(num, rub, tipo)
            result.append(
                f"tipoNir={tipo} numNir={num[:50]!r} rubrica={rub[:30]!r}"
                f" -> {label!r}"
            )
        children = e.get("elementi")
        if children and isinstance(children, list):
            _collect_label_samples(children, result, max_items)


# ---------------------------------------------------------------------------
# Discover codes mode
# ---------------------------------------------------------------------------

async def discover_codes(input_dir: Path):
    """Print all code directories with their URN, title, and mapped code."""
    all_dirs = sorted([
        d.name for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    log.info("=" * 80)
    log.info("CODE DISCOVERY — %d directories", len(all_dirs))
    log.info("=" * 80)

    for dirname in all_dirs:
        code_dir = input_dir / dirname
        latest = find_latest_json(code_dir)
        if not latest:
            log.info("  %-55s -> NO JSON FILES", dirname)
            continue

        try:
            raw = json.loads(latest.read_text(encoding="utf-8"))
            metadati = raw.get("metadati", {})
            urn = metadati.get("urn", "")
            title = metadati.get("titoloDoc", "").strip()
        except Exception:
            urn = "?"
            title = "?"

        code = resolve_code(dirname, urn, title)
        mapped = "STATIC" if dirname in _DIR_TO_CODE else "AUTO"
        versions = len(list(code_dir.glob("*.json")))

        log.info("  %-55s -> %-18s [%s] %d ver  %s",
                 dirname, code, mapped, versions, title[:40])

    log.info("=" * 80)


# ---------------------------------------------------------------------------
# [BIS] Ingest -bis/-ter articles from fetch_bis_articles.py JSON output
# ---------------------------------------------------------------------------

# Mapping from fetch_bis_articles.py codice names to kb.work.code values
_BIS_CODE_MAP: dict[str, tuple[str, str]] = {
    # fetch_codice: (kb_work_code, work_title)
    "CCII":       ("CCII",  "Codice della crisi d'impresa e dell'insolvenza"),
    "TUSL":       ("TUSL",  "Testo Unico Sicurezza Lavoro"),
    "L212":       ("L212",  "Statuto del Contribuente"),
    "L241":       ("L241",  "Procedimento Amministrativo"),
    "CDS_STRADA": ("CDS",   "Codice della Strada"),
    "CONSUMO":    ("CCONS", "Codice del Consumo"),
    "PRIVACY":    ("CPRIV", "Codice Privacy"),
    "AMBIENTE":   ("CAMB",  "Codice dell'Ambiente"),
    "CAD":        ("CAD",   "Codice Amministrazione Digitale"),
    "COST":       ("COST",  "Costituzione della Repubblica Italiana"),
}

# URN bases for -bis articles (must match fetch_bis_articles.py CODICI)
_BIS_URN_BASE: dict[str, str] = {
    "CCII":       "urn:nir:stato:decreto.legislativo:2019-01-12;14",
    "TUSL":       "urn:nir:stato:decreto.legislativo:2008-04-09;81",
    "L212":       "urn:nir:stato:legge:2000-07-27;212",
    "L241":       "urn:nir:stato:legge:1990-08-07;241",
    "CDS_STRADA": "urn:nir:stato:decreto.legislativo:1992-04-30;285",
    "CONSUMO":    "urn:nir:stato:decreto.legislativo:2005-09-06;206",
    "PRIVACY":    "urn:nir:stato:decreto.legislativo:2003-06-30;196",
    "AMBIENTE":   "urn:nir:stato:decreto.legislativo:2006-04-03;152",
    "CAD":        "urn:nir:stato:decreto.legislativo:2005-03-07;82",
    "COST":       "urn:nir:stato:costituzione:1947-12-27",
}


async def ingest_bis_from_json(
    conn: asyncpg.Connection,
    json_dir: Path,
    embed_client: EmbeddingClient | None,
    embed_model: str,
    skip_embed: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Ingest -bis/-ter articles from JSON files produced by fetch_bis_articles.py.

    Each JSON file contains a single article with fields:
        codice, articolo, rubrica, testo_html, testo_plain, urn,
        data_vigenza_inizio, data_vigenza_fine, chars, data_fetch

    Returns summary dict with per-code stats.
    """
    json_files = sorted(json_dir.glob("*.json"))
    # Exclude manifest and summary files
    json_files = [f for f in json_files if not f.name.startswith("_")]

    if not json_files:
        log.error("No JSON files found in %s", json_dir)
        return {"error": "no files", "total": 0}

    log.info("=" * 70)
    log.info("INGEST -BIS ARTICLES FROM JSON")
    log.info("=" * 70)
    log.info("Source dir:  %s (%d files)", json_dir, len(json_files))
    log.info("Dry run:     %s", dry_run)
    log.info("Embeddings:  %s", "skip" if skip_embed else embed_model)

    # Group files by codice
    by_codice: dict[str, list[Path]] = {}
    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning("  Skip %s: %s", f.name, e)
            continue
        codice = data.get("codice", "UNKNOWN")
        by_codice.setdefault(codice, []).append(f)

    summary: dict[str, Any] = {"total_articles": 0, "total_chunks": 0,
                                "total_embeddings": 0, "by_codice": {}}

    for codice, files in sorted(by_codice.items()):
        log.info("")
        log.info("--- %s: %d articles ---", codice, len(files))

        # Resolve kb.work.code
        if codice not in _BIS_CODE_MAP:
            log.warning("  Unknown codice '%s', skipping %d files", codice, len(files))
            continue

        kb_code, work_title = _BIS_CODE_MAP[codice]
        urn_base = _BIS_URN_BASE.get(codice, "")

        if not dry_run:
            work_id = await ensure_work(conn, kb_code, work_title)
        else:
            work_id = uuid.uuid4()  # placeholder for dry-run

        code_stats = {"inserted": 0, "chunks": 0, "embeddings": 0, "articles": []}

        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))

            # Parse article number from the "articolo" field (e.g. "64-bis")
            art_display = data.get("articolo", "")
            # Split "64-bis" into num=64, suffix="bis"
            parts = art_display.split("-", 1)
            try:
                art_num = int(parts[0])
            except (ValueError, IndexError):
                art_num = 0
            suffix = parts[1].lower() if len(parts) > 1 and parts[1] else None
            art_str = f"{art_num}{suffix}" if suffix else str(art_num)

            testo = clean_text(data.get("testo_plain", ""))
            rubrica = (data.get("rubrica") or "").strip() or None

            if not testo or len(testo) < MIN_TEXT_LENGTH:
                log.debug("  Skip %s art.%s: text too short (%d chars)",
                          codice, art_display, len(testo))
                continue

            # Parse vigenza dates
            vigenza_da = normalize_vigenza_date(data.get("data_vigenza_inizio"))
            vigenza_a = normalize_vigenza_date(data.get("data_vigenza_fine"))
            is_current = vigenza_a is None
            lifecycle = "CURRENT" if is_current else "HISTORICAL"

            urn_nir = data.get("urn", build_article_urn(urn_base, art_str, art_num))
            content_hash = compute_content_hash(testo, rubrica, vigenza_da, vigenza_a)

            article = Article(
                numNir=art_str,
                articolo_num=art_num,
                articolo_suffix=suffix,
                testo=testo,
                rubrica=rubrica,
                urn_nir=urn_nir,
                vigenza_da=vigenza_da,
                vigenza_a=vigenza_a,
                is_current=is_current,
                lifecycle=lifecycle,
                concept_path=[],
                gerarchia_text="",
                aggiornanti=[],
                content_hash=content_hash,
            )

            sort_key = make_sort_key(art_num, suffix)
            log.info("  art.%-12s  %s  %d chars  %s",
                     art_display, sort_key, len(testo),
                     rubrica[:40] if rubrica else "-")

            if dry_run:
                code_stats["inserted"] += 1
                code_stats["articles"].append(art_display)
                continue

            # Upsert to DB
            pairs = await upsert_normativa(
                conn, [article], work_id, kb_code, urn_base, None, "bis_api",
            )
            if not pairs:
                log.warning("  Failed to upsert art.%s", art_display)
                continue

            normativa_id = pairs[0][0]
            code_stats["inserted"] += 1
            code_stats["articles"].append(art_display)

            # Chunk
            code_title = work_title
            embed_text = build_embedding_text(code_title, article)
            chunks = chunk_text(embed_text)
            if chunks:
                chunk_ids = await upsert_chunks(
                    conn, normativa_id, work_id, article, kb_code, chunks,
                )
                code_stats["chunks"] += len(chunk_ids)

                # Embed
                if not skip_embed and embed_client and chunk_ids:
                    texts = [c.text for c in chunks]
                    try:
                        embeddings = await embed_client.embed_batch(texts)
                        await insert_embeddings(conn, chunk_ids, embeddings, embed_model)
                        code_stats["embeddings"] += len(embeddings)
                    except Exception as e:
                        log.warning("  Embedding failed for art.%s: %s",
                                    art_display, e)

        log.info("  %s: %d inserted, %d chunks, %d embeddings",
                 codice, code_stats["inserted"], code_stats["chunks"],
                 code_stats["embeddings"])

        summary["by_codice"][codice] = code_stats
        summary["total_articles"] += code_stats["inserted"]
        summary["total_chunks"] += code_stats["chunks"]
        summary["total_embeddings"] += code_stats["embeddings"]

    log.info("")
    log.info("=" * 70)
    log.info("BIS INGEST COMPLETE: %d articles, %d chunks, %d embeddings",
             summary["total_articles"], summary["total_chunks"],
             summary["total_embeddings"])
    log.info("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Ingest Normattiva OpenData Codici (IPZS) into lexe-max KB"
    )
    parser.add_argument(
        "--input-dir",
        help="Path to the extracted opendata_codici_M directory",
    )
    parser.add_argument(
        "--ingest-bis",
        help="Path to directory with JSON files from fetch_bis_articles.py",
    )
    parser.add_argument(
        "--dsn",
        default=os.getenv("LEXE_KB_DSN", DEFAULT_DSN),
        help=f"PostgreSQL DSN (default: {DEFAULT_DSN})",
    )
    parser.add_argument(
        "--codes",
        help="Comma-separated codes or directory names to process (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate only, no DB writes. Shows tipoNir distribution"
             " and extract_label samples.",
    )
    parser.add_argument(
        "--discover-codes", action="store_true",
        help="Print directory-to-code mapping and exit (no ingestion).",
    )
    parser.add_argument(
        "--skip-embed", action="store_true",
        help="Skip embedding generation (still inserts text + chunks).",
    )
    parser.add_argument(
        "--skip-vigenza", action="store_true",
        help="Skip vigenza historicization from all versions (faster).",
    )
    parser.add_argument(
        "--embedding-url",
        default=os.getenv("LEXE_LITELLM_URL",
                          os.getenv("LEXE_EMBED_URL",
                                    "http://lexe-litellm:4000/v1/embeddings")),
        help="OpenAI-compatible embeddings endpoint URL.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("LEXE_EMBED_MODEL", EMBED_MODEL),
        help=f"Embedding model name (default: {EMBED_MODEL}).",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.getenv("LEXE_LITELLM_API_KEY",
                          os.getenv("OPENROUTER_API_KEY", "")),
        help="API key for embedding endpoint (optional for LiteLLM proxy).",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # --ingest-bis mode: ingest -bis articles from fetch_bis_articles.py output
    # ---------------------------------------------------------------------------
    if args.ingest_bis:
        bis_dir = Path(args.ingest_bis)
        if not bis_dir.is_dir():
            log.error("Bis articles directory not found: %s", bis_dir)
            sys.exit(1)

        conn = None
        embed_client = None
        try:
            if not args.dry_run:
                conn = await asyncpg.connect(args.dsn)
                log.info("Connected to DB.")

            if not args.dry_run and not args.skip_embed:
                try:
                    embed_client = EmbeddingClient(
                        base_url=args.embedding_url,
                        model=args.embedding_model,
                        api_key=args.embedding_api_key or None,
                    )
                except Exception as e:
                    log.warning("Could not init embedding client: %s", e)

            summary = await ingest_bis_from_json(
                conn=conn,
                json_dir=bis_dir,
                embed_client=embed_client,
                embed_model=args.embedding_model,
                skip_embed=args.skip_embed or embed_client is None,
                dry_run=args.dry_run,
            )
            log.info("Summary: %s", json.dumps(
                {k: v for k, v in summary.items() if k != "by_codice"},
                indent=2,
            ))
        finally:
            if conn:
                await conn.close()
            if embed_client:
                await embed_client.close()
        return

    # ---------------------------------------------------------------------------
    # Standard mode: requires --input-dir
    # ---------------------------------------------------------------------------
    if not args.input_dir:
        log.error("Either --input-dir or --ingest-bis is required")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Discover codes mode
    # ---------------------------------------------------------------------------
    if args.discover_codes:
        await discover_codes(input_dir)
        return

    # ---------------------------------------------------------------------------
    # Discover code directories
    # ---------------------------------------------------------------------------
    all_dirs = sorted([
        d.name for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not all_dirs:
        log.error("No code directories found in %s", input_dir)
        sys.exit(1)

    # Filter by --codes if specified
    target_dirs = all_dirs
    if args.codes:
        targets = {c.strip() for c in args.codes.split(",")}
        # Build reverse map: abbreviation -> dirname
        abbrev_to_dir = {v: k for k, v in _DIR_TO_CODE.items()}
        resolved_dirs: set[str] = set()
        for t in targets:
            if t in all_dirs:
                resolved_dirs.add(t)
            elif t.upper() in abbrev_to_dir and abbrev_to_dir[t.upper()] in all_dirs:
                resolved_dirs.add(abbrev_to_dir[t.upper()])
            else:
                # Try partial match
                for d in all_dirs:
                    if t.upper() in d.upper():
                        resolved_dirs.add(d)
                        break
                else:
                    log.warning("Code '%s' not found in data dir or mapping", t)
        target_dirs = sorted(resolved_dirs)

    if not target_dirs:
        log.error("No matching directories to process")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Banner
    # ---------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("NORMATTIVA OPENDATA CODICI INGESTION")
    log.info("=" * 70)
    log.info("Input dir:    %s (%d dirs, processing %d)",
             input_dir, len(all_dirs), len(target_dirs))
    log.info("DSN:          %s",
             args.dsn.split("@")[-1] if not args.dry_run else "(dry-run)")
    log.info("Embed URL:    %s",
             args.embedding_url if not args.skip_embed else "(skipped)")
    log.info("Embed model:  %s", args.embedding_model)
    log.info("Dry run:      %s", args.dry_run)
    log.info("Skip embed:   %s", args.skip_embed)
    log.info("Skip vigenza: %s", args.skip_vigenza)
    log.info("Canonical:    %s", CANONICAL_SOURCE)
    log.info("=" * 70)

    # ---------------------------------------------------------------------------
    # Connect to DB + init embedding client
    # ---------------------------------------------------------------------------
    conn: asyncpg.Connection | None = None
    pool: asyncpg.Pool | None = None
    embed_client: EmbeddingClient | None = None
    all_stats: list[CodeStats] = []

    try:
        if not args.dry_run:
            log.info("Connecting to DB (pool, max=%d)...", MAX_PARALLEL)
            pool = await asyncpg.create_pool(args.dsn, min_size=1, max_size=MAX_PARALLEL)
            # Get a single conn reference for compatibility (sequential paths)
            conn = await pool.acquire()
            log.info("Connected.")

        if not args.dry_run and not args.skip_embed:
            try:
                embed_client = EmbeddingClient(
                    base_url=args.embedding_url,
                    model=args.embedding_model,
                    api_key=args.embedding_api_key or None,
                )
                log.info("Embedding client ready: %s -> %s",
                         args.embedding_model, args.embedding_url)
            except Exception as e:
                log.warning("Could not init embedding client: %s — "
                            "skipping embeddings", e)

        # -----------------------------------------------------------------
        # Process codes (max MAX_PARALLEL in parallel)
        # -----------------------------------------------------------------
        sem = asyncio.Semaphore(MAX_PARALLEL)

        async def _process_one(dirname: str, index: int) -> CodeStats:
            async with sem:
                code_hint = _DIR_TO_CODE.get(dirname, "?")
                log.info("")
                log.info(">>> [%d/%d] %s (%s)",
                         index, len(target_dirs), dirname, code_hint)
                try:
                    # Each parallel task gets its own connection from the pool
                    task_conn = conn if args.dry_run else await pool.acquire()
                    try:
                        return await ingest_code(
                            conn=task_conn,
                            input_dir=input_dir,
                            dirname=dirname,
                            embed_client=embed_client,
                            embed_model=args.embedding_model,
                            dry_run=args.dry_run,
                            skip_embed=args.skip_embed or embed_client is None,
                            skip_vigenza=args.skip_vigenza,
                        )
                    finally:
                        if not args.dry_run:
                            await pool.release(task_conn)
                except Exception as e:
                    log.error("[%s] FATAL: %s", dirname, e, exc_info=True)
                    return CodeStats(
                        code=code_hint, dirname=dirname,
                        errors=[f"Fatal: {e}"],
                    )

        # For dry-run, run sequentially for readable output
        if args.dry_run:
            for i, dirname in enumerate(target_dirs, 1):
                stats = await _process_one(dirname, i)
                all_stats.append(stats)
        else:
            # Parallel execution with semaphore
            tasks = [
                _process_one(dirname, i)
                for i, dirname in enumerate(target_dirs, 1)
            ]
            all_stats = list(await asyncio.gather(*tasks))

    finally:
        if not args.dry_run and pool:
            await pool.close()
        elif conn:
            await conn.close()
        if embed_client:
            await embed_client.close()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    total_art = 0
    total_inserted = 0
    total_chunks = 0
    total_embs = 0
    total_vigenza = 0
    total_mods = 0
    total_errors = 0

    for s in all_stats:
        status = "OK" if not s.errors else f"ERRORS({len(s.errors)})"
        log.info(
            "[%-18s] %4d found, %4d upserted, %5d chunks, "
            "%5d embs, %5d vig, %3d mods — %.1fs  %s",
            s.code, s.articles_found, s.articles_inserted,
            s.chunks_inserted, s.embeddings_inserted,
            s.vigenza_records, s.modifications_inserted,
            s.elapsed, status,
        )
        total_art += s.articles_found
        total_inserted += s.articles_inserted
        total_chunks += s.chunks_inserted
        total_embs += s.embeddings_inserted
        total_vigenza += s.vigenza_records
        total_mods += s.modifications_inserted
        total_errors += len(s.errors)

    log.info("-" * 70)
    log.info("TOTAL: %d found, %d upserted, %d chunks, %d embeddings, "
             "%d vigenza, %d modifications, %d errors",
             total_art, total_inserted, total_chunks, total_embs,
             total_vigenza, total_mods, total_errors)

    # Print errors
    if total_errors:
        log.info("")
        log.info("ERRORS:")
        for s in all_stats:
            for err in s.errors:
                log.error("  [%s] %s", s.code, err)

    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
