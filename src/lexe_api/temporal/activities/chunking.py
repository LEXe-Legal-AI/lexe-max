"""Chunking activity for KB nightly sync.

Re-chunks modified normativa articles using the same legal-aware
algorithm as scripts/chunk_normativa.py (TARGET_CHARS=1000, OVERLAP=150).

Schema reference: kb.normativa_chunk (055_chunking_schema.sql)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from uuid import UUID

import asyncpg
from temporalio import activity

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
)

# ---------------------------------------------------------------------------
# Chunking parameters (mirror of scripts/chunk_normativa.py)
# ---------------------------------------------------------------------------

TARGET_CHARS = 1000
MIN_CHARS = 900
MAX_CHARS = 1200
OVERLAP_CHARS = 150
MIN_CHUNK_LEN = 30  # DB CHECK constraint on kb.normativa_chunk

# Legal split patterns (Italian legal text markers)
LEGAL_SPLIT_PATTERNS = [
    r"^\s*Art\.\s",
    r"^\s*Articolo\s",
    r"^\s*Comma\s",
    r"^\s*Lettera\s",
    r"^\s*Capo\s",
    r"^\s*Sezione\s",
    r"^\s*\d+\.\s",
    r"^\s*[a-z]\)\s",
    r"^\s*\d+-[a-z]+\.\s",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    """Result of a rechunking operation."""

    articles_rechunked: int = 0
    chunks_created: int = 0
    new_chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text normalisation (same as chunk_normativa.py)
# ---------------------------------------------------------------------------


def _normalize_for_chunking(text: str) -> str:
    """Normalize text for chunking and embedding.

    - Typographic quotes/apostrophes to standard
    - NBSP to regular space
    - Collapse multiple spaces
    - Limit consecutive newlines to 2
    """
    if not text:
        return ""

    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking logic (same algorithm as chunk_normativa.py)
# ---------------------------------------------------------------------------


@dataclass
class _Chunk:
    """Internal chunk representation."""

    chunk_no: int
    char_start: int
    char_end: int
    text: str
    token_est: int


def _find_split_point(text: str, target_pos: int, window: int = 100) -> int:
    """Find best split point near target_pos.

    Priority: legal marker > paragraph break > sentence end > fallback.
    """
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    search_text = text[start:end]

    # 1. Legal markers
    for pattern in LEGAL_SPLIT_PATTERNS:
        match = re.search(pattern, search_text, re.MULTILINE)
        if match:
            return start + match.start()

    # 2. Paragraph break
    para_break = search_text.find("\n\n")
    if para_break != -1:
        return start + para_break

    # 3. Sentence end
    sentence_end = re.search(r"\.\s", search_text)
    if sentence_end:
        return start + sentence_end.end()

    # 4. Fallback
    return target_pos


def _chunk_text(text: str) -> list[_Chunk]:
    """Split text into chunks with legal-aware split points."""
    if not text or len(text.strip()) < MIN_CHUNK_LEN:
        return []

    normalised = _normalize_for_chunking(text)
    if len(normalised) < MIN_CHUNK_LEN:
        return []

    chunks: list[_Chunk] = []
    pos = 0
    chunk_no = 0

    while pos < len(normalised):
        if pos + MAX_CHARS >= len(normalised):
            end = len(normalised)
        else:
            target = pos + TARGET_CHARS
            end = _find_split_point(normalised, target)
            if end <= pos:
                end = min(pos + MAX_CHARS, len(normalised))
            if end > pos + MAX_CHARS:
                end = pos + MAX_CHARS

        chunk_text = normalised[pos:end].strip()

        if len(chunk_text) < MIN_CHUNK_LEN:
            if chunks and chunk_no > 0:
                prev = chunks[-1]
                new_text = prev.text + " " + chunk_text
                chunks[-1] = _Chunk(
                    chunk_no=prev.chunk_no,
                    char_start=prev.char_start,
                    char_end=end,
                    text=new_text,
                    token_est=len(new_text) // 4,
                )
            break

        chunks.append(
            _Chunk(
                chunk_no=chunk_no,
                char_start=pos,
                char_end=end,
                text=chunk_text,
                token_est=len(chunk_text) // 4,
            )
        )

        chunk_no += 1
        pos = end - OVERLAP_CHARS
        if pos < end - MIN_CHARS:
            pos = end

    return chunks


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


@activity.defn
async def rechunk_articles(normativa_ids: list[str]) -> dict:
    """Re-chunk modified normativa articles.

    For each normativa_id:
    1. DELETE existing chunks from kb.normativa_chunk (CASCADE cleans FTS + embeddings)
    2. Fetch article text from kb.normativa
    3. Chunk with legal-aware algorithm
    4. INSERT new chunks

    Args:
        normativa_ids: List of normativa UUID strings to rechunk.

    Returns:
        Dict serialisation of ChunkResult.
    """
    if not normativa_ids:
        return asdict(ChunkResult())

    result = ChunkResult()

    conn: asyncpg.Connection = await asyncpg.connect(DATABASE_URL)
    try:
        for idx, nid_str in enumerate(normativa_ids):
            if idx % 20 == 0:
                activity.heartbeat()

            nid = UUID(nid_str)

            # Fetch article data
            row = await conn.fetchrow(
                """
                SELECT id, work_id, articolo_sort_key, articolo_num,
                       articolo_suffix, testo
                FROM kb.normativa
                WHERE id = $1
                """,
                nid,
            )

            if row is None:
                logger.warning("rechunk_articles: normativa %s not found, skipping", nid_str)
                continue

            text = row["testo"]
            if not text or len(text.strip()) < MIN_CHUNK_LEN:
                logger.debug("rechunk_articles: normativa %s text too short, skipping", nid_str)
                continue

            chunks = _chunk_text(text)
            if not chunks:
                continue

            async with conn.transaction():
                # Delete old chunks (CASCADE removes FTS + embeddings)
                await conn.execute(
                    "DELETE FROM kb.normativa_chunk WHERE normativa_id = $1",
                    nid,
                )

                # Insert new chunks
                for chunk in chunks:
                    chunk_id = await conn.fetchval(
                        """
                        INSERT INTO kb.normativa_chunk (
                            normativa_id, work_id, articolo_sort_key,
                            articolo_num, articolo_suffix,
                            chunk_no, char_start, char_end, text, token_est
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id::text
                        """,
                        nid,
                        row["work_id"],
                        row["articolo_sort_key"],
                        row["articolo_num"],
                        row["articolo_suffix"],
                        chunk.chunk_no,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.text,
                        chunk.token_est,
                    )
                    if chunk_id:
                        result.new_chunk_ids.append(chunk_id)

            result.articles_rechunked += 1
            result.chunks_created += len(chunks)

    finally:
        await conn.close()

    logger.info(
        "rechunk_articles: rechunked=%d chunks_created=%d",
        result.articles_rechunked,
        result.chunks_created,
    )
    return asdict(result)
