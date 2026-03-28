"""KB upsert activity for nightly sync.

Inserts or updates normativa articles in the KB, maintaining version
history via kb.normativa_vigenza and is_current flagging.

Schema references:
- kb.normativa (010_normativa_schema.sql)
- kb.normativa_vigenza (083_opendata_platinum.sql)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import asyncpg
from temporalio import activity

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class UpsertResult:
    """Result of a batch normativa upsert."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    modified_normativa_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    """SHA-256 hash of normalised text for change detection."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


@activity.defn
async def upsert_normativa(articles: list[dict]) -> dict:
    """Upsert normativa articles into kb.normativa with version tracking.

    For each article (identified by codice_redazionale + article number):
    - If no current row exists: INSERT as is_current=true
    - If current row exists and text hash differs: mark old is_current=false,
      INSERT new with is_current=true, and record in kb.normativa_vigenza
    - If current row exists and text hash matches: skip (unchanged)

    Args:
        articles: List of dicts from fetch_act_articles with keys:
            codice_redazionale, article, text, vigenza_inizio, vigenza_fine, html.

    Returns:
        Dict serialisation of UpsertResult.
    """
    if not articles:
        return asdict(UpsertResult())

    result = UpsertResult()

    conn: asyncpg.Connection = await asyncpg.connect(DATABASE_URL)
    try:
        for idx, art in enumerate(articles):
            if idx % 50 == 0:
                activity.heartbeat()

            codice = art.get("codice_redazionale", "")
            articolo = art.get("article", "")
            text = art.get("text", "")
            html = art.get("html", "")
            vigenza_inizio = art.get("vigenza_inizio")
            vigenza_fine = art.get("vigenza_fine")

            if not codice or not text.strip():
                continue

            new_hash = _content_hash(text)

            # ----- Look up existing current row -----
            existing = await conn.fetchrow(
                """
                SELECT id, content_hash, testo
                FROM kb.normativa
                WHERE codice = $1
                  AND articolo = $2
                  AND is_current = true
                LIMIT 1
                """,
                codice,
                articolo,
            )

            if existing is None:
                # --- INSERT new article ---
                new_id = await conn.fetchval(
                    """
                    INSERT INTO kb.normativa (
                        codice, articolo, testo,
                        canonical_source, canonical_hash, content_hash,
                        identity_class, quality,
                        data_vigenza_da, is_current, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3,
                        'normattiva_opendata', $4::varchar(64), $4::text,
                        'BASE', 'VALID_STRONG',
                        CURRENT_DATE, true, NOW(), NOW()
                    )
                    RETURNING id::text
                    """,
                    codice,
                    articolo,
                    text,
                    new_hash,
                )
                result.inserted += 1
                if new_id:
                    result.modified_normativa_ids.append(new_id)

                    # Record initial vigenza entry
                    await _insert_vigenza(
                        conn,
                        normativa_id=new_id,
                        text=text,
                        content_hash=new_hash,
                        vigenza_inizio=vigenza_inizio,
                        vigenza_fine=vigenza_fine,
                        is_current=True,
                    )

            else:
                old_hash = existing["content_hash"] or ""
                old_id = str(existing["id"])

                if old_hash == new_hash:
                    # --- UNCHANGED ---
                    result.unchanged += 1
                    continue

                # --- UPDATE: retire old, insert new ---
                async with conn.transaction():
                    # Mark old as not current
                    await conn.execute(
                        """
                        UPDATE kb.normativa
                        SET is_current = false,
                            data_vigenza_a = CURRENT_DATE,
                            updated_at = NOW()
                        WHERE id = $1
                        """,
                        existing["id"],
                    )

                    # Mark old vigenza entry as not current
                    await conn.execute(
                        """
                        UPDATE kb.normativa_vigenza
                        SET is_current = false,
                            fine_vigore = CURRENT_DATE
                        WHERE normativa_id = $1
                          AND is_current = true
                        """,
                        existing["id"],
                    )

                    # Insert new version
                    new_id = await conn.fetchval(
                        """
                        INSERT INTO kb.normativa (
                            codice, articolo, testo,
                            canonical_source, canonical_hash, content_hash,
                            identity_class, quality,
                            data_vigenza_da, is_current,
                            previous_version_id,
                            created_at, updated_at
                        ) VALUES (
                            $1, $2, $3,
                            'normattiva_opendata', $4::varchar(64), $4::text,
                            'BASE', 'VALID_STRONG',
                            CURRENT_DATE, true,
                            $5,
                            NOW(), NOW()
                        )
                        RETURNING id::text
                        """,
                        codice,
                        articolo,
                        text,
                        new_hash,
                        existing["id"],
                    )

                    if new_id:
                        await _insert_vigenza(
                            conn,
                            normativa_id=new_id,
                            text=text,
                            content_hash=new_hash,
                            vigenza_inizio=vigenza_inizio,
                            vigenza_fine=vigenza_fine,
                            is_current=True,
                        )

                result.updated += 1
                if new_id:
                    result.modified_normativa_ids.append(new_id)

                logger.info(
                    "upsert_normativa: updated %s art.%s (old=%s new=%s)",
                    codice,
                    articolo,
                    old_id,
                    new_id,
                )

    finally:
        await conn.close()

    logger.info(
        "upsert_normativa: inserted=%d updated=%d unchanged=%d",
        result.inserted,
        result.updated,
        result.unchanged,
    )
    return asdict(result)


async def _insert_vigenza(
    conn: asyncpg.Connection,
    *,
    normativa_id: str,
    text: str,
    content_hash: str,
    vigenza_inizio: str | None,
    vigenza_fine: str | None,
    is_current: bool,
) -> None:
    """Insert a record into kb.normativa_vigenza for version tracking."""
    import uuid
    from datetime import date

    # Parse vigenza dates from YYYYMMDD format
    inizio: date | None = None
    fine: date | None = None

    if vigenza_inizio and vigenza_inizio != "99999999":
        try:
            inizio = datetime.strptime(vigenza_inizio[:8], "%Y%m%d").date()
        except (ValueError, TypeError):
            pass

    if not inizio:
        inizio = date.today()

    if vigenza_fine and vigenza_fine != "99999999":
        try:
            fine = datetime.strptime(vigenza_fine[:8], "%Y%m%d").date()
        except (ValueError, TypeError):
            pass

    await conn.execute(
        """
        INSERT INTO kb.normativa_vigenza (
            normativa_id, testo, inizio_vigore, fine_vigore,
            is_current, content_hash, opendata_version, created_at
        ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, NOW())
        ON CONFLICT (normativa_id, opendata_version) DO NOTHING
        """,
        uuid.UUID(normativa_id),
        text,
        inizio,
        fine,
        is_current,
        content_hash,
        datetime.now().strftime("%Y%m%d"),
    )
