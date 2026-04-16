"""Massime enrichment activities for MassimeEnrichWorkflow.

Sprint 27 T9 S7.1.

Two activities:

1. ``get_pending_massime_activity``: fetch kb.massime rows missing
   source_url (uses migration 081 index ``idx_massime_source_url_null``).
2. ``enrich_massime_batch_activity``: for each massima with numero+anno,
   call the ItalGiure resolver (via lexe-core internal endpoint) and
   persist source_url back to kb.massime.

The resolver integration is HTTP-based (cross-service): the authoritative
resolver lives in lexe-core at ``lexe_core.utils.italgiure_resolver`` and
is exposed via a thin internal endpoint ``POST /api/internal/italgiure/resolve``.
We avoid duplicating the Solr + LLM tool-loop cascade here — lexe-max
stays the KB owner, lexe-core stays the domain-logic owner.

If the internal endpoint is unreachable (feature-flag off or network split),
the activity degrades gracefully: all rows reported as unresolved, no DB
writes. The workflow retries the batch up to ``_ENRICH_RETRY.maximum_attempts``
times before marking it as failed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import asyncpg
import httpx
from temporalio import activity

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexe_kb:lexe_kb_secret@localhost:5436/lexe_kb",
)

# lexe-core internal endpoint for the ItalGiure resolver cascade.
# Keep it internal-only (no auth) — network boundary is the lexe_internal
# Docker network. For staging/prod, the hostname resolves via Docker DNS.
LEXE_CORE_BASE_URL = os.getenv(
    "LEXE_CORE_INTERNAL_URL",
    "http://lexe-core:8100",
)

_RESOLVE_TIMEOUT_S = 10.0
_RESOLVE_PER_ITEM_TIMEOUT_S = 4.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MassimaPending:
    """Minimal fields needed to call ItalGiure resolver."""

    id: str
    numero: int | None
    anno: int | None
    sezione: str | None


# ---------------------------------------------------------------------------
# Activity: get_pending_massime
# ---------------------------------------------------------------------------


@activity.defn(name="get_pending_massime_activity")
async def get_pending_massime_activity(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch next batch of massime missing source_url.

    Uses migration 081 partial index ``idx_massime_source_url_null`` for
    efficient scan even as the table grows.

    Args:
        payload: {"batch_size": int}.

    Returns:
        List of dicts ``{id, numero, anno, sezione}`` (empty when no pending).
    """
    batch_size = int(payload.get("batch_size", 500))
    batch_size = max(1, min(batch_size, 5000))  # clamp

    conn: asyncpg.Connection = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT id::text AS id, numero, anno, sezione
            FROM kb.massime
            WHERE source_url IS NULL
              AND is_active = true
              AND numero IS NOT NULL
              AND anno IS NOT NULL
            ORDER BY id
            LIMIT $1
            """,
            batch_size,
        )
    finally:
        await conn.close()

    out = [dict(r) for r in rows]
    activity.logger.info(
        "[get_pending_massime] fetched=%d batch_size=%d", len(out), batch_size,
    )
    return out


# ---------------------------------------------------------------------------
# Activity: enrich_massime_batch
# ---------------------------------------------------------------------------


async def _call_italgiure_resolve(
    client: httpx.AsyncClient,
    numero: int,
    anno: int,
    sezione: str | None,
) -> str | None:
    """Call lexe-core ItalGiure resolver endpoint for a single massima.

    Returns:
        Tier 1 URL if resolver returns a deterministic match, else None.
    """
    try:
        resp = await client.post(
            f"{LEXE_CORE_BASE_URL}/api/internal/italgiure/resolve",
            json={"numero": numero, "anno": anno, "sezione": sezione},
            timeout=_RESOLVE_PER_ITEM_TIMEOUT_S,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        tier = data.get("tier", 0)
        url = data.get("url")
        # Only accept Tier 1 (institutional / Solr-backed) — Tier 3 (search
        # fallback) is intentionally excluded to avoid persistent poisoning
        # of source_url with ephemeral search pages.
        if url and tier == 1:
            return url
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        activity.logger.debug(
            "[italgiure_resolve] err numero=%s anno=%s exc=%s",
            numero, anno, exc,
        )
    return None


@activity.defn(name="enrich_massime_batch_activity")
async def enrich_massime_batch_activity(payload: dict[str, Any]) -> dict[str, int]:
    """Enrich a batch of massime with source_url.

    For each massima with numero+anno, call ItalGiure resolver. On Tier 1
    hit, persist source_url to kb.massime. Unresolved massime are left
    with source_url = NULL (retried next week).

    Args:
        payload: {"massime": [{id, numero, anno, sezione}, ...]}.

    Returns:
        Stats dict ``{resolved, unresolved, skipped, errors}``.
    """
    items = payload.get("massime") or []
    if not items:
        return {"resolved": 0, "unresolved": 0, "skipped": 0, "errors": 0}

    stats = {"resolved": 0, "unresolved": 0, "skipped": 0, "errors": 0}

    # Split the async-ctx: httpx supports `async with`, asyncpg.connect
    # does not (returns a Connection, not a ctx mgr). Use explicit close.
    db_conn: asyncpg.Connection = await asyncpg.connect(DATABASE_URL)
    async with httpx.AsyncClient(timeout=_RESOLVE_TIMEOUT_S) as client:
        try:
            for row in items:
                numero = row.get("numero")
                anno = row.get("anno")
                mid = row.get("id")
                if numero is None or anno is None:
                    stats["skipped"] += 1
                    continue

                try:
                    url = await _call_italgiure_resolve(
                        client, int(numero), int(anno), row.get("sezione"),
                    )
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    activity.logger.warning(
                        "[enrich_massime] resolver raised id=%s err=%s", mid, exc,
                    )
                    continue

                if not url:
                    stats["unresolved"] += 1
                    continue

                # Persist. Use is_active guard so we never overwrite a
                # manually-curated source_url on a deactivated row.
                try:
                    updated = await db_conn.fetchval(
                        """
                        UPDATE kb.massime
                        SET source_url = $1
                        WHERE id = $2::uuid
                          AND source_url IS NULL
                          AND is_active = true
                        RETURNING id
                        """,
                        url, mid,
                    )
                    if updated is not None:
                        stats["resolved"] += 1
                    else:
                        # Row changed under us (race with another enrich
                        # run) — treat as unresolved for this batch.
                        stats["unresolved"] += 1
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    activity.logger.warning(
                        "[enrich_massime] db update err id=%s err=%s", mid, exc,
                    )
        finally:
            await db_conn.close()

    activity.logger.info(
        "[enrich_massime_batch] resolved=%d unresolved=%d skipped=%d errors=%d",
        stats["resolved"], stats["unresolved"], stats["skipped"], stats["errors"],
    )
    return stats
