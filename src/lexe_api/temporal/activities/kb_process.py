"""KB processing activities — wired to real implementations."""

import logging
from typing import Any

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def upsert_normativa_activity(
    articles: list[dict[str, Any]],
) -> int:
    """Upsert articles into kb.normativa with version tracking."""
    from lexe_api.temporal.activities.kb_upsert import upsert_normativa

    result = await upsert_normativa(articles)
    return result.get("inserted", 0) + result.get("updated", 0)


@activity.defn
async def rechunk_articles_activity(
    article_ids: list[str],
) -> int:
    """Rechunk updated articles."""
    from lexe_api.temporal.activities.chunking import rechunk_articles

    result = await rechunk_articles(article_ids)
    return result.get("chunks_created", 0)


@activity.defn
async def reembed_chunks_activity(
    article_ids: list[str],
) -> int:
    """Regenerate embeddings for chunks of updated articles."""
    from lexe_api.temporal.activities.embedding import reembed_chunks

    result = await reembed_chunks(article_ids)
    return result.get("chunks_embedded", 0)


@activity.defn
async def notify_sync_complete_activity(
    result: dict[str, Any],
) -> None:
    """Send sync completion notification."""
    from lexe_api.temporal.activities.notification import notify_sync_complete

    await notify_sync_complete(result)
