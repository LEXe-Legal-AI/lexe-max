"""Notification activity for KB nightly sync.

Emits a structured log summary at the end of the sync workflow.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog
from temporalio import activity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


@activity.defn
async def notify_sync_complete(summary: dict) -> None:
    """Log a structured summary of the completed KB nightly sync.

    Args:
        summary: Dict with keys:
            - date: ISO date of the sync run
            - acts_checked: Number of acts scanned from Normattiva
            - articles_updated: Number of KB articles inserted or updated
            - chunks_created: Number of new chunks generated
            - embeddings_generated: Number of embeddings produced
            - duration_s: Total workflow duration in seconds
            - errors: List of error messages (if any)
    """
    slog: Any = structlog.get_logger("kb.sync")

    sync_date = summary.get("date", "unknown")
    acts_checked = summary.get("acts_checked", 0)
    articles_updated = summary.get("articles_updated", 0)
    chunks_created = summary.get("chunks_created", 0)
    embeddings_generated = summary.get("embeddings_generated", 0)
    duration_s = summary.get("duration_s", 0)
    errors = summary.get("errors", [])

    status = "SUCCESS" if not errors else "COMPLETED_WITH_ERRORS"

    slog.info(
        "kb_nightly_sync_complete",
        status=status,
        sync_date=sync_date,
        acts_checked=acts_checked,
        articles_updated=articles_updated,
        chunks_created=chunks_created,
        embeddings_generated=embeddings_generated,
        duration_s=round(duration_s, 2),
        error_count=len(errors),
        errors=errors[:10] if errors else [],  # cap at 10 to avoid log bloat
    )

    # Also emit via stdlib logger for systems that don't use structlog
    logger.info(
        "KB nightly sync %s: date=%s acts=%d articles=%d chunks=%d embeddings=%d duration=%.1fs errors=%d",
        status,
        sync_date,
        acts_checked,
        articles_updated,
        chunks_created,
        embeddings_generated,
        duration_s,
        len(errors),
    )
