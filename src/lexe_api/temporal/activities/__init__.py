"""Temporal activities for KB nightly sync workflow."""

from lexe_api.temporal.activities.chunking import rechunk_articles
from lexe_api.temporal.activities.embedding import reembed_chunks
from lexe_api.temporal.activities.kb_upsert import upsert_normativa
from lexe_api.temporal.activities.normattiva_fetch import (
    fetch_act_articles,
    fetch_updated_acts,
)
from lexe_api.temporal.activities.notification import notify_sync_complete

__all__ = [
    "fetch_updated_acts",
    "fetch_act_articles",
    "upsert_normativa",
    "rechunk_articles",
    "reembed_chunks",
    "notify_sync_complete",
]
