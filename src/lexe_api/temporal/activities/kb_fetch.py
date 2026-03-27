"""KB fetch activities - retrieve data from OpenData sources.

These activities handle all external I/O for the KB sync workflow:
fetching updated act lists and individual article content from
dati.normattiva.it OpenData API.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def fetch_updated_acts_activity(
    collections: list[str],
    full_sync: bool,
) -> list[dict[str, Any]]:
    """Fetch list of acts updated since last sync.

    Queries the OpenData API for acts modified after the last
    sync timestamp. If full_sync is True, returns all acts
    regardless of modification date.

    Args:
        collections: List of collection codes to check (e.g. ["codici", "leggi"]).
                     Empty list means all collections.
        full_sync: If True, ignore last sync time and return all acts.

    Returns:
        List of act metadata dicts with keys: urn, title, last_modified, collection.
    """
    activity.heartbeat(f"Fetching updated acts: collections={collections}, full_sync={full_sync}")

    # TODO: Implement OpenData API integration
    # 1. Read last_sync_ts from kb.sync_metadata table
    # 2. Query dati.normattiva.it for modified acts
    # 3. Filter by collections if specified
    # 4. Return act metadata list

    logger.info(
        f"fetch_updated_acts: collections={collections}, "
        f"full_sync={full_sync} [NOT YET IMPLEMENTED]"
    )

    return []


@activity.defn
async def fetch_act_articles_activity(
    act: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fetch all updated articles for a specific act.

    Downloads article content from OpenData for the given act,
    comparing with existing KB content to identify changes.

    Args:
        act: Act metadata dict with keys: urn, title, last_modified, collection.

    Returns:
        List of article dicts with keys: id, urn, act_urn, articolo,
        testo, testo_html, is_vigente, metadata.
    """
    urn = act.get("urn", "unknown")
    activity.heartbeat(f"Fetching articles for act: {urn}")

    # TODO: Implement article fetching
    # 1. Call OpenData API for act URN
    # 2. Parse article XML/JSON response
    # 3. Diff against existing KB content (hash comparison)
    # 4. Return only changed/new articles

    logger.info(f"fetch_act_articles: urn={urn} [NOT YET IMPLEMENTED]")

    return []
