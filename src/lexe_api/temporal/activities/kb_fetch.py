"""KB fetch activities — wired to real Normattiva OpenData implementations."""

import logging
from datetime import date
from typing import Any

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def fetch_updated_acts_activity(
    collections: list[str],
    full_sync: bool,
) -> list[dict[str, Any]]:
    """Fetch updated acts from Normattiva OpenData API."""
    from lexe_api.temporal.activities.normattiva_fetch import fetch_updated_acts

    # TODO #KB-COLLECTIONS: collections parameter is accepted but not forwarded.
    # Currently fetches ALL updated acts regardless of collections filter.
    if collections:
        logger.warning(
            "collections parameter passed (%s) but not yet implemented — "
            "fetching ALL updated acts. See TODO #KB-COLLECTIONS",
            collections,
        )

    sync_date = date.today().isoformat()
    lookback_days = 30 if full_sync else 7
    return await fetch_updated_acts(sync_date, lookback_days)


@activity.defn
async def fetch_act_articles_activity(
    act: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fetch articles for a specific act from Normattiva OpenData API."""
    from lexe_api.temporal.activities.normattiva_fetch import fetch_act_articles

    return await fetch_act_articles(act)
