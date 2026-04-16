"""Sanity tests for massime_enrich Temporal activities.

Sprint 27 T9 S7.1. These tests exercise the shape of the activities
(input/output contracts), not the full resolver cascade — that's covered
by lexe-core's italgiure_resolver tests.

Run: pytest tests/kb/test_massime_enrich_activity.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_pending_massime_clamps_batch_size():
    """batch_size is clamped to [1, 5000]."""
    from lexe_api.temporal.activities.massime_enrich import (
        get_pending_massime_activity,
    )

    # Mock asyncpg.connect to avoid needing a live DB.
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []
    mock_conn.close = AsyncMock()

    with patch("asyncpg.connect", return_value=mock_conn):
        # Excessive batch_size is clamped to 5000.
        out = await get_pending_massime_activity({"batch_size": 999_999})
        assert out == []
        # The query was called with 5000 (clamped), not 999_999.
        args = mock_conn.fetch.call_args
        assert args[0][1] == 5000

    # Zero/negative clamped to 1.
    mock_conn.fetch.reset_mock()
    with patch("asyncpg.connect", return_value=mock_conn):
        out = await get_pending_massime_activity({"batch_size": 0})
        assert out == []
        args = mock_conn.fetch.call_args
        assert args[0][1] == 1


@pytest.mark.asyncio
async def test_enrich_massime_batch_empty_input():
    """Empty input returns zeroed stats without touching DB or HTTP."""
    from lexe_api.temporal.activities.massime_enrich import (
        enrich_massime_batch_activity,
    )

    stats = await enrich_massime_batch_activity({"massime": []})
    assert stats == {"resolved": 0, "unresolved": 0, "skipped": 0, "errors": 0}


@pytest.mark.asyncio
async def test_enrich_massime_batch_skips_without_numero_anno():
    """Rows lacking numero or anno are counted as skipped, no HTTP call."""
    from lexe_api.temporal.activities.massime_enrich import (
        enrich_massime_batch_activity,
    )

    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock()

    with patch("asyncpg.connect", return_value=mock_conn):
        stats = await enrich_massime_batch_activity(
            {
                "massime": [
                    {"id": "a", "numero": None, "anno": 2020, "sezione": None},
                    {"id": "b", "numero": 123, "anno": None, "sezione": None},
                ],
            },
        )

    assert stats["skipped"] == 2
    assert stats["resolved"] == 0
    assert stats["unresolved"] == 0


@pytest.mark.asyncio
async def test_enrich_massime_batch_only_persists_tier1():
    """Resolver returning tier != 1 is treated as unresolved.

    This guards against accidentally persisting ephemeral Tier 3
    (search-fallback) URLs into kb.massime.source_url.
    """
    from lexe_api.temporal.activities import massime_enrich as mod

    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value="deadbeef-0000-0000-0000-000000000000")

    # Resolver returns Tier 3 → must NOT persist (unresolved).
    mock_resolve = AsyncMock(return_value=None)  # our helper returns None for non-tier1

    with patch("asyncpg.connect", return_value=mock_conn), \
         patch.object(mod, "_call_italgiure_resolve", mock_resolve):
        stats = await mod.enrich_massime_batch_activity(
            {
                "massime": [
                    {
                        "id": "deadbeef-0000-0000-0000-000000000000",
                        "numero": 123,
                        "anno": 2020,
                        "sezione": "III",
                    },
                ],
            },
        )

    assert stats["unresolved"] == 1
    assert stats["resolved"] == 0
    # No DB update must have been called.
    mock_conn.fetchval.assert_not_called()
