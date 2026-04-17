"""Unit tests for CGUE EUR-Lex CELLAR ingestion skeleton.

Sprint 30 P1.1. Covers:
  (a) CELEX filter query built correctly
  (b) Malformed response handled gracefully
  (c) Upsert idempotent (same input -> single row, updated_at advances)
  (d) Rate limiter respects 2 req/s budget
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lexe_api.ingestion.cgue_cellar import (
    CELEX_PREFIX_PATTERN,
    DEFAULT_RATE_PER_SEC,
    CgueCellarClient,
    CgueDocument,
    RateLimiter,
    RetryableStatusError,
    build_discovery_sparql,
    parse_cellar_html,
    upsert_sentence,
)


# =============================================================================
# (a) CELEX filter query built correctly
# =============================================================================


class TestCelexFilterQuery:
    def test_query_contains_celex_regex_anchor(self) -> None:
        q = build_discovery_sparql(
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            until=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        # The glob "6*CJ*" must be translated to "^6.*CJ.*" and embedded in REGEX().
        assert "REGEX(STR(?celex)" in q
        assert "^6.*CJ.*" in q

    def test_query_uses_iso_datetime_bounds(self) -> None:
        since = datetime(2024, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
        until = datetime(2024, 3, 16, 0, 0, 0, tzinfo=timezone.utc)
        q = build_discovery_sparql(since=since, until=until)
        assert "2024-03-15T10:30:00Z" in q
        assert "2024-03-16T00:00:00Z" in q
        assert "xsd:dateTime" in q

    def test_query_has_limit_clause(self) -> None:
        q = build_discovery_sparql(
            since=datetime(2024, 1, 1, tzinfo=timezone.utc), limit=250,
        )
        assert "LIMIT 250" in q

    def test_query_default_prefix_is_6_CJ(self) -> None:
        # Sanity: module-level constant matches the plan spec.
        assert CELEX_PREFIX_PATTERN == "6*CJ*"

    def test_custom_prefix_without_regex_injection(self) -> None:
        # Only alphanumerics + "*" allowed. Anything else must raise.
        with pytest.raises(ValueError, match="disallowed chars"):
            build_discovery_sparql(
                since=datetime(2024, 1, 1, tzinfo=timezone.utc),
                celex_prefix="6*CJ*'; DROP",
            )

    @pytest.mark.asyncio
    async def test_discover_passes_query_to_http(self) -> None:
        """Client.discover builds the query and invokes the HTTP layer with it."""
        mock_response = AsyncMock()
        mock_response.json = lambda: {"results": {"bindings": []}}
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None

        # httpx.AsyncClient with `request` mocked. AsyncMock returns awaitable.
        mock_http = AsyncMock()
        mock_http.request = AsyncMock(return_value=mock_response)
        mock_http.aclose = AsyncMock()

        client = CgueCellarClient(http=mock_http, rate_limiter=RateLimiter(100.0))
        try:
            await client.discover(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
        finally:
            await client.close()

        # Inspect the URL passed to httpx.
        args, kwargs = mock_http.request.call_args
        # (method, url) positional.
        assert args[0] == "GET"
        url = args[1]
        assert "publications.europa.eu/webapi/rdf/sparql" in url
        # The query is URL-encoded; check a distinctive fragment.
        assert "query=" in url


# =============================================================================
# (b) Malformed response handled gracefully
# =============================================================================


class TestMalformedHtml:
    def test_empty_html_returns_doc_with_nulls(self) -> None:
        doc = parse_cellar_html("", celex="62023CJ0001")
        assert doc.celex == "62023CJ0001"
        assert doc.ecli is None
        assert doc.data is None
        assert doc.parti is None
        assert doc.testo_integrale is None
        assert doc.massime is None
        assert doc.lingua == "it"

    def test_truncated_html_returns_doc_without_crash(self) -> None:
        truncated = "<html><body><p class='sum-title-1'>Meta v Bundeskartellamt</p"
        doc = parse_cellar_html(truncated, celex="62022CJ0252")
        # bs4 should recover the partial <p> and get its text.
        assert doc.celex == "62022CJ0252"
        assert doc.parti is not None

    def test_missing_meta_date_tries_time_tag(self) -> None:
        html = """
        <html><head></head><body>
          <time datetime="2023-07-04">4 July 2023</time>
          <div id="text">Body text</div>
        </body></html>
        """
        doc = parse_cellar_html(html, celex="62022CJ0252")
        assert doc.data == date(2023, 7, 4)
        assert doc.testo_integrale is not None
        assert "Body text" in doc.testo_integrale

    def test_garbage_date_yields_none_not_exception(self) -> None:
        html = """
        <html><head><meta name="WKF-CASE-DATE" content="not-a-date"/></head>
        <body><time datetime="also-bad">x</time></body></html>
        """
        doc = parse_cellar_html(html, celex="62022CJ0999")
        assert doc.data is None

    def test_malformed_sparql_bindings_filtered(self) -> None:
        """Rows without a celex binding are silently dropped (no crash)."""
        import asyncio

        mock_response = AsyncMock()
        mock_response.json = lambda: {
            "results": {
                "bindings": [
                    {"celex": {"value": "62022CJ0252"}, "date": {"value": "2024-01-01"}},
                    {"ecli": {"value": "ECLI:EU:C:2024:1"}},  # no celex -> drop
                    {},  # empty row -> drop
                    {"celex": {"value": "62023CJ0010"}},
                ],
            },
        }
        mock_response.raise_for_status = lambda: None
        mock_response.status_code = 200

        mock_http = AsyncMock()
        mock_http.request = AsyncMock(return_value=mock_response)
        mock_http.aclose = AsyncMock()

        async def run() -> list[dict[str, Any]]:
            client = CgueCellarClient(http=mock_http, rate_limiter=RateLimiter(100.0))
            try:
                return await client.discover(
                    since=datetime(2024, 1, 1, tzinfo=timezone.utc),
                )
            finally:
                await client.close()

        rows = asyncio.run(run())
        celexes = [r["celex"] for r in rows]
        assert celexes == ["62022CJ0252", "62023CJ0010"]


# =============================================================================
# (c) Upsert idempotent
# =============================================================================


class TestUpsertIdempotent:
    @pytest.mark.asyncio
    async def test_upsert_uses_on_conflict_celex(self) -> None:
        """Verify the UPSERT SQL references ON CONFLICT (celex)."""
        from lexe_api.ingestion.cgue_cellar import UPSERT_SQL

        normalized = " ".join(UPSERT_SQL.split())
        assert "ON CONFLICT (celex) DO UPDATE" in normalized
        assert "kb.cgue_sentences" in normalized
        # updated_at is bumped on conflict; ingested_at is NOT touched.
        assert "updated_at      = NOW()" in UPSERT_SQL or "updated_at = NOW()" in normalized
        assert "ingested_at" not in normalized.split("DO UPDATE SET", 1)[1]

    @pytest.mark.asyncio
    async def test_upsert_called_twice_same_celex_single_row(self) -> None:
        """Two consecutive upserts with same celex invoke the same query; conn.fetchrow
        is called twice with identical bind values. The DB-level guarantee is given
        by ON CONFLICT (celex); here we assert the client contract."""
        mock_conn = AsyncMock()
        # First call -> inserted=True, second call -> inserted=False.
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                {"celex": "62022CJ0252", "inserted": True},
                {"celex": "62022CJ0252", "inserted": False},
            ],
        )

        doc = CgueDocument(
            celex="62022CJ0252",
            ecli="ECLI:EU:C:2023:537",
            data=date(2023, 7, 4),
            parti="Meta Platforms Ireland v Bundeskartellamt",
            testo_integrale="Full text...",
            massime="Dispositivo...",
            source_url="https://eur-lex.europa.eu/...",
            metadata={"eurovoc": ["4708"]},
        )

        first = await upsert_sentence(mock_conn, doc)
        second = await upsert_sentence(mock_conn, doc)

        assert first is True
        assert second is False

        # Both calls used identical bind values.
        assert mock_conn.fetchrow.call_count == 2
        call1_args = mock_conn.fetchrow.call_args_list[0].args
        call2_args = mock_conn.fetchrow.call_args_list[1].args
        # args[0] is the SQL, args[1..] are the binds.
        assert call1_args[1:] == call2_args[1:]
        # Distinctive: first bind is the celex.
        assert call1_args[1] == "62022CJ0252"


# =============================================================================
# (d) Rate limiter respects 2 req/s budget
# =============================================================================


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_five_sequential_acquires_at_2rps_take_at_least_2_seconds(self) -> None:
        """5 acquires at 2 req/s have 4 gaps of 0.5s => >= 2.0s total."""
        limiter = RateLimiter(rate_per_sec=2.0)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        # Allow tiny scheduler slack on slow runners (tolerance 100ms below target).
        assert elapsed >= 1.9, (
            f"5 acquires at 2 req/s took {elapsed:.3f}s, expected >=1.9s"
        )

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_first_acquire_immediately(self) -> None:
        limiter = RateLimiter(rate_per_sec=2.0)
        start = time.monotonic()
        await limiter.acquire()
        assert (time.monotonic() - start) < 0.05

    def test_invalid_rate_rejected(self) -> None:
        with pytest.raises(ValueError):
            RateLimiter(rate_per_sec=0)
        with pytest.raises(ValueError):
            RateLimiter(rate_per_sec=-1)

    def test_default_rate_is_2_rps(self) -> None:
        assert DEFAULT_RATE_PER_SEC == 2.0


# =============================================================================
# Bonus sanity: retry classifies 429/5xx as retryable
# =============================================================================


class TestRetryClassification:
    @pytest.mark.asyncio
    async def test_429_raises_retryable_status_error(self) -> None:
        from lexe_api.ingestion.cgue_cellar import RETRYABLE_STATUS

        assert 429 in RETRYABLE_STATUS
        assert 503 in RETRYABLE_STATUS
        # 404 is NOT retryable.
        assert 404 not in RETRYABLE_STATUS

    def test_retryable_status_error_type(self) -> None:
        # Purely a type check — class exists and is an Exception.
        err = RetryableStatusError("boom")
        assert isinstance(err, Exception)
