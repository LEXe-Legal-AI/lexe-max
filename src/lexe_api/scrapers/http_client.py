"""Throttled HTTP Client with retry logic.

Inspired by VisuaLexAPI patterns for resilient web scraping.
"""

import random
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from lexe_api.config import settings

logger = structlog.get_logger(__name__)


class ScrapingError(Exception):
    """Error during web scraping."""

    def __init__(self, message: str, source: str, status_code: int | None = None):
        super().__init__(message)
        self.source = source
        self.status_code = status_code


class RateLimitError(ScrapingError):
    """Rate limit exceeded."""

    pass


class ThrottledHttpClient:
    """HTTP client with throttling, retries, and user-agent rotation."""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]

    def __init__(self, source: str):
        self.source = source
        self.timeout = httpx.Timeout(settings.http_timeout_seconds)

    def _get_headers(self) -> dict[str, str]:
        """Get headers with random user agent."""
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
        }

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        reraise=True,
    )
    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make GET request with retries."""
        headers = self._get_headers()
        headers.update(kwargs.pop("headers", {}))

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.debug("HTTP GET", url=url, source=self.source)

            response = await client.get(url, headers=headers, **kwargs)

            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("Rate limited", source=self.source, url=url)
                raise RateLimitError(
                    "Rate limit exceeded",
                    source=self.source,
                    status_code=429,
                )

            # Handle other errors
            if response.status_code >= 400:
                logger.warning(
                    "HTTP error",
                    source=self.source,
                    url=url,
                    status=response.status_code,
                )
                raise ScrapingError(
                    f"HTTP {response.status_code}",
                    source=self.source,
                    status_code=response.status_code,
                )

            return response

    async def get_html(self, url: str) -> str:
        """Get HTML content as string."""
        response = await self.get(url)
        return response.text

    async def get_json(self, url: str, **kwargs: Any) -> dict:
        """Get JSON response."""
        headers = {"Accept": "application/json"}
        response = await self.get(url, headers=headers, **kwargs)
        return response.json()


class SparqlClient(ThrottledHttpClient):
    """Client for SPARQL endpoints (EUR-Lex)."""

    EURLEX_SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

    def __init__(self):
        super().__init__(source="eurlex")

    async def query(self, sparql: str) -> dict:
        """Execute SPARQL query."""
        params = {
            "query": sparql,
            "format": "application/sparql-results+json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self.EURLEX_SPARQL_ENDPOINT,
                params=params,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise ScrapingError(
                    f"SPARQL query failed: {response.status_code}",
                    source="eurlex",
                    status_code=response.status_code,
                )

            return response.json()
