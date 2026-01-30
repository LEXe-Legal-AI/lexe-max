"""LEXe Cache Client.

Async Redis/Valkey client for caching scraped documents.
"""

import json
from typing import Any

import redis.asyncio as redis
import structlog

from lexe_api.config import settings

logger = structlog.get_logger(__name__)


class CacheClient:
    """Async Redis/Valkey client for LEXe."""

    PREFIX = "lexe:"

    def __init__(self, url: str | None = None):
        self.url = url or settings.redis_url
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Create Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await self._client.ping()
            logger.info("Cache connected", url=self.url[:20] + "...")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Cache disconnected")

    def _key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.PREFIX}{key}"

    # =========================================================================
    # Basic Operations
    # =========================================================================

    async def get(self, key: str) -> str | None:
        """Get a string value."""
        if self._client is None:
            await self.connect()
        return await self._client.get(self._key(key))

    async def set(
        self,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Set a string value with optional TTL."""
        if self._client is None:
            await self.connect()
        ttl = ttl_seconds or (settings.cache_ttl_hours * 3600)
        await self._client.setex(self._key(key), ttl, value)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        if self._client is None:
            await self.connect()
        await self._client.delete(self._key(key))

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if self._client is None:
            await self.connect()
        return await self._client.exists(self._key(key)) > 0

    # =========================================================================
    # JSON Operations
    # =========================================================================

    async def get_json(self, key: str) -> Any | None:
        """Get and deserialize JSON value."""
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Serialize and set JSON value."""
        await self.set(key, json.dumps(value, default=str), ttl_seconds)

    # =========================================================================
    # Document Cache
    # =========================================================================

    async def get_document(self, urn: str) -> dict | None:
        """Get cached document by URN."""
        return await self.get_json(f"doc:{urn}")

    async def set_document(self, urn: str, document: dict) -> None:
        """Cache document by URN."""
        await self.set_json(f"doc:{urn}", document)

    async def invalidate_document(self, urn: str) -> None:
        """Invalidate cached document."""
        await self.delete(f"doc:{urn}")

    # =========================================================================
    # Vigenza Cache (shorter TTL for validity checks)
    # =========================================================================

    async def get_vigenza(self, urn: str) -> dict | None:
        """Get cached vigenza status."""
        return await self.get_json(f"vig:{urn}")

    async def set_vigenza(self, urn: str, vigenza: dict) -> None:
        """Cache vigenza status (1 hour TTL)."""
        await self.set_json(f"vig:{urn}", vigenza, ttl_seconds=3600)

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    async def check_rate_limit(self, source: str) -> bool:
        """Check if rate limit allows a request.

        Returns True if request is allowed, False if rate limited.
        """
        if self._client is None:
            await self.connect()

        key = self._key(f"rate:{source}")
        pipe = self._client.pipeline()

        # Get current count
        await pipe.get(key)
        # Increment
        await pipe.incr(key)
        # Set expiry if new key
        await pipe.expire(key, 60)

        results = await pipe.execute()
        current = int(results[0] or 0)

        limits = {
            "normattiva": settings.rate_limit_normattiva,
            "eurlex": settings.rate_limit_eurlex,
            "brocardi": settings.rate_limit_brocardi,
        }
        limit = limits.get(source, 30)

        return current < limit


# Global instance
cache = CacheClient()
