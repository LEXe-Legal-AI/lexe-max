"""Base Legal Tool class.

Common functionality for all legal tools including caching,
circuit breaking, and health monitoring.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TypeVar

import structlog

from lexe_api.cache import cache
from lexe_api.database import db

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitOpenError(Exception):
    """Circuit breaker is open, tool unavailable."""

    def __init__(self, tool_name: str, retry_at: datetime | None = None):
        self.tool_name = tool_name
        self.retry_at = retry_at
        msg = f"Tool {tool_name} circuit is open"
        if retry_at:
            msg += f", retry at {retry_at.isoformat()}"
        super().__init__(msg)


class BaseLegalTool(ABC):
    """Base class for legal tools.

    Provides:
    - Cache-first lookup
    - Database persistence
    - Circuit breaker pattern
    - Health monitoring integration
    """

    # Override in subclasses
    TOOL_NAME: str = "base"
    CACHE_PREFIX: str = "doc"

    def __init__(self):
        self._circuit_state: str = "closed"
        self._failure_count: int = 0

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def _fetch(self, **kwargs: Any) -> dict:
        """Fetch data from external source.

        Must be implemented by subclasses.
        Should raise ScrapingError on failure.
        """
        pass

    @abstractmethod
    def _build_cache_key(self, **kwargs: Any) -> str:
        """Build cache key from request parameters."""
        pass

    # =========================================================================
    # Public Interface
    # =========================================================================

    async def search(self, **kwargs: Any) -> dict:
        """Execute search with caching and circuit breaker.

        Flow:
        1. Check circuit breaker
        2. Check cache
        3. Check database
        4. Fetch from external source
        5. Store in database + cache
        6. Update health status
        """
        # Check circuit breaker
        await self._check_circuit()

        cache_key = self._build_cache_key(**kwargs)

        # Try cache first
        cached = await cache.get_json(f"{self.CACHE_PREFIX}:{cache_key}")
        if cached:
            logger.debug("Cache hit", tool=self.TOOL_NAME, key=cache_key)
            cached["cached"] = True
            return cached

        # Try database
        db_result = await self._find_in_db(cache_key)
        if db_result:
            # Populate cache
            await cache.set_json(f"{self.CACHE_PREFIX}:{cache_key}", db_result)
            db_result["cached"] = True
            return db_result

        # Fetch from external source
        try:
            result = await self._fetch(**kwargs)
            result["cached"] = False
            result["scraped_at"] = datetime.utcnow().isoformat()

            # Store in database and cache
            await self._store_in_db(result)
            await cache.set_json(f"{self.CACHE_PREFIX}:{cache_key}", result)

            # Record success
            await self._record_success()

            return result

        except Exception as e:
            logger.error(
                "Tool fetch failed",
                tool=self.TOOL_NAME,
                error=str(e),
                error_type=type(e).__name__,
            )
            await self._record_failure(e)
            raise

    # =========================================================================
    # Database Operations
    # =========================================================================

    async def _find_in_db(self, cache_key: str) -> dict | None:
        """Find document in database by cache key (usually URN)."""
        return await db.find_document_by_urn(cache_key)

    async def _store_in_db(self, result: dict) -> None:
        """Store result in database.

        Override in subclasses for custom storage logic.
        """
        if "urn" in result:
            await db.store_document(
                source=self.TOOL_NAME,
                urn=result.get("urn"),
                act_type=result.get("act_type"),
                act_number=result.get("act_number"),
                title=result.get("title"),
                content=result.get("text") or result.get("content"),
                is_vigente=result.get("vigente", True),
                metadata=result.get("metadata", {}),
            )

    # =========================================================================
    # Circuit Breaker
    # =========================================================================

    async def _check_circuit(self) -> None:
        """Check if circuit breaker allows requests."""
        health = await db.get_tool_health(self.TOOL_NAME)

        if health and health.get("circuit_state") == "open":
            retry_at = health.get("circuit_retry_at")
            if retry_at and retry_at > datetime.utcnow():
                raise CircuitOpenError(self.TOOL_NAME, retry_at)

            # Try half-open
            await db.update_tool_health(
                self.TOOL_NAME,
                circuit_state="half_open",
            )

    async def _record_success(self) -> None:
        """Record successful request."""
        await db.increment_tool_success(self.TOOL_NAME)

    async def _record_failure(self, error: Exception) -> None:
        """Record failed request and check circuit breaker threshold."""
        from lexe_api.config import settings

        failure_count = await db.increment_tool_failure(self.TOOL_NAME)

        if failure_count >= settings.health_failure_threshold:
            # Open circuit
            from datetime import timedelta

            retry_at = datetime.utcnow() + timedelta(minutes=5)
            await db.update_tool_health(
                self.TOOL_NAME,
                state="degraded",
                circuit_state="open",
                circuit_opened_at=datetime.utcnow(),
                circuit_retry_at=retry_at,
                last_error_message=str(error),
                last_error_type=type(error).__name__,
            )
            logger.warning(
                "Circuit opened",
                tool=self.TOOL_NAME,
                failures=failure_count,
                retry_at=retry_at,
            )
